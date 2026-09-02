import Foundation
import MLX
import MLXNN
import ModelRunnerProtocol

/// Native MLX implementation of the decoder-only Voxtral 4B text-to-speech
/// generation path. The actor serializes generation because the language
/// model's KV cache and MLX random state are request-local mutable state.
public actor VoxtralTTSSynthesizer: LocalSpeechSynthesizing {
  public nonisolated let modelPath: String
  public nonisolated let servedModelName: String
  public nonisolated let engine: ModelEngine
  public nonisolated let voiceCatalog: VoxtralVoiceCatalog
  public nonisolated let supportedAudioFormats: Set<LocalSpeechAudioFormat> = [.wav, .pcm]
  public nonisolated let supportsInstructions = false
  public nonisolated let supportedSpeedRange: ClosedRange<Double> = 1...1

  private let modelDirectory: URL
  private let configuration: VoxtralTTSConfiguration
  private let tokenizer: VoxtralTekkenTokenizer
  private let languageModel: VoxtralLanguageBackbone
  private let acousticTransformer: VoxtralAcousticTransformer
  private let audioCodebookEmbeddings: VoxtralAudioCodeEmbeddings
  private let audioDecoder: VoxtralAudioDecoder
  private let device: Device
  private let maximumFrames: Int
  private let randomSeed: UInt64
  private let verbose: Bool

  private var cachedVoiceEmbeddings: [String: MLXArray] = [:]
  private var isGenerating = false

  public init(
    modelPath: String,
    servedModelName: String? = nil,
    engine requestedEngine: ModelEngine = .auto,
    maximumFrames: Int = 512,
    randomSeed: UInt64 = 42,
    verbose: Bool = false
  ) async throws {
    guard maximumFrames > 0 else {
      throw VoxtralTTSError.invalidConfiguration("maximumFrames must be greater than zero")
    }

    let expandedPath = NSString(string: modelPath).expandingTildeInPath
    let modelDirectory = URL(fileURLWithPath: expandedPath, isDirectory: true)
      .standardizedFileURL
      .resolvingSymlinksInPath()
    try Self.validateModelDirectory(modelDirectory)

    let configuration = try VoxtralTTSConfiguration(modelDirectory: modelDirectory)
    let tokenizer = try VoxtralTekkenTokenizer(modelDirectory: modelDirectory)
    guard let voiceCatalog = try VoxtralVoiceCatalog(modelDirectory: modelDirectory.path) else {
      throw VoxtralTTSError.invalidConfiguration(
        "config.json does not contain a valid preset voice catalog"
      )
    }

    let engine = try requestedEngine.resolve()
    let device: Device = engine == .cpu ? .cpu : .gpu
    let resourceLimits = try MLXResourceLimits.resolve(
      for: engine,
      physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory
    )

    let modules = try Device.withDefaultDevice(device) {
      print(
        "MLX resource guard: memory=\(resourceLimits.memoryLimitBytes) bytes "
          + "cache=\(resourceLimits.cacheLimitBytes) bytes"
      )
      try MLXResourceGuard.apply(resourceLimits)

      let languageModel = VoxtralLanguageBackbone(configuration: configuration)
      let acousticTransformer = VoxtralAcousticTransformer(configuration: configuration)
      let audioCodebookEmbeddings = VoxtralAudioCodeEmbeddings(configuration: configuration)
      let audioDecoder = VoxtralAudioDecoder(configuration: configuration)

      try Self.loadModelWeights(
        from: modelDirectory,
        languageModel: languageModel,
        acousticTransformer: acousticTransformer,
        audioCodebookEmbeddings: audioCodebookEmbeddings,
        audioDecoder: audioDecoder
      )
      eval(languageModel, acousticTransformer, audioCodebookEmbeddings, audioDecoder)
      Memory.clearCache()
      return (languageModel, acousticTransformer, audioCodebookEmbeddings, audioDecoder)
    }

    self.modelPath = modelDirectory.path
    self.servedModelName =
      servedModelName?
      .trimmingCharacters(in: .whitespacesAndNewlines)
      .nilIfEmpty ?? modelDirectory.lastPathComponent
    self.engine = engine
    self.voiceCatalog = voiceCatalog
    self.modelDirectory = modelDirectory
    self.configuration = configuration
    self.tokenizer = tokenizer
    self.languageModel = modules.0
    self.acousticTransformer = modules.1
    self.audioCodebookEmbeddings = modules.2
    self.audioDecoder = modules.3
    self.device = device
    self.maximumFrames = maximumFrames
    self.randomSeed = randomSeed
    self.verbose = verbose
  }

  public func stream(
    request: LocalSpeechSynthesisRequest
  ) async -> AsyncThrowingStream<LocalSpeechSynthesisEvent, Error> {
    AsyncThrowingStream { continuation in
      let generationTask = Task {
        do {
          let result = try synthesize(request: request)
          continuation.yield(.audio(result.audio))
          continuation.yield(.completed(result.usage))
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
      continuation.onTermination = { _ in generationTask.cancel() }
    }
  }

  private func synthesize(
    request: LocalSpeechSynthesisRequest
  ) throws -> (audio: Data, usage: LocalSpeechUsage) {
    guard !isGenerating else { throw VoxtralTTSError.busy }
    isGenerating = true
    defer { isGenerating = false }

    guard supportedAudioFormats.contains(request.format) else {
      throw VoxtralTTSError.unsupportedAudioFormat(request.format.rawValue)
    }
    guard request.speed == 1 else {
      throw VoxtralTTSError.generationFailed("Voxtral currently supports only speed 1.0")
    }
    guard request.instructions?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false
    else {
      throw VoxtralTTSError.generationFailed("Voxtral does not currently support instructions")
    }
    guard let voice = voiceCatalog.voice(id: request.voiceID) else {
      throw VoxtralTTSError.voiceNotFound(request.voiceID)
    }

    let text = Self.normalizedSpeechText(request.input)
    guard !text.isEmpty else {
      throw VoxtralTTSError.generationFailed("speech input must not be empty")
    }

    return try Device.withDefaultDevice(device) {
      try Task.checkCancellation()
      MLXRandom.seed(randomSeed)

      let voiceEmbedding = try loadVoiceEmbedding(for: voice)
      let promptTokenIDs = try tokenizer.speechPrompt(
        text: text,
        voice: voice.id,
        configuration: configuration,
        voiceEmbeddingRows: voiceEmbedding.dim(0)
      )
      // The release generation path advances once with the ordinary AUDIO
      // token after prefill. That decode position produces the hidden state
      // for frame one; every later frame consumes one feedback embedding.
      let availableFrameCount =
        configuration.maximumPositionEmbeddings - promptTokenIDs.count
      guard availableFrameCount > 0 else {
        throw VoxtralTTSError.generationFailed(
          "the speech prompt exceeds the model's context window"
        )
      }
      let frameLimit = min(maximumFrames, availableFrameCount)

      let voiceRowCount = voiceEmbedding.dim(0)
      let prefixTokenIDs = Array(promptTokenIDs.prefix(2))
      let suffixTokenIDs = Array(promptTokenIDs.dropFirst(2 + voiceRowCount))
      guard prefixTokenIDs.count == 2, !suffixTokenIDs.isEmpty else {
        throw VoxtralTTSError.generationFailed("the generated speech prompt is malformed")
      }

      let prefixEmbedding = languageModel.embed(Self.batchedTokenArray(prefixTokenIDs))
      let suffixEmbedding = languageModel.embed(Self.batchedTokenArray(suffixTokenIDs))
      let promptEmbedding = concatenated(
        [
          prefixEmbedding,
          voiceEmbedding.expandedDimensions(axis: 0),
          suffixEmbedding,
        ],
        axis: 1
      )

      let cache = languageModel.makeCache()
      let prefillHidden = languageModel.forward(
        embeddings: promptEmbedding,
        cache: cache
      )
      eval(prefillHidden)

      let audioTokenEmbedding = languageModel.embed(
        Self.batchedTokenArray([configuration.audioTokenID])
      )
      var frameHidden = languageModel.forward(
        embeddings: audioTokenEmbedding,
        cache: cache
      )
      eval(frameHidden)

      var frameCodes: [MLXArray] = []
      frameCodes.reserveCapacity(frameLimit)
      let generationStart = ContinuousClock.now

      for frameIndex in 0..<frameLimit {
        try Task.checkCancellation()
        let codes = acousticTransformer.decodeOneFrame(hidden: frameHidden)
        eval(codes)

        let semanticCode = codes[0, 0].item(Int32.self)
        if verbose {
          let acoustic = codes[0, 1...]
          let acousticMinimum = acoustic.min().item(Int32.self)
          let acousticMaximum = acoustic.max().item(Int32.self)
          print(
            "[verbose] voxtral frame=\(frameIndex + 1) semantic_code=\(semanticCode) "
              + "acoustic_codes=\(acousticMinimum)...\(acousticMaximum)"
          )
        }
        if semanticCode <= 1 { break }

        frameCodes.append(codes.expandedDimensions(axis: 1))
        if frameIndex + 1 == frameLimit { break }

        // Subsequent frames feed back the sum of all 37 generated
        // codebook embeddings instead of another text-vocabulary token.
        let feedbackEmbedding = audioCodebookEmbeddings.feedbackEmbedding(codes: codes)
        frameHidden = languageModel.forward(
          embeddings: feedbackEmbedding,
          cache: cache
        )
        eval(frameHidden)

        if (frameIndex + 1).isMultiple(of: 32) {
          Memory.clearCache()
        }
      }

      guard !frameCodes.isEmpty else {
        throw VoxtralTTSError.generationFailed(
          "the model emitted end-of-audio before producing a frame"
        )
      }

      let codes = concatenated(frameCodes, axis: 1)
      let waveform = audioDecoder.decode(codes: codes)
      eval(waveform)
      let samples = waveform.asType(.float32).asArray(Float.self)
      if verbose {
        let elapsed = generationStart.duration(to: .now)
        let seconds = Self.seconds(elapsed)
        let peak = samples.reduce(Float(0)) { max($0, abs($1)) }
        let sumOfSquares = samples.reduce(Double(0)) {
          $0 + Double($1) * Double($1)
        }
        let rootMeanSquare =
          samples.isEmpty
          ? 0
          : Float((sumOfSquares / Double(samples.count)).squareRoot())
        let framesPerSecond = seconds > 0 ? Double(frameCodes.count) / seconds : 0
        print(
          "[verbose] voxtral complete frames=\(frameCodes.count) samples=\(samples.count) "
            + "elapsed=\(String(format: "%.3fs", seconds)) "
            + "frames_per_second=\(String(format: "%.2f", framesPerSecond)) "
            + "peak=\(String(format: "%.6f", peak)) "
            + "rms=\(String(format: "%.6f", rootMeanSquare))"
        )
      }
      let audio = try VoxtralAudioEncoding.encode(
        samples: samples,
        sampleRate: configuration.sampleRate,
        format: request.format,
        pcmEncoding: request.pcmEncoding
      )
      Memory.clearCache()
      return (
        audio,
        LocalSpeechUsage(
          promptTokens: promptTokenIDs.count,
          completionTokens: frameCodes.count
        )
      )
    }
  }

  private func loadVoiceEmbedding(for voice: VoxtralPresetVoice) throws -> MLXArray {
    if let cached = cachedVoiceEmbeddings[voice.id] { return cached }

    let url =
      modelDirectory
      .appendingPathComponent("voice_embedding", isDirectory: true)
      .appendingPathComponent("\(voice.id).safetensors")
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw VoxtralTTSError.missingFile(url.path)
    }

    // Safetensor Load nodes are CPU-only in MLX. Once materialized, their
    // unified-memory buffers can be consumed by the Metal execution stream.
    let arrays = try loadArrays(url: url, stream: .cpu)
    guard arrays.count == 1, let embedding = arrays["embedding"] else {
      let keys = arrays.keys.sorted().joined(separator: ", ")
      throw VoxtralTTSError.invalidWeights(
        "voice \(voice.id) must contain only 'embedding'; found [\(keys)]"
      )
    }
    guard embedding.ndim == 2,
      embedding.dim(1) == configuration.dimension,
      embedding.dim(0) == tokenizer.voiceTokenCounts[voice.id]
    else {
      throw VoxtralTTSError.invalidWeights(
        "voice \(voice.id) embedding has shape \(embedding.shape); expected "
          + "[\(tokenizer.voiceTokenCounts[voice.id] ?? -1), \(configuration.dimension)]"
      )
    }
    eval(embedding)
    cachedVoiceEmbeddings[voice.id] = embedding
    return embedding
  }

  private static func validateModelDirectory(_ directory: URL) throws {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory),
      isDirectory.boolValue
    else {
      throw VoxtralTTSError.missingFile(directory.path)
    }

    for fileName in ["config.json", "tekken.json"] {
      let url = directory.appendingPathComponent(fileName)
      guard FileManager.default.fileExists(atPath: url.path) else {
        throw VoxtralTTSError.missingFile(url.path)
      }
    }

    let voiceDirectory = directory.appendingPathComponent("voice_embedding", isDirectory: true)
    var isVoiceDirectory: ObjCBool = false
    guard
      FileManager.default.fileExists(
        atPath: voiceDirectory.path,
        isDirectory: &isVoiceDirectory
      ), isVoiceDirectory.boolValue
    else {
      throw VoxtralTTSError.missingFile(voiceDirectory.path)
    }

    guard try !modelWeightFiles(in: directory).isEmpty else {
      throw VoxtralTTSError.invalidWeights("the model directory has no model*.safetensors files")
    }
  }

  private static func loadModelWeights(
    from directory: URL,
    languageModel: VoxtralLanguageBackbone,
    acousticTransformer: VoxtralAcousticTransformer,
    audioCodebookEmbeddings: VoxtralAudioCodeEmbeddings,
    audioDecoder: VoxtralAudioDecoder
  ) throws {
    var languageWeights: [String: MLXArray] = [:]
    var acousticWeights: [String: MLXArray] = [:]
    var codebookWeights: [String: MLXArray] = [:]
    var decoderWeights: [String: MLXArray] = [:]

    for url in try modelWeightFiles(in: directory) {
      // Safetensor Load nodes themselves must be evaluated on CPU.
      let shard = try loadArrays(url: url, stream: .cpu)
      for (key, value) in shard {
        if let relativeKey = key.removingPrefix("language_model.model.model.") {
          try insertWeight(value, key: relativeKey, originalKey: key, into: &languageWeights)
        } else if let relativeKey = key.removingPrefix("acoustic_transformer.") {
          try insertWeight(value, key: relativeKey, originalKey: key, into: &acousticWeights)
        } else if let relativeKey = key.removingPrefix("audio_codebook_embeddings.") {
          try insertWeight(value, key: relativeKey, originalKey: key, into: &codebookWeights)
        } else if let relativeKey = key.removingPrefix("audio_tokenizer.") {
          try insertWeight(value, key: relativeKey, originalKey: key, into: &decoderWeights)
        } else {
          throw VoxtralTTSError.invalidWeights("unhandled checkpoint key: \(key)")
        }
      }
    }

    try updateStrictly(languageModel, weights: languageWeights, name: "language_model")
    try updateStrictly(
      acousticTransformer,
      weights: acousticWeights,
      name: "acoustic_transformer"
    )
    try updateStrictly(
      audioCodebookEmbeddings,
      weights: codebookWeights,
      name: "audio_codebook_embeddings"
    )
    try updateStrictly(audioDecoder, weights: decoderWeights, name: "audio_tokenizer")
  }

  private static func insertWeight(
    _ value: MLXArray,
    key: String,
    originalKey: String,
    into weights: inout [String: MLXArray]
  ) throws {
    guard !key.isEmpty else {
      throw VoxtralTTSError.invalidWeights("empty module key derived from \(originalKey)")
    }
    guard weights.updateValue(value, forKey: key) == nil else {
      throw VoxtralTTSError.invalidWeights("duplicate checkpoint key: \(originalKey)")
    }
  }

  private static func updateStrictly(
    _ module: Module,
    weights: [String: MLXArray],
    name: String
  ) throws {
    guard !weights.isEmpty else {
      throw VoxtralTTSError.invalidWeights("no weights were found for \(name)")
    }
    do {
      try module.update(
        parameters: ModuleParameters.unflattened(weights),
        verify: .all
      )
    } catch {
      throw VoxtralTTSError.invalidWeights(
        "could not load \(name) strictly: \(error.localizedDescription)"
      )
    }
  }

  private static func modelWeightFiles(in directory: URL) throws -> [URL] {
    try FileManager.default.contentsOfDirectory(
      at: directory,
      includingPropertiesForKeys: nil,
      options: [.skipsHiddenFiles]
    )
    .filter {
      $0.pathExtension == "safetensors"
        && $0.deletingPathExtension().lastPathComponent.hasPrefix("model")
    }
    .sorted { $0.lastPathComponent < $1.lastPathComponent }
  }

  private static func batchedTokenArray(_ tokenIDs: [Int]) -> MLXArray {
    MLXArray(tokenIDs.map(Int32.init)).reshaped(1, tokenIDs.count)
  }

  private static func normalizedSpeechText(_ input: String) -> String {
    var text =
      input
      .split(whereSeparator: { $0.isWhitespace })
      .joined(separator: " ")
    guard !text.isEmpty else { return text }

    if let finalCharacter = text.last,
      ![".", "!", "?", ";", ":", "…"].contains(finalCharacter)
    {
      text.append(".")
    }
    return text
  }

  private static func seconds(_ duration: Duration) -> Double {
    let components = duration.components
    return Double(components.seconds)
      + Double(components.attoseconds) / 1_000_000_000_000_000_000
  }
}

extension String {
  fileprivate var nilIfEmpty: String? { isEmpty ? nil : self }

  fileprivate func removingPrefix(_ prefix: String) -> String? {
    guard hasPrefix(prefix) else { return nil }
    return String(dropFirst(prefix.count))
  }
}
