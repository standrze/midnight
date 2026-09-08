import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import ModelRunnerProtocol
import Tokenizers

public enum ModelInspectionError: LocalizedError, Equatable, Sendable {
  case invalidRequest(String)
  case unsupported(String)
  case invalidGraph(String)

  public var errorDescription: String? {
    switch self {
    case .invalidRequest(let message), .unsupported(let message), .invalidGraph(let message): message
    }
  }
}

struct ModelInspectionMetadata: Sendable {
  var modelType = "unknown"
  var hiddenSize: Int?
  var attentionHeads: Int?
  var kvHeads: Int?
  var expertCount: Int?
  var expertsPerToken: Int?
  var configuredContextLength: Int?
  var activation: String?
  var layerTypes: [String] = []
  var mlpLayerTypes: [String] = []

  static func load(modelPath: String) -> Self {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: modelPath).appendingPathComponent("config.json"))
    else { return Self() }
    return decode(data)
  }

  static func decode(_ data: Data) -> Self {
    guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return Self() }
    let config = (root["text_config"] as? [String: Any]) ?? root
    return Self(
      modelType: (config["model_type"] as? String) ?? (root["model_type"] as? String) ?? "unknown",
      hiddenSize: (config["hidden_size"] as? Int) ?? (config["n_embd"] as? Int),
      attentionHeads: (config["num_attention_heads"] as? Int) ?? (config["n_head"] as? Int),
      kvHeads: config["num_key_value_heads"] as? Int,
      expertCount: (config["num_local_experts"] as? Int) ?? (config["num_experts"] as? Int),
      expertsPerToken: config["num_experts_per_tok"] as? Int,
      configuredContextLength: (config["max_position_embeddings"] as? Int) ?? (config["n_positions"] as? Int),
      activation: (config["hidden_act"] as? String) ?? (config["activation_function"] as? String),
      layerTypes: (config["layer_types"] as? [String]) ?? [],
      mlpLayerTypes: (config["mlp_layer_types"] as? [String]) ?? [])
  }
}

struct InspectionLayerAddress: Equatable {
  var index: Int
  var path: String

  static func parse(_ path: String) -> Self? {
    let parts = path.split(separator: ".").map(String.init)
    guard parts.count >= 2 else { return nil }
    for index in 0..<(parts.count - 1) {
      if ["layers", "h", "blocks"].contains(parts[index]), let number = Int(parts[index + 1]), number >= 0 {
        return Self(index: number, path: parts[0...(index + 1)].joined(separator: "."))
      }
    }
    return nil
  }
}

enum ModelInspection {
  static let promptTokenLimit = 256
  static let outputTokenLimit = 64
  static let channelBinCount = 16
  static let prefillChunkSize = 16
  static let measurement = "Measured pre-normalization residual activations: layer input and after attention. RMS and maximum absolute value summarize each token's hidden vector; channels are 16 contiguous channel-group RMS values. These are not attention probabilities, feature meanings, or explanations."

  static func maximumTokens(_ request: InspectorTraceRequest) throws -> Int {
    guard !request.question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw ModelInspectionError.invalidRequest("Enter a question to inspect.")
    }
    // Bound work before chat-template/tokenizer allocation as well as after it.
    guard request.question.utf8.count <= 16_384 else {
      throw ModelInspectionError.invalidRequest("Inspector questions must be shorter than 16 KiB.")
    }
    let result = request.maxTokens ?? 32
    guard (1...outputTokenLimit).contains(result) else {
      throw ModelInspectionError.invalidRequest("Inspector maxTokens must be between 1 and \(outputTokenLimit).")
    }
    return result
  }

  static func descriptor(model: any LanguageModel, id: String,
                         metadata: ModelInspectionMetadata, contextLength: Int) -> InspectorModel {
    let leaves = model.leafModules().flattened().sorted { $0.0 < $1.0 }
    let parameters = model.parameters().flattened()
    var grouped: [String: [(String, Module)]] = [:]
    for (path, module) in leaves {
      guard let address = InspectionLayerAddress.parse(path) else { continue }
      grouped[address.path, default: []].append((path, module))
    }
    let layers = grouped.keys.sorted {
      let left = InspectionLayerAddress.parse($0)!, right = InspectionLayerAddress.parse($1)!
      return left.index == right.index ? left.path < right.path : left.index < right.index
    }.map { path in
      let index = InspectionLayerAddress.parse(path)!.index
      let layerParameters = parameters.filter { $0.0.hasPrefix(path + ".") }
      let modules = grouped[path, default: []].map { modulePath, module in
        let tensors = module.parameters().flattened().sorted { $0.0 < $1.0 }
        let quantized = module as? any Quantized
        return InspectorModule(path: modulePath, kind: String(describing: type(of: module)),
          storedElementCount: tensors.reduce(0) { $0 + $1.1.size },
          parameters: tensors.map {
            InspectorParameter(name: $0.0, shape: $0.1.shape, dtype: String(describing: $0.1.dtype))
          }, quantization: quantized.map {
            InspectorQuantization(format: $0.mode.rawValue, bits: $0.bits, groupSize: $0.groupSize)
          })
      }
      var kind: [String] = []
      if metadata.layerTypes.indices.contains(index) { kind.append(metadata.layerTypes[index]) }
      if metadata.mlpLayerTypes.indices.contains(index) { kind.append(metadata.mlpLayerTypes[index]) }
      if kind.isEmpty { kind = ["transformer"] }
      return InspectorLayer(index: index, path: path, kind: kind.joined(separator: " · "),
        storedElementCount: layerParameters.reduce(0) { $0 + $1.1.size },
        weightBytes: layerParameters.reduce(0) { $0 + $1.1.nbytes }, modules: modules)
    }
    let supportReason = traceSupportReason(model: model, leaves: leaves, layers: layers)
    let inferredWidth = leaves.first { $0.0.hasSuffix(".input_layernorm") }
      .flatMap { ($0.1 as? RMSNorm)?.weight.dim(0) }
    return InspectorModel(id: id, modelType: metadata.modelType, runtimeType: String(describing: type(of: model)),
      layerCount: layers.count, hiddenSize: inferredWidth ?? metadata.hiddenSize,
      storedElementCount: parameters.reduce(0) { $0 + $1.1.size },
      weightBytes: parameters.reduce(0) { $0 + $1.1.nbytes }, traceSupported: supportReason == nil,
      traceReason: supportReason, layers: layers, attentionHeads: metadata.attentionHeads,
      kvHeads: metadata.kvHeads, expertCount: metadata.expertCount,
      expertsPerToken: metadata.expertsPerToken, contextLength: contextLength,
      configuredContextLength: metadata.configuredContextLength, activation: model is LFM2Model ? "SwiGLU (native LFM2 runtime)" : metadata.activation)
  }

  private static func traceSupportReason(model: any LanguageModel, leaves: [(String, Module)],
                                         layers: [InspectorLayer]) -> String? {
    guard model is LagunaModel || model is Mistral3TextModel || model is GPTOSSModel else {
      return "Live activation capture currently supports native Laguna, Mistral3/Ministral3, and GPT-OSS runtimes. Loaded architecture inspection remains available."
    }
    guard !layers.isEmpty, Set(layers.map(\.index)) == Set(0..<layers.count) else {
      return "This runtime does not expose a complete, contiguous decoder layer stack."
    }
    let norms = Dictionary(uniqueKeysWithValues: leaves.compactMap { path, module -> (String, RMSNorm)? in
      guard let norm = module as? RMSNorm else { return nil }
      return (path, norm)
    })
    guard layers.allSatisfy({ norms[$0.path + ".input_layernorm"] != nil
      && norms[$0.path + ".post_attention_layernorm"] != nil }) else {
      return "This runtime does not expose both supported residual observation sites for every layer."
    }
    return nil
  }

  /// All MLX references stay inside the model container's serialized closure.
  /// This uses a fresh cache, synchronous greedy decode, and a bounded scratch graph.
  static func trace(context: ModelContext, descriptor: InspectorModel, question: String,
                    promptTokens: [Int], maximumTokens: Int) throws -> InspectorTrace {
    guard descriptor.traceSupported else {
      throw ModelInspectionError.unsupported(descriptor.traceReason ?? "Activation capture is unavailable.")
    }
    guard !promptTokens.isEmpty, promptTokens.count <= promptTokenLimit,
      (1...outputTokenLimit).contains(maximumTokens) else {
      throw ModelInspectionError.invalidRequest("The rendered inspector prompt must contain 1...\(promptTokenLimit) tokens. Shorten the question.")
    }
    let model = context.model
    let recorder = InspectionActivationRecorder()
    let installation = try InspectionNormInstallation(model: model, layers: descriptor.layers, recorder: recorder)
    // Installation also restores partially installed hooks if an update throws.
    defer { installation.restore() }
    defer { StreamOrDevice.default.stream.synchronize() }
    let cache = try model.newCache(parameters: GenerateParameters(maxTokens: maximumTokens, temperature: 0))
    var stopTokens = context.configuration.eosTokenIds
    if let eos = context.tokenizer.eosTokenId { stopTokens.insert(eos) }
    if let unknown = context.tokenizer.unknownTokenId { stopTokens.insert(unknown) }
    for text in context.configuration.extraEOSTokens {
      if let token = context.tokenizer.convertTokenToId(text) { stopTokens.insert(token) }
    }
    stopTokens.formUnion(semanticStopTokenIDs(tokenizer: context.tokenizer, isGPTOSS: model is GPTOSSModel))
    var predictions: [InspectorPrediction] = []
    var nextToken: Int?
    for start in stride(from: 0, to: promptTokens.count, by: prefillChunkSize) {
      try Task.checkCancellation()
      let end = min(start + prefillChunkSize, promptTokens.count)
      nextToken = try forward(model: model, tokens: Array(promptTokens[start..<end]), cache: cache,
                             recorder: recorder, start: start, tokenizer: context.tokenizer, predictions: &predictions)
    }
    var generated: [Int] = []
    var stopReason = "length"
    for _ in 0..<maximumTokens {
      try Task.checkCancellation()
      guard let token = nextToken else { throw ModelInspectionError.invalidGraph("The model returned no next token.") }
      if stopTokens.contains(token) { stopReason = "stop"; break }
      generated.append(token)
      // Evaluate even the last emitted token so each displayed answer token has
      // real input activations. The extra prediction is discarded at the limit.
      nextToken = try forward(model: model, tokens: [token], cache: cache, recorder: recorder,
                             start: promptTokens.count + generated.count - 1, tokenizer: context.tokenizer, predictions: &predictions)
    }
    let allTokens = promptTokens + generated
    let tokens = allTokens.enumerated().map { position, token in
      InspectorToken(id: token, text: context.tokenizer.decode(tokenIds: [token]), position: position,
                     phase: position < promptTokens.count ? "prompt" : "answer")
    }
    return InspectorTrace(model: descriptor.id, question: question,
      answer: context.tokenizer.decode(tokenIds: generated, skipSpecialTokens: false),
      measurement: measurement, tokens: tokens, layers: try recorder.finalize(tokenCount: allTokens.count),
      stopReason: stopReason, maxTokens: maximumTokens,
      promptTokenCount: promptTokens.count, generatedTokenCount: generated.count, outputKind: "raw", predictions: predictions)
  }

  /// Converted Harmony checkpoints may omit generation_config.json. Confirm
  /// each spelling with the tokenizer so an unknown-token fallback never
  /// invents a stop ID; support both published and renamed control spellings.
  static func semanticStopTokenIDs(tokenizer: any MLXLMCommon.Tokenizer, isGPTOSS: Bool) -> Set<Int> {
    guard isGPTOSS else { return [] }
    return Set(["<|call|>", "<|return|>", "<|ghissue|>", "<|fim_suffix|>"].compactMap { spelling in
      guard let id = tokenizer.convertTokenToId(spelling), id != tokenizer.unknownTokenId,
        tokenizer.convertIdToToken(id) == spelling
          || tokenizer.decode(tokenIds: [id], skipSpecialTokens: false) == spelling else { return nil }
      return id
    })
  }

  private static func forward(model: any LanguageModel, tokens: [Int], cache: [KVCache],
                              recorder: InspectionActivationRecorder, start: Int, tokenizer: any MLXLMCommon.Tokenizer,
                              predictions: inout [InspectorPrediction]) throws -> Int {
    try autoreleasepool {
      recorder.startPosition = start
      let logits = model(MLXArray(tokens).reshaped(1, tokens.count), cache: cache)
      guard logits.ndim == 3, logits.dim(0) == 1, logits.dim(1) > 0 else {
        throw ModelInspectionError.invalidGraph("The model returned an unsupported logits shape.")
      }
      let last = logits[0, -1, 0...].asType(.float32)
      let valid = MLX.any(MLX.isFinite(last))
      let next = MLX.argMax(MLX.which(MLX.isFinite(last), last, MLXArray(-Float.infinity)))
      try recorder.flush(evaluating: [next, valid] + cache.flatMap(\.state))
      guard valid.item(Bool.self) else {
        throw ModelInspectionError.invalidGraph("The model produced non-finite logits during inspection.")
      }
      let probabilities = MLX.softmax(MLX.which(MLX.isFinite(last), last, MLXArray(-Float.infinity))).asArray(Float.self)
      var top: [(Int, Float)] = []
      for (id, probability) in probabilities.enumerated() {
        if top.count < 5 || probability > top.last!.1 {
          top.append((id, probability))
          top.sort { $0.1 == $1.1 ? $0.0 < $1.0 : $0.1 > $1.1 }
          if top.count > 5 { top.removeLast() }
        }
      }
      predictions.append(InspectorPrediction(afterTokenIndex: start + tokens.count - 1,
        candidates: top.map { InspectorTokenProbability(id: $0.0,
          text: tokenizer.decode(tokenIds: [$0.0], skipSpecialTokens: false), probability: $0.1) }))
      return next.item(Int.self)
    }
  }
}

/// Wraps the original normalization operation exactly; never reinitializes or
/// copies trained weights. Wrappers are installed only during an inspection.
final class InspectionRecordingNorm: RMSNorm {
  private let original: RMSNorm
  private let path: String
  private let recorder: InspectionActivationRecorder

  init(original: RMSNorm, path: String, recorder: InspectionActivationRecorder) {
    self.original = original; self.path = path; self.recorder = recorder
    super.init(dimensions: original.weight.dim(0), eps: original.eps)
  }

  override func callAsFunction(_ input: MLXArray) -> MLXArray {
    recorder.observe(path: path, input: input)
    return original(input)
  }
}

final class InspectionNormInstallation {
  private let model: Module
  private var originals: [(String, Module)] = []

  init(model: Module, layers: [InspectorLayer], recorder: InspectionActivationRecorder) throws {
    self.model = model
    let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
    for layer in layers {
      for (suffix, site) in [("input_layernorm", "layer_input"), ("post_attention_layernorm", "after_attention")] {
        let path = layer.path + "." + suffix
        guard let norm = leaves[path] as? RMSNorm else {
          throw ModelInspectionError.invalidGraph("Missing supported normalization at \(path).")
        }
        originals.append((path, norm))
        recorder.register(path: path, layer: layer.index, site: site)
      }
    }
    do {
      try model.update(modules: ModuleChildren.unflattened(originals.map { path, module in
        (path, InspectionRecordingNorm(original: module as! RMSNorm, path: path, recorder: recorder) as Module)
      }), verify: [.noUnusedKeys])
    } catch {
      restore()
      throw ModelInspectionError.invalidGraph("Unable to install activation observers: \(error.localizedDescription)")
    }
  }

  func restore() {
    guard !originals.isEmpty else { return }
    // Every path already referred to this exact assignable module before the
    // transaction. Updating with no validation cannot fail for these paths.
    model.update(modules: ModuleChildren.unflattened(originals))
    originals.removeAll()
  }
}

final class InspectionActivationRecorder {
  struct Pending {
    var path: String
    var start: Int
    var count: Int
    var rms: MLXArray
    var maxAbs: MLXArray
    var channels: MLXArray
  }

  var startPosition = 0
  private var layers: [String: InspectorActivationLayer] = [:]
  private var pending: [Pending] = []
  private var failure: String?

  func register(path: String, layer: Int, site: String) {
    layers[path] = InspectorActivationLayer(index: layer, path: path, site: site, samples: [])
  }

  func observe(path: String, input: MLXArray) {
    guard failure == nil else { return }
    guard layers[path] != nil, input.ndim == 3, input.dim(0) == 1,
      input.dim(1) > 0, input.dim(1) <= ModelInspection.prefillChunkSize, input.dim(2) > 0 else {
      failure = "Unsupported residual activation geometry at \(path)."
      return
    }
    let x = input[0, 0..., 0...].asType(.float32)
    let squared = MLX.square(x)
    let width = x.dim(1)
    // Empty bins on extremely small test models use the nearest channel.
    let bins = (0..<ModelInspection.channelBinCount).map { bin in
      let start = min(width - 1, bin * width / ModelInspection.channelBinCount)
      let end = max(start + 1, (bin + 1) * width / ModelInspection.channelBinCount)
      return MLX.sqrt(squared[0..., start..<end].mean(axis: -1))
    }
    pending.append(Pending(path: path, start: startPosition, count: input.dim(1),
      rms: MLX.sqrt(squared.mean(axis: -1)), maxAbs: MLX.abs(x).max(axis: -1),
      channels: MLX.stacked(bins, axis: -1)))
  }

  func flush(evaluating additionalArrays: [MLXArray] = []) throws {
    if let failure { throw ModelInspectionError.invalidGraph(failure) }
    guard pending.count == layers.count else {
      throw ModelInspectionError.invalidGraph("Not every layer observation site executed exactly once.")
    }
    try MLX.checkedEval(additionalArrays + pending.flatMap { [$0.rms, $0.maxAbs, $0.channels] })
    for item in pending {
      let rms = item.rms.asArray(Float.self), maxAbs = item.maxAbs.asArray(Float.self)
      let bins = item.channels.asArray(Float.self)
      guard (rms + maxAbs + bins).allSatisfy({ $0.isFinite && $0 >= 0 }) else {
        throw ModelInspectionError.invalidGraph("Non-finite activation summary at \(item.path).")
      }
      for token in 0..<item.count {
        let start = token * ModelInspection.channelBinCount
        layers[item.path]!.samples.append(InspectorActivationSample(
          tokenIndex: item.start + token, rms: rms[token], maxAbs: maxAbs[token],
          channels: Array(bins[start..<(start + ModelInspection.channelBinCount)])))
      }
    }
    pending.removeAll(keepingCapacity: true)
  }

  func finalize(tokenCount: Int) throws -> [InspectorActivationLayer] {
    guard !layers.isEmpty, pending.isEmpty,
      layers.values.allSatisfy({ $0.samples.map(\.tokenIndex) == Array(0..<tokenCount) }) else {
      throw ModelInspectionError.invalidGraph("Activation observations did not cover the displayed token sequence.")
    }
    return layers.values.sorted {
      $0.index == $1.index ? $0.path < $1.path : $0.index < $1.index
    }
  }
}
