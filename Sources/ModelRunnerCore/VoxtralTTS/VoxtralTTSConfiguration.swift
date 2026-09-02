import Foundation

struct VoxtralTTSConfiguration: Sendable {
  let dimension: Int
  let layerCount: Int
  let headDimension: Int
  let hiddenDimension: Int
  let attentionHeadCount: Int
  let keyValueHeadCount: Int
  let vocabularySize: Int
  let ropeTheta: Float
  let normalizationEpsilon: Float
  let usesBiases: Bool
  let maximumPositionEmbeddings: Int

  let sampleRate: Int
  let frameRate: Double
  let semanticCodebookSize: Int
  let acousticCodebookSize: Int
  let acousticCodebookCount: Int

  let acousticDimension: Int
  let acousticLayerCount: Int
  let acousticHeadDimension: Int
  let acousticHiddenDimension: Int
  let acousticAttentionHeadCount: Int
  let acousticKeyValueHeadCount: Int
  let acousticNormalizationEpsilon: Float
  let acousticSigmaMaximum: Float

  let codecPatchSize: Int
  let codecPatchProjectionKernelSize: Int
  let codecSemanticDimension: Int
  let codecAcousticDimension: Int
  let codecDimension: Int
  let codecHiddenDimension: Int
  let codecHeadDimension: Int
  let codecAttentionHeadCount: Int
  let codecKeyValueHeadCount: Int
  let codecQKNormEpsilon: Float
  let codecNormEpsilon: Float
  let codecTransformerLayerCounts: [Int]
  let codecConvolutionKernelSizes: [Int]
  let codecConvolutionStrides: [Int]

  let beginningOfSequenceTokenID: Int
  let audioTokenID: Int
  let beginAudioTokenID: Int

  /// The release runner performs seven Euler updates. This is the model's
  /// advertised eight-point schedule, including both endpoints.
  let acousticDenoisingStepCount = 7
  let classifierFreeGuidanceScale: Float = 1.2

  init(modelDirectory: URL) throws {
    let url = modelDirectory.appendingPathComponent("config.json")
    let root = try JSONDecoder().decode(Root.self, from: Data(contentsOf: url))
    guard root.modelType == "voxtral_tts" else {
      throw VoxtralTTSError.invalidConfiguration("model_type must be voxtral_tts")
    }

    let audioModel = root.multimodal.audioModelArguments
    let encoding = audioModel.audioEncodingArguments
    let acoustic = audioModel.acousticTransformerArguments
    let codec = root.multimodal.audioTokenizerArguments

    dimension = root.dimension
    layerCount = root.layerCount
    headDimension = root.headDimension
    hiddenDimension = root.hiddenDimension
    attentionHeadCount = root.attentionHeadCount
    keyValueHeadCount = root.keyValueHeadCount
    vocabularySize = root.vocabularySize
    ropeTheta = root.ropeTheta
    normalizationEpsilon = root.normalizationEpsilon
    usesBiases = root.usesBiases
    maximumPositionEmbeddings = root.maximumPositionEmbeddings

    sampleRate = encoding.sampleRate
    frameRate = encoding.frameRate
    semanticCodebookSize = audioModel.semanticCodebookSize
    acousticCodebookSize = audioModel.acousticCodebookSize
    acousticCodebookCount = audioModel.acousticCodebookCount

    acousticDimension = acoustic.dimension
    acousticLayerCount = acoustic.layerCount
    acousticHeadDimension = acoustic.headDimension
    acousticHiddenDimension = acoustic.hiddenDimension
    acousticAttentionHeadCount = acoustic.attentionHeadCount
    acousticKeyValueHeadCount = acoustic.keyValueHeadCount
    acousticNormalizationEpsilon = root.normalizationEpsilon
    acousticSigmaMaximum = acoustic.sigmaMaximum

    codecPatchSize = codec.patchSize
    codecPatchProjectionKernelSize = codec.patchProjectionKernelSize
    codecSemanticDimension = codec.semanticDimension
    codecAcousticDimension = codec.acousticDimension
    codecDimension = codec.dimension
    codecHiddenDimension = codec.hiddenDimension
    codecHeadDimension = codec.headDimension
    codecAttentionHeadCount = codec.attentionHeadCount
    codecKeyValueHeadCount = codec.keyValueHeadCount
    codecQKNormEpsilon = codec.qkNormEpsilon
    codecNormEpsilon = codec.normEpsilon
    codecTransformerLayerCounts = try Self.parseList(codec.decoderTransformerLengths)
    codecConvolutionKernelSizes = try Self.parseList(codec.decoderConvolutionKernels)
    codecConvolutionStrides = try Self.parseList(codec.decoderConvolutionStrides)

    beginningOfSequenceTokenID = root.multimodal.beginningOfSequenceTokenID
    audioTokenID = audioModel.audioTokenID
    beginAudioTokenID = audioModel.beginAudioTokenID

    guard dimension > 0, layerCount > 0, headDimension > 0,
      attentionHeadCount > 0, keyValueHeadCount > 0,
      attentionHeadCount % keyValueHeadCount == 0,
      codecTransformerLayerCounts.count == codecConvolutionKernelSizes.count,
      codecConvolutionKernelSizes.count == codecConvolutionStrides.count,
      !codecConvolutionStrides.isEmpty,
      sampleRate > 0, frameRate > 0,
      semanticCodebookSize > 0, acousticCodebookSize > 1,
      acousticCodebookCount > 0
    else {
      throw VoxtralTTSError.invalidConfiguration("checkpoint dimensions are inconsistent")
    }
  }

  var audioEmbeddingCount: Int {
    let semanticPadded = (semanticCodebookSize / 128 + 1) * 128
    let acousticPadded = Self.padToMultiple(acousticCodebookSize * acousticCodebookCount, 128)
    return semanticPadded + acousticPadded
  }

  var samplesPerFrame: Int {
    codecConvolutionStrides.reduce(codecPatchSize, *)
  }

  static func padToMultiple(_ value: Int, _ multiple: Int) -> Int {
    (value + multiple - 1) / multiple * multiple
  }

  private static func parseList(_ value: String) throws -> [Int] {
    let result = try value.split(separator: ",").map { component in
      guard let number = Int(component.trimmingCharacters(in: .whitespaces)), number > 0 else {
        throw VoxtralTTSError.invalidConfiguration("invalid integer list: \(value)")
      }
      return number
    }
    guard !result.isEmpty else {
      throw VoxtralTTSError.invalidConfiguration("integer list must not be empty")
    }
    return result
  }
}

private extension VoxtralTTSConfiguration {
  struct Root: Decodable {
    let modelType: String
    let dimension: Int
    let layerCount: Int
    let headDimension: Int
    let hiddenDimension: Int
    let attentionHeadCount: Int
    let keyValueHeadCount: Int
    let vocabularySize: Int
    let ropeTheta: Float
    let normalizationEpsilon: Float
    let usesBiases: Bool
    let maximumPositionEmbeddings: Int
    let multimodal: Multimodal

    enum CodingKeys: String, CodingKey {
      case modelType = "model_type"
      case dimension = "dim"
      case layerCount = "n_layers"
      case headDimension = "head_dim"
      case hiddenDimension = "hidden_dim"
      case attentionHeadCount = "n_heads"
      case keyValueHeadCount = "n_kv_heads"
      case vocabularySize = "vocab_size"
      case ropeTheta = "rope_theta"
      case normalizationEpsilon = "norm_eps"
      case usesBiases = "use_biases"
      case maximumPositionEmbeddings = "max_position_embeddings"
      case multimodal
    }
  }

  struct Multimodal: Decodable {
    let beginningOfSequenceTokenID: Int
    let audioModelArguments: AudioModelArguments
    let audioTokenizerArguments: AudioTokenizerArguments

    enum CodingKeys: String, CodingKey {
      case beginningOfSequenceTokenID = "bos_token_id"
      case audioModelArguments = "audio_model_args"
      case audioTokenizerArguments = "audio_tokenizer_args"
    }
  }

  struct AudioModelArguments: Decodable {
    let semanticCodebookSize: Int
    let acousticCodebookSize: Int
    let acousticCodebookCount: Int
    let audioTokenID: Int
    let beginAudioTokenID: Int
    let audioEncodingArguments: AudioEncodingArguments
    let acousticTransformerArguments: AcousticTransformerArguments

    enum CodingKeys: String, CodingKey {
      case semanticCodebookSize = "semantic_codebook_size"
      case acousticCodebookSize = "acoustic_codebook_size"
      case acousticCodebookCount = "n_acoustic_codebook"
      case audioTokenID = "audio_token_id"
      case beginAudioTokenID = "begin_audio_token_id"
      case audioEncodingArguments = "audio_encoding_args"
      case acousticTransformerArguments = "acoustic_transformer_args"
    }
  }

  struct AudioEncodingArguments: Decodable {
    let sampleRate: Int
    let frameRate: Double

    enum CodingKeys: String, CodingKey {
      case sampleRate = "sampling_rate"
      case frameRate = "frame_rate"
    }
  }

  struct AcousticTransformerArguments: Decodable {
    let dimension: Int
    let layerCount: Int
    let headDimension: Int
    let hiddenDimension: Int
    let attentionHeadCount: Int
    let keyValueHeadCount: Int
    let sigmaMaximum: Float

    enum CodingKeys: String, CodingKey {
      case dimension = "dim"
      case layerCount = "n_layers"
      case headDimension = "head_dim"
      case hiddenDimension = "hidden_dim"
      case attentionHeadCount = "n_heads"
      case keyValueHeadCount = "n_kv_heads"
      case sigmaMaximum = "sigma_max"
    }
  }

  struct AudioTokenizerArguments: Decodable {
    let patchSize: Int
    let patchProjectionKernelSize: Int
    let semanticDimension: Int
    let acousticDimension: Int
    let dimension: Int
    let hiddenDimension: Int
    let headDimension: Int
    let attentionHeadCount: Int
    let keyValueHeadCount: Int
    let qkNormEpsilon: Float
    let normEpsilon: Float
    let decoderTransformerLengths: String
    let decoderConvolutionKernels: String
    let decoderConvolutionStrides: String

    enum CodingKeys: String, CodingKey {
      case patchSize = "pretransform_patch_size"
      case patchProjectionKernelSize = "patch_proj_kernel_size"
      case semanticDimension = "semantic_dim"
      case acousticDimension = "acoustic_dim"
      case dimension = "dim"
      case hiddenDimension = "hidden_dim"
      case headDimension = "head_dim"
      case attentionHeadCount = "n_heads"
      case keyValueHeadCount = "n_kv_heads"
      case qkNormEpsilon = "qk_norm_eps"
      case normEpsilon = "norm_eps"
      case decoderTransformerLengths = "decoder_transformer_lengths_str"
      case decoderConvolutionKernels = "decoder_convs_kernels_str"
      case decoderConvolutionStrides = "decoder_convs_strides_str"
    }
  }
}

enum VoxtralTTSError: LocalizedError {
  case invalidConfiguration(String)
  case missingFile(String)
  case invalidTokenizer(String)
  case invalidWeights(String)
  case unsupportedAudioFormat(String)
  case voiceNotFound(String)
  case generationFailed(String)
  case busy

  var errorDescription: String? {
    switch self {
    case .invalidConfiguration(let message): "Invalid Voxtral configuration: \(message)"
    case .missingFile(let path): "Required Voxtral file is missing: \(path)"
    case .invalidTokenizer(let message): "Invalid Voxtral tokenizer: \(message)"
    case .invalidWeights(let message): "Invalid Voxtral weights: \(message)"
    case .unsupportedAudioFormat(let format):
      "Voxtral currently supports wav and pcm output; requested \(format)."
    case .voiceNotFound(let voice): "Voxtral voice is not available: \(voice)"
    case .generationFailed(let message): "Voxtral generation failed: \(message)"
    case .busy: "The Voxtral speech generator is already processing a request."
    }
  }
}
