import MLX
import MLXLMCommon
import MLXNN

final class VoxtralLanguageBackbone: Module {
  @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
  @ModuleInfo(key: "layers") fileprivate var layers: [VoxtralLanguageDecoderLayer]
  @ModuleInfo(key: "norm") var norm: RMSNorm

  init(configuration: VoxtralTTSConfiguration) {
    self._embedTokens.wrappedValue = Embedding(
      embeddingCount: configuration.vocabularySize,
      dimensions: configuration.dimension
    )
    self._layers.wrappedValue = (0 ..< configuration.layerCount).map { _ in
      VoxtralLanguageDecoderLayer(configuration: configuration)
    }
    self._norm.wrappedValue = RMSNorm(
      dimensions: configuration.dimension,
      eps: configuration.normalizationEpsilon
    )
    super.init()
  }

  func embed(_ tokenIDs: MLXArray) -> MLXArray {
    embedTokens(tokenIDs)
  }

  func makeCache() -> [KVCache] {
    layers.map { _ in KVCacheSimple() }
  }

  func forward(embeddings: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
    precondition(
      cache == nil || cache?.count == layers.count,
      "Voxtral language cache must contain one entry per decoder layer"
    )

    var hidden = embeddings
    let mask = createAttentionMask(h: hidden, cache: cache?.first)
    for (index, layer) in layers.enumerated() {
      hidden = layer(hidden, mask: mask, cache: cache?[index])
    }
    return norm(hidden)
  }

  func callAsFunction(embeddings: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
    forward(embeddings: embeddings, cache: cache)
  }
}

private final class VoxtralLanguageDecoderLayer: Module {
  @ModuleInfo(key: "self_attn") var selfAttention: VoxtralLanguageAttention
  @ModuleInfo(key: "mlp") var mlp: VoxtralLanguageMLP
  @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
  @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

  init(configuration: VoxtralTTSConfiguration) {
    self._selfAttention.wrappedValue = VoxtralLanguageAttention(configuration: configuration)
    self._mlp.wrappedValue = VoxtralLanguageMLP(configuration: configuration)
    self._inputLayerNorm.wrappedValue = RMSNorm(
      dimensions: configuration.dimension,
      eps: configuration.normalizationEpsilon
    )
    self._postAttentionLayerNorm.wrappedValue = RMSNorm(
      dimensions: configuration.dimension,
      eps: configuration.normalizationEpsilon
    )
    super.init()
  }

  func callAsFunction(
    _ input: MLXArray,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    cache: KVCache?
  ) -> MLXArray {
    let afterAttention = input + selfAttention(inputLayerNorm(input), mask: mask, cache: cache)
    return afterAttention + mlp(postAttentionLayerNorm(afterAttention))
  }
}

private final class VoxtralLanguageAttention: Module {
  @ModuleInfo(key: "q_proj") var queryProjection: Linear
  @ModuleInfo(key: "k_proj") var keyProjection: Linear
  @ModuleInfo(key: "v_proj") var valueProjection: Linear
  @ModuleInfo(key: "o_proj") var outputProjection: Linear

  private let attentionHeadCount: Int
  private let keyValueHeadCount: Int
  private let headDimension: Int
  private let scale: Float
  private let rope: RoPE

  init(configuration: VoxtralTTSConfiguration) {
    self.attentionHeadCount = configuration.attentionHeadCount
    self.keyValueHeadCount = configuration.keyValueHeadCount
    self.headDimension = configuration.headDimension
    self.scale = 1 / Float(configuration.headDimension).squareRoot()
    self.rope = RoPE(
      dimensions: configuration.headDimension,
      traditional: true,
      base: configuration.ropeTheta
    )

    self._queryProjection.wrappedValue = Linear(
      configuration.dimension,
      configuration.attentionHeadCount * configuration.headDimension,
      bias: configuration.usesBiases
    )
    self._keyProjection.wrappedValue = Linear(
      configuration.dimension,
      configuration.keyValueHeadCount * configuration.headDimension,
      bias: configuration.usesBiases
    )
    self._valueProjection.wrappedValue = Linear(
      configuration.dimension,
      configuration.keyValueHeadCount * configuration.headDimension,
      bias: configuration.usesBiases
    )
    self._outputProjection.wrappedValue = Linear(
      configuration.attentionHeadCount * configuration.headDimension,
      configuration.dimension,
      bias: configuration.usesBiases
    )
    super.init()
  }

  func callAsFunction(
    _ input: MLXArray,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    cache: KVCache?
  ) -> MLXArray {
    let batchSize = input.dim(0)
    let sequenceLength = input.dim(1)

    var queries = queryProjection(input)
      .reshaped(batchSize, sequenceLength, attentionHeadCount, headDimension)
      .transposed(0, 2, 1, 3)
    var keys = keyProjection(input)
      .reshaped(batchSize, sequenceLength, keyValueHeadCount, headDimension)
      .transposed(0, 2, 1, 3)
    let values = valueProjection(input)
      .reshaped(batchSize, sequenceLength, keyValueHeadCount, headDimension)
      .transposed(0, 2, 1, 3)

    queries = applyRotaryPosition(rope, to: queries, offset: cache?.ropeOffset)
    keys = applyRotaryPosition(rope, to: keys, offset: cache?.ropeOffset)

    let attended = attentionWithCacheUpdate(
      queries: queries,
      keys: keys,
      values: values,
      cache: cache,
      scale: scale,
      mask: mask
    )
    .transposed(0, 2, 1, 3)
    .reshaped(batchSize, sequenceLength, attentionHeadCount * headDimension)

    return outputProjection(attended)
  }
}

private final class VoxtralLanguageMLP: Module {
  @ModuleInfo(key: "gate_proj") var gateProjection: Linear
  @ModuleInfo(key: "up_proj") var upProjection: Linear
  @ModuleInfo(key: "down_proj") var downProjection: Linear

  init(configuration: VoxtralTTSConfiguration) {
    self._gateProjection.wrappedValue = Linear(
      configuration.dimension,
      configuration.hiddenDimension,
      bias: configuration.usesBiases
    )
    self._upProjection.wrappedValue = Linear(
      configuration.dimension,
      configuration.hiddenDimension,
      bias: configuration.usesBiases
    )
    self._downProjection.wrappedValue = Linear(
      configuration.hiddenDimension,
      configuration.dimension,
      bias: configuration.usesBiases
    )
    super.init()
  }

  func callAsFunction(_ input: MLXArray) -> MLXArray {
    downProjection(silu(gateProjection(input)) * upProjection(input))
  }
}
