import Foundation
import MLX
import MLXNN

final class VoxtralAcousticTransformer: Module {
  @ModuleInfo(key: "input_projection") var inputProjection: Linear
  @ModuleInfo(key: "time_projection") var timeProjection: Linear
  @ModuleInfo(key: "llm_projection") var languageProjection: Linear
  @ModuleInfo(key: "semantic_codebook_output") var semanticCodebookOutput: Linear
  @ModuleInfo(key: "acoustic_codebook_output") var acousticCodebookOutput: Linear
  @ModuleInfo(key: "layers") fileprivate var layers: [VoxtralAcousticDecoderLayer]
  @ModuleInfo(key: "norm") var norm: RMSNorm

  private let languageDimension: Int
  private let acousticDimension: Int
  private let acousticCodebookCount: Int
  private let acousticCodebookSize: Int
  private let semanticValidUpperBound: Int
  private let semanticOutputSize: Int
  private let denoisingStepCount: Int
  private let classifierFreeGuidanceScale: Float
  private let sigmaMaximum: Float
  // Keep generated constants behind a non-Module reference so MLXNN's
  // reflection-based checkpoint loader does not mistake them for weights.
  private let runtimeConstants: VoxtralAcousticRuntimeConstants

  init(configuration: VoxtralTTSConfiguration) {
    self.languageDimension = configuration.dimension
    self.acousticDimension = configuration.acousticDimension
    self.acousticCodebookCount = configuration.acousticCodebookCount
    self.acousticCodebookSize = configuration.acousticCodebookSize
    self.semanticValidUpperBound = configuration.semanticCodebookSize + 2
    self.semanticOutputSize = VoxtralTTSConfiguration.padToMultiple(
      configuration.semanticCodebookSize + 2,
      128
    )
    self.denoisingStepCount = configuration.acousticDenoisingStepCount
    self.classifierFreeGuidanceScale = configuration.classifierFreeGuidanceScale
    self.sigmaMaximum = configuration.acousticSigmaMaximum

    self.runtimeConstants = VoxtralAcousticRuntimeConstants(
      dimension: configuration.acousticDimension
    )

    self._inputProjection.wrappedValue = Linear(
      configuration.acousticCodebookCount,
      configuration.acousticDimension,
      bias: configuration.usesBiases
    )
    self._timeProjection.wrappedValue = Linear(
      configuration.acousticDimension,
      configuration.acousticDimension,
      bias: configuration.usesBiases
    )
    self._languageProjection.wrappedValue = Linear(
      configuration.dimension,
      configuration.acousticDimension,
      bias: configuration.usesBiases
    )
    self._semanticCodebookOutput.wrappedValue = Linear(
      configuration.dimension,
      semanticOutputSize,
      bias: configuration.usesBiases
    )
    self._acousticCodebookOutput.wrappedValue = Linear(
      configuration.acousticDimension,
      configuration.acousticCodebookCount,
      bias: configuration.usesBiases
    )
    self._layers.wrappedValue = (0 ..< configuration.acousticLayerCount).map { _ in
      VoxtralAcousticDecoderLayer(configuration: configuration)
    }
    self._norm.wrappedValue = RMSNorm(
      dimensions: configuration.acousticDimension,
      eps: configuration.acousticNormalizationEpsilon
    )
    super.init()
  }

  /// Returns one semantic code followed by all 36 acoustic FSQ codes.
  /// The result has shape `[batch, 37]` and dtype `int32`.
  func decodeOneFrame(hidden: MLXArray) -> MLXArray {
    let frameHidden = normalizedFrameHidden(hidden)
    let batchSize = frameHidden.dim(0)

    let semanticLogits = semanticCodebookOutput(frameHidden).asType(.float32)
    let classIDs = MLXArray(0 ..< semanticOutputSize).asType(.int32)
    let validSemanticClass = (classIDs .> Int32(0)) .&&
      (classIDs .< Int32(semanticValidUpperBound))
    let suppressed = MLXArray.full(
      semanticLogits.shape,
      values: MLXArray(-Float.infinity),
      dtype: .float32
    )
    let semanticCode = which(validSemanticClass, semanticLogits, suppressed)
      .argMax(axis: -1, keepDims: true)
      .asType(.int32)

    var sample = MLXRandom.normal(
      [batchSize, acousticCodebookCount],
      dtype: frameHidden.dtype
    ) * sigmaMaximum

    let projectedLanguage = languageProjection(
      concatenated(
        [frameHidden, MLXArray.zeros(like: frameHidden)],
        axis: 0
      )
    )
    let stepSize = 1 / Float(denoisingStepCount)

    for step in 0 ..< denoisingStepCount {
      let time = Float(step) * stepSize
      let projectedTime = timeProjection(
        timeEmbedding(time: time, batchSize: batchSize, dtype: frameHidden.dtype)
      )
      let doubledTime = concatenated([projectedTime, projectedTime], axis: 0)
      let doubledSample = concatenated([sample, sample], axis: 0)
      let velocity = predictVelocity(
        sample: doubledSample,
        projectedTime: doubledTime,
        projectedLanguage: projectedLanguage
      )

      let conditionalVelocity = velocity[..<batchSize]
      let unconditionalVelocity = velocity[batchSize...]
      let guidedVelocity =
        classifierFreeGuidanceScale * conditionalVelocity
        + (1 - classifierFreeGuidanceScale) * unconditionalVelocity
      sample = sample + stepSize * guidedVelocity
    }

    let acousticCodes = (
      MLX.clip(sample, min: -1, max: 1) + 1
    ) * (Float(acousticCodebookSize - 1) / 2)
    let quantizedCodes = acousticCodes.round().asType(.int32) + Int32(2)
    let emittedCodes = which(
      semanticCode .!= Int32(1),
      quantizedCodes,
      MLXArray.full(
        quantizedCodes.shape,
        values: MLXArray(Int32(2)),
        dtype: .int32
      )
    )
    return concatenated([semanticCode, emittedCodes], axis: -1)
  }

  private func predictVelocity(
    sample: MLXArray,
    projectedTime: MLXArray,
    projectedLanguage: MLXArray
  ) -> MLXArray {
    var hidden = concatenated(
      [
        inputProjection(sample).expandedDimensions(axis: 1),
        projectedTime.expandedDimensions(axis: 1),
        projectedLanguage.expandedDimensions(axis: 1),
      ],
      axis: 1
    )
    for layer in layers {
      hidden = layer(hidden)
    }
    hidden = norm(hidden)
    return acousticCodebookOutput(hidden[0..., 0, 0...])
  }

  private func timeEmbedding(time: Float, batchSize: Int, dtype: DType) -> MLXArray {
    let times = MLXArray.full(
      [batchSize, 1],
      values: MLXArray(time),
      dtype: .float32
    )
    let angles = times * runtimeConstants.inverseFrequencies[.newAxis, 0...]
    return concatenated([MLX.cos(angles), MLX.sin(angles)], axis: -1).asType(dtype)
  }

  private func normalizedFrameHidden(_ hidden: MLXArray) -> MLXArray {
    let result: MLXArray
    switch hidden.ndim {
    case 1:
      result = hidden.expandedDimensions(axis: 0)
    case 2:
      result = hidden
    case 3:
      precondition(hidden.dim(1) == 1, "Voxtral frame hidden state must have sequence length 1")
      result = hidden.squeezed(axis: 1)
    default:
      preconditionFailure("Voxtral frame hidden state must have rank 1, 2, or 3")
    }
    precondition(
      result.dim(-1) == languageDimension,
      "Voxtral frame hidden state has an unexpected feature dimension"
    )
    return result
  }
}

private final class VoxtralAcousticRuntimeConstants {
  let inverseFrequencies: MLXArray

  init(dimension: Int) {
    let halfDimension = dimension / 2
    let frequencyIndices = MLXArray(0 ..< halfDimension).asType(.float32)
    self.inverseFrequencies = MLX.exp(
      -Float(Foundation.log(10_000.0)) * frequencyIndices / Float(halfDimension)
    )
  }
}

private final class VoxtralAcousticDecoderLayer: Module {
  @ModuleInfo(key: "attention") var attention: VoxtralAcousticAttention
  @ModuleInfo(key: "feed_forward") var feedForward: VoxtralAcousticFeedForward
  @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
  @ModuleInfo(key: "ffn_norm") var feedForwardNorm: RMSNorm

  init(configuration: VoxtralTTSConfiguration) {
    self._attention.wrappedValue = VoxtralAcousticAttention(configuration: configuration)
    self._feedForward.wrappedValue = VoxtralAcousticFeedForward(configuration: configuration)
    self._attentionNorm.wrappedValue = RMSNorm(
      dimensions: configuration.acousticDimension,
      eps: configuration.acousticNormalizationEpsilon
    )
    self._feedForwardNorm.wrappedValue = RMSNorm(
      dimensions: configuration.acousticDimension,
      eps: configuration.acousticNormalizationEpsilon
    )
    super.init()
  }

  func callAsFunction(_ input: MLXArray) -> MLXArray {
    let afterAttention = input + attention(attentionNorm(input))
    return afterAttention + feedForward(feedForwardNorm(afterAttention))
  }
}

private final class VoxtralAcousticAttention: Module {
  @ModuleInfo(key: "wq") var queryProjection: Linear
  @ModuleInfo(key: "wk") var keyProjection: Linear
  @ModuleInfo(key: "wv") var valueProjection: Linear
  @ModuleInfo(key: "wo") var outputProjection: Linear

  private let attentionHeadCount: Int
  private let keyValueHeadCount: Int
  private let headDimension: Int
  private let scale: Float

  init(configuration: VoxtralTTSConfiguration) {
    self.attentionHeadCount = configuration.acousticAttentionHeadCount
    self.keyValueHeadCount = configuration.acousticKeyValueHeadCount
    self.headDimension = configuration.acousticHeadDimension
    self.scale = 1 / Float(configuration.acousticHeadDimension).squareRoot()

    self._queryProjection.wrappedValue = Linear(
      configuration.acousticDimension,
      configuration.acousticAttentionHeadCount * configuration.acousticHeadDimension,
      bias: configuration.usesBiases
    )
    self._keyProjection.wrappedValue = Linear(
      configuration.acousticDimension,
      configuration.acousticKeyValueHeadCount * configuration.acousticHeadDimension,
      bias: configuration.usesBiases
    )
    self._valueProjection.wrappedValue = Linear(
      configuration.acousticDimension,
      configuration.acousticKeyValueHeadCount * configuration.acousticHeadDimension,
      bias: configuration.usesBiases
    )
    self._outputProjection.wrappedValue = Linear(
      configuration.acousticAttentionHeadCount * configuration.acousticHeadDimension,
      configuration.acousticDimension,
      bias: configuration.usesBiases
    )
    super.init()
  }

  func callAsFunction(_ input: MLXArray) -> MLXArray {
    let batchSize = input.dim(0)
    let sequenceLength = input.dim(1)
    let queries = queryProjection(input)
      .reshaped(batchSize, sequenceLength, attentionHeadCount, headDimension)
      .transposed(0, 2, 1, 3)
    let keys = keyProjection(input)
      .reshaped(batchSize, sequenceLength, keyValueHeadCount, headDimension)
      .transposed(0, 2, 1, 3)
    let values = valueProjection(input)
      .reshaped(batchSize, sequenceLength, keyValueHeadCount, headDimension)
      .transposed(0, 2, 1, 3)

    let attended = MLXFast.scaledDotProductAttention(
      queries: queries,
      keys: keys,
      values: values,
      scale: scale,
      mask: .none
    )
    .transposed(0, 2, 1, 3)
    .reshaped(batchSize, sequenceLength, attentionHeadCount * headDimension)
    return outputProjection(attended)
  }
}

private final class VoxtralAcousticFeedForward: Module {
  @ModuleInfo(key: "w1") var gateProjection: Linear
  @ModuleInfo(key: "w2") var outputProjection: Linear
  @ModuleInfo(key: "w3") var upProjection: Linear

  init(configuration: VoxtralTTSConfiguration) {
    self._gateProjection.wrappedValue = Linear(
      configuration.acousticDimension,
      configuration.acousticHiddenDimension,
      bias: configuration.usesBiases
    )
    self._outputProjection.wrappedValue = Linear(
      configuration.acousticHiddenDimension,
      configuration.acousticDimension,
      bias: configuration.usesBiases
    )
    self._upProjection.wrappedValue = Linear(
      configuration.acousticDimension,
      configuration.acousticHiddenDimension,
      bias: configuration.usesBiases
    )
    super.init()
  }

  func callAsFunction(_ input: MLXArray) -> MLXArray {
    outputProjection(silu(gateProjection(input)) * upProjection(input))
  }
}
