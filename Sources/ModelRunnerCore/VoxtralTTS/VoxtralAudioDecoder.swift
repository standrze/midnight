import Foundation
import MLX
import MLXNN

/// Voxtral's decoder-only audio tokenizer. The module/property keys in this
/// file intentionally mirror the checkpoint below `audio_tokenizer.` so a
/// strict `Module.update(parameters:verify:)` can load the weights directly.
final class VoxtralAudioDecoder: Module {
  @ModuleInfo fileprivate var quantizer: VoxtralCodecQuantizer
  @ModuleInfo(key: "decoder_blocks") var decoderBlocks: [Module]
  @ModuleInfo(key: "output_proj") fileprivate var outputProjection: VoxtralCodecConvBlock

  private let strides: [Int]
  private let slidingWindows = [2, 4, 8, 16]
  private let alibiSlopeValues: [Float]

  init(configuration: VoxtralTTSConfiguration) {
    precondition(
      configuration.codecTransformerLayerCounts.count == 4
        && configuration.codecTransformerLayerCounts.allSatisfy { $0 == 2 },
      "Voxtral's audio decoder requires four two-layer transformer stages"
    )
    precondition(
      configuration.codecConvolutionKernelSizes.count == 4
        && configuration.codecConvolutionStrides.count == 4,
      "Voxtral's audio decoder requires four convolution stages"
    )

    _quantizer.wrappedValue = VoxtralCodecQuantizer(configuration: configuration)

    var blocks: [Module] = []
    for stage in 0 ..< 4 {
      let inputChannels = stage == 0
        ? configuration.codecSemanticDimension + configuration.codecAcousticDimension
        : configuration.codecDimension
      blocks.append(
        VoxtralCodecConvBlock(
          outputChannels: configuration.codecDimension,
          inputChannels: inputChannels,
          kernelSize: configuration.codecConvolutionKernelSizes[stage],
          paddingMode: .replicate
        )
      )
      blocks.append(
        VoxtralCodecTransformerBlock(
          layerCount: configuration.codecTransformerLayerCounts[stage],
          configuration: configuration
        )
      )
    }
    _decoderBlocks.wrappedValue = blocks

    _outputProjection.wrappedValue = VoxtralCodecConvBlock(
      outputChannels: configuration.codecPatchSize,
      inputChannels: configuration.codecDimension,
      kernelSize: configuration.codecPatchProjectionKernelSize,
      paddingMode: .reflect
    )

    strides = configuration.codecConvolutionStrides
    alibiSlopeValues = Self.alibiSlopes(headCount: configuration.codecAttentionHeadCount)
    super.init()
  }

  /// Decodes `[batch, frames, 37]` semantic/acoustic codes into a
  /// `[batch, samples]` 24-kHz waveform. Production generation uses batch 1.
  func decode(codes: MLXArray) -> MLXArray {
    precondition(codes.ndim == 3, "Voxtral audio codes must have rank three")

    var hidden = quantizer.decode(codes: codes)
    let slopes = MLXArray(alibiSlopeValues)

    for stage in 0 ..< 4 {
      guard
        let convolution = decoderBlocks[stage * 2] as? VoxtralCodecConvBlock,
        let transformer = decoderBlocks[stage * 2 + 1] as? VoxtralCodecTransformerBlock
      else {
        preconditionFailure("Voxtral decoder block hierarchy was replaced with incompatible modules")
      }

      let stride = strides[stage]
      hidden = convolution(
        hidden,
        stride: stride,
        transposed: stride > 1
      )
      hidden = transformer(
        hidden,
        alibiSlopes: slopes,
        slidingWindow: slidingWindows[stage]
      )
    }

    hidden = outputProjection(hidden, stride: 1, transposed: false)
    return hidden.reshaped(hidden.dim(0), -1)
  }

  /// Matches the slope ordering used by the release checkpoint. For eight
  /// heads this is exactly `[.5, .25, .125, ...]`.
  private static func alibiSlopes(headCount: Int) -> [Float] {
    precondition(headCount > 0)

    func powerOfTwoSlopes(_ count: Int) -> [Float] {
      let log2Count = Double(count.trailingZeroBitCount)
      let startExponent = Foundation.pow(2.0, -(log2Count - 3.0))
      let start = Float(Foundation.pow(2.0, -startExponent))
      var result: [Float] = []
      result.reserveCapacity(count)
      var value = start
      for _ in 0 ..< count {
        result.append(value)
        value *= start
      }
      return result
    }

    if headCount.nonzeroBitCount == 1 {
      return powerOfTwoSlopes(headCount)
    }

    let closestPowerOfTwo = 1 << (Int.bitWidth - 1 - headCount.leadingZeroBitCount)
    let base = powerOfTwoSlopes(closestPowerOfTwo)
    let extras = powerOfTwoSlopes(closestPowerOfTwo * 2)
      .enumerated()
      .compactMap { index, value in index.isMultiple(of: 2) ? value : nil }
    return base + Array(extras.prefix(headCount - closestPowerOfTwo))
  }
}

// MARK: - Quantizer

private final class VoxtralCodecQuantizer: Module {
  @ModuleInfo(key: "semantic_codebook") var semanticCodebook: VoxtralSemanticCodebook

  private let semanticCodebookSize: Int
  private let acousticCodebookSize: Int
  private let acousticCodebookCount: Int

  init(configuration: VoxtralTTSConfiguration) {
    semanticCodebookSize = configuration.semanticCodebookSize
    acousticCodebookSize = configuration.acousticCodebookSize
    acousticCodebookCount = configuration.acousticCodebookCount
    _semanticCodebook.wrappedValue = VoxtralSemanticCodebook(
      entryCount: configuration.semanticCodebookSize,
      dimensions: configuration.codecSemanticDimension
    )
    super.init()
  }

  func decode(codes: MLXArray) -> MLXArray {
    precondition(
      codes.dim(-1) == acousticCodebookCount + 1,
      "Voxtral expects one semantic and 36 acoustic codes per frame"
    )

    // Generated codes reserve indices 0 and 1 for empty/end audio.
    let semanticIndices = (codes[0..., 0..., 0] - 2).asType(.int32)
    let acousticIndices = (codes[0..., 0..., 1...] - 2).asType(.float32)

    let semantic = semanticCodebook.decode(indices: semanticIndices)
    let acoustic =
      (2 * acousticIndices / Float(acousticCodebookSize - 1)) - 1
    return concatenated([semantic, acoustic], axis: -1)
  }
}

private final class VoxtralSemanticCodebook: Module {
  @ParameterInfo(key: "cluster_usage") var clusterUsage: MLXArray
  @ParameterInfo(key: "embedding_sum") var embeddingSum: MLXArray

  init(entryCount: Int, dimensions: Int) {
    _clusterUsage.wrappedValue = MLXArray.ones([entryCount])
    _embeddingSum.wrappedValue = MLXArray.zeros([entryCount, dimensions])
    super.init()
  }

  func decode(indices: MLXArray) -> MLXArray {
    // The checkpoint stores EMA totals rather than normalized embeddings.
    let usage = maximum(clusterUsage.asType(.float32), Float(1e-5))
    let codebook = embeddingSum.asType(.float32) / usage[0..., .newAxis]
    return codebook[indices]
  }
}

// MARK: - Weight-normalized causal convolution

private enum VoxtralCodecPaddingMode {
  case replicate
  case reflect
}

private final class VoxtralCodecWeightNormValues: Module {
  @ParameterInfo var original0: MLXArray
  @ParameterInfo var original1: MLXArray

  init(outputChannels: Int, inputChannels: Int, kernelSize: Int) {
    _original0.wrappedValue = MLXArray.ones([outputChannels, 1, 1])
    _original1.wrappedValue = MLXArray.zeros([outputChannels, inputChannels, kernelSize])
    super.init()
  }
}

private final class VoxtralCodecWeightParametrization: Module {
  @ModuleInfo var weight: VoxtralCodecWeightNormValues

  init(outputChannels: Int, inputChannels: Int, kernelSize: Int) {
    _weight.wrappedValue = VoxtralCodecWeightNormValues(
      outputChannels: outputChannels,
      inputChannels: inputChannels,
      kernelSize: kernelSize
    )
    super.init()
  }
}

private final class VoxtralCodecWeightNormConv: Module {
  @ModuleInfo var parametrizations: VoxtralCodecWeightParametrization

  private let paddingMode: VoxtralCodecPaddingMode

  init(
    outputChannels: Int,
    inputChannels: Int,
    kernelSize: Int,
    paddingMode: VoxtralCodecPaddingMode
  ) {
    self.paddingMode = paddingMode
    _parametrizations.wrappedValue = VoxtralCodecWeightParametrization(
      outputChannels: outputChannels,
      inputChannels: inputChannels,
      kernelSize: kernelSize
    )
    super.init()
  }

  func callAsFunction(
    _ input: MLXArray,
    stride: Int,
    transposed: Bool
  ) -> MLXArray {
    let weight = normalizedWeight()
    return transposed
      ? transposedConvolution(input, weight: weight, stride: stride)
      : convolution(input, weight: weight, stride: stride)
  }

  private func normalizedWeight() -> MLXArray {
    let gain = parametrizations.weight.original0
    let direction = parametrizations.weight.original1
    let norm = sqrt((direction * direction).sum(axes: [1, 2], keepDims: true) + Float(1e-12))
    return gain * direction / norm
  }

  private func convolution(
    _ input: MLXArray,
    weight: MLXArray,
    stride: Int
  ) -> MLXArray {
    let kernelSize = weight.dim(2)
    let totalPadding = kernelSize - stride
    let extraRightPadding = (stride - (input.dim(1) % stride)) % stride
    let paddedInput = paddedInput(
      input,
      left: totalPadding,
      right: extraRightPadding
    )

    // MLX uses channels-last input and `[C_out, kernel, C_in]` weights.
    return conv1d(
      paddedInput,
      weight.transposed(0, 2, 1),
      stride: stride
    )
  }

  private func transposedConvolution(
    _ input: MLXArray,
    weight: MLXArray,
    stride: Int
  ) -> MLXArray {
    // PyTorch stores ConvTranspose1d as `[C_in, C_out, kernel]`; MLX
    // expects `[C_out, kernel, C_in]`.
    let mlxWeight = weight.transposed(1, 2, 0)
    let output = convTransposed1d(input, mlxWeight, stride: stride)

    // Cropping the right tail is equivalent to causal left padding K - 1.
    return output[0..., 0 ..< (input.dim(1) * stride), 0...]
  }

  private func paddedInput(
    _ input: MLXArray,
    left: Int,
    right: Int
  ) -> MLXArray {
    guard left > 0 || right > 0 else { return input }

    switch paddingMode {
    case .replicate:
      return padded(
        input,
        widths: [0, .init((left, right)), 0],
        mode: .edge
      )

    case .reflect:
      // MLX's native pad supports constant and edge modes. Reproduce the
      // checkpoint's reflection rule explicitly, including its short-input
      // fallback before taking the reflected slices.
      let maximumPadding = max(left, right)
      let extraPadding = input.dim(1) <= maximumPadding
        ? maximumPadding - input.dim(1) + 1
        : 0
      let extended = extraPadding > 0
        ? padded(input, widths: [0, .init((0, extraPadding)), 0])
        : input

      var pieces: [MLXArray] = []
      if left > 0 {
        let indices = MLXArray((1 ... left).reversed().map(Int32.init))
        pieces.append(take(extended, indices, axis: 1))
      }
      pieces.append(extended)
      if right > 0 {
        let end = extended.dim(1) - 1
        let start = end - right
        let indices = MLXArray((start ..< end).reversed().map(Int32.init))
        pieces.append(take(extended, indices, axis: 1))
      }

      let result = concatenated(pieces, axis: 1)
      return extraPadding > 0
        ? result[0..., 0 ..< (result.dim(1) - extraPadding), 0...]
        : result
    }
  }
}

private final class VoxtralCodecConvBlock: Module {
  @ModuleInfo var conv: VoxtralCodecWeightNormConv

  init(
    outputChannels: Int,
    inputChannels: Int,
    kernelSize: Int,
    paddingMode: VoxtralCodecPaddingMode
  ) {
    _conv.wrappedValue = VoxtralCodecWeightNormConv(
      outputChannels: outputChannels,
      inputChannels: inputChannels,
      kernelSize: kernelSize,
      paddingMode: paddingMode
    )
    super.init()
  }

  func callAsFunction(
    _ input: MLXArray,
    stride: Int,
    transposed: Bool
  ) -> MLXArray {
    conv(input, stride: stride, transposed: transposed)
  }
}

// MARK: - ALiBi transformer decoder

private final class VoxtralCodecAttention: Module {
  @ModuleInfo var wq: Linear
  @ModuleInfo var wk: Linear
  @ModuleInfo var wv: Linear
  @ModuleInfo var wo: Linear
  @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
  @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm

  private let headCount: Int
  private let keyValueHeadCount: Int
  private let headDimension: Int
  private let scale: Float

  init(configuration: VoxtralTTSConfiguration) {
    headCount = configuration.codecAttentionHeadCount
    keyValueHeadCount = configuration.codecKeyValueHeadCount
    headDimension = configuration.codecHeadDimension
    scale = 1 / Float(configuration.codecHeadDimension).squareRoot()

    _wq.wrappedValue = Linear(
      configuration.codecDimension,
      configuration.codecAttentionHeadCount * configuration.codecHeadDimension,
      bias: false
    )
    _wk.wrappedValue = Linear(
      configuration.codecDimension,
      configuration.codecKeyValueHeadCount * configuration.codecHeadDimension,
      bias: false
    )
    _wv.wrappedValue = Linear(
      configuration.codecDimension,
      configuration.codecKeyValueHeadCount * configuration.codecHeadDimension,
      bias: false
    )
    _wo.wrappedValue = Linear(
      configuration.codecAttentionHeadCount * configuration.codecHeadDimension,
      configuration.codecDimension,
      bias: false
    )
    _queryNorm.wrappedValue = RMSNorm(
      dimensions: configuration.codecAttentionHeadCount * configuration.codecHeadDimension,
      eps: configuration.codecQKNormEpsilon
    )
    _keyNorm.wrappedValue = RMSNorm(
      dimensions: configuration.codecKeyValueHeadCount * configuration.codecHeadDimension,
      eps: configuration.codecQKNormEpsilon
    )
    super.init()
  }

  func callAsFunction(
    _ input: MLXArray,
    alibiSlopes: MLXArray,
    slidingWindow: Int
  ) -> MLXArray {
    let batchSize = input.dim(0)
    let sequenceLength = input.dim(1)

    let queries = queryNorm(wq(input))
      .reshaped(batchSize, sequenceLength, headCount, headDimension)
      .transposed(0, 2, 1, 3)
    var keys = keyNorm(wk(input))
      .reshaped(batchSize, sequenceLength, keyValueHeadCount, headDimension)
      .transposed(0, 2, 1, 3)
    var values = wv(input)
      .reshaped(batchSize, sequenceLength, keyValueHeadCount, headDimension)
      .transposed(0, 2, 1, 3)

    if keyValueHeadCount < headCount {
      let repetitionCount = headCount / keyValueHeadCount
      keys = repeated(keys, count: repetitionCount, axis: 1)
      values = repeated(values, count: repetitionCount, axis: 1)
    }

    var scores = matmul(queries, keys.transposed(0, 1, 3, 2)) * scale
    scores = scores.asType(.float32)

    let positions = MLXArray(Int32(0) ..< Int32(sequenceLength)).asType(.float32)
    let distance = positions[.newAxis, 0...] - positions[0..., .newAxis]
    let alibi =
      alibiSlopes[0..., .newAxis, .newAxis]
      * distance[.newAxis, 0..., 0...]
    scores = scores + alibi[.newAxis, 0..., 0..., 0...]

    let allowed = (distance .<= Float(0)) .&& (distance .>= Float(-slidingWindow))
    scores = MLX.where(
      allowed[.newAxis, .newAxis, 0..., 0...],
      scores,
      MLXArray(Float(-1e9), dtype: .float32)
    )

    let weights = softmax(scores, axis: -1).asType(input.dtype)
    let output = matmul(weights, values)
      .transposed(0, 2, 1, 3)
      .reshaped(batchSize, sequenceLength, -1)
    return wo(output)
  }
}

private final class VoxtralCodecFeedForward: Module {
  @ModuleInfo var w1: Linear
  @ModuleInfo var w2: Linear
  @ModuleInfo var w3: Linear

  init(configuration: VoxtralTTSConfiguration) {
    _w1.wrappedValue = Linear(
      configuration.codecDimension,
      configuration.codecHiddenDimension,
      bias: false
    )
    _w2.wrappedValue = Linear(
      configuration.codecHiddenDimension,
      configuration.codecDimension,
      bias: false
    )
    _w3.wrappedValue = Linear(
      configuration.codecDimension,
      configuration.codecHiddenDimension,
      bias: false
    )
    super.init()
  }

  func callAsFunction(_ input: MLXArray) -> MLXArray {
    w2(silu(w1(input)) * w3(input))
  }
}

private final class VoxtralCodecTransformerLayer: Module {
  @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
  @ModuleInfo(key: "ffn_norm") var feedForwardNorm: RMSNorm
  @ModuleInfo var attention: VoxtralCodecAttention
  @ModuleInfo(key: "feed_forward") var feedForward: VoxtralCodecFeedForward
  @ParameterInfo(key: "attention_scale") var attentionScale: MLXArray
  @ParameterInfo(key: "ffn_scale") var feedForwardScale: MLXArray

  init(configuration: VoxtralTTSConfiguration) {
    _attentionNorm.wrappedValue = RMSNorm(
      dimensions: configuration.codecDimension,
      eps: configuration.codecNormEpsilon
    )
    _feedForwardNorm.wrappedValue = RMSNorm(
      dimensions: configuration.codecDimension,
      eps: configuration.codecNormEpsilon
    )
    _attention.wrappedValue = VoxtralCodecAttention(configuration: configuration)
    _feedForward.wrappedValue = VoxtralCodecFeedForward(configuration: configuration)
    _attentionScale.wrappedValue = MLXArray.full(
      [configuration.codecDimension],
      values: MLXArray(Float(0.01))
    )
    _feedForwardScale.wrappedValue = MLXArray.full(
      [configuration.codecDimension],
      values: MLXArray(Float(0.01))
    )
    super.init()
  }

  func callAsFunction(
    _ input: MLXArray,
    alibiSlopes: MLXArray,
    slidingWindow: Int
  ) -> MLXArray {
    var hidden = input
    hidden = hidden
      + attention(
        attentionNorm(hidden),
        alibiSlopes: alibiSlopes,
        slidingWindow: slidingWindow
      ) * attentionScale
    hidden = hidden + feedForward(feedForwardNorm(hidden)) * feedForwardScale
    return hidden
  }
}

private final class VoxtralCodecTransformerBlock: Module {
  @ModuleInfo var layers: [VoxtralCodecTransformerLayer]

  init(layerCount: Int, configuration: VoxtralTTSConfiguration) {
    _layers.wrappedValue = (0 ..< layerCount).map { _ in
      VoxtralCodecTransformerLayer(configuration: configuration)
    }
    super.init()
  }

  func callAsFunction(
    _ input: MLXArray,
    alibiSlopes: MLXArray,
    slidingWindow: Int
  ) -> MLXArray {
    var hidden = input
    for layer in layers {
      hidden = layer(
        hidden,
        alibiSlopes: alibiSlopes,
        slidingWindow: slidingWindow
      )
    }
    return hidden
  }
}
