import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@Suite("Mistral hybrid attention")
struct MistralHybridAttentionTests {
  @Test("Classic Mistral infers all-sliding layers from a non-null window")
  func classicMistralUsesBoundedCaches() throws {
    let model = LlamaModel(
      try decodeLlama(
        modelType: "mistral",
        slidingWindow: "8"
      )
    )

    let cache = try model.newCache(parameters: nil)

    #expect(cache.count == 4)
    #expect(cache.allSatisfy { $0 is RotatingKVCache })
    #expect(cache.allSatisfy { $0.maxSize == 8 })
  }

  @Test("Explicit Ministral metadata preserves each full and sliding layer")
  func ministralUsesHybridCaches() throws {
    let model = LlamaModel(
      try decodeLlama(
        modelType: "mistral",
        slidingWindow: "8",
        layerTypes:
          #", "layer_types":["full_attention","sliding_attention","sliding_attention","sliding_attention"]"#
      )
    )

    let cache = try model.newCache(parameters: nil)

    #expect(cache[0] is KVCacheSimple)
    #expect(cache[1] is RotatingKVCache)
    #expect(cache[2] is RotatingKVCache)
    #expect(cache[3] is RotatingKVCache)
    #expect(cache[1].maxSize == 8)
  }

  @Test("Null windows and non-Mistral Llama models retain full attention")
  func fullAttentionRemainsUnchanged() throws {
    let mistral = LlamaModel(
      try decodeLlama(modelType: "mistral", slidingWindow: "null")
    )
    let llama = LlamaModel(
      try decodeLlama(
        modelType: "llama",
        slidingWindow: "8",
        layerTypes:
          #", "layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention"]"#
      )
    )

    #expect(try mistral.newCache(parameters: nil).allSatisfy { $0 is KVCacheSimple })
    #expect(try llama.newCache(parameters: nil).allSatisfy { $0 is KVCacheSimple })
  }

  @Test("Cached classic Mistral decode matches a cold pass beyond the sliding window")
  func cachedDecodeMatchesColdWindowedAttention() throws {
    let model = LlamaModel(
      try decodeLlama(
        modelType: "mistral",
        slidingWindow: "8",
        hiddenSize: 16,
        hiddenLayers: 2,
        attentionHeads: 4,
        kvHeads: 2
      )
    )
    model.update(
      parameters: model.parameters().mapValues { parameter in
        if parameter.shape.count == 1 {
          return MLXArray.ones(like: parameter)
        }
        let positions = MLXArray(0 ..< Int32(parameter.size)).asType(.float32)
        return (sin(positions * 0.173) * 0.05)
          .reshaped(parameter.shape)
          .asType(parameter.dtype)
      }
    )
    eval(model)

    let tokens = (1 ... 13).map(Int32.init)
    let cold = model(MLXArray(tokens).reshaped(1, -1), cache: nil)[0..., -1, 0...]
    eval(cold)

    let cache = try model.newCache(parameters: nil)
    var warmLogits: MLXArray?
    for token in tokens {
      let logits = model(MLXArray([token]).reshaped(1, 1), cache: cache)
      eval(logits)
      warmLogits = logits[0..., -1, 0...]
    }
    let warm = try #require(warmLogits)
    eval(warm)

    let noWindow = LlamaModel(
      try decodeLlama(
        modelType: "llama",
        slidingWindow: "null",
        hiddenSize: 16,
        hiddenLayers: 2,
        attentionHeads: 4,
        kvHeads: 2
      )
    )
    noWindow.update(parameters: model.parameters())
    eval(noWindow)
    let unwindowed = noWindow(MLXArray(tokens).reshaped(1, -1), cache: nil)[0..., -1, 0...]
    eval(unwindowed)

    let maximumDifference = MLX.max(abs(cold - warm)).item(Float.self)
    let missingWindowDifference = MLX.max(abs(cold - unwindowed)).item(Float.self)
    #expect(
      maximumDifference < 0.005,
      "maximum cold/warm logit difference: \(maximumDifference)"
    )
    #expect(
      missingWindowDifference > maximumDifference * 4,
      "windowed delta \(maximumDifference), missing-window delta \(missingWindowDifference)"
    )
    #expect(cache.allSatisfy { $0.offset == tokens.count })
    #expect(
      cache.allSatisfy { layer in
        layer.state.allSatisfy { $0.dim(2) <= 8 }
      }
    )
  }

  @Test("Malformed classic Mistral hybrid metadata is rejected")
  func rejectsMalformedMetadata() {
    let invalidConfigurations = [
      #", "layer_types":["sliding_attention"]"#,
      #", "layer_types":["full_attention","local_attention","full_attention","full_attention"]"#,
    ]

    for layerTypes in invalidConfigurations {
      #expect(throws: DecodingError.self) {
        try decodeLlama(
          modelType: "mistral",
          slidingWindow: "8",
          layerTypes: layerTypes
        )
      }
    }
    #expect(throws: DecodingError.self) {
      try decodeLlama(
        modelType: "mistral",
        slidingWindow: "null",
        layerTypes:
          #", "layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention"]"#
      )
    }
  }

  @Test("Mistral 3 keeps its existing hybrid cache implementation")
  func mistral3RemainsHybrid() throws {
    let configuration = try JSONDecoder().decode(
      Mistral3TextConfiguration.self,
      from: Data(
        """
        {
          "model_type":"ministral3",
          "hidden_size":8,
          "num_hidden_layers":4,
          "intermediate_size":16,
          "num_attention_heads":2,
          "rms_norm_eps":0.000001,
          "vocab_size":32,
          "num_key_value_heads":1,
          "head_dim":4,
          "sliding_window":8,
          "layer_types":[
            "sliding_attention","full_attention",
            "sliding_attention","full_attention"
          ]
        }
        """.utf8
      )
    )

    let cache = try Mistral3TextModel(configuration).newCache(parameters: nil)

    #expect(cache[0] is RotatingKVCache)
    #expect(cache[1] is KVCacheSimple)
    #expect(cache[2] is RotatingKVCache)
    #expect(cache[3] is KVCacheSimple)
  }

  @Test("Q4 Mistral 3 cached tokenwise logits remain close to a cold batch")
  func quantizedMistral3CachedTokenwiseMatchesColdBatch() throws {
    let model = Mistral3TextModel(
      Mistral3TextConfiguration(
        hiddenSize: 64,
        hiddenLayers: 2,
        intermediateSize: 128,
        attentionHeads: 4,
        rmsNormEps: 0.000001,
        vocabularySize: 128,
        headDimensions: 16,
        kvHeads: 2,
        tieWordEmbeddings: false,
        layerTypes: ["full_attention", "full_attention"]
      )
    )
    model.update(
      parameters: model.parameters().mapValues { parameter in
        if parameter.shape.count == 1 {
          return MLXArray.ones(like: parameter)
        }
        let positions = MLXArray(0 ..< Int32(parameter.size)).asType(.float32)
        return (sin(positions * 0.173) * 0.05)
          .reshaped(parameter.shape)
          .asType(parameter.dtype)
      }
    )
    quantize(model: model, groupSize: 64, bits: 4)
    model.train(false)
    eval(model)

    let tokens = (1 ... 16).map(Int32.init)
    let cold = model(MLXArray(tokens).reshaped(1, -1), cache: nil)[0..., -1, 0...]
    eval(cold)

    let cache = try model.newCache(parameters: nil)
    var cachedLogits: MLXArray?
    for token in tokens {
      let logits = model(MLXArray([token]).reshaped(1, 1), cache: cache)
      eval(logits)
      cachedLogits = logits[0..., -1, 0...]
    }
    let cached = try #require(cachedLogits)
    eval(cold, cached)

    let maximumDifference = MLX.max(abs(cold - cached)).item(Float.self)
    // Q4 decode uses the one-row quantized matrix-vector path, while the cold
    // control uses batched quantized matrix multiplication. Their reduction
    // orders need not be bit-identical, but the resulting logits must remain
    // numerically close enough to represent the same model computation. The
    // pinned M5 Max path measures about 0.00038; 0.005 leaves backend headroom
    // while remaining an order-of-magnitude numerical-parity bound.
    #expect(
      maximumDifference < 0.005,
      "maximum Q4 cold-batch/tokenwise-cache logit difference: \(maximumDifference)"
    )
    #expect(cache.allSatisfy { $0.offset == tokens.count })
    #expect(
      cache.allSatisfy { layer in
        layer.state.count == 2 && layer.state.allSatisfy { $0.dim(2) == tokens.count }
      }
    )
  }

  private func decodeLlama(
    modelType: String,
    slidingWindow: String,
    layerTypes: String = "",
    hiddenSize: Int = 8,
    hiddenLayers: Int = 4,
    attentionHeads: Int = 2,
    kvHeads: Int = 1
  ) throws -> LlamaConfiguration {
    try JSONDecoder().decode(
      LlamaConfiguration.self,
      from: Data(
        """
        {
          "model_type":"\(modelType)",
          "vocab_size":32,
          "hidden_size":\(hiddenSize),
          "intermediate_size":\(hiddenSize * 2),
          "num_hidden_layers":\(hiddenLayers),
          "num_attention_heads":\(attentionHeads),
          "num_key_value_heads":\(kvHeads),
          "rms_norm_eps":0.000001,
          "sliding_window":\(slidingWindow)\(layerTypes)
        }
        """.utf8
      )
    )
  }
}
