import Foundation
import MLX
import Testing

@testable import ModelRunnerCore

@Suite("Native Laguna model")
struct LagunaModelTests {
  @Test("Poolside's hybrid configuration decodes without Python metadata")
  func decodesHybridConfiguration() throws {
    let configuration = try decodeConfiguration()

    #expect(configuration.modelType == "laguna")
    #expect(configuration.layerTypes == ["full_attention", "sliding_attention"])
    #expect(configuration.mlpLayerTypes == ["dense", "sparse"])
    #expect(configuration.attentionHeadsPerLayer == [2, 4])
    #expect(configuration.gating == .perHead)
    #expect(
      configuration.ropeParameters?["full_attention"]?["rope_type"]
        == .string("yarn")
    )
  }

  @Test("Layer-specific configuration arrays must match the decoder depth")
  func rejectsMismatchedLayerMetadata() {
    let data = Data(
      """
      {
        "model_type": "laguna",
        "num_hidden_layers": 2,
        "layer_types": ["full_attention"],
        "mlp_layer_types": ["dense", "sparse"],
        "num_attention_heads_per_layer": [2, 4],
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 4
      }
      """.utf8
    )

    #expect(throws: LagunaConfigurationError.self) {
      try JSONDecoder().decode(LagunaConfiguration.self, from: data)
    }
  }

  @Test("The local architecture is registered before model loading")
  func registersArchitecture() async {
    await LagunaModelRegistration.register()
    #expect(await LagunaModelRegistration.isRegistered())
  }

  @Test("MLX conversion prefixes match mixed-precision module paths")
  func normalizesLanguageModelPrefix() {
    #expect(
      LagunaModel.normalizedWeightKey(
        "language_model.model.norm.weight", tiedWordEmbeddings: false)
        == "language_model.model.norm.weight"
    )
    #expect(
      LagunaModel.normalizedWeightKey(
        "language_model.lm_head.weight", tiedWordEmbeddings: false)
        == "language_model.lm_head.weight"
    )
    #expect(
      LagunaModel.normalizedWeightKey(
        "model.norm.weight", tiedWordEmbeddings: false)
        == "language_model.model.norm.weight"
    )
    #expect(
      LagunaModel.normalizedWeightKey(
        "language_model.lm_head.weight", tiedWordEmbeddings: true) == nil
    )
  }

  @Test("Compiled MoE fusion preserves Laguna logits")
  func compiledMoEFusionPreservesLogits() throws {
    let configuration = try decodeConfiguration(
      numberOfExperts: 8,
      expertsPerToken: 8)
    let model = LagunaModel(configuration)
    let tokens = MLXArray([Int32(1), 2, 3])[.newAxis]

    let unfused = LagunaRuntimeTuning.$useCompiledMoEFusion.withValue(false) {
      model(tokens, cache: nil)
    }
    let fused = LagunaRuntimeTuning.$useCompiledMoEFusion.withValue(true) {
      model(tokens, cache: nil)
    }
    eval(unfused, fused)

    #expect(allClose(unfused, fused, rtol: 1e-2, atol: 1e-2).item(Bool.self))
  }

  @Test("Compiled per-head attention gate preserves Laguna logits")
  func compiledAttentionGatePreservesLogits() throws {
    let configuration = try decodeConfiguration()
    let model = LagunaModel(configuration)
    let tokens = MLXArray([Int32(1), 2, 3])[.newAxis]

    let eager = LagunaRuntimeTuning.$useCompiledAttentionGate.withValue(false) {
      model(tokens, cache: nil)
    }
    let compiled = LagunaRuntimeTuning.$useCompiledAttentionGate.withValue(true) {
      model(tokens, cache: nil)
    }
    eval(eager, compiled)

    #expect(allClose(eager, compiled, rtol: 1e-2, atol: 1e-2).item(Bool.self))
  }

  @Test("Compiled block tail is restricted to ordinary cached decode")
  func compiledBlockTailEligibilityIsDecodeOnly() {
    func allows(
      runtimeEnabled: Bool = true,
      capturesHiddenStates: Bool = false,
      batchSize: Int = 1,
      sequenceLength: Int = 1,
      cacheOffset: Int? = 3,
      useCompiledAttentionGate: Bool = true,
      useCompiledMoEFusion: Bool = true
    ) -> Bool {
      LagunaCompiledBlockTailEligibility.allows(
        runtimeEnabled: runtimeEnabled,
        capturesHiddenStates: capturesHiddenStates,
        batchSize: batchSize,
        sequenceLength: sequenceLength,
        cacheOffset: cacheOffset,
        useCompiledAttentionGate: useCompiledAttentionGate,
        useCompiledMoEFusion: useCompiledMoEFusion)
    }

    #expect(allows())
    #expect(!allows(runtimeEnabled: false))
    #expect(!allows(capturesHiddenStates: true))
    #expect(!allows(batchSize: 2))
    #expect(!allows(sequenceLength: 2))
    #expect(!allows(cacheOffset: nil))
    #expect(!allows(cacheOffset: 0))
    #expect(!allows(useCompiledAttentionGate: false))
    #expect(!allows(useCompiledMoEFusion: false))
  }

  @Test("Production Laguna decode fast paths are Metal-only and remain overridable")
  func productionDecodeFastPathSelection() {
    #expect(
      LagunaDecodeFastPathSelection.resolve(
        engine: .metal,
        compiledBlockTailOverride: nil,
        fusedRouterTopKOverride: nil)
        == LagunaDecodeFastPathSelection(
          useCompiledBlockTail: true,
          useFusedRouterTopK: true))
    #expect(
      LagunaDecodeFastPathSelection.resolve(
        engine: .cpu,
        compiledBlockTailOverride: nil,
        fusedRouterTopKOverride: nil)
        == LagunaDecodeFastPathSelection(
          useCompiledBlockTail: false,
          useFusedRouterTopK: false))
    #expect(
      LagunaDecodeFastPathSelection.resolve(
        engine: .cuda,
        compiledBlockTailOverride: true,
        fusedRouterTopKOverride: false)
        == LagunaDecodeFastPathSelection(
          useCompiledBlockTail: true,
          useFusedRouterTopK: false))
  }

  @Test("Compiled block tail preserves cached one-token decode logits")
  func compiledBlockTailPreservesCachedDecodeLogits() throws {
    let configuration = try decodeConfiguration()
    let model = LagunaModel(configuration)
    let eagerCache = try model.newCache(parameters: nil)
    let compiledCache = try model.newCache(parameters: nil)
    let prompt = MLXArray([Int32(1), 2, 3])[.newAxis]

    let eagerPrefill = LagunaRuntimeTuning.$useCompiledBlockTail.withValue(false) {
      model(prompt, cache: eagerCache)
    }
    let compiledPrefill = LagunaRuntimeTuning.$useCompiledBlockTail.withValue(true) {
      model(prompt, cache: compiledCache)
    }
    eval(eagerPrefill, compiledPrefill)
    eval(eagerCache)
    eval(compiledCache)

    #expect(eagerCache.allSatisfy { $0.offset == 3 })
    #expect(compiledCache.allSatisfy { $0.offset == 3 })
    #expect(
      allClose(eagerPrefill, compiledPrefill, rtol: 1e-2, atol: 1e-2)
        .item(Bool.self))

    let nextToken = MLXArray([Int32(4)])[.newAxis]
    let eager = LagunaRuntimeTuning.$useCompiledBlockTail.withValue(false) {
      model(nextToken, cache: eagerCache)
    }
    let compiled = LagunaRuntimeTuning.$useCompiledBlockTail.withValue(true) {
      model(nextToken, cache: compiledCache)
    }
    eval(eager, compiled)
    eval(eagerCache)
    eval(compiledCache)

    #expect(eagerCache.allSatisfy { $0.offset == 4 })
    #expect(compiledCache.allSatisfy { $0.offset == 4 })
    #expect(allClose(eager, compiled, rtol: 1e-2, atol: 1e-2).item(Bool.self))
  }

  @Test("Compiled block tail flag falls back for multi-token cached input")
  func compiledBlockTailFallsBackForMultiTokenInput() throws {
    let configuration = try decodeConfiguration()
    let model = LagunaModel(configuration)
    let eagerCache = try model.newCache(parameters: nil)
    let fallbackCache = try model.newCache(parameters: nil)
    let prompt = MLXArray([Int32(1), 2])[.newAxis]

    let eagerPrefill = model(prompt, cache: eagerCache)
    let fallbackPrefill = model(prompt, cache: fallbackCache)
    eval(eagerPrefill, fallbackPrefill)
    eval(eagerCache)
    eval(fallbackCache)

    let continuation = MLXArray([Int32(3), 4])[.newAxis]
    let eager = LagunaRuntimeTuning.$useCompiledBlockTail.withValue(false) {
      model(continuation, cache: eagerCache)
    }
    let fallback = LagunaRuntimeTuning.$useCompiledBlockTail.withValue(true) {
      model(continuation, cache: fallbackCache)
    }
    eval(eager, fallback)
    eval(eagerCache)
    eval(fallbackCache)

    #expect(eagerCache.allSatisfy { $0.offset == 4 })
    #expect(fallbackCache.allSatisfy { $0.offset == 4 })
    #expect(allClose(eager, fallback, rtol: 1e-2, atol: 1e-2).item(Bool.self))
  }

  @Test("Fused Laguna router exactly preserves stable top-k indices and weights")
  func fusedRouterTopKIsExact() {
    let logits = MLXArray(
      (0..<256).map { index in
        Float((index * 73) % 257 - 128) / 19
      }
    ).reshaped(1, 1, 256)
    let correctionBias = MLXArray(
      (0..<256).map { index in
        Float((index * 29) % 31 - 15) / 1_000
      })

    for dtype: DType in [.float32, .float16, .bfloat16] {
      let typedLogits = logits.asType(dtype)
      let legacy = lagunaSigmoidTopK8Router(
        typedLogits, correctionBias: correctionBias)
      let fused = fusedLagunaSigmoidTopK8Router(
        typedLogits, correctionBias: correctionBias)
      eval(legacy.weights, legacy.indices, fused.weights, fused.indices)

      #expect(MLX.arrayEqual(legacy.indices, fused.indices).item(Bool.self))
      #expect(MLX.arrayEqual(legacy.weights, fused.weights).item(Bool.self))
    }

    // Every selection score ties. The legacy stable sort keeps lower expert
    // indices first; this also covers the candidate's explicit tie-break.
    let tiedLogits = MLXArray.zeros([1, 1, 256])
    let tiedBias = MLXArray.zeros([256])
    let legacyTie = lagunaSigmoidTopK8Router(
      tiedLogits, correctionBias: tiedBias)
    let fusedTie = fusedLagunaSigmoidTopK8Router(
      tiedLogits, correctionBias: tiedBias)
    eval(legacyTie.weights, legacyTie.indices, fusedTie.weights, fusedTie.indices)

    #expect(MLX.arrayEqual(legacyTie.indices, fusedTie.indices).item(Bool.self))
    #expect(MLX.arrayEqual(legacyTie.weights, fusedTie.weights).item(Bool.self))
    #expect(fusedTie.indices.asArray(UInt32.self) == Array(0..<8).map(UInt32.init))

    // sigmoid(0) + (-0.5) produces zero-valued selection keys. The kernel
    // deliberately canonicalizes signed zero before applying the stable
    // lower-index tie-break, matching the legacy comparator.
    let zeroSelectionBias = MLXArray(
      Array(repeating: Float(-0.5), count: 256))
    let legacyZero = lagunaSigmoidTopK8Router(
      tiedLogits, correctionBias: zeroSelectionBias)
    let fusedZero = fusedLagunaSigmoidTopK8Router(
      tiedLogits, correctionBias: zeroSelectionBias)
    eval(legacyZero.weights, legacyZero.indices, fusedZero.weights, fusedZero.indices)
    #expect(MLX.arrayEqual(legacyZero.indices, fusedZero.indices).item(Bool.self))
    #expect(MLX.arrayEqual(legacyZero.weights, fusedZero.weights).item(Bool.self))
  }

  @Test("Fused Laguna router preserves cached one-token logits")
  func fusedRouterPreservesCachedDecodeLogits() throws {
    let configuration = try decodeConfiguration(
      numberOfExperts: 256,
      expertsPerToken: 8)
    let model = LagunaModel(configuration)
    let legacyCache = try model.newCache(parameters: nil)
    let fusedCache = try model.newCache(parameters: nil)
    let prompt = MLXArray([Int32(1), 2, 3])[.newAxis]

    let legacyPrefill = LagunaRuntimeTuning.$useCompiledBlockTail.withValue(true) {
      LagunaRuntimeTuning.$useFusedRouterTopK.withValue(false) {
        model(prompt, cache: legacyCache)
      }
    }
    let fusedPrefill = LagunaRuntimeTuning.$useCompiledBlockTail.withValue(true) {
      LagunaRuntimeTuning.$useFusedRouterTopK.withValue(true) {
        model(prompt, cache: fusedCache)
      }
    }
    eval(legacyPrefill, fusedPrefill)
    eval(legacyCache)
    eval(fusedCache)
    #expect(MLX.arrayEqual(legacyPrefill, fusedPrefill).item(Bool.self))

    let nextToken = MLXArray([Int32(4)])[.newAxis]
    let legacy = LagunaRuntimeTuning.$useCompiledBlockTail.withValue(true) {
      LagunaRuntimeTuning.$useFusedRouterTopK.withValue(false) {
        model(nextToken, cache: legacyCache)
      }
    }
    let fused = LagunaRuntimeTuning.$useCompiledBlockTail.withValue(true) {
      LagunaRuntimeTuning.$useFusedRouterTopK.withValue(true) {
        model(nextToken, cache: fusedCache)
      }
    }
    eval(legacy, fused)
    eval(legacyCache)
    eval(fusedCache)

    #expect(legacyCache.allSatisfy { $0.offset == 4 })
    #expect(fusedCache.allSatisfy { $0.offset == 4 })
    #expect(MLX.arrayEqual(legacy, fused).item(Bool.self))
  }

  private func decodeConfiguration(
    numberOfExperts: Int = 4,
    expertsPerToken: Int = 2
  ) throws -> LagunaConfiguration {
    let data = Data(
      """
      {
        "model_type": "laguna",
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_attention_heads_per_layer": [2, 4],
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 128,
        "rms_norm_eps": 0.000001,
        "attention_bias": false,
        "qkv_bias": false,
        "gating": "per-head",
        "tie_word_embeddings": false,
        "sliding_window": 8,
        "layer_types": ["full_attention", "sliding_attention"],
        "mlp_layer_types": ["dense", "sparse"],
        "num_experts": \(numberOfExperts),
        "num_experts_per_tok": \(expertsPerToken),
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 8,
        "moe_routed_scaling_factor": 2.5,
        "moe_router_score_func": "sigmoid",
        "rope_parameters": {
          "full_attention": {
            "rope_type": "yarn",
            "rope_theta": 500000.0,
            "factor": 2.0,
            "original_max_position_embeddings": 64,
            "partial_rotary_factor": 0.5
          },
          "sliding_attention": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0
          }
        }
      }
      """.utf8
    )
    return try JSONDecoder().decode(LagunaConfiguration.self, from: data)
  }
}
