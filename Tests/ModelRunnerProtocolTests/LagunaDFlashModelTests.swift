import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import ModelRunnerCore

@Suite("Laguna DFlash model", .serialized)
struct LagunaDFlashModelTests {
  @Test("Poolside Laguna-XS-2.1 DFlash configuration decodes exactly")
  func decodesOfficialConfiguration() throws {
    let configuration = try decodeOfficialConfiguration()

    #expect(configuration.architectures == ["DFlashLagunaForCausalLM"])
    #expect(configuration.hiddenLayers == 5)
    #expect(configuration.hiddenSize == 2_048)
    #expect(configuration.intermediateSize == 8_192)
    #expect(configuration.attentionHeads == 64)
    #expect(configuration.keyValueHeads == 8)
    #expect(configuration.headDimension == 128)
    #expect(configuration.ropeTheta == 500_000)
    #expect(configuration.slidingWindow == 512)
    #expect(configuration.block.blockSize == 16)
    #expect(configuration.block.maskTokenID == 12)
    #expect(configuration.block.numberOfTargetLayers == 40)
    #expect(configuration.block.targetLayerIDs == [1, 13, 25, 33, 39])
  }

  @Test("Registration selects Laguna DFlash independently from Laguna target models")
  func registersDraftArchitecture() async {
    await LagunaDFlashRegistration.register()
    #expect(await LagunaDFlashRegistration.isRegistered())
  }

  @Test("Fused QKV and SwiGLU checkpoint rows map to runtime modules")
  func sanitizesFusedProjections() throws {
    let configuration = try decodeTinyConfiguration()
    let model = LagunaDFlashModel(configuration)
    let qkv = MLXArray(0..<128).reshaped(16, 8)
    let gate = MLXArray(0..<128).reshaped(16, 8)
    let up = MLXArray(128..<256).reshaped(16, 8)

    let sanitized = model.sanitize(weights: [
      "layers.0.self_attn.qkv_proj.weight": qkv,
      "layers.0.mlp.gate_proj.weight": gate,
      "layers.0.mlp.up_proj.weight": up,
    ])

    #expect(sanitized["layers.0.self_attn.qkv_proj.weight"] == nil)
    #expect(sanitized["layers.0.self_attn.q_proj.weight"]?.shape == [8, 8])
    #expect(sanitized["layers.0.self_attn.kv_proj.weight"]?.shape == [8, 8])
    #expect(sanitized["layers.0.mlp.gate_up_proj.weight"]?.shape == [32, 8])
  }

  @Test("Duplicate or unordered target taps are rejected")
  func rejectsInvalidTargetLayers() {
    let data = Data(
      tinyConfigurationJSON.replacingOccurrences(
        of: "\"target_layer_ids\": [0, 2]",
        with: "\"target_layer_ids\": [2, 2]"
      ).utf8
    )
    #expect(throws: LagunaDFlashConfigurationError.self) {
      try JSONDecoder().decode(LagunaDFlashConfiguration.self, from: data)
    }
  }

  @Test("Target capture preserves logits and the block drafter runs end to end")
  func runsTinyTargetAndDraft() throws {
    let targetConfiguration = try JSONDecoder().decode(
      LagunaConfiguration.self,
      from: Data(Self.tinyTargetConfigurationJSON.utf8)
    )
    let draft = LagunaDFlashModel(try decodeTinyConfiguration())
    let target = LagunaModel(targetConfiguration)
    try target.configureDFlash(draft.targetDescriptor)

    let batchedTokens = MLXArray([Int32(1), 2, 3])[.newAxis]
    let plainLogits = target(batchedTokens, cache: nil as [KVCache]?)
    var emitState = LMOutput.State()
    emitState[mtpEmitFlagKey] = true
    let captured = target(
      LMInput.Text(tokens: batchedTokens), cache: nil as [KVCache]?, state: emitState)
    eval(plainLogits, captured.logits)

    #expect(plainLogits.shape == captured.logits.shape)
    #expect(
      plainLogits.asArray(Float.self) == captured.logits.asArray(Float.self))
    let targetHidden = try #require(captured.state?[mtpLastHiddenStatesKey])
    #expect(targetHidden.shape == [1, 3, 16])

    var draftState = draft.makeState(parameters: nil)
    draft.prepareDrafterState(
      target: target,
      promptTokens: batchedTokens[0],
      targetHidden: targetHidden,
      firstBonus: MLXArray([Int32(4)]),
      positionDeltas: nil,
      state: &draftState,
      sampler: ArgMaxSampler()
    )
    let proposals = draft.draftBlock(
      target: target,
      lastToken: MLXArray([Int32(4)]),
      lastHidden: targetHidden[0..., (-1)..., 0...],
      sharedKV: [:],
      positionDeltas: nil,
      queryOffset: 3,
      blockSize: 4,
      state: &draftState,
      sampler: ArgMaxSampler()
    )
    eval(proposals)
    #expect(proposals.shape == [1, 3])
  }

  @Test("Greedy DFlash stays token-identical across chunked prefill and cache wrap")
  func greedyIteratorMatchesTargetOnly() throws {
    let targetConfiguration = try JSONDecoder().decode(
      LagunaConfiguration.self,
      from: Data(Self.tinyTargetConfigurationJSON.utf8)
    )
    let draftConfiguration = try decodeTinyConfiguration()

    for seed in UInt64(0)..<64 {
      let (draft, target) = withRandomState(MLXRandom.RandomState(seed: seed)) {
        (LagunaDFlashModel(draftConfiguration), LagunaModel(targetConfiguration))
      }
      try target.configureDFlash(draft.targetDescriptor)

      let prompt: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      let parameters = GenerateParameters(
        maxTokens: 24,
        temperature: 0,
        prefill: PrefillParameters(stepSize: 3, chunking: .balanced)
      )

      var ordinary = try TokenIterator(
        input: LMInput(tokens: MLXArray(prompt)),
        model: target,
        cache: target.newCache(parameters: parameters),
        parameters: parameters
      )
      var ordinaryTokens = [Int]()
      while let token = ordinary.next() { ordinaryTokens.append(token) }

      var speculative = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray(prompt)),
        mainModel: target,
        drafter: draft,
        mainCache: target.newCache(parameters: parameters),
        parameters: parameters,
        blockSize: 4
      )
      var speculativeTokens = [Int]()
      while let token = speculative.next() { speculativeTokens.append(token) }

      #expect(speculative.proposedDraftTokens > 0)
      guard speculativeTokens == ordinaryTokens else {
        Issue.record(
          "seed \(seed): speculative \(speculativeTokens) != ordinary \(ordinaryTokens)")
        return
      }
    }
  }

  @Test("DFlash commits only the target-verified prefix across its rotating window")
  func commitsAcceptedTargetContextOnly() throws {
    let targetConfiguration = try JSONDecoder().decode(
      LagunaConfiguration.self,
      from: Data(Self.tinyTargetConfigurationJSON.utf8)
    )
    let draftConfiguration = try decodeTinyConfiguration()
    let (draft, target) = withRandomState(MLXRandom.RandomState(seed: 101)) {
      (LagunaDFlashModel(draftConfiguration), LagunaModel(targetConfiguration))
    }
    try target.configureDFlash(draft.targetDescriptor)

    var state = draft.makeState(parameters: nil)
    let promptTokens = MLXArray(Array(Int32(1)...Int32(10)))
    let promptSuffix = MLXArray.zeros([1, 8, 16])
    draft.prepareDrafterState(
      target: target,
      promptTokens: promptTokens,
      targetHidden: promptSuffix,
      firstBonus: MLXArray([Int32(11)]),
      positionDeltas: nil,
      state: &state,
      sampler: ArgMaxSampler()
    )
    #expect(state.nextPosition == 10)
    #expect(state.cache.allSatisfy { $0.offset == 8 })

    let proposals = draft.draftBlock(
      target: target,
      lastToken: MLXArray([Int32(11)]),
      lastHidden: promptSuffix[0..., (-1)..., 0...],
      sharedKV: [:],
      positionDeltas: nil,
      queryOffset: 10,
      blockSize: 4,
      state: &state,
      sampler: ArgMaxSampler()
    )
    eval(proposals)
    #expect(state.nextPosition == 10)
    #expect(state.cache.allSatisfy { $0.offset == 8 })

    // Four verifier rows were evaluated, but two accepted drafts mean only
    // [incoming bonus, accepted_1, accepted_2] become persistent context.
    draft.commitDrafterState(
      target: target,
      targetHidden: MLXArray.zeros([1, 4, 16]),
      draftTokens: proposals,
      acceptedCount: 2,
      finalToken: MLXArray([Int32(12)]),
      positionDeltas: nil,
      state: &state,
      sampler: ArgMaxSampler()
    )
    #expect(state.nextPosition == 13)
    #expect(state.cache.allSatisfy { $0.offset == 11 })
  }

  private func decodeOfficialConfiguration() throws -> LagunaDFlashConfiguration {
    try JSONDecoder().decode(
      LagunaDFlashConfiguration.self,
      from: Data(
        """
        {
          "architectures": ["DFlashLagunaForCausalLM"],
          "attention_bias": false,
          "head_dim": 128,
          "hidden_size": 2048,
          "intermediate_size": 8192,
          "max_position_embeddings": 262144,
          "model_type": "laguna",
          "num_attention_heads": 64,
          "num_hidden_layers": 5,
          "num_key_value_heads": 8,
          "rms_norm_eps": 0.000001,
          "rope_theta": 500000,
          "sliding_window": 512,
          "vocab_size": 100352,
          "layer_types": [
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention"
          ],
          "dflash_config": {
            "block_size": 16,
            "mask_token_id": 12,
            "num_target_layers": 40,
            "target_layer_ids": [1, 13, 25, 33, 39],
            "causal": true
          }
        }
        """.utf8
      )
    )
  }

  private func decodeTinyConfiguration() throws -> LagunaDFlashConfiguration {
    try JSONDecoder().decode(
      LagunaDFlashConfiguration.self,
      from: Data(Self.tinyConfigurationJSON.utf8)
    )
  }

  private static let tinyConfigurationJSON =
    """
    {
      "architectures": ["DFlashLagunaForCausalLM"],
      "attention_bias": false,
      "head_dim": 4,
      "hidden_size": 8,
      "intermediate_size": 16,
      "max_position_embeddings": 128,
      "model_type": "laguna",
      "num_attention_heads": 2,
      "num_hidden_layers": 1,
      "num_key_value_heads": 1,
      "rms_norm_eps": 0.000001,
      "rope_theta": 500000,
      "sliding_window": 8,
      "vocab_size": 32,
      "layer_types": ["sliding_attention"],
      "dflash_config": {
        "block_size": 4,
        "mask_token_id": 3,
        "num_target_layers": 4,
        "target_layer_ids": [0, 2],
        "causal": true
      }
    }
    """

  private static let tinyTargetConfigurationJSON =
    """
    {
      "model_type": "laguna",
      "vocab_size": 32,
      "hidden_size": 8,
      "intermediate_size": 16,
      "num_hidden_layers": 4,
      "num_attention_heads": 2,
      "num_attention_heads_per_layer": [2, 2, 2, 2],
      "num_key_value_heads": 1,
      "head_dim": 4,
      "max_position_embeddings": 128,
      "rms_norm_eps": 0.000001,
      "attention_bias": false,
      "qkv_bias": false,
      "gating": "per-head",
      "tie_word_embeddings": false,
      "sliding_window": 8,
      "rope_theta": 500000,
      "layer_types": [
        "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention"
      ],
      "mlp_layer_types": ["dense", "dense", "dense", "dense"],
      "num_experts": 2,
      "num_experts_per_tok": 1,
      "moe_intermediate_size": 8,
      "shared_expert_intermediate_size": 8,
      "moe_routed_scaling_factor": 1,
      "moe_router_score_func": "sigmoid"
    }
    """

  private var tinyConfigurationJSON: String { Self.tinyConfigurationJSON }
}
