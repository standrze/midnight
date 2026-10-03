import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import Testing

@Suite("Gemma text speculative target state")
struct GemmaMTPStateTests {
    @Test(
        "Official A4B assistant weights load",
        .enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_TEST_GEMMA_ASSISTANT"] != nil))
    func officialAssistantWeights() throws {
        let path = try #require(ProcessInfo.processInfo.environment["MIDNIGHT_TEST_GEMMA_ASSISTANT"])
        try Device.withDefaultDevice(.cpu) {
            let directory = URL(fileURLWithPath: path)
            let config = try JSONDecoder().decode(
                Gemma4AssistantConfiguration.self,
                from: Data(contentsOf: directory.appendingPathComponent("config.json")))
            let model = Gemma4AssistantDraftModel(config)
            try loadWeights(modelDirectory: directory, model: model, perLayerQuantization: nil)
            eval(model)
            #expect(config.backboneHiddenSize == 2816)
            #expect(model.parameters().flattened().count == 48)
        }
    }

    @Test("MTP state preserves logits, cache sources and absolute offsets", arguments: [false, true])
    func targetState(_ wrapped: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let textConfig: [String: Any] = [
                "model_type": "gemma4_text", "hidden_size": 16, "intermediate_size": 32,
                "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 8, "global_head_dim": 8, "sliding_window": 4, "vocab_size": 32,
                "layer_types": ["sliding_attention", "full_attention"],
                "hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0,
            ]
            let model: any LanguageModel
            if wrapped {
                let data = try JSONSerialization.data(withJSONObject: [
                    "model_type": "gemma4", "text_config": textConfig,
                ])
                model = Gemma4Model(try JSONDecoder().decode(MLXLLM.Gemma4Configuration.self, from: data))
            } else {
                let data = try JSONSerialization.data(withJSONObject: textConfig)
                model = Gemma4TextModel(try JSONDecoder().decode(MLXLLM.Gemma4TextConfiguration.self, from: data))
            }
            let tokens = MLXArray(Array(1...6)).reshaped(1, 6)
            let ordinaryCache = try model.newCache(parameters: nil)
            let draftCache = try model.newCache(parameters: nil)
            let expected = model(tokens, cache: ordinaryCache)
            var request = LMOutput.State()
            request[mtpEmitFlagKey] = true
            let result = model(LMInput(tokens: tokens).text, cache: draftCache, state: request)
            eval(expected, result.logits)
            #expect(MLX.max(abs(expected - result.logits)).item(Float.self) < 0.0001)
            let state = try #require(result.state)
            #expect(state[mtpLastHiddenStatesKey]?.shape == [1, 6, 16])
            #expect(state[mtpSharedKVSourceIndicesKey] == ["sliding_attention": 0, "full_attention": 1])
            #expect(state[mtpSharedKVOffsetsKey] == ["sliding_attention": 6, "full_attention": 6])
            // Keep complete retained snapshots; the drafter narrows only its read view.
            #expect(state[mtpSharedKVStatesKey]?["sliding_attention"]?.0.dim(2) == 6)
            let embedding = try #require(model as? any Gemma4MTPEmbeddingProvider)
            #expect(embedding.mtpInputEmbedding(tokens, scaleDType: .float32).shape == [1, 6, 16])
            let draftConfig = try JSONSerialization.data(withJSONObject: [
                "text_config": textConfig, "backbone_hidden_size": 16, "block_size": 2,
                "tie_word_embeddings": true, "use_ordered_embeddings": false,
            ])
            let drafter = Gemma4AssistantDraftModel(
                try JSONDecoder().decode(Gemma4AssistantConfiguration.self, from: draftConfig))
            let shared = try #require(state[mtpSharedKVStatesKey])
            let hidden = try #require(state[mtpLastHiddenStatesKey])[0..., (-1)..., 0...]
            let proposed = drafter.draftBlock(
                target: model, lastToken: MLXArray([7]), lastHidden: hidden,
                sharedKV: shared, positionDeltas: nil, queryOffset: 6,
                blockSize: 2, sampler: ArgMaxSampler())
            var bounded = shared
            let sliding = try #require(shared["sliding_attention"])
            bounded["sliding_attention"] = (
                sliding.0[.ellipsis, 2..., 0...], sliding.1[.ellipsis, 2..., 0...]
            )
            let boundedProposal = drafter.draftBlock(
                target: model, lastToken: MLXArray([7]), lastHidden: hidden,
                sharedKV: bounded, positionDeltas: nil, queryOffset: 6,
                blockSize: 2, sampler: ArgMaxSampler())
            #expect(proposed.shape == [1, 1])
            #expect(proposed.asArray(Int32.self) == boundedProposal.asArray(Int32.self))
            #expect(shared["sliding_attention"]?.0.dim(2) == 6)
            let next = MLXArray([7]).reshaped(1, 1)
            let ordinaryNext = model(LMInput(tokens: next).text, cache: ordinaryCache, state: nil)
            let draftNext = model(LMInput(tokens: next).text, cache: draftCache, state: state)
            eval(ordinaryNext.logits, draftNext.logits)
            #expect(MLX.max(abs(ordinaryNext.logits - draftNext.logits)).item(Float.self) < 0.0001)
            #expect(ordinaryNext.state?[mtpLastHiddenStatesKey] == nil)
            #expect(draftNext.state?[mtpSharedKVOffsetsKey]?["full_attention"] == 7)
            #expect(draftNext.state?[mtpSharedKVStatesKey]?["sliding_attention"]?.0.dim(2) == 7)

            let parameters = GenerateParameters(maxTokens: 12, temperature: 0)
            let input = LMInput(tokens: MLXArray(Array(1...6)))
            var baseline = try TokenIterator(input: input, model: model, parameters: parameters)
            var expectedTokens: [Int] = []
            while let token = baseline.next() {
                expectedTokens.append(token)
            }
            for blockSize in [2, 4] {
                var speculative = try MTPSpeculativeTokenIterator(
                    input: input, mainModel: model, drafter: drafter,
                    parameters: parameters, blockSize: blockSize)
                var actualTokens: [Int] = []
                while let token = speculative.next() {
                    actualTokens.append(token)
                }
                #expect(actualTokens == expectedTokens)
                #expect(speculative.proposedCount > 0)
            }
        }
    }
}
