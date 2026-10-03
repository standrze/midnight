import Foundation
import MLX
@_spi(Testing) import MLXLLM
import MLXLMCommon
import MLXNN
@_spi(Testing) import MLXVLM
import Testing

@Suite("Gemma assistant explicit-zero mask parity", .serialized)
struct GemmaAssistantUnmaskedTests {
    @Test("Text and VLM assistants preserve hidden states, logits and proposals", arguments: [false, true])
    func maskParity(_ quantized: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let text: [String: Any] = [
                "model_type": "gemma4_text", "hidden_size": 64, "intermediate_size": 128,
                "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 32, "global_head_dim": 64, "num_global_key_value_heads": 1,
                "attention_k_eq_v": true, "sliding_window": 4, "vocab_size": 128,
                "layer_types": ["sliding_attention", "full_attention"],
                "hidden_size_per_layer_input": 0, "num_kv_shared_layers": 2,
                "use_double_wide_mlp": false,
            ]
            let data = try JSONSerialization.data(withJSONObject: [
                "text_config": text, "backbone_hidden_size": 64,
                "tie_word_embeddings": true, "use_ordered_embeddings": false,
            ])
            let native = Gemma4TextAssistantDraftModel(
                try JSONDecoder().decode(Gemma4TextAssistantConfiguration.self, from: data))
            let vlm = Gemma4AssistantDraftModel(
                try JSONDecoder().decode(Gemma4AssistantConfiguration.self, from: data))
            try native.update(parameters: vlm.parameters(), verify: .all)
            if quantized {
                quantize(model: native, groupSize: 32, bits: 4)
                quantize(model: vlm, groupSize: 32, bits: 4)
            }
            for queryLength in [1, 3] {
                for slidingLength in [1, 4] {
                    let shared: [String: (MLXArray, MLXArray)] = [
                        "sliding_attention": (
                            values([1, 1, slidingLength, 32]), values([1, 1, slidingLength, 32])
                        ),
                        "full_attention": (values([1, 1, 7, 64]), values([1, 1, 7, 64])),
                    ]
                    let input = values([1, queryLength, 128])
                    for offset in [7, 1031] {
                        let nativeMasked = native.forwardHidden(
                            inputsEmbeds: input, sharedKV: shared, queryOffset: offset, useUnmaskedAttention: false)
                        let nativeUnmasked = native.forwardHidden(
                            inputsEmbeds: input, sharedKV: shared, queryOffset: offset, useUnmaskedAttention: true)
                        let vlmMasked = vlm.forwardHidden(
                            inputsEmbeds: input, sharedKV: shared, queryOffset: offset, useUnmaskedAttention: false)
                        let vlmUnmasked = vlm.forwardHidden(
                            inputsEmbeds: input, sharedKV: shared, queryOffset: offset, useUnmaskedAttention: true)
                        for (expected, actual) in [
                            (nativeMasked, nativeUnmasked), (vlmMasked, vlmUnmasked),
                        ] {
                            #expect(MLX.max(abs(expected.lastHidden - actual.lastHidden)).item(Float.self) < 0.0001)
                            #expect(MLX.max(abs(expected.logits - actual.logits)).item(Float.self) < 0.0001)
                            #expect(
                                argMax(expected.logits, axis: -1).asArray(Int32.self)
                                    == argMax(actual.logits, axis: -1).asArray(Int32.self))
                        }
                    }
                }
            }
        }
    }

    private func values(_ shape: [Int]) -> MLXArray {
        sin(MLXArray(0..<shape.reduce(1, *)).asType(.float32) * 0.137).reshaped(shape)
    }
}
