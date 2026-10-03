import Foundation
import MLX
@_spi(Testing) import MLXLLM
import MLXLMCommon
import MLXNN
@_spi(Testing) import MLXVLM
import Testing

@testable import ModelRunnerCore

@Suite("Text-only Gemma assistant reference parity")
struct GemmaTextAssistantTests {
    @Test(
        "Official A4B assistant matches reference proposals",
        .enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_TEST_GEMMA_ASSISTANT"] != nil))
    func officialWeights() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["MIDNIGHT_TEST_GEMMA_ASSISTANT"])
        try await MLXPinnedRuntime.shared.run {
            try Device.withDefaultDevice(.gpu) {
                let directory = URL(fileURLWithPath: path)
                let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
                let reference = Gemma4AssistantDraftModel(
                    try JSONDecoder().decode(Gemma4AssistantConfiguration.self, from: data))
                let native = Gemma4TextAssistantDraftModel(
                    try JSONDecoder().decode(Gemma4TextAssistantConfiguration.self, from: data))
                try loadWeights(modelDirectory: directory, model: reference, perLayerQuantization: nil)
                try loadWeights(modelDirectory: directory, model: native, perLayerQuantization: nil)
                #expect(native.parameters().flattened().count == 48)
                for bits in [0, 4] {
                    if bits > 0 {
                        quantize(model: reference, groupSize: 64, bits: bits)
                        quantize(model: native, groupSize: 64, bits: bits)
                    }
                    eval(reference, native)
                    let shared: [String: (MLXArray, MLXArray)] = [
                        "sliding_attention": (
                            values([1, 8, 16, 256]).asType(.bfloat16), values([1, 8, 16, 256]).asType(.bfloat16)
                        ),
                        "full_attention": (
                            values([1, 2, 32, 512]).asType(.bfloat16), values([1, 2, 32, 512]).asType(.bfloat16)
                        ),
                    ]
                    for offset in [32, 1031] {
                        let inputs = values([1, 1, 5632]).asType(.bfloat16)
                        let expected = reference.forwardHidden(
                            inputsEmbeds: inputs, sharedKV: shared, queryOffset: offset)
                        let actual = native.forwardHidden(inputsEmbeds: inputs, sharedKV: shared, queryOffset: offset)
                        let hiddenError = MLX.max(abs(expected.lastHidden - actual.lastHidden)).item(Float.self)
                        let logitsError = MLX.max(abs(expected.logits - actual.logits)).item(Float.self)
                        print(
                            "Official assistant bits=\(bits) offset=\(offset) hidden_error=\(hiddenError) logits_error=\(logitsError)"
                        )
                        #expect(hiddenError < 0.0001)
                        #expect(logitsError < 0.0001)
                        #expect(
                            argMax(expected.logits, axis: -1).asArray(Int32.self)
                                == argMax(actual.logits, axis: -1).asArray(Int32.self))
                    }
                }
            }
        }
    }

    @Test("Shared-only decoder matches the reference", arguments: [false, true])
    func referenceParity(_ quantized: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let text: [String: Any] = [
                "model_type": "gemma4_text", "hidden_size": 64, "intermediate_size": 128,
                "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 32, "global_head_dim": 64, "num_global_key_value_heads": 1,
                "attention_k_eq_v": true, "sliding_window": 4, "vocab_size": 128,
                "layer_types": ["sliding_attention", "full_attention"],
                "hidden_size_per_layer_input": 0, "num_kv_shared_layers": 2,
                "use_double_wide_mlp": false,
                "rope_parameters": [
                    "sliding_attention": ["rope_type": "default", "rope_theta": 10000],
                    "full_attention": [
                        "rope_type": "proportional", "rope_theta": 1_000_000,
                        "partial_rotary_factor": 0.25,
                    ],
                ],
            ]
            let data = try JSONSerialization.data(withJSONObject: [
                "text_config": text, "backbone_hidden_size": 64,
                "tie_word_embeddings": true, "use_ordered_embeddings": false,
            ])
            let reference = Gemma4AssistantDraftModel(
                try JSONDecoder().decode(Gemma4AssistantConfiguration.self, from: data))
            let native = Gemma4TextAssistantDraftModel(
                try JSONDecoder().decode(Gemma4TextAssistantConfiguration.self, from: data))
            // Strict loading catches accidental private K/V projections or MLP shape changes.
            try native.update(parameters: reference.parameters(), verify: .all)
            if quantized {
                quantize(model: reference, groupSize: 32, bits: 4)
                quantize(model: native, groupSize: 32, bits: 4)
            }
            let shared: [String: (MLXArray, MLXArray)] = [
                "sliding_attention": (values([1, 1, 4, 32]), values([1, 1, 4, 32])),
                "full_attention": (values([1, 1, 7, 64]), values([1, 1, 7, 64])),
            ]
            for offset in [7, 1031] {
                let inputs = values([1, 1, 128])
                let expected = reference.forwardHidden(inputsEmbeds: inputs, sharedKV: shared, queryOffset: offset)
                let actual = native.forwardHidden(inputsEmbeds: inputs, sharedKV: shared, queryOffset: offset)
                #expect(MLX.max(abs(expected.lastHidden - actual.lastHidden)).item(Float.self) < 0.0001)
                #expect(MLX.max(abs(expected.logits - actual.logits)).item(Float.self) < 0.0001)
                #expect(
                    argMax(expected.logits, axis: -1).asArray(Int32.self)
                        == argMax(actual.logits, axis: -1).asArray(Int32.self))
            }
        }
    }

    private func values(_ shape: [Int]) -> MLXArray {
        let count = shape.reduce(1, *)
        return sin(MLXArray(0..<count).asType(.float32) * 0.137).reshaped(shape)
    }
}
