import Foundation
import Testing

@testable import ModelRunnerCore

struct GemmaAssistantCompatibilityTests {
    private func pair() throws -> (target: [String: Any], assistant: [String: Any]) {
        let text: [String: Any] = [
            "model_type": "gemma4_text", "hidden_size": 2816, "intermediate_size": 2112,
            "num_hidden_layers": 30, "num_attention_heads": 16, "num_key_value_heads": 8,
            "num_global_key_value_heads": 2, "head_dim": 256, "global_head_dim": 512,
            "sliding_window": 1024, "vocab_size": 262144, "attention_k_eq_v": true,
            "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
            "layer_types": Array(
                repeating: [
                    "sliding_attention", "sliding_attention",
                    "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
                ], count: 5
            ).flatMap { $0 },
        ]
        var draft = text
        draft["hidden_size"] = 1024
        draft["intermediate_size"] = 8192
        draft["num_hidden_layers"] = 4
        draft["num_kv_shared_layers"] = 4
        draft["layer_types"] = ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
        return (
            text,
            [
                "model_type": "gemma4_assistant", "text_config": draft,
                "backbone_hidden_size": 2816, "block_size": 4, "use_ordered_embeddings": false,
            ]
        )
    }

    private func validate(_ target: [String: Any], _ assistant: [String: Any], block: Int? = nil, bits: Int? = nil)
        throws -> Int
    {
        try GemmaAssistantCompatibility.validate(
            target: JSONSerialization.data(withJSONObject: ["model_type": "gemma4", "text_config": target]),
            assistant: JSONSerialization.data(withJSONObject: assistant), blockSize: block, quantizationBits: bits)
    }

    @Test func matchingA4B() throws {
        let (target, assistant) = try pair()
        #expect(try validate(target, assistant) == 4)
        #expect(try validate(target, assistant, block: 2) == 2)
        #expect(throws: (any Error).self) { try validate(target, assistant, block: 1) }
        #expect(throws: (any Error).self) { try validate(target, assistant, block: 5) }
    }

    @Test func validatesAssistantQuantization() throws {
        var (target, assistant) = try pair()
        #expect(try validate(target, assistant, bits: 4) == 4)
        #expect(try validate(target, assistant, bits: 8) == 4)
        #expect(throws: (any Error).self) { try validate(target, assistant, bits: 3) }
        assistant["quantization"] = ["bits": 4, "group_size": 64]
        #expect(throws: (any Error).self) { try validate(target, assistant, bits: 4) }
    }

    @Test(arguments: [
        "hidden_size", "vocab_size", "num_key_value_heads", "num_global_key_value_heads",
        "head_dim", "global_head_dim", "sliding_window",
    ])
    func rejectsShapeMismatch(_ key: String) throws {
        var (target, assistant) = try pair()
        target[key] = 1
        #expect(throws: (any Error).self) { try validate(target, assistant) }
    }

    @Test func rejectsRotaryMismatch() throws {
        var (target, assistant) = try pair()
        target["rope_parameters"] = ["sliding_attention": ["rope_theta": 9999]]
        #expect(throws: (any Error).self) { try validate(target, assistant) }
    }
}
