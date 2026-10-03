import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@Suite("Gemma 3 attention layout and small-model support")
struct Gemma3LayoutTests {
    @Test("Canonical and private sliding-pattern keys decode identically", arguments: [false, true])
    func patternAlias(_ nested: Bool) throws {
        let canonical = try decode(["sliding_window_pattern": 2], nested: nested)
        let legacy = try decode(["_sliding_window_pattern": 2], nested: nested)
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        #expect(try encoder.encode(canonical) == encoder.encode(legacy))
    }

    @Test("Canonical pattern takes precedence over the private alias")
    func canonicalPrecedence() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma3TextModel(
                try decode([
                    "sliding_window_pattern": 1, "_sliding_window_pattern": 2,
                ]))
            #expect(try model.newCache().allSatisfy { $0 is StandardKVCache })
        }
    }

    @Test("270M metadata preserves its explicit 18-layer layout")
    func smallModelMetadata() throws {
        let layout = (0..<18).map { ($0 + 1) % 6 == 0 ? "full_attention" : "sliding_attention" }
        let configuration = try decode([
            "hidden_size": 640, "intermediate_size": 2048, "num_hidden_layers": 18,
            "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 256,
            "vocab_size": 262_144, "sliding_window": 512, "_sliding_window_pattern": 6,
            "layer_types": layout,
        ])
        let encoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(configuration)) as? [String: Any]
        #expect(encoded?["hidden_size"] as? Int == 640)
        #expect(encoded?["num_hidden_layers"] as? Int == 18)
        #expect(encoded?["layer_types"] as? [String] == layout)
    }

    @Test("Invalid layout metadata fails before allocating a model")
    func invalidMetadata() throws {
        for invalid: [String: Any] in [
            ["sliding_window_pattern": 0],
            ["_sliding_window_pattern": -1],
            ["layer_types": ["full_attention"]],
            ["layer_types": ["full_attention", "unknown_attention"]],
        ] {
            #expect(throws: DecodingError.self) { try decode(invalid) }
        }
    }

    @Test("Explicit layer types choose matching caches")
    func explicitCacheLayout() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma3TextModel(
                try decode([
                    "sliding_window_pattern": 2,
                    "layer_types": ["full_attention", "sliding_attention"],
                ]))
            let cache = try model.newCache()
            #expect(cache[0] is StandardKVCache)
            #expect(cache[1] is RotatingKVCache)
        }
    }

    @Test("Tiny and reordered layouts preserve cached logits", arguments: [false, true], [1, 2, 4])
    func cachedLogits(_ quantized: Bool, _ chunk: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            for layout in [
                ["full_attention", "sliding_attention"],
                ["sliding_attention", "sliding_attention"],
                ["full_attention", "full_attention"],
            ] {
                let model = Gemma3TextModel(try decode(["layer_types": layout]))
                if quantized {
                    quantize(model: model, groupSize: 64, bits: 4)
                }
                let tokens = MLXArray(Array(1...10)).reshaped(1, 10)
                let expected = model(tokens, cache: nil)
                eval(expected)
                let cache = try model.newCache()
                var pieces: [MLXArray] = []
                for start in stride(from: 0, to: 10, by: chunk) {
                    let logits = model(tokens[0..., start..<min(start + chunk, 10)], cache: cache)
                    eval(logits)
                    pieces.append(logits)
                }
                let actual = concatenated(pieces, axis: 1)
                #expect(MLX.max(abs(expected - actual)).item(Float.self) < 0.0001)
            }
        }
    }

    @Test("A reduced-layer pattern-only fixture has no nonexistent global cache")
    func tinyPatternOnly() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma3TextModel(try decode([:]))
            let cache = try model.newCache()
            #expect(cache.allSatisfy { $0 is RotatingKVCache })
            let logits = model(MLXArray([1, 2]).reshaped(1, 2), cache: cache)
            eval(logits)
            #expect(logits.shape == [1, 2, 128])
        }
    }

    private func decode(_ overrides: [String: Any], nested: Bool = false) throws -> Gemma3TextConfiguration {
        let base: [String: Any] = [
            "model_type": "gemma3_text", "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 32, "vocab_size": 128, "sliding_window": 4,
        ]
        let text = base.merging(overrides) { _, new in new }
        let root = nested ? ["model_type": "gemma3", "text_config": text] : text
        return try JSONDecoder().decode(
            Gemma3TextConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
    }
}
