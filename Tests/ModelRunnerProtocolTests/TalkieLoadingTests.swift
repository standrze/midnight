import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import ModelRunnerCore

@Suite("Talkie checkpoint loading", .serialized)
struct TalkieLoadingTests {
    private var fixture: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Fixtures/TalkieTiny")
    }

    @Test("Only uniform affine quantization is eligible for projection fusion")
    func fusionQuantizationPolicy() throws {
        let data = try Data(contentsOf: fixture.appendingPathComponent("config.json"))
        let original = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let cases: [([String: Any], Bool)] = [
            (["bits": 4, "group_size": 32], true),
            (["bits": 8, "group_size": 32, "mode": "affine"], true),
            (["bits": 4, "group_size": 16, "mode": "nvfp4"], false),
            (["bits": 4, "group_size": 32, "model.blocks.0.attn.attn_query": ["bits": 8]], false),
        ]
        for (quantization, expected) in cases {
            var root = original
            root["quantization"] = quantization
            let config = try JSONDecoder().decode(
                TalkieConfiguration.self,
                from: JSONSerialization.data(withJSONObject: root))
            #expect(config.permitsProjectionFusion == expected)
        }
    }

    @Test("Fusion cannot hide compensating malformed projection dimensions")
    func validatesEachProjection() throws {
        try Device.withDefaultDevice(.cpu) {
            let config = try JSONDecoder().decode(
                TalkieConfiguration.self,
                from: Data(contentsOf: fixture.appendingPathComponent("config.json")))
            let model = TalkieModel(config, fuseProjections: true)
            var weights = try MLX.loadArrays(url: fixture.appendingPathComponent("weights-fp32.safetensors"))
            weights["model.blocks.0.attn.attn_query.weight"] = MLXArray.zeros([15, 16])
            weights["model.blocks.0.attn.attn_key.weight"] = MLXArray.zeros([17, 16])
            #expect(throws: (any Error).self) {
                try model.update(
                    parameters: ModuleParameters.unflattened(model.sanitize(weights: weights)), verify: .all)
            }
        }
    }

    @Test("Talkie stop markers require an explicit model identity")
    func stopMarkers() {
        #expect(
            TalkieLoadingOptions.additionalEOSTokens(configuration: Data(#"{"model_type":"talkie"}"#.utf8))
                == ["<|end|>", "<|user|>", "<|assistant|>", "<|system|>"])
        for json in ["{}", #"{"model_type":"llama"}"#, "invalid"] {
            #expect(TalkieLoadingOptions.additionalEOSTokens(configuration: Data(json.utf8)).isEmpty)
        }
    }

    @Test("Adapter fusion override is scoped to its loading task")
    func adapterFusionScope() async {
        #expect(TalkieLoadingOptions.fuseProjections)
        await TalkieLoadingOptions.$fuseProjections.withValue(false) {
            let inherited = await Task { TalkieLoadingOptions.fuseProjections }.value
            #expect(!inherited)
        }
        #expect(TalkieLoadingOptions.fuseProjections)
    }
}
