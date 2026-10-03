import Foundation
import ModelRunnerProtocol
import Testing

@testable import ModelRunnerCore

struct GemmaExpertProjectionCompatibilityTests {
    private func configuration(_ edit: (inout [String: Any]) -> Void = { _ in }) throws -> Data {
        var root: [String: Any] = [
            "model_type": "gemma4",
            "text_config": [
                "hidden_size": 2816, "num_hidden_layers": 30, "num_experts": 128,
                "top_k_experts": 8, "moe_intermediate_size": 704, "enable_moe_block": true,
            ], "quantization": ["bits": 4, "group_size": 64, "mode": "affine"],
        ]
        edit(&root)
        return try JSONSerialization.data(withJSONObject: root)
    }

    @Test func acceptsMatchingA4BAndDenseOverrides() throws {
        let data = try configuration { root in
            var q = root["quantization"] as! [String: Any]
            q["language_model.model.layers.0.mlp.gate_proj"] = ["bits": 8, "group_size": 64]
            q["language_model.model.layers.0.experts.switch_glu.gate_proj"] = ["bits": 4, "group_size": 64]
            root["quantization"] = q
        }
        try GemmaExpertProjectionCompatibility.validate(configuration: data, engine: .metal, hasAdapter: false)
    }

    @Test(arguments: ["hidden_size", "num_hidden_layers", "num_experts", "top_k_experts", "moe_intermediate_size"])
    func rejectsOtherGeometry(_ key: String) throws {
        let data = try configuration { root in
            var text = root["text_config"] as! [String: Any]
            text[key] = 1
            root["text_config"] = text
        }
        #expect(throws: (any Error).self) {
            try GemmaExpertProjectionCompatibility.validate(configuration: data, engine: .metal, hasAdapter: false)
        }
    }

    @Test func rejectsUnsupportedProjectionPrecision() throws {
        for setting in [
            false, ["bits": 8, "group_size": 64],
            ["bits": 4, "group_size": 32], ["bits": 4, "group_size": 64, "mode": 1],
        ] as [Any] {
            let data = try configuration { root in
                var q = root["quantization"] as! [String: Any]
                q["language_model.model.layers.0.experts.switch_glu.up_proj"] = setting
                root["quantization"] = q
            }
            #expect(throws: (any Error).self) {
                try GemmaExpertProjectionCompatibility.validate(configuration: data, engine: .metal, hasAdapter: false)
            }
        }
    }

    @Test func rejectsAdaptersAndOtherEngines() throws {
        let data = try configuration()
        for (engine, adapter) in [(ModelEngine.cpu, false), (.cuda, false), (.metal, true)] {
            #expect(throws: (any Error).self) {
                try GemmaExpertProjectionCompatibility.validate(
                    configuration: data, engine: engine, hasAdapter: adapter)
            }
        }
    }
}
