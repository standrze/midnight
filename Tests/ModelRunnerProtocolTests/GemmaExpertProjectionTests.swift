import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import ModelRunnerCore

@Suite("Gemma combined expert projections")
struct GemmaExpertProjectionTests {
    @Test(
        "Actual A4B Q4 expert gate/up rows can be combined",
        .enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_TEST_GEMMA_TARGET"] != nil))
    func actualExpert() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["MIDNIGHT_TEST_GEMMA_TARGET"])
        let output = try #require(ProcessInfo.processInfo.environment["MIDNIGHT_TEST_EXPERT_PROJECTION_OUTPUT"])
        #expect(!FileManager.default.fileExists(atPath: output))
        guard !FileManager.default.fileExists(atPath: output) else {
            return
        }
        try await MLXPinnedRuntime.shared.run {
            try Device.withDefaultDevice(.gpu) {
                let root = URL(fileURLWithPath: path)
                let configData = try Data(contentsOf: root.appendingPathComponent("config.json"))
                try GemmaExpertProjectionCompatibility.validate(
                    configuration: configData,
                    engine: .metal, hasAdapter: false)
                let config = try #require(JSONSerialization.jsonObject(with: configData) as? [String: Any])
                let text = try #require(config["text_config"] as? [String: Any])
                #expect(text["hidden_size"] as? Int == 2816)
                #expect(text["num_experts"] as? Int == 128)
                #expect(text["moe_intermediate_size"] as? Int == 704)
                let prefix = "language_model.model.layers.0.experts.switch_glu."
                let weights = try Self.expertWeights(root, prefix: prefix)
                let reference = SwitchGLU(
                    inputDims: 2816, hiddenDims: 704, numExperts: 128,
                    activation: geluApproximate)
                let combined = FusedGateUpSwitchGLU(
                    inputDims: 2816, hiddenDims: 704, numExperts: 128,
                    activation: geluApproximate)
                quantize(model: reference, groupSize: 64, bits: 4)
                quantize(model: combined, groupSize: 64, bits: 4)
                try reference.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
                var fused: [String: MLXArray] = [:]
                for suffix in ["weight", "scales", "biases"] {
                    let gate = try #require(weights["gate_proj.\(suffix)"])
                    let up = try #require(weights["up_proj.\(suffix)"])
                    #expect(gate.shape == up.shape)
                    fused["gate_up_proj.\(suffix)"] = concatenated([gate, up], axis: 1)
                    fused["down_proj.\(suffix)"] = try #require(weights["down_proj.\(suffix)"])
                }
                try combined.update(parameters: ModuleParameters.unflattened(fused), verify: .all)
                eval(reference, combined)
                var rows: [[String: Any]] = []
                for count in [1, 4, 8] {
                    let input = sin(MLXArray(0..<(count * 2816)).asType(.float32) * 0.137)
                        .reshaped(count, 2816).asType(.bfloat16)
                    let indices = MLXArray((0..<(count * 8)).map { Int32(($0 * 17) % 128) }).reshaped(count, 8)
                    let expected = reference(input, indices)
                    let actual = combined(input, indices)
                    let error = MLX.max(abs(expected - actual)).item(Float.self)
                    let relative = error / max(MLX.max(abs(expected)).item(Float.self), 1e-10)
                    #expect(error == 0)
                    #expect(actual.shape == [count, 8, 2816])
                    // Do not conceal differences caused by changing GEMM shape.
                    print("A4B expert tokens=\(count) max_error=\(error) relative_error=\(relative)")
                    for _ in 0..<4 {
                        eval(reference(input, indices), combined(input, indices))
                    }
                    var control: [Double] = []
                    var candidate: [Double] = []
                    for pair in 0..<10 {
                        for enabled in pair.isMultiple(of: 2) ? [false, true] : [true, false] {
                            let start = ContinuousClock.now
                            for _ in 0..<20 {
                                if enabled {
                                    eval(combined(input, indices))
                                } else {
                                    eval(reference(input, indices))
                                }
                            }
                            let duration = start.duration(to: .now).components
                            let ms = (Double(duration.seconds) * 1000 + Double(duration.attoseconds) / 1e15) / 20
                            if enabled {
                                candidate.append(ms)
                            } else {
                                control.append(ms)
                            }
                        }
                    }
                    rows.append([
                        "tokens": count, "maximum_absolute_error": error,
                        "relative_maximum_error": relative, "exact_output": error == 0,
                        "reference_milliseconds": control, "combined_milliseconds": candidate,
                    ])
                }
                let report: [String: Any] = [
                    "scope":
                        "One actual A4B Q4 expert layer; synthetic hidden states and routing; not full-model throughput",
                    "model_path": path, "layer": 0, "bits": 4, "group_size": 64, "rows": rows,
                ]
                try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                    .write(to: URL(fileURLWithPath: output), options: .withoutOverwriting)
            }
        }
    }

    private static func expertWeights(_ root: URL, prefix: String) throws -> [String: MLXArray] {
        let index = try JSONSerialization.jsonObject(
            with: Data(contentsOf: root.appendingPathComponent("model.safetensors.index.json")))
        let map = try #require((index as? [String: Any])?["weight_map"] as? [String: String])
        let selected = map.filter { $0.key.hasPrefix(prefix) }
        #expect(selected.count == 9)
        var result: [String: MLXArray] = [:]
        for shard in Set(selected.values) {
            let arrays = try MLX.loadArrays(url: root.appendingPathComponent(shard))
            for (name, _) in selected where map[name] == shard {
                result[String(name.dropFirst(prefix.count))] = try #require(arrays[name])
            }
        }
        return result
    }
}
