#if os(macOS)
    import Foundation
    import MLX
    import MLXLLM
    import MLXLMCommon
    import MLXNN
    import Testing

    @Suite(
        "Local Gemma 270M Q4 projection capture", .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_CAPTURE_GEMMA270_PROJECTIONS"] == "1"))
    struct Gemma270MProjectionCaptureTests {
        @Test("Capture identical teacher-forced inputs at each eligible projection")
        func capture() throws {
            let environment = ProcessInfo.processInfo.environment
            let directory = URL(fileURLWithPath: try #require(environment["MIDNIGHT_Q4_REAL_CHECKPOINT"]))
            let tokensURL = URL(fileURLWithPath: try #require(environment["MIDNIGHT_Q4_REAL_TOKEN_IDS"]))
            let output = URL(fileURLWithPath: try #require(environment["MIDNIGHT_Q4_REAL_CAPTURE_DIR"]))
            let arm = try #require(environment["MLX_METAL_AFFINE_Q4_QMV_TAIL"])
            try #require(arm == "0" || arm == "1")
            try #require(!FileManager.default.fileExists(atPath: output.path))
            let tokens = try JSONDecoder().decode([Int].self, from: Data(contentsOf: tokensURL))
            try #require((1...32).contains(tokens.count))
            try Device.withDefaultDevice(.gpu) {
                let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
                let config = try JSONDecoder().decode(Gemma3TextConfiguration.self, from: data)
                let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
                let model = Gemma3TextModel(config)
                try loadWeights(
                    modelDirectory: directory, model: model, perLayerQuantization: base.perLayerQuantization)
                let store = CaptureStore()
                let updates = model.leafModules().flattened().compactMap { path, module -> (String, Module)? in
                    guard let linear = module as? QuantizedLinear,
                        linear.bits == 4, linear.groupSize == 64, linear.mode == .affine,
                        linear.shape.1 == 640, linear.scales.dtype == .bfloat16
                    else { return nil }
                    return (path, CapturingLinear(linear, path: path, store: store))
                }
                try #require(updates.count > 80, "Expect Gemma 270M eligible projections, including lm_head.")
                model.update(modules: ModuleChildren.unflattened(updates))
                model.train(false)
                let cache = try model.newCache()
                try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
                for (position, token) in tokens.enumerated() {
                    store.arrays.removeAll(keepingCapacity: true)
                    let logits = model(MLXArray([token]).reshaped(1, 1), cache: cache)
                    // Keep the original forward graph intact. Materialize once at
                    // the token boundary; captures merely retain its input/output views.
                    eval(logits)
                    try #require(store.arrays.count == updates.count * 2)
                    try save(
                        arrays: store.arrays,
                        metadata: ["position": String(position), "token_id": String(token), "tail_gate": arm],
                        url: output.appendingPathComponent(String(format: "%03d.safetensors", position)))
                }
                let report: [String: Any] = [
                    "checkpoint": directory.path, "tail_gate": arm, "token_ids": tokens,
                    "captured_projections": updates.map(\.0).sorted(),
                    "scope": "Teacher-forced diagnostic capture; extra saves are not a speed benchmark",
                ]
                try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                    .write(to: output.appendingPathComponent("manifest.json"), options: .withoutOverwriting)
            }
        }
    }

    private final class CaptureStore {
        var arrays: [String: MLXArray] = [:]
    }

    private final class CapturingLinear: QuantizedLinear {
        let capturePath: String
        let store: CaptureStore

        init(_ original: QuantizedLinear, path: String, store: CaptureStore) {
            self.capturePath = path
            self.store = store
            super.init(
                weight: original.weight, bias: original.bias, scales: original.scales, biases: original.biases,
                groupSize: original.groupSize, bits: original.bits, mode: original.mode,
                globalScale: original.globalScale)
        }

        override func callAsFunction(_ x: MLXArray) -> MLXArray {
            let output = super.callAsFunction(x)
            store.arrays[capturePath + ".input"] = x
            store.arrays[capturePath + ".output"] = output
            return output
        }
    }
#endif
