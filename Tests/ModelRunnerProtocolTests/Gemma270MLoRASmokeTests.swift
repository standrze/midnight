#if os(macOS)
    import CryptoKit
    import Foundation
    import MLX
    import MLXLLM
    import MLXLMCommon
    import MLXNN
    import Testing

    @Suite(
        "Opt-in real Gemma270M mixed Q4/Q8 adapter smoke", .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_RUN_GEMMA270_LORA_SMOKE"] == "1"))
    struct Gemma270MLoRASmokeTests {
        @Test("Two updates preserve the frozen mixed base and produce a reloadable adapter")
        func trainSaveReload() throws {
            let environment = ProcessInfo.processInfo.environment
            let source = URL(fileURLWithPath: try #require(environment["MIDNIGHT_GEMMA270_LORA_BASE"]))
            let output = URL(fileURLWithPath: try #require(environment["MIDNIGHT_GEMMA270_LORA_OUTPUT"]))
            try #require(!FileManager.default.fileExists(atPath: output.path))
            try Device.withDefaultDevice(.gpu) {
                let configData = try Data(contentsOf: source.appendingPathComponent("config.json"))
                let config = try JSONDecoder().decode(Gemma3TextConfiguration.self, from: configData)
                let base = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
                let sourceHashes = try fileHashes(source)
                func load() throws -> Gemma3TextModel {
                    let model = Gemma3TextModel(config)
                    try loadWeights(
                        modelDirectory: source, model: model, quantization: base.quantization,
                        perLayerQuantization: base.perLayerQuantization)
                    model.train(false)
                    return model
                }
                let model = try load()
                try #require(model.loraLayers.count == 18 && model.vocabularySize == 262144)
                let frozenBefore = tensorHashes(model)
                let geometryBefore = try geometry(model)
                try verifyMixedGeometry(geometryBefore)

                let configuration = LoRAConfiguration(
                    numLayers: 1, loraParameters: .init(rank: 4, scale: 16, dropout: 0))
                MLXRandom.seed(27_020_260_929)
                _ = try LoRAContainer.from(model: model, configuration: configuration)
                let expectedKeys = Set(
                    Self.moduleNames.flatMap {
                        ["model.layers.17.\($0).lora_a", "model.layers.17.\($0).lora_b"]
                    })
                let initial = Dictionary(uniqueKeysWithValues: model.trainableParameters().flattened())
                try #require(Set(initial.keys) == expectedKeys && initial.values.reduce(0, { $0 + $1.size }) == 52736)
                try #require(initial.values.allSatisfy { $0.dtype == .float32 })
                let adapterBefore = tensorHashes(model, adaptersOnly: true)
                try #require(tensorHashes(model) == frozenBefore)
                try #require(try geometry(model) == geometryBefore)

                // Fixed valid IDs isolate the native gradient/save/reload contract from tokenizer or dataset behavior.
                // The separate existing Studio-worker smoke covers actual text tokenization and its Adam path.
                let tokenIDs = [2, 100, 101, 102, 103]
                let inputs = MLXArray(Array(tokenIDs.dropLast())).reshaped(1, tokenIDs.count - 1)
                let targets = MLXArray(Array(tokenIDs.dropFirst())).reshaped(1, tokenIDs.count - 1)
                let zeroAdapterLogits = model(inputs, cache: nil)
                eval(zeroAdapterLogits)
                model.setLoRAEnabled(false)
                let disabledLogits = model(inputs, cache: nil)
                eval(disabledLogits)
                let zeroAdapterDifference = maxDifference(zeroAdapterLogits, disabledLogits)
                model.setLoRAEnabled(true)
                model.train(true)

                let valueGradient = valueAndGrad(model: model) { model, arrays in
                    let logits = model(arrays[0], cache: nil).asType(.float32)
                    return [crossEntropy(logits: logits, targets: arrays[1]).mean()]
                }
                var losses: [Float] = []
                var nonzeroGradient = false
                for _ in 0..<2 {
                    let (values, derivatives) = valueGradient(model, [inputs, targets])
                    eval(values)
                    let loss = values[0].item(Float.self)
                    try #require(loss.isFinite)
                    losses.append(loss)
                    let gradients = Dictionary(uniqueKeysWithValues: derivatives.flattened())
                    try #require(Set(gradients.keys) == expectedKeys)
                    let current = Dictionary(uniqueKeysWithValues: model.trainableParameters().flattened())
                    var updates: [String: MLXArray] = [:]
                    for key in expectedKeys {
                        let gradient = try #require(gradients[key])
                        try #require(all(isFinite(gradient)).item(Bool.self))
                        nonzeroGradient = nonzeroGradient || max(abs(gradient)).item(Float.self) > 0
                        updates[key] = try #require(current[key]) - Float(0.0001) * gradient
                    }
                    try model.update(parameters: ModuleParameters.unflattened(updates), verify: .noUnusedKeys)
                    eval(model)
                    try #require(tensorHashes(model) == frozenBefore, "Frozen base changed after an optimizer step")
                    try #require(try geometry(model) == geometryBefore)
                }
                try #require(nonzeroGradient)
                let adapterAfter = tensorHashes(model, adaptersOnly: true)
                try #require(adapterAfter != adapterBefore)
                try #require(expectedKeys.contains { $0.hasSuffix(".lora_b") && adapterAfter[$0] != adapterBefore[$0] })
                model.train(false)
                let trainedLogits = model(inputs, cache: nil)
                eval(trainedLogits)
                try #require(all(isFinite(trainedLogits)).item(Bool.self))

                try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
                try JSONEncoder().encode(configuration).write(
                    to: output.appendingPathComponent("adapter_config.json"), options: .withoutOverwriting)
                try LoRATrain.saveLoRAWeights(model: model, url: output.appendingPathComponent("adapters.safetensors"))
                let saved = try loadArrays(url: output.appendingPathComponent("adapters.safetensors"))
                try #require(Set(saved.keys) == expectedKeys)
                try #require(arrayHashes(saved) == adapterAfter)

                let fresh = try load()
                try #require(tensorHashes(fresh) == frozenBefore)
                let adapter = try LoRAContainer.from(directory: output)
                try adapter.load(into: fresh)
                try #require(tensorHashes(fresh) == frozenBefore)
                try #require(try geometry(fresh) == geometryBefore)
                try #require(tensorHashes(fresh, adaptersOnly: true) == adapterAfter)
                let reloadedLogits = fresh(inputs, cache: nil)
                eval(reloadedLogits)
                let reloadDifference = maxDifference(trainedLogits, reloadedLogits)
                try #require(reloadDifference <= 1e-5)
                adapter.unload(from: fresh)
                try #require(tensorHashes(fresh) == frozenBefore)
                try #require(try geometry(fresh) == geometryBefore)
                try #require(tensorHashes(fresh, adaptersOnly: true).isEmpty)
                let unloadedLogits = fresh(inputs, cache: nil)
                eval(unloadedLogits)
                try #require(maxDifference(disabledLogits, unloadedLogits) <= 1e-5)
                try #require(try fileHashes(source) == sourceHashes)

                let report: [String: Any] = [
                    "status": "passed", "scope": "Two-step functionality only; no quality or speed conclusion",
                    "base": source.path, "optimizer": "explicit SGD", "learning_rate": 0.0001,
                    "steps": 2, "losses": losses, "rank": 4, "layers": 1, "scale": 16,
                    "trainable_tensor_count": expectedKeys.count, "trainable_parameter_count": 52736,
                    "adapter_dtype": "float32", "token_ids": tokenIDs,
                    "frozen_base_tensors_exact": true, "quantization_geometry_exact": true,
                    "source_files_unchanged": true, "saved_adapter_arrays_exact": true,
                    "adapter_updated": true, "reload_maximum_logit_difference": reloadDifference,
                    "zero_adapter_vs_disabled_maximum_difference": zeroAdapterDifference,
                    "zero_adapter_logit_dtype": String(describing: zeroAdapterLogits.dtype),
                    "disabled_logit_dtype": String(describing: disabledLogits.dtype),
                    "frozen_tensor_hashes": frozenBefore, "source_file_hashes": sourceHashes,
                    "base_geometry": geometryBefore, "adapter_hashes": adapterAfter,
                ]
                try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                    .write(to: output.appendingPathComponent("smoke-report.json"), options: .withoutOverwriting)
            }
        }

        private static let moduleNames = [
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        ]

        private func maxDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
            max(abs(lhs.asType(.float32) - rhs.asType(.float32))).item(Float.self)
        }

        private func tensorHashes(_ model: Module, adaptersOnly: Bool = false) -> [String: String] {
            let tensors = Dictionary(
                uniqueKeysWithValues: model.parameters().flattened().filter {
                    let isAdapter = $0.0.hasSuffix(".lora_a") || $0.0.hasSuffix(".lora_b")
                    return isAdapter == adaptersOnly
                })
            return arrayHashes(tensors)
        }

        private func arrayHashes(_ tensors: [String: MLXArray]) -> [String: String] {
            tensors.mapValues { tensor in
                let bytes = tensor.asData(access: .copy).data
                let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
                return "\(tensor.dtype):\(tensor.shape):\(digest)"
            }
        }

        private func geometry(_ model: Module) throws -> [String: [Int]] {
            var result: [String: [Int]] = [:]
            for (key, module) in model.namedModules() {
                guard let linear = module as? QuantizedLinear else {
                    continue
                }
                try #require(linear.mode == .affine && linear.scales.dtype == .bfloat16)
                try #require(linear.biases?.dtype == .bfloat16)
                result[key] = [linear.bits, linear.groupSize, linear.shape.0, linear.shape.1]
            }
            return result
        }

        private func verifyMixedGeometry(_ geometry: [String: [Int]]) throws {
            let q8 = geometry.filter { $0.value[0] == 8 }
            try #require(q8.count == 36)
            for layer in 0..<18 {
                for suffix in ["q_proj", "k_proj"] {
                    try #require(geometry["model.layers.\(layer).self_attn.\(suffix)"]?.prefix(2) == [8, 64])
                }
            }
            try #require(geometry.allSatisfy { $0.value[1] == 64 && [4, 8].contains($0.value[0]) })
        }

        private func fileHashes(_ directory: URL) throws -> [String: String] {
            var result: [String: String] = [:]
            for file in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            where file.pathExtension == "safetensors" || file.pathExtension == "json" || file.pathExtension == "model" {
                let handle = try FileHandle(forReadingFrom: file)
                defer { try? handle.close() }
                var digest = SHA256()
                while let bytes = try handle.read(upToCount: 1024 * 1024), !bytes.isEmpty {
                    digest.update(data: bytes)
                }
                result[file.lastPathComponent] = digest.finalize().map { String(format: "%02x", $0) }.joined()
            }
            return result
        }
    }
#endif
