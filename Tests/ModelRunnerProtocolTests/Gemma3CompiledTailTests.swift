#if os(macOS)
    import Foundation
    import MLX
    @_spi(GemmaCompiledTailTesting) @_spi(GemmaEncoder) import MLXLLM
    import MLXLMCommon
    import MLXNN
    import Testing

    /// Run separately with MIDNIGHT_RUN_GEMMA3_COMPILED_TAIL_TESTS=1. These tests
    /// require Metal because a CPU fallback cannot prove a warm Metal graph is reused.
    @Suite(
        "Gemma 3 compiled tail lifecycle and parity", .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_RUN_GEMMA3_COMPILED_TAIL_TESTS"] == "1"))
    struct Gemma3CompiledTailTests {
        @Test("Cached decode reuses fixed traces across sliding rollover and new prefill")
        func cacheGrowthAndReuse() throws {
            try Device.withDefaultDevice(.gpu) {
                let model = try fixture()
                let keysBefore = Set(model.parameters().flattened().map(\.0))
                _ = try compare(model, tokens: Array(1...12))
                #expect(model.model.layers.map(\.compiledTailTraceCount) == [1, 1])
                // A new request has a new cache, but the fixed tail graph stays valid.
                _ = try compare(model, tokens: [9, 8, 7, 6])
                #expect(model.model.layers.map(\.compiledTailTraceCount) == [1, 1])
                #expect(Set(model.parameters().flattened().map(\.0)) == keysBefore)
            }
        }

        @Test("Warm norm and quantized parameter replacement uses current values")
        func parameterReplacement() throws {
            try Device.withDefaultDevice(.gpu) {
                let model = try fixture()
                let before = try compare(model, tokens: [1, 2, 3, 4])
                let traceCounts = model.model.layers.map(\.compiledTailTraceCount)
                let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
                let normKey = "model.layers.0.post_feedforward_layernorm.weight"
                let scalesKey = "model.layers.0.mlp.gate_proj.scales"
                let norm = try #require(parameters[normKey])
                let scales = try #require(parameters[scalesKey])
                model.update(
                    parameters: ModuleParameters.unflattened([
                        normKey: norm + 0.5,
                        scalesKey: scales * 1.75,
                    ]))
                let after = try compare(model, tokens: [1, 2, 3, 4])
                #expect(maxDifference(before, after) > 0.01, "The update must have an observable effect")
                #expect(model.model.layers.map(\.compiledTailTraceCount) == traceCounts)
            }
        }

        @Test("Replacing a projection within the same MLP rebuilds its trace")
        func projectionReplacement() throws {
            try Device.withDefaultDevice(.gpu) {
                let model = try fixture()
                _ = try compare(model, tokens: [1, 2, 3, 4])
                let key = "model.layers.0.mlp.up_proj"
                let original = try projection(model, key)
                let quantizationBiases = try #require(original.biases)
                let probe = (sin(MLXArray(0..<64).asType(.float32) * 0.17) * 0.5)
                    .asType(.bfloat16).reshaped(1, 1, 64)
                let beforeLayer = layerOutput(model.model.layers[0], input: probe, compiled: false)
                // Negating both affine terms reverses the entire projection.
                // A scales-only perturbation can be absorbed by the later RMSNorm
                // and produce only one BF16 ULP of difference in final logits.
                let replacement = QuantizedLinear(
                    weight: original.weight, bias: original.bias.map { -$0 },
                    scales: -original.scales, biases: -quantizationBiases,
                    groupSize: original.groupSize, bits: original.bits, mode: original.mode)
                replacement.train(false)
                let originalProjection = original(probe)
                let replacementProjection = replacement(probe)
                eval(originalProjection, replacementProjection)
                #expect(maxDifference(originalProjection, replacementProjection) > 1)
                #expect(maxDifference(-originalProjection, replacementProjection) == 0)

                model.update(modules: ModuleChildren.unflattened([key: replacement]))
                let installed = try projection(model, key)
                #expect(installed === replacement)
                _ = try compare(model, tokens: [1, 2, 3, 4])
                let afterLayer = layerOutput(model.model.layers[0], input: probe, compiled: false)
                let compiledLayer = layerOutput(model.model.layers[0], input: probe, compiled: true)
                let changedOutput = maxDifference(beforeLayer, afterLayer)
                let compiledError = maxDifference(afterLayer, compiledLayer)
                #expect(changedOutput > 0.25, "The replacement must materially change the decoder output")
                #expect(
                    compiledError < changedOutput * 0.1,
                    "A stale projection must fail even when final-logit parity permits BF16 rounding")
                #expect(model.model.layers.map(\.compiledTailTraceCount) == [2, 1])
            }
        }

        @Test("QLoRA replacement and its mutable enable flag stay on the eager path")
        func loraReplacement() throws {
            try Device.withDefaultDevice(.gpu) {
                let model = try fixture()
                _ = try compare(model, tokens: [1, 2, 3, 4])
                let key = "model.layers.0.mlp.up_proj"
                let original = try projection(model, key)
                let lora = try #require(QLoRALinear.from(linear: original, rank: 2, scale: 1) as? QLoRALinear)
                lora.update(
                    parameters: ModuleParameters.unflattened([
                        "lora_a": MLXArray.ones([64, 2], dtype: .bfloat16) * 0.125,
                        "lora_b": (MLXArray(0..<256).asType(.float32) / 256 - 0.5)
                            .asType(.bfloat16).reshaped(2, 128),
                    ]))
                lora.train(false)
                model.update(modules: ModuleChildren.unflattened([key: lora]))
                let enabled = try compare(model, tokens: [1, 2, 3, 4])
                lora.loraEnabled = false
                let disabled = try compare(model, tokens: [1, 2, 3, 4])
                #expect(maxDifference(enabled, disabled) > 0.001)
                #expect(model.model.layers.map(\.compiledTailTraceCount) == [1, 1])

                model.update(modules: ModuleChildren.unflattened([key: original]))
                _ = try compare(model, tokens: [1, 2, 3, 4])
                #expect(model.model.layers.map(\.compiledTailTraceCount) == [2, 1])
            }
        }

        @Test("Training after warm inference preserves eager gradients and invalidates traces")
        func trainingAfterWarmup() throws {
            try Device.withDefaultDevice(.gpu) {
                let model = try fixture()
                _ = try compare(model, tokens: [1, 2, 3, 4])
                let layers = model.model.layers
                model.train(true)
                let input = (MLXArray(0..<64).asType(.float32) / 64 - 0.5)
                    .asType(.bfloat16).reshaped(1, 1, 64)
                func gradient(_ enabled: Bool) -> MLXArray {
                    let cache = RotatingKVCache(maxSize: 4)
                    let prefix = MLXArray.zeros([1, 1, 2, 64], dtype: .bfloat16)
                    let state = cache.update(keys: prefix, values: prefix)
                    eval(state.0, state.1)
                    return Gemma3RuntimeTuning.$useCompiledTail.withValue(enabled) {
                        let derivative = grad { x in
                            layers[0](x, mask: .none, cache: cache).asType(.float32).square().mean()
                        }
                        let result = derivative(input)
                        eval(result)
                        return result
                    }
                }
                let expected = gradient(false)
                let actual = gradient(true)
                #expect(all(isFinite(actual)).item(Bool.self))
                #expect(MLX.max(abs(actual)).item(Float.self) > 0)
                #expect(maxDifference(expected, actual) == 0)
                #expect(layers.map(\.compiledTailTraceCount) == [1, 1])
                model.train(false)
                _ = try compare(model, tokens: [1, 2, 3, 4])
                #expect(layers.map(\.compiledTailTraceCount) == [2, 2])
            }
        }

        @Test("Compiled children do not retain the owning decoder or appear as parameters")
        func moduleLifetime() throws {
            try Device.withDefaultDevice(.gpu) {
                weak var layer: Gemma3TransformerBlock?
                weak var gate: QuantizedLinear?
                try autoreleasepool {
                    let model = try fixture()
                    layer = model.model.layers[0]
                    gate = try projection(model, "model.layers.0.mlp.gate_proj")
                    _ = try compare(model, tokens: [1, 2, 3, 4])
                    #expect(model.model.layers[0].compiledTailTraceCount == 1)
                }
                #expect(layer == nil)
                #expect(gate == nil)
            }
        }

        private func fixture() throws -> Gemma3TextModel {
            let json = """
                {"model_type":"gemma3_text","hidden_size":64,"intermediate_size":128,
                 "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
                 "head_dim":64,"vocab_size":128,"sliding_window":4,
                 "layer_types":["sliding_attention","full_attention"]}
                """
            let config = try JSONDecoder().decode(Gemma3TextConfiguration.self, from: Data(json.utf8))
            let model = Gemma3TextModel(config)
            // Deterministic loaded parameters, independent of global random state.
            let parameters = model.parameters().mapValues { array in
                let values = MLXArray(0..<array.size).asType(.float32)
                return (sin(values * 0.17) * 0.08).reshaped(array.shape).asType(.bfloat16)
            }
            model.update(parameters: parameters)
            quantize(model: model, groupSize: 64, bits: 4)
            model.train(false)
            eval(model)
            return model
        }

        private func projection(_ model: Gemma3TextModel, _ path: String) throws -> QuantizedLinear {
            let modules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            return try #require(modules[path] as? QuantizedLinear)
        }

        @discardableResult
        private func compare(_ model: Gemma3TextModel, tokens: [Int]) throws -> MLXArray {
            let baseline = try model.newCache()
            let candidate = try model.newCache()
            let prompt = MLXArray(Array(tokens.prefix(2))).reshaped(1, 2)
            let a = Gemma3RuntimeTuning.$useCompiledTail.withValue(false) { model(prompt, cache: baseline) }
            let b = Gemma3RuntimeTuning.$useCompiledTail.withValue(true) { model(prompt, cache: candidate) }
            eval(a, b)
            #expect(maxDifference(a, b) == 0, "Prefill must stay on the same eager path")
            var result = a
            for token in tokens.dropFirst(2) {
                let input = MLXArray([token]).reshaped(1, 1)
                let expected = Gemma3RuntimeTuning.$useCompiledTail.withValue(false) { model(input, cache: baseline) }
                let actual = Gemma3RuntimeTuning.$useCompiledTail.withValue(true) { model(input, cache: candidate) }
                eval(expected, actual)
                #expect(all(isFinite(actual)).item(Bool.self))
                // BF16 fusion may change intermediate rounding. This is primitive
                // parity, not a substitute for checkpoint NLL/task-quality evaluation.
                #expect(
                    allClose(expected.asType(.float32), actual.asType(.float32), rtol: 0.03, atol: 0.04).item(Bool.self)
                )
                result = expected
            }
            #expect(baseline.map(\.offset) == candidate.map(\.offset))
            return result
        }

        private func layerOutput(
            _ layer: Gemma3TransformerBlock, input: MLXArray, compiled: Bool
        ) -> MLXArray {
            let cache = RotatingKVCache(maxSize: 4)
            let prefix = MLXArray.zeros([1, 1, 2, 64], dtype: .bfloat16)
            let state = cache.update(keys: prefix, values: prefix)
            eval(state.0, state.1)
            return Gemma3RuntimeTuning.$useCompiledTail.withValue(compiled) {
                let output = layer(input, mask: .none, cache: cache)
                eval(output)
                return output
            }
        }

        private func maxDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
            MLX.max(abs(lhs.asType(.float32) - rhs.asType(.float32))).item(Float.self)
        }
    }
#endif
