import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import ModelRunnerCore

@Suite("Native Talkie numerical reference", .serialized)
struct TalkieModelTests {
    private var fixtureDirectory: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/TalkieTiny", isDirectory: true)
    }

    private let tokenIDs: [Int32] = [3, 1, 4, 1, 5, 9, 2]

    private func configuration(_ variant: String = "fp32") throws -> TalkieConfiguration {
        let filename = variant == "q4" ? "config-q4.json" : "config.json"
        return try JSONDecoder().decode(
            TalkieConfiguration.self,
            from: Data(contentsOf: fixtureDirectory.appendingPathComponent(filename)))
    }

    private func arrays(_ filename: String) throws -> [String: MLXArray] {
        try MLX.loadArrays(url: fixtureDirectory.appendingPathComponent(filename))
    }

    private func loadedModel(
        _ variant: String = "fp32", fused: Bool = true
    ) throws -> TalkieModel {
        let model = TalkieModel(try configuration(variant), fuseProjections: fused)
        if variant == "q4" {
            quantize(model: model, groupSize: 32, bits: 4)
        }
        let weights = model.sanitize(weights: try arrays("weights-\(variant).safetensors"))
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model)
        return model
    }

    private func tokens(_ range: Range<Int>) -> MLXArray {
        MLXArray(Array(tokenIDs[range]))[.newAxis]
    }

    private func maximumError(_ actual: MLXArray, _ expected: MLXArray) -> Float {
        MLX.max(abs(actual.asType(.float32) - expected.asType(.float32))).item(Float.self)
    }

    @Test(
        "Fused and unfused FP32 logits match independent Python equations",
        arguments: [false, true])
    func fullPromptReference(_ fused: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadedModel(fused: fused)
            let expected = try #require(arrays("expected-fp32.safetensors")["full"])
            let actual = model(tokens(0..<7), cache: nil)
            eval(actual)

            #expect(actual.shape == [1, 7, 32])
            #expect(maximumError(actual, expected) < 0.00002)
            #expect(
                argMax(actual, axis: -1).asArray(Int32.self)
                    == argMax(expected, axis: -1).asArray(Int32.self))
        }
    }

    @Test(
        "BF16 gains, weightless norms, and the folded output head preserve reference logits",
        arguments: [false, true])
    func bfloat16Reference(_ fused: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadedModel("bf16", fused: fused)
            let expected = try #require(arrays("expected-bf16.safetensors")["full"])
            let actual = model(tokens(0..<7), cache: nil)
            eval(actual)

            #expect(actual.dtype == .bfloat16)
            #expect(maximumError(actual, expected) < 0.04)
        }
    }

    @Test(
        "Prefill, a cached chunk, and two decode steps preserve positions and causality",
        arguments: [false, true])
    func chunkedCacheReference(_ fused: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadedModel(fused: fused)
            let expected = try arrays("expected-fp32.safetensors")
            let cache = try model.newCache(parameters: nil)
            let stages: [(Range<Int>, String)] = [
                (0..<2, "prefill"), (2..<5, "continuation"),
                (5..<6, "decode"), (6..<7, "last"),
            ]

            for (range, name) in stages {
                let actual = model(tokens(range), cache: cache)
                eval(actual)
                eval(cache)
                #expect(cache.count == 2)
                #expect(cache.allSatisfy { $0.offset == range.upperBound })
                #expect(maximumError(actual, try #require(expected[name])) < 0.00002)
            }
        }
    }

    @Test(
        "Affine Q4 checkpoints with a pre-folded LM head load and retain cached parity",
        arguments: [false, true])
    func quantizedCheckpointReference(_ fused: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadedModel("q4", fused: fused)
            let expected = try arrays("expected-q4.safetensors")
            let actual = model(tokens(0..<7), cache: nil)
            eval(actual)
            // Independent model equations use packed QMM for the BF16 reference:
            // CPU QMM and a dense GEMM differ in their accumulator rounding.
            let reference = try #require(expected["full"]).asType(.float32)
            let difference = actual.asType(.float32) - reference
            let relativeRMS = sqrt(
                mean(difference * difference, axis: -1) / mean(reference * reference, axis: -1))
            #expect(all(isFinite(actual)).item(Bool.self))
            // Bound each token's logit-vector error by two BF16 epsilons. This is
            // scale-aware; elementwise relative/ULP errors are unstable near zero.
            // Fixture mutations all exceed this bound (minimum: head gain, 2.19%).
            #expect(MLX.max(relativeRMS).item(Float.self) < 1.0 / 64.0)
            #expect(
                argMax(actual, axis: -1).asArray(Int32.self)
                    == argMax(try #require(expected["full"]), axis: -1).asArray(Int32.self))
            let otherModel = try loadedModel("q4", fused: !fused)
            let otherActual = otherModel(tokens(0..<7), cache: nil)
            eval(otherActual)
            #expect(maximumError(actual, otherActual) == 0)

            let cache = try model.newCache(parameters: nil)
            let prefill = model(tokens(0..<5), cache: cache)
            eval(prefill)
            let continuation = model(tokens(5..<7), cache: cache)
            eval(continuation)
            #expect(cache.allSatisfy { $0.offset == 7 })
            #expect(maximumError(continuation, actual[0..., 5..<7, 0...]) < 0.06)

            // The same packed integers with FP32 scales and activations compare
            // tightly to an independently unpacked affine-weight reference. This
            // covers packed loading and fusion without BF16 accumulator ambiguity.
            let floatModel = TalkieModel(try configuration("q4"), fuseProjections: fused)
            quantize(model: floatModel, groupSize: 32, bits: 4)
            let floatWeights = try arrays("weights-q4.safetensors").mapValues {
                $0.dtype == .uint32 ? $0 : $0.asType(.float32)
            }
            try floatModel.update(
                parameters: ModuleParameters.unflattened(floatModel.sanitize(weights: floatWeights)),
                verify: .all)
            let floatExpected = try #require(arrays("expected-q4-fp32.safetensors")["full"])
            let floatActual = floatModel(tokens(0..<7), cache: nil)
            eval(floatActual)
            #expect(floatActual.dtype == .float32)
            #expect(maximumError(floatActual, floatExpected) < 0.00002)
        }
    }

    @Test("Official bare output weights and an explicitly folded head produce identical logits")
    func foldedHeadCompatibility() throws {
        try Device.withDefaultDevice(.cpu) {
            let canonical = try loadedModel("bf16")
            let folded = TalkieModel(try configuration(), fuseProjections: true)
            var weights = try arrays("weights-bf16.safetensors")
            let removedHead = weights.removeValue(forKey: "lm_head")
            let removedGain = weights.removeValue(forKey: "lm_head_gain.w_g")
            let head = try #require(removedHead)
            let gain = try #require(removedGain)
            weights["lm_head.weight"] = head * gain
            try folded.update(
                parameters: ModuleParameters.unflattened(folded.sanitize(weights: weights)), verify: .all)
            let expected = canonical(tokens(0..<7), cache: nil)
            let actual = folded(tokens(0..<7), cache: nil)
            eval(expected, actual)
            #expect(maximumError(actual, expected) == 0)
        }
    }

    @Test("Strict loading rejects an incomplete checkpoint and unexplained quantized head gain")
    func rejectsIncompleteOrAmbiguousWeights() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = TalkieModel(try configuration(), fuseProjections: false)
            var incomplete = try arrays("weights-fp32.safetensors")
            incomplete.removeValue(forKey: "model.blocks.0.attn.head_gain.head_g")
            #expect(throws: (any Error).self) {
                try model.update(
                    parameters: ModuleParameters.unflattened(model.sanitize(weights: incomplete)), verify: .all)
            }

            let quantized = TalkieModel(try configuration("q4"), fuseProjections: false)
            quantize(model: quantized, groupSize: 32, bits: 4)
            var ambiguous = try arrays("weights-q4.safetensors")
            ambiguous["lm_head_gain.w_g"] = MLXArray([Float(1.25)])
            #expect(throws: (any Error).self) {
                try quantized.update(
                    parameters: ModuleParameters.unflattened(quantized.sanitize(weights: ambiguous)), verify: .all)
            }
        }
    }

    @Test("Invalid head geometry is rejected before allocating model tensors")
    func rejectsInvalidConfiguration() throws {
        let data = try Data(contentsOf: fixtureDirectory.appendingPathComponent("config.json"))
        let original = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        for (key, value) in [
            ("num_attention_heads", 0), ("head_dim", 7), ("hidden_size", 15),
            ("num_key_value_heads", 0), ("num_key_value_heads", 3),
        ] {
            var invalid = original
            invalid[key] = value
            let invalidData = try JSONSerialization.data(withJSONObject: invalid)
            #expect(throws: (any Error).self) {
                try JSONDecoder().decode(TalkieConfiguration.self, from: invalidData)
            }
        }
    }

    @Test(
        "GQA matches explicitly repeated MHA K/V weights and retains smaller caches",
        arguments: ["fp32", "q4"])
    func groupedQueryReference(_ variant: String) throws {
        try Device.withDefaultDevice(.cpu) {
            let filename = variant == "q4" ? "config-q4.json" : "config.json"
            let data = try Data(contentsOf: fixtureDirectory.appendingPathComponent(filename))
            var object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
            object["num_key_value_heads"] = 1
            let config = try JSONDecoder().decode(
                TalkieConfiguration.self,
                from: JSONSerialization.data(withJSONObject: object))
            let grouped = TalkieModel(config, fuseProjections: true)
            let expanded = TalkieModel(try configuration(variant), fuseProjections: false)
            #expect(!grouped.usesFusedProjections)
            #expect(grouped.kvHeads == [1, 1])
            if variant == "q4" {
                quantize(model: grouped, groupSize: 32, bits: 4)
                quantize(model: expanded, groupSize: 32, bits: 4)
            }
            var groupedWeights = try arrays("weights-\(variant).safetensors")
            var expandedWeights = groupedWeights
            for layer in 0..<2 {
                for projection in ["key", "value"] {
                    for suffix in ["weight", "scales", "biases"] {
                        let key = "model.blocks.\(layer).attn.attn_\(projection).\(suffix)"
                        if let weight = groupedWeights[key] {
                            // One stored head versus two explicitly identical heads. Packed
                            // rows and their metadata are repeated together in the Q4 case.
                            let head = weight[0..<config.headDimension, 0...]
                            groupedWeights[key] = head
                            expandedWeights[key] = concatenated([head, head], axis: 0)
                        }
                    }
                }
            }
            try grouped.update(
                parameters: ModuleParameters.unflattened(grouped.sanitize(weights: groupedWeights)), verify: .all)
            try expanded.update(
                parameters: ModuleParameters.unflattened(expanded.sanitize(weights: expandedWeights)), verify: .all)
            let expected = expanded(tokens(0..<7), cache: nil)
            let actual = grouped(tokens(0..<7), cache: nil)
            eval(expected, actual)
            let tolerance: Float = variant == "q4" ? 0.08 : 0.00002
            #expect(maximumError(actual, expected) < tolerance)
            let cache = try grouped.newCache(parameters: nil)
            for range in [0..<2, 2..<5, 5..<7] {
                let output = grouped(tokens(range), cache: cache)
                eval(output)
                eval(cache)
                #expect(maximumError(output, expected[0..., range, 0...]) < tolerance)
                #expect(cache.allSatisfy { $0.offset == range.upperBound })
                #expect(cache.allSatisfy { $0.innerState().allSatisfy { $0.dim(1) == 1 } })
            }
        }
    }

    @Test("Native Talkie architecture is available to the common model loader")
    func registersArchitecture() async {
        await TalkieModelRegistration.register()
        #expect(await TalkieModelRegistration.isRegistered())
    }

    @Test("GQA FP32 recovery master weights reproduce their BF16 export exactly")
    func recoveryExportPrecision() throws {
        try Device.withDefaultDevice(.cpu) {
            let data = try Data(contentsOf: fixtureDirectory.appendingPathComponent("config.json"))
            var object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
            object["num_key_value_heads"] = 1
            let config = try JSONDecoder().decode(
                TalkieConfiguration.self,
                from: JSONSerialization.data(withJSONObject: object))
            var masters = try arrays("weights-bf16.safetensors")
            for layer in 0..<2 {
                for projection in ["key", "value"] {
                    let path = "model.blocks.\(layer).attn.attn_\(projection).weight"
                    let weight = try #require(masters[path])
                    masters[path] = weight[0..<8, 0...].asType(.float32) + Float(0.00005)
                }
            }
            let exported = masters.mapValues { $0.asType(.bfloat16) }
            let trainingModel = TalkieModel(config)
            let exportModel = TalkieModel(config)
            try trainingModel.update(
                parameters: ModuleParameters.unflattened(trainingModel.sanitize(weights: masters)), verify: .all)
            try exportModel.update(
                parameters: ModuleParameters.unflattened(exportModel.sanitize(weights: exported)), verify: .all)
            let trainingOutput = trainingModel(tokens(0..<7), cache: nil)
            let exportOutput = exportModel(tokens(0..<7), cache: nil)
            eval(trainingOutput, exportOutput)
            #expect(trainingOutput.dtype == .bfloat16)
            #expect(maximumError(trainingOutput, exportOutput) == 0)
        }
    }
}
