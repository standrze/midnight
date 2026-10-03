#if os(macOS)
    import Foundation
    import MLX
    import Testing

    @Suite(
        "Real Gemma expert affine Q4 projections", .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_RUN_REAL_GEMMA_Q4_PROJECTIONS"] == "1"))
    struct RealGemmaExpertQ4ProjectionTests {
        @Test("Real Q4/G64 tensors and implicit gather indices match an FP32 oracle")
        func realExpertProjections() throws {
            let environment = ProcessInfo.processInfo.environment
            let directory = URL(fileURLWithPath: try #require(environment["MIDNIGHT_Q4_REAL_CHECKPOINT"]))
            let reportURL = URL(fileURLWithPath: try #require(environment["MIDNIGHT_Q4_REAL_REPORT"]))
            let arm = try #require(environment["MLX_METAL_AFFINE_Q4_QMV_TAIL"])
            try #require(arm == "0" || arm == "1")
            try #require(!FileManager.default.fileExists(atPath: reportURL.path))
            let indexData = try Data(contentsOf: directory.appendingPathComponent("model.safetensors.index.json"))
            let index = try #require(JSONSerialization.jsonObject(with: indexData) as? [String: Any])
            let map = try #require(index["weight_map"] as? [String: String])
            var reports: [[String: Any]] = []
            try Device.withDefaultDevice(.gpu) {
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    let suffix = "model.layers.0.experts.switch_glu.\(projection)"
                    let weightKey = try #require(map.keys.first { $0.hasSuffix(suffix + ".weight") })
                    let prefix = String(weightKey.dropLast(".weight".count))
                    var tensors: [String: MLXArray] = [:]
                    let keys = [prefix + ".weight", prefix + ".scales", prefix + ".biases"]
                    for file in Set(try keys.map { try #require(map[$0]) }) {
                        let loaded = try loadArrays(url: directory.appendingPathComponent(file))
                        for key in keys {
                            if let value = loaded[key] {
                                tensors[key] = value
                            }
                        }
                    }
                    let weight = try #require(tensors[keys[0]])
                    let scales = try #require(tensors[keys[1]])
                    let biases = try #require(tensors[keys[2]])
                    let k = weight.dim(-1) * 8
                    let n = weight.dim(-2)
                    try #require(weight.dtype == .uint32 && weight.dim(0) == 128)
                    try #require(scales.dtype == .bfloat16 && biases.dtype == .bfloat16)
                    try #require(scales.shape == [128, n, k / 64] && biases.shape == scales.shape)
                    try #require((k == 2816 && n == 704) || (k == 704 && n == 2816))
                    let selected = MLXArray([UInt32(0), 127, 63, 7, 29, 44, 89, 101])
                    let restored = dequantized(
                        weight[selected], scales: scales[selected].asType(.float32),
                        biases: biases[selected].asType(.float32), groupSize: 64, bits: 4, dtype: .float32
                    )
                    .swappedAxes(-1, -2)
                    let expanded = projection == "down_proj"
                    for (caseIndex, name) in ["rms_one", "large", "cancellation", "outlier", "tail_only"].enumerated() {
                        for batches in [1, 2] {
                            let length = batches == 1 ? 1 : 3
                            let rows = batches * length * (expanded ? 8 : 1)
                            let values = (0..<(rows * k)).map { offset -> Float in
                                let i = offset % k
                                let base = sin(Float(offset + 1) * 1.713) + cos(Float(offset + 3) * 0.917)
                                switch name {
                                case "large": return base * 32
                                case "cancellation": return (i % 2 == 0 ? 1 : -1) * Float(1 + (i / 2) % 64)
                                case "outlier": return i % 131 == 0 ? (i % 2 == 0 ? 256 : -256) : base * 0.125
                                case "tail_only": return i < k - k % 512 ? 0 : base
                                default: return base
                                }
                            }
                            let x = MLXArray(values).asType(.bfloat16)
                                .reshaped([batches, length, expanded ? 8 : 1, 1, k])
                            let ids = broadcast(selected, to: [batches, length, 8])
                            let remapped = broadcast(MLXArray(0..<8).asType(.uint32), to: ids.shape)
                            // Deliberately omit lhsIndices. These are the exact
                            // broadcast gate/up and expanded down layouts used by Gemma.
                            let actual = gatherQuantizedMM(
                                x, weight, scales: scales, biases: biases, rhsIndices: ids,
                                groupSize: 64, bits: 4, sortedIndices: false)
                            let reference = gatherMM(x.asType(.float32), restored, rhsIndices: remapped)
                            eval(actual, reference)
                            let difference = actual.asType(.float32) - reference
                            let maximum = abs(difference).max().item(Float.self)
                            let relative = sqrt(mean(square(difference)) / mean(square(reference))).item(Float.self)
                            let referenceMax = abs(reference).max().item(Float.self)
                            #expect(actual.shape == [batches, length, 8, 1, n])
                            #expect(maximum.isFinite && maximum / max(referenceMax, 1e-6) < 0.03)
                            #expect(relative.isFinite && relative < 0.03)
                            reports.append([
                                "projection": prefix, "case": name, "case_index": caseIndex,
                                "input_shape": x.shape, "indices_shape": ids.shape, "lhs_indices": "implicit",
                                "weight_dtype": String(describing: weight.dtype),
                                "metadata_dtype": String(describing: scales.dtype), "weight_shape": weight.shape,
                                "maximum_error": maximum, "relative_rms": relative, "reference_maximum": referenceMax,
                                "output_shape": actual.shape, "output": actual.asType(.float32).asArray(Float.self),
                            ])
                        }
                    }
                }
            }
            let report: [String: Any] = [
                "checkpoint": directory.path, "tail_gate": arm, "rows": reports,
                "scope": "Real layer0 expert tensors with controlled activations; no model-quality claim",
            ]
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                .write(to: reportURL, options: .withoutOverwriting)
        }
    }
#endif
