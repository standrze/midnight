#if os(macOS)
    import CryptoKit
    import Foundation
    import MLX
    import Testing

    @Suite(
        "Experimental Metal affine Q4 QMV tails", .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_RUN_Q4_QMV_TAIL_REGRESSION"] == "1"))
    struct MetalAffineQ4QMVTailTests {
        // Run separate processes with MLX_METAL_AFFINE_Q4_QMV_TAIL=0 and =1: the backend caches the gate.
        // Optional reports contain complete outputs for numerical A/B comparison, plus reproducibility hashes.
        // Selector expectations are source-derived; use a Metal capture to independently confirm kernel names.
        @Test("Gemma dense and gathered Q4 tails match a dequantized FP32 oracle")
        func matchesReference() throws {
            let environment = ProcessInfo.processInfo.environment
            let arm = try #require(environment["MLX_METAL_AFFINE_Q4_QMV_TAIL"])
            try #require(arm == "0" || arm == "1", "Set the tail gate before starting each test process.")
            let scope = environment["MLX_METAL_AFFINE_Q4_QMV_TAIL_SCOPE"] ?? "all"
            let iterations = Int(environment["MIDNIGHT_Q4_QMV_TAIL_TIMING_ITERATIONS"] ?? "0") ?? -1
            try #require((0...200).contains(iterations))
            let outputPath = environment["MIDNIGHT_Q4_QMV_TAIL_REPORT"]
            if let outputPath {
                try #require(!FileManager.default.fileExists(atPath: outputPath))
            }

            let cases: [Shape] =
                [
                    .init(name: "gemma3_270m_gate", k: 640, n: 2048),
                    .init(name: "gemma4_31b_gate", k: 5376, n: 21504),
                    .init(name: "gemma4_a4b_shared_gate", k: 2816, n: 2112),
                    .init(name: "gemma4_a4b_expert_gate", k: 2816, n: 704, gathered: true),
                    .init(name: "gemma4_a4b_combined_gate_up", k: 2816, n: 1408, gathered: true),
                    .init(name: "gemma4_a4b_expert_down", k: 704, n: 2816, gathered: true),
                    .init(name: "batched_tail", k: 640, n: 24, batches: 2),
                    .init(name: "aligned_control", k: 2048, n: 640),
                    .init(name: "gather_aligned_control", k: 512, n: 24, gathered: true),
                    .init(name: "unaligned_n_control", k: 640, n: 17),
                    .init(name: "small_k_control", k: 448, n: 24),
                ]
                + [576, 640, 704, 960, 2816, 5376].map {
                    .init(name: "tail_only_\($0)", k: $0, n: 24, tailOnly: true)
                }
            var rows: [[String: Any]] = []
            try Device.withDefaultDevice(.gpu) {
                for dtype in [DType.float16, .bfloat16] {
                    for shape in cases {
                        // Keep the 31B-shaped FP32 oracle's lifetime bounded to this one case.
                        let row = try autoreleasepool {
                            try check(shape: shape, dtype: dtype, arm: arm, scope: scope, iterations: iterations)
                        }
                        rows.append(row)
                    }
                }
            }
            if let outputPath {
                let report: [String: Any] = [
                    "scope": "Synthetic packed Q4/G64 primitive correctness; not model quality or serving throughput",
                    "tail_gate": arm, "tail_scope": scope, "bits": 4, "group_size": 64,
                    "gpu_architecture": GPU.deviceInfo().architecture,
                    "timing_iterations": iterations, "rows": rows,
                ]
                try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                    .write(to: URL(fileURLWithPath: outputPath), options: .withoutOverwriting)
            }
        }

        private struct Shape {
            let name: String
            let k: Int
            let n: Int
            var batches: Int = 1
            var gathered: Bool = false
            var tailOnly: Bool = false

            var weightBatches: Int { gathered ? 3 : batches }
            var inputBatches: Int { gathered ? 2 : batches }
            var tailEligible: Bool { k >= 512 && k % 512 != 0 && n % 8 == 0 }
        }

        private func check(shape: Shape, dtype: DType, arm: String, scope: String, iterations: Int) throws -> [String:
            Any]
        {
            let weightShape = shape.weightBatches == 1 ? [shape.n] : [shape.weightBatches, shape.n]
            let packedCount = shape.weightBatches * shape.n * shape.k / 8
            let packed = MLXArray((0..<packedCount).map { mixedWord(UInt32($0)) })
                .reshaped(weightShape + [shape.k / 8])
            let groupCount = shape.weightBatches * shape.n * shape.k / 64
            let scaleValues = (0..<groupCount).map { index -> Float in
                let sign: Float = index.isMultiple(of: 2) ? 1 : -1
                return sign * Float(index % 13 + 1) / 1024
            }
            let biasValues = (0..<groupCount).map { Float(($0 * 11) % 17 - 8) / 128 }
            let scales = MLXArray(scaleValues).reshaped(weightShape + [shape.k / 64]).asType(dtype)
            let biases = MLXArray(biasValues).reshaped(weightShape + [shape.k / 64]).asType(dtype)
            let activeInputWidth = shape.tailOnly ? shape.k % 512 : shape.k
            let inputValues = (0..<(shape.inputBatches * shape.k)).map { index -> Float in
                if shape.tailOnly && index % shape.k < shape.k - shape.k % 512 {
                    return 0
                }
                return Float(Int(mixedWord(UInt32(index + 12345)) % 1021) - 510)
                    / (510 * sqrt(Float(activeInputWidth)))
            }
            let x = MLXArray(inputValues).reshaped([shape.inputBatches, 1, shape.k]).asType(dtype)
            let lhs = MLXArray([UInt32(1), 0, 1, 1, 0, 0, 1, 0])
            let rhs = MLXArray([UInt32(2), 0, 2, 1, 0, 2, 1, 1])
            // Convert stored metadata before dequantizing, avoiding an intermediate FP16/BF16 weight rounding.
            let restored = dequantized(
                packed, scales: scales.asType(.float32), biases: biases.asType(.float32),
                groupSize: 64, bits: 4, dtype: .float32
            ).swappedAxes(-1, -2)
            let reference =
                shape.gathered
                ? gatherMM(x.asType(.float32), restored, lhsIndices: lhs, rhsIndices: rhs)
                : matmul(x.asType(.float32), restored)
            eval(x, packed, scales, biases, reference)

            func operation() -> MLXArray {
                if shape.gathered {
                    return gatherQuantizedMM(
                        x, packed, scales: scales, biases: biases, lhsIndices: lhs, rhsIndices: rhs,
                        transpose: true, groupSize: 64, bits: 4, sortedIndices: false)
                }
                return quantizedMM(x, packed, scales: scales, biases: biases, groupSize: 64, bits: 4)
            }

            let actual = try withError { error in
                let result = operation()
                try error.check()
                eval(result)
                return result
            }
            let difference = actual.asType(.float32) - reference
            let maximumError = abs(difference).max().item(Float.self)
            let relativeRMS = sqrt(mean(square(difference)) / mean(square(reference))).item(Float.self)
            let referenceMaximum = abs(reference).max().item(Float.self)
            let absoluteTolerance: Float = dtype == .bfloat16 ? 0.002 : 0.0004
            let relativeTolerance: Float = dtype == .bfloat16 ? 0.012 : 0.002
            let context = "case=\(shape.name), dtype=\(dtype), tail_gate=\(arm), tail_scope=\(scope)"
            #expect(actual.shape == reference.shape, "\(context)")
            #expect(referenceMaximum > absoluteTolerance * 5, "Oracle must detect missing writes: \(context)")
            #expect(maximumError.isFinite && maximumError < absoluteTolerance, "\(context), maxError=\(maximumError)")
            #expect(relativeRMS.isFinite && relativeRMS < relativeTolerance, "\(context), relativeRMS=\(relativeRMS)")

            var milliseconds: [Double] = []
            if iterations > 0 {
                for _ in 0..<3 {
                    eval(operation())
                }
                for _ in 0..<iterations {
                    Stream.defaultStream(.gpu).synchronize()
                    let start = ContinuousClock.now
                    eval(operation())
                    Stream.defaultStream(.gpu).synchronize()
                    let duration = start.duration(to: .now).components
                    milliseconds.append(Double(duration.seconds) * 1000 + Double(duration.attoseconds) / 1e15)
                }
            }
            let output = actual.asType(.float32).asArray(Float.self)
            let bytes = output.withUnsafeBytes { Data($0) }
            let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
            print("Q4 tail \(context), max_error=\(maximumError), relative_rms=\(relativeRMS), sha256=\(digest)")
            return [
                "case": shape.name, "dtype": String(describing: dtype), "k": shape.k, "n": shape.n,
                "weight_batches": shape.weightBatches, "gathered": shape.gathered, "tail_only": shape.tailOnly,
                "expected_tail_selection": arm == "1" && shape.tailEligible
                    && (scope == "all" || scope == (shape.gathered ? "gather" : "dense")),
                "output_shape": actual.shape, "maximum_absolute_error": maximumError,
                "relative_rms_error": relativeRMS, "reference_maximum": referenceMaximum,
                "output_sha256": digest, "output": output, "wall_milliseconds": milliseconds,
            ]
        }

        private func mixedWord(_ index: UInt32) -> UInt32 {
            var value = index &+ 0x9e37_79b9
            value = (value ^ (value >> 16)) &* 0x85eb_ca6b
            value = (value ^ (value >> 13)) &* 0xc2b2_ae35
            return value ^ (value >> 16)
        }
    }
#endif
