#if os(macOS)
    import Foundation
    import MLX
    import Testing

    @Suite(
        "Experimental Metal D512 decode", .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_RUN_SDPA_D512_REGRESSION"] == "1"))
    struct MetalSDPAD512Tests {
        @Test("Fused FP16/BF16 D512 decode matches an independent FP32 oracle")
        func fusedMatchesReference() throws {
            try #require(ProcessInfo.processInfo.environment["MIDNIGHT_METAL_SDPA_D512"] == "1")
            try Device.withDefaultDevice(.gpu) {
                // Empty partitions, SIMD boundaries, dispatch thresholds, and real Gemma GQA shapes.
                let shapes = [
                    (heads: 4, kvHeads: 4, length: 1),
                    (heads: 8, kvHeads: 2, length: 31),
                    (heads: 16, kvHeads: 2, length: 33),
                    (heads: 32, kvHeads: 4, length: 1023),
                    (heads: 16, kvHeads: 2, length: 1024),
                    (heads: 32, kvHeads: 4, length: 4097),
                ]
                for dtype in [DType.float16, .bfloat16] {
                    for shape in shapes {
                        let q = MLXRandom.normal(
                            [1, shape.heads, 1, 512], key: MLXRandom.key(17)
                        ).asType(dtype)
                        // Slice a padded cache so the KV head stride differs from length * D.
                        let k = MLXRandom.normal(
                            [1, shape.kvHeads, shape.length + 17, 512], key: MLXRandom.key(23)
                        ).asType(dtype)[.ellipsis, 1..<(shape.length + 1), 0...]
                        let v = MLXRandom.normal(
                            [1, shape.kvHeads, shape.length + 17, 512], key: MLXRandom.key(29)
                        ).asType(dtype)[.ellipsis, 1..<(shape.length + 1), 0...]
                        try checkFused(q: q, k: k, v: v, dtype: dtype)
                    }
                }
            }
        }

        @Test("Sinks and concentrated scores preserve the softmax denominator")
        func sinksAndLargeScores() throws {
            try #require(ProcessInfo.processInfo.environment["MIDNIGHT_METAL_SDPA_D512"] == "1")
            try Device.withDefaultDevice(.gpu) {
                for dtype in [DType.float16, .bfloat16] {
                    let q = MLXRandom.normal([1, 16, 1, 512], scale: 8, key: MLXRandom.key(31)).asType(dtype)
                    let k = MLXRandom.normal([1, 2, 257, 512], key: MLXRandom.key(37)).asType(dtype)
                    let v = MLXRandom.normal([1, 2, 257, 512], key: MLXRandom.key(41)).asType(dtype)
                    let sinks = MLXArray.full([16], values: MLXArray(Float(16)), dtype: dtype)
                    try checkFused(q: q, k: k, v: v, dtype: dtype, sinks: sinks)
                }
            }
        }

        @Test("Unsupported shapes stay outside the experimental fused path")
        func rejectsBroaderShapes() throws {
            try Device.withDefaultDevice(.gpu) {
                let cases: [(batch: Int, heads: Int, queries: Int, dtype: DType, mask: Bool)] = [
                    (2, 8, 1, .float16, false),
                    (1, 8, 2, .float16, false),
                    (1, 8, 1, .float32, false),
                    (1, 32, 1, .float16, false),
                    (1, 8, 1, .float16, true),
                ]
                for shape in cases {
                    let q = MLXArray.zeros([shape.batch, shape.heads, shape.queries, 512], dtype: shape.dtype)
                    let k = MLXArray.zeros([shape.batch, 2, 8, 512], dtype: shape.dtype)
                    let v = MLXArray.zeros([shape.batch, 2, 8, 512], dtype: shape.dtype)
                    let mask = shape.mask ? MLXArray.ones([shape.queries, 8], dtype: .bool) : nil
                    do {
                        _ = try withError {
                            MLXFast.scaledDotProductAttention(
                                queries: q, keys: k, values: v, scale: 1, mask: mask, forceFused: true)
                        }
                        Issue.record("Unsupported D512 shape unexpectedly admitted: \(shape)")
                    } catch let MLXError.caught(message) {
                        #expect(message.contains("force_fused=True but no fused kernel is available"))
                    }
                    let fallback = MLXFast.scaledDotProductAttention(
                        queries: q, keys: k, values: v, scale: 1, mask: mask)
                    #expect(abs(fallback).max().item(Float.self) == 0)
                }
            }
        }

        private func checkFused(
            q: MLXArray, k: MLXArray, v: MLXArray, dtype: DType, sinks: MLXArray? = nil
        ) throws {
            let scale: Float = 1 / sqrt(512)
            let repetitions = q.dim(1) / k.dim(1)
            let expandedK = repeated(k.asType(.float32), count: repetitions, axis: 1)
            let expandedV = repeated(v.asType(.float32), count: repetitions, axis: 1)
            var scores = matmul(q.asType(.float32) * scale, expandedK.swappedAxes(-1, -2))
            if let sinks {
                scores = concatenated([sinks.asType(.float32).reshaped([1, q.dim(1), 1, 1]), scores], axis: -1)
            }
            var probabilities = softmax(scores, axis: -1)
            if sinks != nil {
                probabilities = probabilities[.ellipsis, 1...]
            }
            let reference = matmul(probabilities, expandedV)
            let actual = try withError { error in
                // forceFused makes a missing/disabled specialization fail instead of passing via fallback.
                let result = MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: scale, mask: nil,
                    sinks: sinks, forceFused: true)
                try error.check()
                eval(result)
                return result
            }
            let difference = actual.asType(.float32) - reference
            let maximumError = abs(difference).max().item(Float.self)
            let relativeRMSError = sqrt(mean(square(difference)) / mean(square(reference))).item(Float.self)
            let absoluteTolerance: Float = dtype == .bfloat16 ? 0.025 : 0.004
            let relativeTolerance: Float = dtype == .bfloat16 ? 0.018 : 0.004
            #expect(actual.shape == q.shape)
            #expect(
                maximumError.isFinite && maximumError < absoluteTolerance,
                "dtype=\(dtype), shape=\(k.shape), sinks=\(sinks != nil), maxError=\(maximumError)")
            #expect(
                relativeRMSError.isFinite && relativeRMSError < relativeTolerance,
                "dtype=\(dtype), shape=\(k.shape), relativeRMS=\(relativeRMSError)")
        }
    }
#endif
