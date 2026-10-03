#if os(macOS)
    import CryptoKit
    import Foundation
    import MLX
    import Testing

    @Suite(
        "Opt-in Q4_0 mixed dtype reference primitive", .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_RUN_Q4ZERO_PROTOTYPE"] == "1"))
    struct Q4ZeroMetalPrototypeTests {
        @Test("BF16 activations and untouched F16 scales match the source-grid CPU dot product")
        func syntheticShapes() throws {
            try Device.withDefaultDevice(.gpu) {
                for (rows, columns) in [(7, 640), (8, 704), (33, 2112), (3, 2816), (32, 5376)] {
                    var source = Data()
                    for block in 0..<(rows * columns / 32) {
                        let scale = Float16(Float(block % 11 + 1) / 1024 * (block % 2 == 0 ? 1 : -1))
                        source.append(Q4ZeroPrototypeChecks.block(scale: scale.bitPattern, seed: block % 16))
                    }
                    let descriptor = try Q4ZeroPrototypeDescriptor(
                        rows: rows, columns: columns, sourceByteCount: source.count)
                    let grid = try Q4ZeroPrototypeCodec.repack(source, descriptor: descriptor)
                    try check(grid)
                }
            }
        }

        @Test("Optional pinned official row slice retains its verified source bytes")
        func officialSlice() throws {
            guard let path = ProcessInfo.processInfo.environment["MIDNIGHT_Q4ZERO_FIXTURE_DIR"] else {
                return
            }
            let directory = URL(fileURLWithPath: path)
            let document = try JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appendingPathComponent("provenance.json")))
            let manifest = try #require(document as? [String: Any])
            try #require(manifest["status"] as? String == "acquired")
            try #require(manifest["revision"] as? String == "59dde24573e7e61570dba08b18a2e1fe246955ed")
            let slice = try #require(manifest["slice"] as? [String: Any])
            let rows = try #require(slice["rows"] as? Int)
            let columns = try #require(slice["columns"] as? Int)
            let byteCount = try #require(slice["byte_count"] as? Int)
            let type = try #require(slice["ggml_type"] as? Int)
            let group = try #require(slice["group_size"] as? Int)
            let bits = try #require(slice["bits"] as? Int)
            let scaleType = try #require(slice["scale_dtype"] as? String)
            let offset = try #require(slice["implicit_offset"] as? Int)
            let hasBias = try #require(slice["has_bias_tensor"] as? Bool)
            let descriptor = try Q4ZeroPrototypeDescriptor(
                rows: rows, columns: columns, sourceByteCount: byteCount,
                ggmlType: type, groupSize: group, bits: bits, scaleType: scaleType,
                implicitOffset: offset, hasBiasTensor: hasBias)
            let grid = try Q4ZeroPrototypeCodec.readRows(
                from: directory.appendingPathComponent("slice.q4_0"), descriptor: descriptor, rows: 0..<rows)
            let digest = SHA256.hash(data: grid.sourceBytes()).map { String(format: "%02x", $0) }.joined()
            try #require(digest == slice["sha256"] as? String)
            try Device.withDefaultDevice(.gpu) { try check(grid) }
        }

        private func check(_ grid: Q4ZeroPrototypeGrid) throws {
            let columns = grid.columns
            let values = (0..<columns).map { Float(($0 * 37 + 17) % 251 - 125) / (125 * sqrt(Float(columns))) }
            let input = MLXArray(values).asType(.bfloat16)
            let packed = MLXArray(grid.packed).reshaped([grid.rows, columns / 8])
            // Reinterpret the original scale bytes. There is no floating-point cast or scale fitting.
            let scales = MLXArray(grid.scaleBits).view(dtype: .float16).reshaped([grid.rows, columns / 32])
            eval(input, packed, scales)
            let roundedInput = input.asType(.float32).asArray(Float.self)
            #expect(scales.view(dtype: .uint16).asArray(UInt16.self) == grid.scaleBits)
            let result = try withError { error in
                let output = Self.kernel(
                    [input, packed, scales], template: [("K", columns), ("N", grid.rows)],
                    grid: (grid.rows * 32, 1, 1), threadGroup: (32, 1, 1),
                    outputShapes: [[grid.rows], [grid.rows]], outputDTypes: [.bfloat16, .float32])
                try error.check()
                eval(output)
                return output
            }
            #expect(input.dtype == .bfloat16 && scales.dtype == .float16 && packed.dtype == .uint32)
            #expect(result[0].dtype == .bfloat16 && result[1].dtype == .float32)
            let actual = result[0].asType(.float32).asArray(Float.self)
            let sums = result[1].asArray(Float.self)
            var maximumError: Double = 0
            for row in 0..<grid.rows {
                var expected: Double = 0
                var absoluteSum: Double = 0
                for column in 0..<columns {
                    let term = Double(roundedInput[column]) * Double(grid.referenceValue(row: row, column: column))
                    expected += term
                    absoluteSum += abs(term)
                }
                let accumulationTolerance = max(1e-6, absoluteSum * 2e-5)
                #expect(sums[row].isFinite && abs(Double(sums[row]) - expected) <= accumulationTolerance)
                // BF16 output rounds once after FP32 accumulation. Near cancellation uses an absolute bound.
                let outputTolerance = accumulationTolerance + abs(expected) / 128 + 1e-7
                #expect(actual[row].isFinite && abs(Double(actual[row]) - expected) <= outputTolerance)
                maximumError = max(maximumError, abs(Double(actual[row]) - expected))
            }
            print(
                "Q4_0 source-grid parity N=\(grid.rows) K=\(columns) bytes=\(grid.storedByteCount) max=\(maximumError)")
        }

        // One SIMD group per output row: deliberately simple correctness fixture, not a tuned kernel.
        // Per-element float conversions remain register-local; no tensor promotion or affine bias allocation.
        private static let kernel = MLXFast.metalKernel(
            name: "midnight_test_q4zero_f16_scales_bf16_activation_v1",
            inputNames: ["x", "packed", "scales"], outputNames: ["output", "accumulated"],
            source: """
                const uint row = threadgroup_position_in_grid.x;
                const uint lane = thread_position_in_threadgroup.x;
                float total = 0.0f;
                for (uint k = lane; k < K; k += 32) {
                    const uint word = packed[row * (K / 8) + k / 8];
                    const int code = int((word >> (4 * (k % 8))) & 15u) - 8;
                    const float weight = float(scales[row * (K / 32) + k / 32]) * float(code);
                    total = fma(float(x[k]), weight, total);
                }
                const float sum = simd_sum(total);
                if (lane == 0 && row < N) {
                    output[row] = bfloat16_t(sum);
                    accumulated[row] = sum;
                }
                """, ensureRowContiguous: true)
    }
#endif
