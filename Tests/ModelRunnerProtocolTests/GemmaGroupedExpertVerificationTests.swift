import Foundation
import MLX
@_spi(Testing) import MLXLMCommon
import Testing

@Suite("Gemma grouped expert verification eligibility", .serialized)
struct GemmaGroupedExpertVerificationEligibilityTests {
    @Test("Only BF16 Q4/G64 Gemma A4B verification shapes enter the experiment")
    func supportedShapes() {
        Device.withDefaultDevice(.cpu) {
            let weight = MLXArray.zeros([128, 704, 352], dtype: .uint32)
            let metadata = MLXArray.zeros([128, 704, 44], dtype: .bfloat16)
            func supports(_ x: MLXArray, _ ids: MLXArray, scales: MLXArray? = nil) -> Bool {
                GemmaGroupedExpertVerificationKernel.supports(
                    input: x, weight: weight, scales: scales ?? metadata, biases: metadata, indices: ids)
            }
            for tokens in [2, 3, 4, 5, 8] {
                let ids = MLXArray.zeros([tokens, 8], dtype: .uint32)
                let x = MLXArray.zeros([tokens, 1, 1, 2816], dtype: .bfloat16)
                #expect(supports(x, ids))
                #expect(supports(MLXArray.zeros([tokens, 8, 1, 2816], dtype: .bfloat16), ids))
                #expect(supports(MLXArray.zeros([tokens * 8, 1, 2816], dtype: .bfloat16), ids.flattened()))
                #expect(!supports(x.asType(.float16), ids))
                #expect(!supports(x.asType(.float32), ids))
                #expect(!supports(x, ids, scales: metadata.asType(.float16)))
                #expect(!supports(x, ids.asType(.int64)))
                #expect(
                    GemmaGroupedExpertVerificationKernel.callAsFunction(
                        input: x, weight: weight, scales: metadata, biases: metadata, indices: ids) == nil)
            }
            for tokens in [1, 9] {
                #expect(
                    !supports(
                        MLXArray.zeros([tokens, 1, 1, 2816], dtype: .bfloat16),
                        MLXArray.zeros([tokens, 8], dtype: .uint32)))
            }
            #expect(
                !supports(
                    MLXArray.zeros([2, 1, 1, 2816], dtype: .bfloat16),
                    MLXArray.zeros([2, 4], dtype: .uint32)))
            #expect(
                !supports(
                    MLXArray.zeros([2, 1, 1, 2816], dtype: .bfloat16),
                    MLXArray.zeros([1, 2, 8], dtype: .uint32)))
        }
    }
}

#if os(macOS)
    @Suite(
        "Gemma grouped expert verification Metal parity", .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_RUN_GROUPED_EXPERT_REGRESSION"] == "1"))
    struct GemmaGroupedExpertVerificationParityTests {
        @Test("Repeated experts preserve all assignments, including groups larger than four")
        func projectionParity() throws {
            try Device.withDefaultDevice(.gpu) {
                let cases: [Fixture] = [
                    .init(name: "no_reuse_gate", tokens: 2, k: 2816, n: 704, ids: Array(0..<16)),
                    .init(
                        name: "mixed_fused_gate_up", tokens: 3, k: 2816, n: 1408,
                        ids: (0..<24).map { [127, 0, 3, 17, 0, 127, 63][$0 % 7] }),
                    .init(
                        name: "all_same_down", tokens: 4, k: 704, n: 2816,
                        ids: Array(repeating: 97, count: 32), expanded: true),
                    .init(
                        name: "five_token_down", tokens: 5, k: 704, n: 2816,
                        ids: (0..<40).map { [127, 1, 7, 31, 1, 7, 0, 63, 31][$0 % 9] }, expanded: true),
                    .init(
                        name: "sorted_eight_token_gate", tokens: 8, k: 2816, n: 704,
                        ids: (0..<64).map { [0, 7, 7, 31, 63, 127][$0 % 6] }.sorted(), flattened: true),
                    .init(
                        name: "batched_two_by_three_gate", tokens: 6, k: 2816, n: 704,
                        ids: (0..<48).map { [2, 5, 17, 29, 61, 89, 101, 127][$0 % 8] }),
                ]
                for fixture in cases {
                    try autoreleasepool { try check(fixture) }
                }
            }
        }

        private struct Fixture {
            let name: String
            let tokens: Int
            let k: Int
            let n: Int
            let ids: [Int]
            var expanded = false
            var flattened = false
        }

        private func check(_ fixture: Fixture) throws {
            let k = fixture.k
            let n = fixture.n
            let assignments = fixture.ids.count
            // A compact packed expert is broadcast, while expert-specific affine
            // metadata exposes any incorrect expert selection. Inputs differ in
            // every assignment so skipped scatters and wrong membership are visible.
            let values = MLXRandom.normal(
                [1, n, k], scale: 0.02, key: MLXRandom.key(1801)
            ).asType(.bfloat16)
            let q = quantized(values, groupSize: 64, bits: 4)
            let packed = broadcast(q.wq, to: [128, n, k / 8])
            let factors = (MLXArray(0..<128).asType(.float32) / 128 + 0.5)
                .asType(.bfloat16).reshaped([128, 1, 1])
            let scales = broadcast(q.scales, to: [128, n, k / 64]) * factors
            let biases = broadcast(try #require(q.biases), to: [128, n, k / 64]) * factors
            let expanded = fixture.expanded || fixture.flattened
            let rows = expanded ? assignments : fixture.tokens
            let storage = MLXRandom.normal(
                [rows, k, 2], scale: 0.5, key: MLXRandom.key(1802)
            ).asType(.bfloat16)
            let inputShape = fixture.flattened ? [assignments, 1, k] : [fixture.tokens, expanded ? 8 : 1, 1, k]
            let x = storage[0..., 0..., 0].reshaped(inputShape)
            let idStorage = MLXArray(fixture.ids.flatMap { [UInt32($0), 0] }).reshaped([assignments, 2])
            let ids = idStorage[0..., 0].reshaped(fixture.flattened ? [assignments] : [fixture.tokens, 8])

            // Dequantize only selected experts. Conversion BEFORE dequantization
            // keeps the oracle independent of intermediate BF16 weight rounding.
            let unique = Array(Set(fixture.ids)).sorted()
            let selected = MLXArray(unique.map(UInt32.init))
            let positions = Dictionary(uniqueKeysWithValues: unique.enumerated().map { ($1, UInt32($0)) })
            let remapped = MLXArray(fixture.ids.map { positions[$0]! }).reshaped(ids.shape)
            let restored = dequantized(
                packed[selected], scales: scales[selected].asType(.float32),
                biases: biases[selected].asType(.float32), groupSize: 64, bits: 4, dtype: .float32
            )
            .swappedAxes(-1, -2)
            let reference = gatherMM(x.asType(.float32), restored, rhsIndices: remapped)
            let stock = gatherQuantizedMM(
                x, packed, scales: scales, biases: biases, rhsIndices: ids, groupSize: 64, bits: 4,
                sortedIndices: fixture.flattened)
            let actual = try #require(
                GemmaGroupedExpertVerificationKernel.callAsFunction(
                    input: x, weight: packed, scales: scales, biases: biases, indices: ids,
                    outputFill: .nan))
            eval(reference, stock, actual)
            let difference = actual.asType(.float32) - reference
            let maximumError = abs(difference).max().item(Float.self)
            let relativeRMS = sqrt(mean(square(difference)) / mean(square(reference))).item(Float.self)
            let stockRMS = sqrt(mean(square(stock.asType(.float32) - reference)) / mean(square(reference)))
                .item(Float.self)
            let stockDifference = sqrt(mean(square(actual.asType(.float32) - stock.asType(.float32))))
                .item(Float.self)
            #expect(actual.shape == ids.shape + [1, n])
            #expect(abs(reference).max().item(Float.self) > 0.1)
            #expect(maximumError.isFinite && maximumError < 0.025, "\(fixture.name): max=\(maximumError)")
            #expect(relativeRMS.isFinite && relativeRMS < 0.004, "\(fixture.name): rms=\(relativeRMS)")
            // Stock gather arithmetic rounds differently. Report its error too;
            // model fixed-prefix validation is required before enabling this path.
            #expect(stockRMS.isFinite && stockRMS < 0.025, "\(fixture.name): stock rms=\(stockRMS)")
            #expect(stockDifference.isFinite)
            print(
                "Grouped expert \(fixture.name): max_error=\(maximumError), relative_rms=\(relativeRMS), "
                    + "stock_relative_rms=\(stockRMS), candidate_stock_rms=\(stockDifference)")
        }
    }
#endif
