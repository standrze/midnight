#if os(macOS)
    import Foundation
    import MLX
    import Testing

    @Suite(
        "Experimental Metal D256 masked prefill", .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_RUN_SDPA_D256_REGRESSION"] == "1"))
    struct MetalSDPAD256MaskTests {
        @Test("Window tile bounds preserve GQA and uneven cache lengths")
        func windowsMatchReference() throws {
            try requireNAXExperiment()
            try Device.withDefaultDevice(.gpu) {
                for (heads, kvHeads, length) in [
                    (16, 8, 512), (32, 16, 1023), (8, 1, 1025), (16, 8, 1535), (4, 4, 1536),
                ] {
                    let mask = makeMask(length: length, window: 1024)
                    try checkFused(heads: heads, kvHeads: kvHeads, length: length, mask: mask)
                }
            }
        }

        @Test("Per-head masks, holes, and fully masked rows preserve existing semantics")
        func nonWindowMasksMatchReference() throws {
            try requireNAXExperiment()
            try Device.withDefaultDevice(.gpu) {
                let length = 1057
                let masks = (0..<4).map { head in
                    var elements = [Bool]()
                    elements.reserveCapacity(512 * length)
                    for row in 0..<512 {
                        for key in 0..<length {
                            // First block and one interior row are entirely masked.
                            // Other rows have head-specific bounds and interior holes.
                            elements.append(
                                row >= 64 && row != 271 && key >= 17 + head * 32
                                    && key < 801 + head * 32 && (row + key) % 7 != 0)
                        }
                    }
                    return MLXArray(elements, [512, length])
                }
                try checkFused(
                    heads: 4, kvHeads: 1, length: length,
                    mask: stacked(masks, axis: 0).expandedDimensions(axis: 0),
                    fullyMaskedRows: [0, 63, 271])
            }
        }

        @Test("Negative-infinite masks prune while finite negative biases remain active")
        func additiveMasksMatchReference() throws {
            try requireNAXExperiment()
            try Device.withDefaultDevice(.gpu) {
                let length = 1535
                let visible = makeMask(length: length, window: 129)
                for excludedValue in [-Float.infinity, Float(-8)] {
                    let mask = which(visible, MLXArray(Float(0)), MLXArray(excludedValue)).asType(.bfloat16)
                    try checkFused(heads: 8, kvHeads: 1, length: length, mask: mask)
                }
            }
        }

        private func requireNAXExperiment() throws {
            try #require(ProcessInfo.processInfo.environment["MIDNIGHT_METAL_SDPA_D256_MASKED"] == "1")
            try #require(
                ProcessInfo.processInfo.isOperatingSystemAtLeast(
                    OperatingSystemVersion(majorVersion: 26, minorVersion: 2, patchVersion: 0)))
            let architecture = GPU.deviceInfo().architecture
            let match = try #require(architecture.range(of: "g[0-9]+[a-z]$", options: .regularExpression))
            let suffix = architecture[match]
            let generation = try #require(Int(suffix.dropFirst().dropLast()))
            try #require(generation >= (suffix.last == "p" ? 18 : 17), "Run on a NAX-capable GPU.")
        }

        private func makeMask(length: Int, window: Int) -> MLXArray {
            let offset = length - 512
            let queries = MLXArray(offset..<(offset + 512)).expandedDimensions(axis: 1)
            let keys = MLXArray(0..<length).expandedDimensions(axis: 0)
            return (queries .>= keys) & (queries .< keys + window)
        }

        private func checkFused(
            heads: Int, kvHeads: Int, length: Int, mask: MLXArray, fullyMaskedRows: [Int] = []
        ) throws {
            let q = MLXRandom.normal([1, heads, 512, 256], key: MLXRandom.key(43)).asType(.bfloat16)
            let k = MLXRandom.normal([1, kvHeads, length + 19, 256], key: MLXRandom.key(47))
                .asType(.bfloat16)[.ellipsis, 3..<(length + 3), 0...]
            let v = MLXRandom.normal([1, kvHeads, length + 19, 256], key: MLXRandom.key(53))
                .asType(.bfloat16)[.ellipsis, 3..<(length + 3), 0...]
            let expandedK = repeated(k.asType(.float32), count: heads / kvHeads, axis: 1)
            let expandedV = repeated(v.asType(.float32), count: heads / kvHeads, axis: 1)
            var scores = matmul(q.asType(.float32) * Float(0.0625), expandedK.swappedAxes(-1, -2))
            if mask.dtype == .bool {
                // Both the backend and fallback use finite_min for bool masks,
                // yielding a uniform mean when every key in a row is masked.
                scores = which(mask, scores, MLXArray(-Float.greatestFiniteMagnitude))
            } else {
                scores = scores + mask.asType(.float32)
            }
            let reference = matmul(softmax(scores, axis: -1), expandedV)
            let actual = try withError { error in
                let result = MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: 0.0625,
                    mask: mask, forceFused: true)
                try error.check()
                eval(result)
                return result
            }
            let difference = actual.asType(.float32) - reference
            let maximumError = abs(difference).max().item(Float.self)
            let relativeRMSError = sqrt(mean(square(difference)) / mean(square(reference))).item(Float.self)
            #expect(actual.shape == q.shape)
            #expect(
                maximumError.isFinite && maximumError < 0.025,
                "heads=\(heads), kvHeads=\(kvHeads), length=\(length), mask=\(mask.dtype), maxError=\(maximumError)")
            #expect(
                relativeRMSError.isFinite && relativeRMSError < 0.025,
                "length=\(length), mask=\(mask.dtype), relativeRMS=\(relativeRMSError)")
            // A global tolerance could hide a denominator that includes padded
            // keys. These rows must average exactly the actual cache length.
            let valueMean = mean(expandedV, axis: 2)[0]
            for row in fullyMaskedRows {
                let rowError = actual[0, 0..., row, 0...].asType(.float32) - valueMean
                let relativeMeanError = sqrt(mean(square(rowError)) / mean(square(valueMean))).item(Float.self)
                #expect(
                    relativeMeanError.isFinite && relativeMeanError < 0.006,
                    "Fully masked row=\(row), actualKeys=\(length), mean relativeRMS=\(relativeMeanError)")
            }
        }
    }
#endif
