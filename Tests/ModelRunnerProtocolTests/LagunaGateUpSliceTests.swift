#if os(macOS)
    import Foundation
    import MLX
    import MLXLMCommon
    import MLXNN
    import Testing

    @testable import ModelRunnerCore

    @Suite("Mac gate/up independent slices", .serialized)
    struct LagunaGateUpSliceTests {
        @Test("Equal gate/up slices preserve contiguous and strided values on CPU")
        func cpuSlices() throws { try checkSlices(device: .cpu) }

        @Test("Routed gate/up slices preserve sorted and unsorted quantized outputs on CPU")
        func cpuRouted() throws { try checkRouted(device: .cpu) }

        @Test(
            "Equal gate/up slices preserve contiguous and strided values on Metal",
            .enabled(if: runMetal))
        func metalSlices() throws { try checkSlices(device: .gpu) }

        @Test(
            "Routed gate/up slices preserve sorted and unsorted quantized outputs on Metal",
            .enabled(if: runMetal))
        func metalRouted() throws { try checkRouted(device: .gpu) }

        @Test(
            "Compiled routed models release materialized captures after each lifetime",
            .enabled(if: runMetal), arguments: ["cpu", "metal"])
        func compiledLifetime(engine: String) throws {
            let device: Device = engine == "cpu" ? .cpu : .gpu
            try Device.withDefaultDevice(device) {
                // Warm global activation/compiler bookkeeping before isolating repeated
                // model-owned captures. All subsequent models use fresh weights.
                let warm = try exerciseLifetime(device: device, seed: 1)
                let before = memory(device)
                for seed in 2...17 {
                    let result = try exerciseLifetime(device: device, seed: seed)
                    #expect(
                        result.heldBytes >= before + 16 * 1024,
                        "The model must actually materialize and hold its captured weights")
                    if seed == 2 {
                        #expect(result.fingerprint != warm.fingerprint)
                    }
                    let after = memory(device)
                    #expect(
                        after <= before + 1024,
                        "\(engine) compiled lifetime \(seed) retained \(after-before) bytes")
                }
            }
        }

        private static var runMetal: Bool {
            ProcessInfo.processInfo.environment["MIDNIGHT_RUN_GATE_UP_LIFETIME"] == "1"
        }

        private func checkSlices(device: Device) throws {
            try Device.withDefaultDevice(device) {
                for dtype in [DType.float32, .bfloat16] {
                    for strided in [false, true] {
                        let source = values([2, 4, 16], seed: 1).asType(dtype)
                        let input = strided ? source.transposed(0, 2, 1) : source
                        let reference = split(input, parts: 2, axis: -1)
                        let result = lagunaGateUpSlices(input)
                        try checkedEval(reference + result)
                        #expect(result.map(\.shape) == reference.map(\.shape))
                        for index in 0..<2 {
                            #expect(bits(result[index]) == bits(reference[index]))
                        }
                    }
                }
            }
        }

        private func checkRouted(device: Device) throws {
            try Device.withDefaultDevice(device) {
                for quantized in [false, true] {
                    // The pinned CPU unquantized gather kernel supports Float32 only.
                    let dtype: DType = device == .cpu && !quantized ? .float32 : .bfloat16
                    let layer = makeLayer(seed: 5, quantized: quantized, dtype: dtype)
                    for count in [1, 16] {
                        // 4 selected experts: count1 is unsorted; count16 reaches the
                        // existing 64-assignment sorting threshold exactly.
                        let input = values([count, 64], seed: 2).asType(dtype)
                        let indices = MLXArray((0..<(count * 4)).map { UInt32(($0 * 3 + 1) % 8) })
                            .reshaped(count, 4)
                        let expected = Self.reference(layer, input: input, indices: indices)
                        let eager = layer(input, indices)
                        let compiled: @Sendable ([MLXArray]) -> [MLXArray] = compile { args in
                            [layer(args[0], args[1])]
                        }
                        let compiledReference: @Sendable ([MLXArray]) -> [MLXArray] = compile { args in
                            [Self.reference(layer, input: args[0], indices: args[1])]
                        }
                        let actual = compiled([input, indices])[0]
                        let expectedCompiled = compiledReference([input, indices])[0]
                        try checkedEval(expected, eager, actual, expectedCompiled)
                        #expect(eager.shape == expected.shape)
                        #expect(actual.shape == expected.shape)
                        let eagerEqual = bits(eager) == bits(expected)
                        let compiledEqual = bits(actual) == bits(expectedCompiled)
                        #expect(eagerEqual, "Eager gate/up slicing changed routed values")
                        #expect(
                            compiledEqual,
                            "Compiled split/slice values differ: quantized=\(quantized), assignments=\(indices.size)")
                    }
                }
            }
        }

        private static func reference(_ layer: FusedGateUpSwitchGLU, input: MLXArray, indices: MLXArray) -> MLXArray {
            let leaves = Dictionary(uniqueKeysWithValues: layer.leafModules().flattened())
            let gate = leaves["gate_up_proj"] as! SwitchLinear
            let down = leaves["down_proj"] as! SwitchLinear
            var x = expandedDimensions(input, axes: [-2, -3])
            var ids = indices
            var inverse = MLXArray()
            let sorted = indices.size >= 64
            if sorted {
                (x, ids, inverse) = gatherSort(x: x, indices: indices)
            }
            let parts = split(gate(x, ids, sortedIndices: sorted), parts: 2, axis: -1)
            x = down(compiledSiluProduct(parts[0], parts[1]), ids, sortedIndices: sorted)
            if sorted {
                x = scatterUnsort(x: x, invOrder: inverse, shape: indices.shape)
            }
            return squeezed(x, axis: -2)
        }

        private func makeLayer(seed: Int, quantized: Bool, dtype: DType = .bfloat16) -> FusedGateUpSwitchGLU {
            let layer = FusedGateUpSwitchGLU(inputDims: 64, hiddenDims: 64, numExperts: 8)
            layer.update(parameters: layer.parameters().mapValues { values($0.shape, seed: seed).asType(dtype) })
            if quantized {
                quantize(model: layer, groupSize: 32, bits: 4)
            }
            eval(layer.parameters())
            return layer
        }

        private func exerciseLifetime(device: Device, seed: Int) throws -> (heldBytes: Int, fingerprint: [UInt32]) {
            try autoreleasepool {
                let layer = makeLayer(seed: seed, quantized: true)
                let input = values([1, 64], seed: seed + 7).asType(.bfloat16)
                let indices = MLXArray([UInt32(0), 2, 4, 6]).reshaped(1, 4)
                let compiled: @Sendable ([MLXArray]) -> [MLXArray] = compile { args in [layer(args[0], args[1])] }
                let output = compiled([input, indices])[0]
                try checkedEval(output)
                StreamOrDevice.device(device).stream.synchronize()
                return (Memory.activeMemory, Array(bits(output).prefix(16)))
            }
        }

        private func values(_ shape: [Int], seed: Int) -> MLXArray {
            let count = shape.reduce(1, *)
            return MLXArray((0..<count).map { Float(($0 * 7 + seed * 3) % 31 - 15) / 128 })
                .reshaped(shape)
        }

        private func bits(_ array: MLXArray) -> [UInt32] {
            array.asType(.float32).asArray(Float.self).map(\.bitPattern)
        }

        private func memory(_ device: Device) -> Int {
            StreamOrDevice.device(device).stream.synchronize()
            Stream.cpu.synchronize()
            Memory.clearCache()
            return Memory.activeMemory
        }
    }
#endif
