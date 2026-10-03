#if os(macOS)
    import MLX
    import MLXNN
    import Testing

    @Suite("Sparse mixed-precision module updates", .serialized)
    struct MLXSparseQuantizationTests {
        final class Block: Module {
            @ModuleInfo var projection: Linear
            override init() {
                _projection.wrappedValue = Linear(128, 8, bias: false)
                super.init()
            }
        }
        final class Blocks: Module {
            let layers = [Block(), Block(), Block(), Block()]
        }
        final class Linears: Module {
            @ModuleInfo var layers: [Linear]
            override init() {
                _layers.wrappedValue = (0..<4).map { _ in Linear(128, 8, bias: false) }
                super.init()
            }
        }

        @Test("Quantizing later nested blocks preserves leading and interior BF16 blocks")
        func nestedSparseUpdates() {
            let model = Blocks()
            let first = model.layers[0].projection
            let middle = model.layers[2].projection
            quantize(model: model) { path, _ in
                if path == "layers.1.projection" {
                    return (128, 4, .affine)
                }
                if path == "layers.3.projection" {
                    return (128, 8, .affine)
                }
                return nil
            }
            #expect(model.layers[0].projection === first)
            #expect(model.layers[2].projection === middle)
            #expect((model.layers[1].projection as? QuantizedLinear)?.bits == 4)
            #expect((model.layers[3].projection as? QuantizedLinear)?.bits == 8)
            let result = model.layers[3].projection(MLXArray.ones([1, 128]))
            #expect(all(isFinite(result)).item(Bool.self))
        }

        @Test("Sparse updates to a direct module array preserve untouched entries")
        func directSparseUpdates() {
            let model = Linears()
            let first = model.layers[0]
            let middle = model.layers[2]
            quantize(model: model) { path, _ in
                (path == "layers.1" || path == "layers.3") ? (128, 4, .affine) : nil
            }
            #expect(model.layers.count == 4)
            #expect(model.layers[0] === first)
            #expect(model.layers[2] === middle)
            #expect(model.layers[1] is QuantizedLinear)
            #expect(model.layers[3] is QuantizedLinear)
        }
    }
#endif
