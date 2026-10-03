import Cmlx
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@Suite("Gemma bounded chronological sliding cache")
struct GemmaBoundedCacheTests {
    @Test("A bounded two-input concat is retained once and returned wrappers remain independent")
    func retainsOwnedConcatenation() throws {
        try Device.withDefaultDevice(.cpu) {
            let cache = WindowedKVCache(window: 4)
            _ = cache.update(keys: rows(0..<2), values: rows(0..<2))
            for position in 2..<8 {
                let (workingKeys, workingValues) = cache.update(
                    keys: rows(position..<(position + 1)), values: rows(position..<(position + 1)))
                eval(workingKeys, workingValues)
                eval(cache)
                let retained = cache.state
                let retainedKeys = retained[0]
                let retainedValues = retained[1]
                eval(retainedKeys, retainedValues)
                let keyAddress = try floatStorageAddress(workingKeys)
                let valueAddress = try floatStorageAddress(workingValues)
                let retainedKeyAddress = try floatStorageAddress(retainedKeys)
                let retainedValueAddress = try floatStorageAddress(retainedValues)
                #expect(retainedKeyAddress == keyAddress)
                #expect(retainedValueAddress == valueAddress)
                let expected = Array(max(0, position - 3)...position).map(Float.init)
                #expect(retained[0].asArray(Float.self) == expected)
                workingKeys[0, 0, 0, 0] = MLXArray(Float(-999))
                workingValues[0, 0, 0, 0] = MLXArray(Float(-888))
                #expect(cache.state[0].asArray(Float.self) == expected)
                #expect(cache.state[1].asArray(Float.self) == expected)
            }
        }
    }

    @Test("Initial views and cropped concatenations still materialize small owned tails")
    func materializesBorrowedOrCroppedInputs() throws {
        try Device.withDefaultDevice(.cpu) {
            let source = rows(0..<256)
            let input = source[.ellipsis, 100..<102, 0...]
            let cache = WindowedKVCache(window: 4)
            let (initial, _) = cache.update(keys: input, values: input)
            eval(initial)
            eval(cache)
            let initialStoredKeys = cache.state[0]
            eval(initialStoredKeys)
            let initialAddress = try floatStorageAddress(initial)
            let initialStoredAddress = try floatStorageAddress(initialStoredKeys)
            #expect(initialStoredAddress != initialAddress)
            #expect(initialStoredKeys.asArray(Float.self) == [100, 101])

            let (working, _) = cache.update(keys: rows(102..<109), values: rows(102..<109))
            let tail = working[.ellipsis, (working.dim(2) - 4)..., 0...]
            eval(working, tail)
            eval(cache)
            let croppedStoredKeys = cache.state[0]
            eval(croppedStoredKeys)
            let tailAddress = try floatStorageAddress(tail)
            let croppedStoredAddress = try floatStorageAddress(croppedStoredKeys)
            #expect(croppedStoredAddress != tailAddress)
            #expect(croppedStoredKeys.asArray(Float.self) == [105, 106, 107, 108])
            #expect(source.asArray(Float.self) == Array(0..<256).map(Float.init))
        }
    }

    @Test("Chunk attention keeps its full union while stored rows stay bounded")
    func chunkUnion() {
        Device.withDefaultDevice(.cpu) {
            let cache = WindowedKVCache(window: 4)
            _ = cache.update(keys: rows(0..<3), values: rows(0..<3))
            let (working, _) = cache.update(keys: rows(3..<8), values: rows(3..<8))
            eval(working)
            eval(cache)
            #expect(working.asArray(Float.self) == Array(0..<8).map(Float.init))
            #expect(cache.state[0].asArray(Float.self) == [4, 5, 6, 7])
            #expect(cache.offset == 8)
            guard case .scalar(let ropeOffset) = cache.ropeOffset else {
                Issue.record("Expected absolute scalar RoPE offset")
                return
            }
            #expect(ropeOffset == 8)
            let (next, _) = cache.update(keys: rows(8..<9), values: rows(8..<9))
            eval(next)
            #expect(next.asArray(Float.self) == [5, 6, 7, 8])
            #expect(cache.state.allSatisfy { $0.dim(2) <= 4 })
        }
    }

    @Test("Copies and restore points survive append, eviction and branch mutation")
    func independentSnapshots() throws {
        try Device.withDefaultDevice(.cpu) {
            let source = WindowedKVCache(window: 4)
            _ = source.update(keys: rows(0..<9), values: rows(0..<9))
            eval(source)
            let snapshot = source.state
            let metadata = source.metaState
            let branch = try #require(source.copy() as? WindowedKVCache)
            _ = source.update(keys: rows(9..<12), values: rows(9..<12))
            _ = branch.update(keys: rows(90..<92), values: rows(90..<92))
            eval(source)
            eval(branch)
            #expect(snapshot[0].asArray(Float.self) == [5, 6, 7, 8])
            #expect(source.state[0].asArray(Float.self) == [8, 9, 10, 11])
            #expect(branch.state[0].asArray(Float.self) == [7, 8, 90, 91])
            source.state = snapshot
            source.metaState = metadata
            #expect(source.offset == 9)
            let (replayed, _) = source.update(keys: rows(9..<10), values: rows(9..<10))
            #expect(replayed.asArray(Float.self) == [6, 7, 8, 9])
        }
    }

    @Test("Trim refuses a rewind once required history has been evicted")
    func trimSafety() {
        Device.withDefaultDevice(.cpu) {
            let cache = WindowedKVCache(window: 4)
            _ = cache.update(keys: rows(0..<3), values: rows(0..<3))
            #expect(cache.isTrimmable(after: 1))
            #expect(!cache.isTrimmable(after: 2))
            #expect(cache.trim(1) == 1)
            #expect(cache.offset == 2)
            _ = cache.update(keys: rows(2..<7), values: rows(2..<7))
            #expect(!cache.isTrimmable)
            #expect(cache.trim(1) == 0)
            #expect(cache.offset == 7)
        }
    }

    @Test("Disk snapshots preserve absolute offsets and chronological tails", arguments: [0, 13])
    func diskSnapshot(_ count: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let source = WindowedKVCache(window: 4)
            if count > 0 {
                _ = source.update(keys: rows(0..<count), values: rows(0..<count))
                eval(source)
            }
            let file = directory.appendingPathComponent("cache.safetensors")
            try savePromptCache(url: file, cache: [source])
            let (restored, _) = try loadPromptCache(url: file)
            let cache = try #require(restored.first as? WindowedKVCache)
            #expect(cache.offset == count)
            #expect(cache.metaState == source.metaState)
            let expected = source.update(keys: rows(count..<(count + 2)), values: rows(count..<(count + 2)))
            let actual = cache.update(keys: rows(count..<(count + 2)), values: rows(count..<(count + 2)))
            #expect(expected.0.asArray(Float.self) == actual.0.asArray(Float.self))
        }
    }

    @Test("Bounded storage is explicit and disabled by any KV policy")
    func optInAndCompression() throws {
        try Device.withDefaultDevice(.cpu) {
            let standard = try makeModel(bounded: false)
            #expect(try standard.newCache(parameters: nil).allSatisfy { $0 is StandardKVCache })
            let bounded = try makeModel(bounded: true)
            let cache = try bounded.newCache(parameters: nil)
            #expect(cache[0] is WindowedKVCache)
            #expect(cache[1] is StandardKVCache)
            for parameters in [
                GenerateParameters(kvBits: 4), GenerateParameters(kvScheme: "affine4"),
                GenerateParameters(maxKVSize: 128),
            ] {
                #expect(try bounded.newCache(parameters: parameters).allSatisfy { !($0 is WindowedKVCache) })
            }
        }
    }

    @Test(
        "Absolute-position cached logits match retained history across chunk and wrap boundaries",
        arguments: [false, true], [1, 3, 5, 17])
    func logitsParity(_ quantized: Bool, _ chunk: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let reference = try makeModel(bounded: false)
            let bounded = try makeModel(bounded: true)
            try bounded.update(parameters: reference.parameters(), verify: .all)
            if quantized {
                quantize(model: reference, groupSize: 64, bits: 4)
                quantize(model: bounded, groupSize: 64, bits: 4)
            }
            let referenceCache = try reference.newCache(parameters: nil)
            let boundedCache = try bounded.newCache(parameters: nil)
            let tokens = MLXArray(Array(1...33)).reshaped(1, 33)
            for start in stride(from: 0, to: 33, by: chunk) {
                let input = tokens[0..., start..<min(start + chunk, 33)]
                let expected = reference(input, cache: referenceCache)
                let actual = bounded(input, cache: boundedCache)
                eval(expected, actual)
                eval(boundedCache)
                #expect(MLX.max(abs(expected - actual)).item(Float.self) < 0.0001)
                #expect(boundedCache[0].state.allSatisfy { $0.dim(2) <= 4 })
                #expect(boundedCache[0].offset == min(start + chunk, 33))
            }
            let cloned = boundedCache.map { $0.copy() }
            let input = MLXArray([34]).reshaped(1, 1)
            let expected = reference(input, cache: referenceCache)
            let actual = bounded(input, cache: cloned)
            #expect(MLX.max(abs(expected - actual)).item(Float.self) < 0.0001)
            #expect(boundedCache[0].offset == 33)
            #expect(cloned[0].offset == 34)
        }
    }

    private func floatStorageAddress(_ array: MLXArray) throws -> UInt {
        try withExtendedLifetime(array) {
            eval(array)
            let pointer = try #require(mlx_array_data_float32(array.ctx))
            return UInt(bitPattern: UnsafeRawPointer(pointer))
        }
    }

    private func rows(_ range: Range<Int>) -> MLXArray {
        MLXArray(range.map(Float.init)).reshaped(1, 1, range.count, 1)
    }

    private func makeModel(bounded: Bool) throws -> Gemma4TextModel {
        try Gemma4RuntimeTuning.$useBoundedWindowCache.withValue(bounded) {
            let configuration = try JSONDecoder().decode(
                Gemma4TextConfiguration.self,
                from: Data(
                    """
                    {"model_type":"gemma4_text","hidden_size":64,"intermediate_size":128,
                    "num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":1,
                    "head_dim":32,"global_head_dim":32,"sliding_window":4,"vocab_size":128,
                    "layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"],
                    "hidden_size_per_layer_input":0,"num_kv_shared_layers":2}
                    """.utf8))
            return Gemma4TextModel(configuration)
        }
    }
}
