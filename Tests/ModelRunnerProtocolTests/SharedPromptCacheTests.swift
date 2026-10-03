import Cmlx
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import ModelRunnerCore

@Suite("Shared prompt checkpoint correctness", .serialized)
struct SharedPromptCacheTests {
    private let memoryLimit = Int.max
    private let budget = 1_048_576

    @Test(
        "Custom prepare tokens and logits retain the same cache as TokenIterator",
        arguments: [PrefixProbeModel.Preparation.tokens, .partial, .logits])
    private func matchesIterator(preparation: PrefixProbeModel.Preparation) throws {
        try Device.withDefaultDevice(.cpu) {
            let tokens = Array(1...600)
            let model = PrefixProbeModel(preparation: preparation)
            let shared = SharedPromptCache(maximumBytes: budget)
            let result = try shared.prepare(
                tokens: tokens, model: model,
                parameters: .init(), memoryLimitBytes: memoryLimit)
            #expect(result.count == 512)
            #expect(result.cachedCount == 0)
            let actual = try #require(result.cache)

            let referenceModel = PrefixProbeModel(preparation: preparation)
            let expected = try referenceModel.newCache(parameters: nil)
            var start = 0
            for end in [128, 256, 512] {
                _ = try TokenIterator(
                    input: LMInput(tokens: MLXArray(Array(tokens[start..<end]))),
                    model: referenceModel, cache: expected, parameters: .init(maxTokens: 1, temperature: 0))
                StreamOrDevice.default.stream.synchronize()
                start = end
            }
            #expect(actual.map(\.offset) == expected.map(\.offset))
            #expect(
                actual.flatMap(\.state).map { $0.asArray(Float.self) }
                    == expected.flatMap(\.state).map { $0.asArray(Float.self) })
            #expect(model.preparedInputs == referenceModel.preparedInputs)
            #expect(model.forwardInputs == referenceModel.forwardInputs)

            // Continue both with the real suffix. A discarded prediction must never
            // become part of the reusable prefix or replace the next prompt token.
            let suffix = LMInput(tokens: MLXArray(Array(tokens.dropFirst(result.count))))
            var resumed = try TokenIterator(
                input: suffix, model: model, cache: actual,
                parameters: .init(maxTokens: 1, temperature: 0))
            var reference = try TokenIterator(
                input: suffix, model: referenceModel, cache: expected,
                parameters: .init(maxTokens: 1, temperature: 0))
            #expect(resumed.next() == reference.next())
            StreamOrDevice.default.stream.synchronize()
            #expect(
                actual.flatMap(\.state).map { $0.asArray(Float.self) }
                    == expected.flatMap(\.state).map { $0.asArray(Float.self) })
        }
    }

    @Test(
        "Auxiliary state from either preparation result prevents reuse",
        arguments: [PrefixProbeModel.Preparation.statefulTokens, .statefulLogits])
    private func rejectsAuxiliaryState(preparation: PrefixProbeModel.Preparation) throws {
        try Device.withDefaultDevice(.cpu) {
            let shared = SharedPromptCache(maximumBytes: budget)
            let result = try shared.prepare(
                tokens: Array(1...300),
                model: PrefixProbeModel(preparation: preparation), parameters: .init(),
                memoryLimitBytes: memoryLimit)
            #expect(result.cache == nil)
            #expect(result.count == 0)
            #expect(result.cachedCount == 0)
            #expect(shared.bytes == 0)
        }
    }

    @Test("Independent copies survive request mutation and longest matching prefix is selected")
    func snapshotsAreIndependent() throws {
        try Device.withDefaultDevice(.cpu) {
            let shared = SharedPromptCache(maximumBytes: budget)
            let model = PrefixProbeModel()
            let tokens = Array(1...600)
            let first = try shared.prepare(
                tokens: tokens, model: model,
                parameters: .init(), memoryLimitBytes: memoryLimit)
            let original = try #require(first.cache)
            let retained = original.flatMap(\.state).map { $0.asArray(Float.self) }
            _ = model(LMInput.Text(tokens: MLXArray([9999], [1, 1])), cache: original, state: nil)
            eval(original)
            let next = try shared.prepare(
                tokens: tokens, model: model,
                parameters: .init(), memoryLimitBytes: memoryLimit)
            #expect(next.cachedCount == 512)
            #expect(try #require(next.cache).flatMap(\.state).map { $0.asArray(Float.self) } == retained)
            let shorter = Array(tokens.prefix(200)) + [777]
            let branch = try shared.prepare(
                tokens: shorter, model: model,
                parameters: .init(), memoryLimitBytes: memoryLimit)
            #expect(branch.cachedCount == 128)
            #expect(branch.count == 128)
        }
    }

    @Test("Byte budget evicts older snapshots without corrupting newer prefixes")
    func eviction() throws {
        try Device.withDefaultDevice(.cpu) {
            // Two 128-position float32 key/value arrays consume 1024 bytes.
            let shared = SharedPromptCache(maximumBytes: 1024)
            let model = PrefixProbeModel()
            let a = Array(repeating: 11, count: 129)
            let b = Array(repeating: 22, count: 129)
            _ = try shared.prepare(tokens: a, model: model, parameters: .init(), memoryLimitBytes: memoryLimit)
            _ = try shared.prepare(tokens: b, model: model, parameters: .init(), memoryLimitBytes: memoryLimit)
            #expect(shared.bytes == 1024)
            let bHit = try shared.prepare(tokens: b, model: model, parameters: .init(), memoryLimitBytes: memoryLimit)
            #expect(bHit.cachedCount == 128)
            let aMiss = try shared.prepare(tokens: a, model: model, parameters: .init(), memoryLimitBytes: memoryLimit)
            #expect(aMiss.cachedCount == 0)
            #expect(shared.bytes == 1024)
            shared.clear()
            #expect(shared.bytes == 0)
        }
    }

    @Test("Entry limit disables retention or evicts incompatible branches")
    func entryLimit() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = PrefixProbeModel()
            let disabled = SharedPromptCache(maximumBytes: budget, maximumEntries: 0)
            _ = try disabled.prepare(
                tokens: Array(1...300), model: model, parameters: .init(), memoryLimitBytes: memoryLimit)
            #expect(model.cacheCreations == 0)
            let shared = SharedPromptCache(maximumBytes: budget, maximumEntries: 1)
            let a = Array(repeating: 11, count: 129)
            let b = Array(repeating: 22, count: 129)
            for tokens in [a, b, a] {
                let result = try shared.prepare(
                    tokens: tokens, model: model, parameters: .init(), memoryLimitBytes: memoryLimit)
                #expect(result.cachedCount == 0)
            }
        }
    }

    @Test("Gemma checkpoints retain absolute positions across sliding windows and token edits")
    func gemmaNormalizedPrefix() throws {
        try Device.withDefaultDevice(.cpu) {
            let configuration = try JSONDecoder().decode(
                Gemma4TextConfiguration.self,
                from: Data(
                    """
                    {"model_type":"gemma4_text","hidden_size":16,"intermediate_size":32,
                    "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
                    "head_dim":8,"global_head_dim":8,"sliding_window":16,"vocab_size":128,
                    "layer_types":["sliding_attention","full_attention"],
                    "hidden_size_per_layer_input":0,"num_kv_shared_layers":0}
                    """.utf8))
            let model = Gemma4TextModel(configuration)
            let shared = SharedPromptCache(maximumBytes: budget)
            // Already normalized BOS + turn prefix. Actual token IDs, including
            // thinking/tool delimiters, are the entire cache key.
            let tokens = [2, 105] + (0..<300).map { 3 + $0 % 100 }
            for expectedHit in [0, 256] {
                let prepared = try shared.prepare(
                    tokens: tokens, model: model, parameters: .init(), memoryLimitBytes: memoryLimit)
                #expect(prepared.cachedCount == expectedHit)
                let cache = try #require(prepared.cache)
                #expect(cache.allSatisfy { $0.offset == 256 })
                let warm = model(MLXArray(Array(tokens.dropFirst(prepared.count))).reshaped(1, -1), cache: cache)
                let cold = model(MLXArray(tokens).reshaped(1, -1), cache: nil)
                let error = MLX.max(abs(warm[0, -1, 0...] - cold[0, -1, 0...])).item(Float.self)
                #expect(error < 0.0001)
            }
            // Editing a rendered delimiter or any earlier token invalidates the
            // checkpoint even if the source message objects happen to match.
            var edited = tokens
            edited[1] = 107
            let branch = try shared.prepare(
                tokens: edited, model: model, parameters: .init(), memoryLimitBytes: memoryLimit)
            #expect(branch.cachedCount == 0)
        }
    }

    @Test("Disabled, empty and short prompts do no model work")
    func noWork() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = PrefixProbeModel()
            for (limit, tokens) in [
                (0, Array(1...200)), (-1, Array(1...200)), (budget, []),
                (budget, Array(1...128)),
            ] {
                let result = try SharedPromptCache(maximumBytes: limit).prepare(
                    tokens: tokens,
                    model: model, parameters: .init(), memoryLimitBytes: memoryLimit)
                #expect(result.cache == nil)
                #expect(result.count == 0)
            }
            #expect(model.preparedInputs.isEmpty)
            #expect(model.cacheCreations == 0)
        }
    }

    @Test("Nondefault cache plans keep iterator evaluation and validation")
    func cachePlanFallback() throws {
        try Device.withDefaultDevice(.cpu) {
            // Exercise explicit configurations and legacy settings that resolve to
            // an otherwise disabled plan. Preparation must retain their validation.
            let overrides: [GenerateParameters] = [
                .init(kvCache: .init(strategy: .fullPrecision)),
                .init(kvGroupSize: 32), .init(quantizedKVStart: 10),
                .init(kvBits: 4, quantizedKVStart: 1024),
                .init(quantizedKVStart: 1024, kvScheme: "affine4"),
                .init(maxKVSize: 256),
            ]
            for parameters in overrides {
                let reader = PrefixLazyTensorReader()
                let model = PrefixProbeModel(preparation: .logits, logitsSource: try reader.load())
                _ = try SharedPromptCache(maximumBytes: budget).prepare(
                    tokens: Array(1...129),
                    model: model, parameters: parameters, memoryLimitBytes: memoryLimit)
                #expect(reader.payloadReads > 0)
            }
            let model = PrefixProbeModel()
            #expect(throws: KVCacheConfigurationError.self) {
                _ = try SharedPromptCache(maximumBytes: budget).prepare(
                    tokens: Array(1...129),
                    model: model, parameters: .init(maxKVSize: -1), memoryLimitBytes: memoryLimit)
            }
            #expect(model.preparedInputs.isEmpty)
        }
    }

    @Test("Token remainders finish progress; logits preparations own their progress")
    func progress() throws {
        try Device.withDefaultDevice(.cpu) {
            for preparation in [PrefixProbeModel.Preparation.partial, .logits] {
                let progress = PrefixProgressRecorder()
                let parameters = GenerateParameters(
                    prefill: .init(progress: { done, total in
                        progress.values.append([done, total])
                    }))
                _ = try SharedPromptCache(maximumBytes: budget).prepare(
                    tokens: Array(1...129),
                    model: PrefixProbeModel(preparation: preparation), parameters: parameters,
                    memoryLimitBytes: memoryLimit)
                #expect(progress.values == (preparation == .partial ? [[64, 128], [128, 128]] : [[128, 128]]))
            }
        }
    }

    @Test("Cancellation before prefill publishes no checkpoint")
    func cancellation() async throws {
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try Device.withDefaultDevice(.cpu) {
                let shared = SharedPromptCache(maximumBytes: budget)
                let model = PrefixProbeModel()
                do {
                    _ = try shared.prepare(
                        tokens: Array(1...300), model: model,
                        parameters: .init(), memoryLimitBytes: memoryLimit)
                    Issue.record("Expected cancellation")
                } catch is CancellationError {
                    #expect(shared.bytes == 0)
                    #expect(model.preparedInputs.isEmpty)
                }
            }
        }
        try await task.value
    }
}

private final class PrefixProgressRecorder: @unchecked Sendable {
    var values: [[Int]] = []
}

private final class PrefixProbeModel: Module, LanguageModel {
    enum Preparation: Sendable { case tokens, partial, logits, statefulTokens, statefulLogits }
    let preparation: Preparation
    let logitsSource: MLXArray?
    var preparedInputs: [[Int]] = []
    var forwardInputs: [[Int]] = []
    var cacheCreations = 0
    var lastLogits: MLXArray?

    init(preparation: Preparation = .tokens, logitsSource: MLXArray? = nil) {
        self.preparation = preparation
        self.logitsSource = logitsSource
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        cacheCreations += 1
        if let limit = parameters?.maxKVSize, limit > 0 {
            return [RotatingKVCache(maxSize: limit, keep: 4)]
        }
        return [KVCacheSimple()]
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        preparedInputs.append(input.text.tokens.asArray(Int.self))
        switch preparation {
        case .tokens, .statefulTokens:
            return .tokens(input.text)
        case .partial:
            let midpoint = input.text.tokens.size / 2
            _ = self(input.text[.newAxis, ..<midpoint], cache: cache, state: state)
            prefill.progress?(midpoint, input.text.tokens.size)
            return .tokens(input.text[midpoint...])
        case .logits, .statefulLogits:
            let output = self(input.text[text: .newAxis], cache: cache, state: state)
            prefill.progress?(input.text.tokens.size, input.text.tokens.size)
            return .logits(output)
        }
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        forwardInputs.append(input.tokens.asArray(Int.self))
        let values = input.tokens.asType(.float32).reshaped([1, 1, -1, 1])
        let retained = cache![0].update(keys: values, values: values).0
        // The optional counted tensor lets cache-plan tests observe evaluation
        // without timers or large allocations.
        let logits = sin(
            retained.sum().reshaped([1, 1, 1])
                + (logitsSource ?? MLXArray(0..<17).asType(.float32)))
        lastLogits = logits
        return LMOutput(
            logits: logits,
            state: preparation == .statefulTokens || preparation == .statefulLogits ? LMOutput.State() : nil)
    }
}

/// A valid in-memory NumPy tensor whose header is read eagerly and whose values
/// are read only when MLX evaluates the lazy load. Counts actual work without
/// invalid operations, timers, allocator estimates, or large tensor allocations.
private final class PrefixLazyTensorReader: @unchecked Sendable {
    let data: Data
    let payloadOffset: Int
    var offset = 0
    var payloadReads = 0

    init() {
        var header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1, 17), }"
        header += String(repeating: " ", count: (64 - (10 + header.utf8.count + 1) % 64) % 64) + "\n"
        let length = header.utf8.count
        payloadOffset = 10 + length
        var bytes: [UInt8] = [0x93, 78, 85, 77, 80, 89, 1, 0, UInt8(length & 255), UInt8(length >> 8)]
        bytes += Array(header.utf8)
        for value in 0..<17 {
            let bits = Float(value).bitPattern
            bytes += (0..<4).map { UInt8((bits >> ($0 * 8)) & 255) }
        }
        data = Data(bytes)
    }

    func read(into destination: UnsafeMutablePointer<CChar>?, count: Int, at position: Int) {
        precondition(position >= 0 && position + count <= data.count)
        if count > 0 && position + count > payloadOffset {
            payloadReads += 1
        }
        if let destination, count > 0 {
            data.withUnsafeBytes { source in
                _ = memcpy(destination, source.baseAddress!.advanced(by: position), count)
            }
        }
    }

    func load() throws -> MLXArray {
        let vtable = mlx_io_vtable {
            $0 != nil
        } good: { _ in
            true
        } tell: { pointer in
            Unmanaged<PrefixLazyTensorReader>.fromOpaque(pointer!).takeUnretainedValue().offset
        } seek: { pointer, delta, whence in
            let reader = Unmanaged<PrefixLazyTensorReader>.fromOpaque(pointer!).takeUnretainedValue()
            switch whence {
            case SEEK_SET: reader.offset = Int(delta)
            case SEEK_CUR: reader.offset += Int(delta)
            case SEEK_END: reader.offset = reader.data.count + Int(delta)
            default: preconditionFailure("Unexpected seek origin")
            }
        } read: { pointer, destination, count in
            let reader = Unmanaged<PrefixLazyTensorReader>.fromOpaque(pointer!).takeUnretainedValue()
            reader.read(into: destination, count: count, at: reader.offset)
            reader.offset += count
        } read_at_offset: { pointer, destination, count, offset in
            Unmanaged<PrefixLazyTensorReader>.fromOpaque(pointer!).takeUnretainedValue()
                .read(into: destination, count: count, at: offset)
        } write: { _, _, _ in
            preconditionFailure("Read-only tensor fixture")
        } label: { _ in
            let label: StaticString = "prefix lazy tensor\0"
            return UnsafeRawPointer(label.utf8Start).assumingMemoryBound(to: CChar.self)
        } free: { pointer in
            Unmanaged<PrefixLazyTensorReader>.fromOpaque(pointer!).release()
        }
        let reader = mlx_io_reader_new(Unmanaged.passRetained(self).toOpaque(), vtable)
        defer { mlx_io_reader_free(reader) }
        var array = mlx_array_new()
        let status = mlx_load_reader(&array, reader, StreamOrDevice.cpu.ctx)
        precondition(status == 0)
        return MLXArray(array)
    }
}
