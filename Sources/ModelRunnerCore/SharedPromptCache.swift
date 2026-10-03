import MLX
import MLXLMCommon

/// Owned by the runner execution gate, including all runtime closures. Exact
/// token keys are local to one loaded model. Every request gets independent
/// cache arrays; generated answers are never retained here.
final class SharedPromptCache: @unchecked Sendable {
    private struct Entry {
        let tokens: [Int]
        let cache: [KVCache]
        let bytes: Int
        var access: UInt64
    }
    private var entries: [Entry] = []
    private var clock: UInt64 = 0
    private(set) var bytes = 0
    let maximumBytes: Int
    let maximumEntries: Int
    init(maximumBytes: Int, maximumEntries: Int = 4) {
        self.maximumBytes = max(0, maximumBytes)
        self.maximumEntries = max(0, maximumEntries)
    }
    func clear() {
        entries = []
        bytes = 0
    }

    static func prefixLength(_ tokens: [Int]) -> Int {
        // Geometric checkpoints retain a short shared system prefix alongside a
        // longer reusable context. Leave at least one token for request prefill.
        [128, 256, 512, 1024].last(where: { $0 < tokens.count }) ?? 0
    }

    func prepare(
        tokens incoming: [Int], model: any LanguageModel,
        parameters: GenerateParameters, memoryLimitBytes: Int
    ) throws -> (cache: [KVCache]?, count: Int, cachedCount: Int) {
        let target = Self.prefixLength(incoming)
        guard maximumBytes > 0, maximumEntries > 0, target > 0 else {
            return (nil, 0, 0)
        }
        let match = entries.indices.filter { incoming.starts(with: entries[$0].tokens) }
            .max { entries[$0].tokens.count < entries[$1].tokens.count }
        var count = 0
        let working: [KVCache]
        if let match,
            entries[match].bytes <= max(0, memoryLimitBytes - Memory.activeMemory - 512 * 1_048_576)
        {
            clock &+= 1
            entries[match].access = clock
            count = entries[match].tokens.count
            working = entries[match].cache.map { $0.copy() }
        } else {
            working = try model.newCache(parameters: parameters)
        }
        let cachedCount = count
        var prefill = parameters
        prefill.temperature = 0  // Argmax consumes no sampling RNG.
        prefill.maxTokens = 1
        for boundary in [128, 256, 512, 1024] where boundary > count && boundary <= target {
            try Task.checkCancellation()
            // The unused greedy prediction realizes exactly this prefix. The next
            // iterator receives actual input tokens, never the predicted token.
            let iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray(Array(incoming[count..<boundary]))),
                model: model, cache: working, parameters: prefill)
            StreamOrDevice.default.stream.synchronize()
            // Auxiliary model position state cannot be dropped or shared blindly.
            guard iterator.state == nil else {
                return (nil, 0, 0)
            }
            count = boundary
            let cost = working.flatMap(\.state).reduce(0) { $0 + $1.nbytes }
            guard cost > 0, cost <= maximumBytes else {
                continue
            }
            while !entries.isEmpty && (entries.count >= maximumEntries || bytes > maximumBytes - cost) {
                let oldest = entries.indices.min { entries[$0].access < entries[$1].access }!
                bytes -= entries.remove(at: oldest).bytes
            }
            guard cost <= max(0, memoryLimitBytes - Memory.activeMemory - 512 * 1_048_576) else {
                continue
            }
            let snapshot = working.map { $0.copy() }
            eval(snapshot.flatMap(\.state))
            clock &+= 1
            let snapshotBytes = snapshot.flatMap(\.state).reduce(0) { $0 + $1.nbytes }
            guard snapshotBytes <= maximumBytes - bytes else {
                continue
            }
            entries.append(
                Entry(
                    tokens: Array(incoming.prefix(count)), cache: snapshot,
                    bytes: snapshotBytes, access: clock))
            bytes += snapshotBytes
        }
        return (working, count, cachedCount)
    }
}
