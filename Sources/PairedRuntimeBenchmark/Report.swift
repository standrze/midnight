import Foundation
import ModelRunnerCore

struct PairedRuntimeReport: Encodable {
    let format = 1
    var status = "running"
    let createdAt = ISO8601DateFormatter().string(from: Date())
    let experiment = "two_resident_gemma3_270m_models"
    let requestedTokens = 512
    let warmupsPerArm = 3
    let maximumResidentModels = 2
    let temperature = 0
    let topP = 1
    let enablePromptCache = false
    let enableSpeculativeDecoding = false
    let engine = "metal"
    let pairs: Int
    let contextLength: Int
    let prefillStepSize = 512
    let prompt: String
    let promptUTF8SHA256: String
    let runtimeEnvironment: [String: String]
    let combinedStoredWeightBytes: Int
    let combinedStoredWeightCeilingBytes: Int
    let processMemoryCeilingBytes: Int
    let allocatorMemoryLimitBytes: Int
    let estimatedMaximumRequestBytesBeforeLoad: Int
    let models: [CheckpointRecord]
    var loads: [LoadRecord] = []
    var renderedPromptTokens: [Int] = []
    var combinedLoadedActiveMemoryBytes: Int?
    var estimatedRequestBytesAfterLoad: Int?
    var trials: [TrialRecord] = []
    var error: String?
    let limitations = [
        "Raw trials only; no acceptance or speedup decision is made by this executable.",
        "Both models stay resident; each arm uses a fresh KV cache and no assistant or prompt reuse.",
        "Each reset also clears the shared MLX allocator cache, identically for both arms.",
        "Load timing includes any default wired-memory tuning; OS file caches are not flushed.",
        "Three explicit generation warmups per arm follow loading and its optional internal tuning.",
        "Peak memory is process-wide MLX active allocation, including both models; it is not RSS or per-model memory.",
        "TTFT measures the first nonempty content event, not an unavailable raw token timestamp; visible output also counts reasoning.",
        "Natural greedy trajectories may differ between quantizations; requested length is 512, while EOS is preserved and flagged.",
        "Identical rendered prompt IDs are required. Generated token IDs are unavailable through this public runtime API.",
        "Checkpoint metadata hashes and stored shard sizes do not replace the campaign's full weight provenance hashes.",
    ]
}

struct LoadRecord: Encodable {
    let arm: String
    let order: Int
    let milliseconds: Double
    let implementation: String
    let activeMemoryBeforeBytes: Int
    let activeMemoryAfterBytes: Int
    let processPeakActiveMemoryBytes: Int
}

struct TrialRecord: Encodable {
    let sequence: Int
    let phase: String
    let pair: Int
    let order: String
    let positionInPair: Int
    let arm: String
    let startedAt: String
    let startElapsedMilliseconds: Double
    let thermalStateBefore: Int?
    var thermalStateAfter: Int?
    var promptTokenCount: Int?
    var promptTokenIDFingerprint: String?
    var metrics: TrialMetrics?
    var timeToFirstTokenMilliseconds: Double?
    var timeToFirstVisibleOutputMilliseconds: Double?
    var totalMilliseconds: Double?
    var peakActiveMemoryBytes: Int?
    var activeMemoryAfterBytes: Int?
    var content = ""
    var reasoning = ""
    var toolCallCount = 0
    var validationIssues: [String] = []
    var error: String?
}

struct TrialMetrics: Encodable {
    let promptTokenCount: Int
    let prefilledPromptTokenCount: Int
    let cachedPromptTokenCount: Int
    let generationTokenCount: Int
    let promptTokensPerSecond: Double?
    let tokensPerSecond: Double?
    let stopReason: String
    let proposedDraftTokens: Int?
    let acceptedDraftTokens: Int?
    let speculativePassthroughReason: String?

    init(_ metrics: LocalModelRunnerMetrics) {
        promptTokenCount = metrics.promptTokenCount
        prefilledPromptTokenCount = metrics.prefilledPromptTokenCount
        cachedPromptTokenCount = metrics.cachedPromptTokenCount
        generationTokenCount = metrics.generationTokenCount
        promptTokensPerSecond = metrics.promptTokensPerSecond.isFinite ? metrics.promptTokensPerSecond : nil
        tokensPerSecond = metrics.tokensPerSecond.isFinite ? metrics.tokensPerSecond : nil
        stopReason = metrics.stopReason
        proposedDraftTokens = metrics.proposedDraftTokens
        acceptedDraftTokens = metrics.acceptedDraftTokens
        speculativePassthroughReason = metrics.speculativePassthroughReason
    }
}

func elapsedMilliseconds(_ duration: Duration) -> Double {
    let value = duration.components
    return Double(value.seconds) * 1_000 + Double(value.attoseconds) / 1_000_000_000_000_000
}

func writeReport(_ report: PairedRuntimeReport, to url: URL, initial: Bool = false) throws {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    encoder.dateEncodingStrategy = .iso8601
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(report).write(to: url, options: initial ? .withoutOverwriting : .atomic)
}
