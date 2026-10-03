import Foundation

// JSONEncoder.convertToSnakeCase is the report's wire format. The Id/Sha
// spelling here keeps acronym-bearing keys unambiguous across Foundation builds.
struct GenerationSample: Encodable, Sendable {
    let sequence: Int
    let id: String
    let category: String
    var status = "pending"
    var generatedText = ""
    var promptSha256: String?
    var promptTokenIdFingerprint: String?
    var promptTokenCount: Int?
    var prefilledPromptTokenCount: Int?
    var cachedPromptTokenCount: Int?
    var generationTokenCount: Int?
    var stopReason: String?
    var outputLimitReached: Bool?
    var promptTruncated = false
    var promptTokensPerSecond: Double?
    var tokensPerSecond: Double?
    var timeToFirstContentMilliseconds: Double?
    var promptPreparationMilliseconds: Double?
    var totalMilliseconds: Double?
    var peakMlxMemoryBytes: Int?
    var error: String?
    var failureStage: String?
}

struct GenerationReport: Encodable {
    var format = 1
    var benchmark = "native_greedy_generation"
    var status = "starting"
    var createdAt: String
    var updatedAt: String
    var elapsedSeconds = 0.0
    var modelPath: String
    var adapter: GenerationAdapterReport?
    var modelConfigSha256: String?
    var modelIndexSha256: String?
    var corpusPath: String
    var corpusSha256: String?
    var corpusFingerprint: String?
    var corpusBytes: Int?
    var inputSampleCount = 0
    var completedSampleCount = 0
    var preparedSampleCount = 0
    var outputLimitReachedCount = 0
    var activeSampleId: String?
    var failureStatus: String?
    var error: String?
    var requestedEngine: String
    var engine: String?
    var requestedTokens: Int
    var requestedContextLength: Int?
    var contextLength: Int?
    var prefillStepSize: Int
    var kvCompression: String
    var memoryLimitBytes: Int?
    var modelLoadMilliseconds: Double?
    var temperature = 0.0
    var topP = 1.0
    var promptFormat = "single_user_message_checkpoint_chat_template"
    var promptTokenFingerprintMethod = "ModelQualityCore.tokenIDFingerprint"
    var promptCache = false
    var speculativeDecoding = false
    var fusedGateUpSilu = false
    var compiledLagunaBlockTail: Bool?
    var fusedLagunaRouterTopK: Bool?
    var modelLoadCount = 0
    var invocation: [String]
    var runtimeEnvironment: [String: String]
    var operatingSystem = ProcessInfo.processInfo.operatingSystemVersionString
    var physicalMemoryBytes = ProcessInfo.processInfo.physicalMemory
    var samples = [GenerationSample]()
}

/// Initial exclusive creation refuses existing reports. Each subsequent snapshot
/// replaces the report atomically, preserving all finished samples on interruption.
/// Resuming a partial report is deliberately unsupported: use a new output path.
func writeGenerationReport(
    _ report: inout GenerationReport, to url: URL,
    startedAt: ContinuousClock.Instant, initial: Bool = false
) throws {
    report.updatedAt = ISO8601DateFormatter().string(from: Date())
    report.elapsedSeconds = generationMilliseconds(startedAt.duration(to: ContinuousClock().now)) / 1_000
    report.completedSampleCount = report.samples.filter { $0.status == "completed" }.count
    report.preparedSampleCount = report.samples.filter { $0.promptTokenCount != nil }.count
    report.outputLimitReachedCount = report.samples.filter { $0.outputLimitReached == true }.count
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    var data = try encoder.encode(report)
    data.append(0x0a)
    try data.write(to: url, options: initial ? [.withoutOverwriting] : [.atomic])
}
