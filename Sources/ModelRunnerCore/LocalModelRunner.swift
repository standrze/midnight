import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import ModelRunnerProtocol
import Synchronization
import Tokenizers

/// Prepared by this runner; reused by the HTTP path to avoid tokenizing twice.
public struct PreparedModelPrompt: Sendable {
    fileprivate var nativeProtocol: String? = nil
    fileprivate let modelPath: String
    fileprivate let messages: [OpenAIMessage]
    fileprivate let tools: [OpenAIToolDefinition]?
    fileprivate let toolChoice: ToolChoicePlan.Constraint
    fileprivate let thinkingEnabled: Bool?
    fileprivate let reasoningEffort: ChatCompletionRequest.ReasoningEffort?
    fileprivate let forcedToolPrefixTokenIDs: [[Int]]
    fileprivate let tokenIDs: [Int]
    fileprivate let responseFormat: OpenAIResponseFormat?
    fileprivate let structuredOutput: StructuredOutputLogitProcessor?
    /// Number of token IDs in the rendered prompt.
    public var promptTokenCount: Int { tokenIDs.count }
    /// Exact rendered input, exposed read-only for reproducible benchmark comparisons.
    public var promptTokenIDs: [Int] { tokenIDs }
}

/// Text, reasoning, tool-call, or metrics output from a local generation.
public enum LocalModelRunnerEvent: Equatable, Sendable {
    case content(String)
    case reasoning(String)
    case toolCall(OpenAIToolCall)
    case metrics(LocalModelRunnerMetrics)
}

/// Token counts, throughput, stop reason, and optional draft acceptance metrics.
public struct LocalModelRunnerMetrics: Codable, Equatable, Sendable {
    /// Total token IDs in the rendered prompt, including tokens reused from cache.
    public let promptTokenCount: Int
    /// Prompt token IDs evaluated for this request after any prefix-cache reuse.
    public let prefilledPromptTokenCount: Int
    /// Prompt token IDs served from a reusable prefix cache.
    public let cachedPromptTokenCount: Int
    /// Token IDs emitted by the model for this request.
    public let generationTokenCount: Int
    /// Evaluated prompt tokens per second, excluding cached prompt tokens.
    public let promptTokensPerSecond: Double
    /// Emitted tokens per second during generation.
    public let tokensPerSecond: Double
    public let stopReason: String
    /// Draft token IDs proposed by speculative decoding, when enabled.
    public let proposedDraftTokens: Int?
    /// Proposed draft token IDs accepted by the target model, when enabled.
    public let acceptedDraftTokens: Int?
    public let speculativePassthroughReason: String?

    /// Records token counts, throughput, stop reason, and optional draft metrics.
    public init(
        promptTokenCount: Int,
        prefilledPromptTokenCount: Int? = nil,
        cachedPromptTokenCount: Int = 0,
        generationTokenCount: Int,
        promptTokensPerSecond: Double,
        tokensPerSecond: Double,
        stopReason: String,
        proposedDraftTokens: Int? = nil,
        acceptedDraftTokens: Int? = nil,
        speculativePassthroughReason: String? = nil
    ) {
        self.promptTokenCount = promptTokenCount
        self.prefilledPromptTokenCount = prefilledPromptTokenCount ?? promptTokenCount
        self.cachedPromptTokenCount = cachedPromptTokenCount
        self.generationTokenCount = generationTokenCount
        self.promptTokensPerSecond = promptTokensPerSecond
        self.tokensPerSecond = tokensPerSecond
        self.stopReason = stopReason
        self.proposedDraftTokens = proposedDraftTokens
        self.acceptedDraftTokens = acceptedDraftTokens
        self.speculativePassthroughReason = speculativePassthroughReason
    }
}

/// Validated request-time tool policy. `tools` is the exact allow-list supplied
/// to both the chat template and MLX's output parser.
public struct ToolChoicePlan: Equatable, Sendable {
    /// The validated requirement applied to model-generated tool calls.
    public enum Constraint: Equatable, Sendable {
        case automatic
        case prohibited
        case required
        case named(String)

        var requiresToolCall: Bool {
            switch self {
            case .required, .named: true
            case .automatic, .prohibited: false
            }
        }
    }

    public let tools: [OpenAIToolDefinition]?
    public let constraint: Constraint

    /// Validates tool choice against declared functions and returns parser authorization.
    public static func resolve(
        choice: OpenAIToolChoice?,
        tools: [OpenAIToolDefinition]?
    ) throws -> Self {
        switch choice {
        case nil, .some(.auto):
            // At the HTTP boundary, no declarations means no authorization. The
            // underlying standard MLX parsers use nil as an unrestricted mode.
            return Self(tools: tools?.isEmpty == false ? tools : [], constraint: .automatic)
        case .some(.none):
            // Preserve an explicit empty list: MLX treats nil as unrestricted parser
            // authorization, while [] authorizes no generated function names.
            return Self(tools: [], constraint: .prohibited)
        case .some(.required):
            guard tools?.isEmpty == false else {
                throw ToolChoiceValidationError.requiredWithoutTools
            }
            return Self(tools: tools, constraint: .required)
        case .some(.function(let name)):
            guard isValidFunctionName(name) else {
                throw ToolChoiceValidationError.invalidFunctionName(name)
            }
            guard
                let selected = tools?.first(where: {
                    $0.type == "function"
                        && $0.function.name.trimmingCharacters(in: .whitespacesAndNewlines) == name
                })
            else {
                throw ToolChoiceValidationError.unknownFunction(name)
            }
            return Self(tools: [selected], constraint: .named(name))
        }
    }

    private static func isValidFunctionName(_ name: String) -> Bool {
        guard !name.isEmpty, name.utf8.count <= 64 else {
            return false
        }
        return name.unicodeScalars.allSatisfy { scalar in
            switch scalar.value {
            case 45, 48...57, 65...90, 95, 97...122: true
            default: false
            }
        }
    }

    func forcedToolCallPrefixes(for format: ToolCallFormat) throws -> [String] {
        guard constraint.requiresToolCall else {
            return []
        }
        guard format == .atem || format == .glm4 else {
            return try forcedToolCallPrefix(for: format).map { [$0] } ?? []
        }
        let names: [String]
        switch constraint {
        case .named(let name): names = [name]
        case .required: names = tools?.map(\.function.name) ?? []
        case .automatic, .prohibited: return []
        }
        return try names.map { name in
            guard Self.isValidFunctionName(name) else {
                throw ToolChoiceValidationError.invalidFunctionName(name)
            }
            if format == .glm4 {
                return "<tool_call>\(name)"
            }
            // The prompt already contains <|start|>assistant. Constrain the recipient
            // and matching invocation together, then let the model supply arguments.
            return " to=\(name)<|message|><atem:function_calls>\n<atem:invoke name=\"\(name)\">\n"
        }
    }

    func forcedToolCallPrefix(for format: ToolCallFormat) throws -> String? {
        guard constraint.requiresToolCall else {
            return nil
        }
        guard format == .glm4 else {
            throw LocalModelRunnerError.unsupportedForcedToolChoiceFormat(format.rawValue)
        }
        switch constraint {
        case .required: return "<tool_call>"
        case .named(let name): return "<tool_call>\(name)"
        case .automatic, .prohibited: return nil
        }
    }
}

/// A requested tool policy has no usable declared function or an invalid name.
public enum ToolChoiceValidationError: LocalizedError, Equatable, Sendable {
    case requiredWithoutTools
    case invalidFunctionName(String)
    case unknownFunction(String)

    /// User-facing explanation for this error.
    public var errorDescription: String? {
        switch self {
        case .requiredWithoutTools:
            "tool_choice 'required' requires at least one declared tool."
        case .invalidFunctionName(let name):
            "Named tool_choice requires a 1-64 character function name containing only letters, digits, '_' or '-', not '\(name)'."
        case .unknownFunction(let name):
            "Named tool_choice references undeclared function '\(name)'."
        }
    }
}

/// Stateful completion check kept separate from MLX parsing so the public API
/// stays fail-closed even if a backend parser regresses.
final class ToolChoiceOutputValidator: @unchecked Sendable {
    private let constraint: ToolChoicePlan.Constraint
    private let allowedToolNames: Set<String>
    private let lock = NSLock()
    private var toolCallCount = 0

    init(constraint: ToolChoicePlan.Constraint, tools: [OpenAIToolDefinition]?) {
        self.constraint = constraint
        self.allowedToolNames = Set(
            tools?.compactMap { definition in
                guard definition.type == "function" else {
                    return nil
                }
                return definition.function.name.trimmingCharacters(in: .whitespacesAndNewlines)
            } ?? []
        )
    }

    func observe(_ event: LocalModelRunnerEvent) throws {
        guard case .toolCall(let call) = event else {
            return
        }
        try lock.withLock {
            switch constraint {
            case .automatic:
                break
            case .prohibited:
                throw LocalModelRunnerError.prohibitedToolCall(call.function.name)
            case .required:
                guard allowedToolNames.contains(call.function.name) else {
                    throw LocalModelRunnerError.undeclaredToolCall(call.function.name)
                }
            case .named(let expected):
                guard call.function.name == expected else {
                    throw LocalModelRunnerError.wrongNamedToolCall(expected: expected, actual: call.function.name)
                }
            }
            toolCallCount += 1
        }
    }

    func validateCompletion() throws {
        try lock.withLock {
            guard !constraint.requiresToolCall || toolCallCount > 0 else {
                throw LocalModelRunnerError.requiredToolCallMissing
            }
        }
    }
}

struct LagunaDecodeFastPathSelection: Equatable {
    var useCompiledBlockTail: Bool
    var useFusedRouterTopK: Bool

    static func resolve(
        engine: ModelEngine,
        compiledBlockTailOverride: Bool?,
        fusedRouterTopKOverride: Bool?
    ) -> Self {
        let metalDefault = engine == .metal
        return Self(
            useCompiledBlockTail: compiledBlockTailOverride ?? metalDefault,
            useFusedRouterTopK: fusedRouterTopKOverride ?? metalDefault)
    }
}

private final class SpeculativeModelReference: @unchecked Sendable {
    let model: any MTPDrafterModel

    init(_ model: any MTPDrafterModel) {
        self.model = model
    }
}

private struct LoadedSpeculativeModel: Sendable {
    // Retain the factory-owned context for the lifetime of the shared model.
    let container: MTPDrafterContainer
    let model: SpeculativeModelReference
    let blockSize: Int
    let modelPath: String
    var quantizedModuleCount: Int = 0
}

/// `ChatSession` is deliberately single-consumer rather than `Sendable`.
/// LocalModelRunner's actor and execution admission queue provide that serialization;
/// this reference keeps the assertion at one explicit boundary for Swift 6.
private final class ChatSessionReference: @unchecked Sendable {
    let session: ChatSession

    init(_ session: ChatSession) {
        self.session = session
    }

    func configure(
        parameters: GenerateParameters,
        components: GenerationComponents,
        additionalContext: [String: any Sendable]?,
        tools: [ToolSpec]?
    ) {
        session.generateParameters = parameters
        session.components = components
        session.additionalContext = additionalContext
        session.tools = tools
    }

    func streamDetails(
        to messages: consuming [Chat.Message]
    ) -> AsyncThrowingStream<Generation, Error> {
        session.streamDetails(to: messages)
    }

    func synchronize() async {
        await session.synchronize()
    }

    func cacheMemoryBytes() async -> Int { await session.cacheMemoryBytes() }

    func processedTokenCount() async throws -> Int? {
        try await session.cacheStatus().processedTokenCount
    }

    func snapshot() async throws -> ChatSessionSnapshotReference {
        ChatSessionSnapshotReference(try await session.snapshot())
    }
}

private final class ChatSessionSnapshotReference: @unchecked Sendable {
    private let snapshot: ChatSessionSnapshot
    let estimatedBytes: Int

    init(_ snapshot: consuming ChatSessionSnapshot) {
        self.estimatedBytes = snapshot.estimatedBytes
        self.snapshot = snapshot
    }

    func restore(
        container: ModelContainer,
        parameters: GenerateParameters,
        components: GenerationComponents,
        additionalContext: [String: any Sendable]?,
        tools: [ToolSpec]?
    ) -> ChatSession {
        ChatSession(
            container,
            restoring: snapshot,
            generateParameters: parameters,
            components: components,
            additionalContext: additionalContext,
            tools: tools
        )
    }
}

private struct HotConversation: Sendable {
    let session: ChatSessionReference
    let committedMessages: [OpenAIMessage]
    let tools: [OpenAIToolDefinition]?
    let reasoningEffort: ChatCompletionRequest.ReasoningEffort?
}

struct ConversationPrefixCacheLimits: Equatable, Sendable {
    static let entriesEnvironmentKey = "MODEL_RUNNER_PREFIX_CACHE_ENTRIES"
    static let memoryEnvironmentKey = "MODEL_RUNNER_PREFIX_CACHE_MIB"
    static let defaultMaximumEntries = 4
    static let defaultMaximumBytes = 2 * 1_024 * 1_024 * 1_024

    let maximumEntries: Int
    let maximumBytes: Int

    static func resolve(environment: [String: String]) -> Self {
        let entries = boundedInteger(
            environment[entriesEnvironmentKey],
            defaultValue: defaultMaximumEntries,
            range: 0...64
        )
        let memoryMiB = boundedInteger(
            environment[memoryEnvironmentKey],
            defaultValue: defaultMaximumBytes / 1_048_576,
            range: 0...16_384
        )
        return Self(maximumEntries: entries, maximumBytes: memoryMiB * 1_048_576)
    }

    private static func boundedInteger(
        _ rawValue: String?,
        defaultValue: Int,
        range: ClosedRange<Int>
    ) -> Int {
        guard let rawValue, let value = Int(rawValue), range.contains(value) else {
            return defaultValue
        }
        return value
    }
}

#if os(macOS) && MODEL_RUNNER_PINNED_MLX
    /// Creates the ordinary MLX default stream on the permanent MLX pthread.
    ///
    /// Unlike mlx-swift's cross-thread-safe static stream, this stream is backed by
    /// MLX's thread-local command encoder. It is safe to retain because every use is
    /// scoped by `withPinnedMLXRuntime` on the same process-lifetime worker.
    private func makePinnedMLXStream(device: Device) async throws -> MLX.Stream {
        try await MLXPinnedRuntime.shared.run {
            Device.withDefaultDevice(device) {
                MLX.Stream()
            }
        }
    }

    /// Runs one complete MLX operation with the worker's executor, device, and
    /// persistent thread-local stream inherited by MLX-LM's producer tasks.
    private func withPinnedMLXRuntime<Result: Sendable>(
        device: Device,
        stream: MLX.Stream,
        operation: @escaping @Sendable () async throws -> Result
    ) async throws -> Result {
        let runtime = MLXPinnedRuntime.shared
        return try await runtime.run {
            try await Device.withDefaultDevice(device) {
                try await MLXTaskExecutorPreference.$current.withValue(runtime.taskExecutor) {
                    try await MLX.Stream.withDefaultStream(stream) {
                        try await operation()
                    }
                }
            }
        }
    }
#endif

/// Owns a loaded text model and serializes its mutable generation state.
public actor LocalModelRunner {
    private nonisolated let generationProgress = GenerationProgressMonitor()
    private nonisolated let reportedGeneration = Mutex<LocalModelRunnerMetrics?>(nil)

    /// Latest backend-reported completion metrics for this loaded model; nil before its first report.
    /// Does not enable per-token monitoring or wait for model execution.
    public nonisolated var latestGenerationMetrics: LocalModelRunnerMetrics? {
        reportedGeneration.withLock { $0 }
    }

    /// Enables passive token measurements for the local dashboard; off by default.
    public nonisolated func setGenerationMonitoringEnabled(_ enabled: Bool) {
        generationProgress.setEnabled(enabled)
    }

    /// Current or most recent text generation, readable without waiting for the decoding actor.
    public nonisolated var generationProgressSnapshot: GenerationProgressSnapshot? {
        generationProgress.snapshot()
    }

    /// Operations waiting for the shared model execution slot.
    public nonisolated var queuedGenerationCount: Int { generationAdmission.queuedCount }

    #if os(macOS) && MODEL_RUNNER_PINNED_MLX
        /// Keep actor-isolated orchestration on the same pthread as MLX evaluation.
        public nonisolated var unownedExecutor: UnownedSerialExecutor {
            MLXPinnedRuntime.shared.unownedSerialExecutor
        }
    #endif

    public nonisolated let modelPath: String
    public nonisolated let servedModelName: String
    public nonisolated let modelImplementation: String
    public nonisolated let engine: ModelEngine
    public nonisolated let dflashModelPath: String?
    public nonisolated let dflashBlockSize: Int?
    public nonisolated let gemmaAssistantModelPath: String?
    public nonisolated let gemmaAssistantBlockSize: Int?
    public nonisolated let gemmaAssistantQuantizationBits: Int?
    public nonisolated let gemmaAssistantQuantizedModuleCount: Int?
    public nonisolated let supportsMistralHotConversationCache: Bool
    /// Supports the shared hot-session path without copying branch snapshots.
    public nonisolated let supportsHotConversationCache: Bool

    public nonisolated let decisionContract: DecisionModelContract?

    public nonisolated var supportsDecisions: Bool { decisionContract != nil }

    public nonisolated let contextLength: Int
    public nonisolated let prefillStepSize: Int
    public nonisolated let kvCompression: String
    public nonisolated let memoryLimitBytes: Int
    private let longContext: LongContextOptions
    private let memoryProfile: ModelMemoryProfile
    private let residentModelBytes: Int
    private var hotCacheBytes = 0
    private let container: ModelContainer
    private let tokenLimit: GenerationTokenLimit
    private let device: Device
    private let normalizesGemma4Prompt: Bool
    private let runtimeCapabilities: ModelRuntimeCapabilities
    private let supportsLagunaPromptCache: Bool
    private let dflash: LoadedSpeculativeModel?
    private var wiredMemoryPlan: MLXWiredMemoryPlan?
    private let cudaWeightPlacement: CUDAWeightPlacement?
    private let cudaDeviceBudgetBytes: Int?
    private var benchmarkWiredMemoryRequests: [BenchmarkWiredMemoryRequest]?
    #if os(macOS) && MODEL_RUNNER_PINNED_MLX
        private nonisolated let mlxStream: MLX.Stream
    #endif
    private var conversationCache: CompletedMessagePrefixLRU<ChatSessionSnapshotReference>
    private var hotConversation: HotConversation?
    private let generationAdmission = GenerationAdmission()
    private let sharedPromptCache: SharedPromptCache
    private let producerLifetime = StreamProducerLifetime()
    private var pendingInspectorRecording: PendingInspectorRecording?

    /// Loads a model and optional adapters under the selected engine and resource limits.
    public init(
        modelPath: String,
        servedModelName: String? = nil,
        engine requestedEngine: ModelEngine = .auto,
        maximumTokens: Int = 4096,
        defaultMaximumTokens: Int? = nil,
        adapterPath: String? = nil,
        adapterScale: Float? = nil,
        dflashModelPath: String? = nil,
        dflashBlockSize: Int? = nil,
        gemmaAssistantModelPath: String? = nil,
        gemmaAssistantBlockSize: Int? = nil,
        gemmaAssistantQuantizationBits: Int? = nil,
        longContext: LongContextOptions = try! LongContextOptions()
    ) async throws {
        let tokenLimit = try GenerationTokenLimit(configuredMaximum: maximumTokens, defaultTokens: defaultMaximumTokens)
        let environment = ProcessInfo.processInfo.environment

        let engine = try requestedEngine.resolve()
        let device: Device = engine == .cpu ? .cpu : .gpu
        let gpuResourceLimits = try MLXResourceGuard.resolve(
            for: engine,
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            recommendedWorkingSetBytes: engine == .metal ? GPU.maxRecommendedWorkingSetBytes() : nil
        )
        let offloadMode = try CUDALayerOffloadMode(environment: environment, engine: engine)
        let usesCUDAOffload = offloadMode != .disabled
        if usesCUDAOffload {
            guard try CUDAMemoryRuntime.snapshot().supportsLayerOffload else {
                throw RequestAdmissionError.configuration(
                    "CUDA RAM offload requires concurrent managed memory access on every visible GPU")
            }
            guard adapterPath == nil, dflashModelPath == nil, gemmaAssistantModelPath == nil else {
                throw RequestAdmissionError.configuration(
                    "CUDA RAM offload currently supports target-only text models without adapters")
            }
        }
        let hostBudget = usesCUDAOffload ? try CUDAHostMemory.resolve(environment: environment) : 0
        let resourceLimits = try gpuResourceLimits.addingHostAllowance(hostBudget)

        let expandedPath = NSString(string: modelPath).expandingTildeInPath
        let modelURL = URL(fileURLWithPath: expandedPath, isDirectory: true)
            .standardizedFileURL
            .resolvingSymlinksInPath()
        try Self.validateModelFolder(modelURL)
        if ProcessInfo.processInfo.environment["MIDNIGHT_GEMMA4_EXPERT_GATE_UP"] == "1" {
            try GemmaExpertProjectionCompatibility.validate(
                configuration: Data(contentsOf: modelURL.appendingPathComponent("config.json")),
                engine: engine, hasAdapter: adapterPath != nil)
        }
        let normalizesGemma4Prompt = try Self.isGemma4Model(modelURL)
        let runtimeCapabilities = try ModelRuntimeCapabilities.load(from: modelURL)
        let useBoundedGemmaKV =
            environment["MIDNIGHT_GEMMA4_BOUNDED_KV"] == "1"
            && engine == .metal && gemmaAssistantModelPath == nil && dflashModelPath == nil
            && longContext.compression == .none
        let memoryProfile = try ModelMemoryProfile(
            configuration: Data(contentsOf: modelURL.appendingPathComponent("config.json")), options: longContext,
            boundedGemmaSlidingCache: useBoundedGemmaKV)
        guard dflashModelPath == nil || gemmaAssistantModelPath == nil else {
            throw RequestAdmissionError.configuration("Select only one speculative assistant")
        }
        if (gemmaAssistantBlockSize != nil || gemmaAssistantQuantizationBits != nil) && gemmaAssistantModelPath == nil {
            throw RequestAdmissionError.configuration("Gemma assistant options require an assistant checkpoint")
        }
        if let bits = gemmaAssistantQuantizationBits, ![4, 8].contains(bits) {
            throw RequestAdmissionError.configuration("Experimental Gemma assistant quantization must be 4 or 8 bits")
        }
        if longContext.compression != .none && (dflashModelPath != nil || gemmaAssistantModelPath != nil) {
            throw RequestAdmissionError.configuration("KV compression with speculative decoding is not supported")
        }
        // Prevent loading a checkpoint whose stored payload alone exceeds the allocator budget.
        let shards = try FileManager.default.contentsOfDirectory(
            at: modelURL,
            includingPropertiesForKeys: [.fileSizeKey]
        ).filter { $0.pathExtension == "safetensors" }
        let storedBytes = try shards.reduce(0) { total, url in
            ModelMemoryProfile.add(total, try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0)
        }
        if usesCUDAOffload, storedBytes > hostBudget {
            throw RequestAdmissionError.configuration(
                "CUDA RAM offload currently stages the complete checkpoint in host RAM; "
                    + "stored weights require \(storedBytes) bytes, host staging budget is \(hostBudget) bytes")
        }
        guard storedBytes < resourceLimits.memoryLimitBytes else {
            throw RequestAdmissionError.memoryExceeded(
                required: storedBytes, available: resourceLimits.memoryLimitBytes)
        }

        let dflashURL: URL? = try (dflashModelPath ?? gemmaAssistantModelPath).map { path in
            let expanded = NSString(string: path).expandingTildeInPath
            let url = URL(fileURLWithPath: expanded, isDirectory: true)
                .standardizedFileURL
                .resolvingSymlinksInPath()
            try Self.validateModelFolder(url)
            return url
        }

        #if os(macOS) && MODEL_RUNNER_PINNED_MLX
            let mlxStream = try await makePinnedMLXStream(device: device)
        #endif

        let loadModels: @Sendable () async throws -> (ModelContainer, LoadedSpeculativeModel?) = {
            // Laguna is implemented by this target so the runner can use Poolside
            // checkpoints natively without a Python sidecar or an mlx-swift-lm fork.
            // Register and construct it on the same permanent pthread used for eval.
            await LagunaModelRegistration.register()
            await TalkieModelRegistration.register()
            await LagunaDFlashRegistration.register()
            await MuseGlimmerDFlashRegistration.register()

            print(
                "MLX resource guard: memory=\(resourceLimits.memoryLimitBytes) bytes "
                    + "cache=\(resourceLimits.cacheLimitBytes) bytes"
            )
            try MLXResourceGuard.apply(resourceLimits)
            let targetModelType = try Self.modelType(at: modelURL)
            let targetConfiguration = LocalModelChatConfiguration.make(
                directory: modelURL,
                modelType: targetModelType,
                additionalEOSTokens: TalkieLoadingOptions.additionalEOSTokens(
                    configuration: try Data(contentsOf: modelURL.appendingPathComponent("config.json")))
            )
            // Muse Glimmer is implemented by MLXVLM.  Its language model remains a
            // normal LanguageModel, so text-only requests use the regular serving
            // path while the VLM factory preserves its hybrid attention and native
            // checkpoint weight layout.  Image input is deliberately not admitted by
            // this runner yet; that is a separate HTTP/API milestone.
            let targetContainer: ModelContainer
            if targetModelType == "muse_glimmer" {
                guard adapterPath == nil else {
                    throw RequestAdmissionError.configuration("Muse Glimmer LoRA adapters are not supported")
                }
                targetContainer = try await VLMModelFactory.shared.loadContainer(
                    from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                    configuration: targetConfiguration
                )
            } else {
                targetContainer = try await Gemma4RuntimeTuning.$useBoundedWindowCache.withValue(useBoundedGemmaKV) {
                    try await TalkieLoadingOptions.$fuseProjections.withValue(adapterPath == nil) {
                        try await LLMModelFactory.shared.loadContainer(
                            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                            configuration: targetConfiguration
                        )
                    }
                }
            }
            guard let dflashURL else {
                return (targetContainer, nil)
            }

            if gemmaAssistantModelPath != nil {
                let blockSize = try GemmaAssistantCompatibility.validate(
                    target: Data(contentsOf: modelURL.appendingPathComponent("config.json")),
                    assistant: Data(contentsOf: dflashURL.appendingPathComponent("config.json")),
                    blockSize: gemmaAssistantBlockSize, quantizationBits: gemmaAssistantQuantizationBits)
                try await targetContainer.perform { context in
                    guard context.model is any Gemma4MTPEmbeddingProvider else {
                        throw RequestAdmissionError.configuration("Gemma assistant requires a Gemma 4 MTP target")
                    }
                }
                await MTPDrafterTypeRegistry.shared.registerModelType(
                    "gemma4_assistant",
                    creator: { data in
                        Gemma4TextAssistantDraftModel(
                            try JSONDecoder().decode(Gemma4TextAssistantConfiguration.self, from: data))
                    })
                let draftContainer = try await MTPDrafterModelFactory.shared.loadContainer(
                    from: dflashURL, using: #huggingFaceTokenizerLoader())
                let (reference, count) = try await draftContainer.perform { context in
                    var count = 0
                    if let bits = gemmaAssistantQuantizationBits {
                        quantize(model: context.model, groupSize: 64, bits: bits)
                        eval(context.model)
                        count = context.model.leafModules().flattened().filter { $0.1 is QuantizedLinear }.count
                        guard count > 0 else {
                            throw RequestAdmissionError.configuration(
                                "Gemma assistant quantization converted no linear modules")
                        }
                    }
                    return (SpeculativeModelReference(context.model), count)
                }
                return (
                    targetContainer,
                    LoadedSpeculativeModel(
                        container: draftContainer,
                        model: reference, blockSize: blockSize, modelPath: dflashURL.path, quantizedModuleCount: count)
                )
            }

            if targetModelType == "muse_glimmer" {
                _ = try MuseGlimmerDFlashCompatibility.validate(
                    target: Data(contentsOf: modelURL.appendingPathComponent("config.json")),
                    assistant: Data(contentsOf: dflashURL.appendingPathComponent("config.json")),
                    blockSize: dflashBlockSize)
                let draftContainer = try await MTPDrafterModelFactory.shared.loadContainer(
                    from: dflashURL, using: #huggingFaceTokenizerLoader())
                let loaded = await draftContainer.perform { context -> (SpeculativeModelReference, Int, [Int], Int)? in
                    guard let model = context.model as? MuseGlimmerDFlashModel else {
                        return nil
                    }
                    return (
                        SpeculativeModelReference(model), model.configuration.blockSize,
                        model.targetLayerIDs, model.captureWindow
                    )
                }
                guard let (reference, maximum, layerIDs, captureWindow) = loaded else {
                    throw LocalModelRunnerError.incompatibleDFlash("checkpoint is not MuseGlimmerAssistantModel")
                }
                // Muse's bundled assistant is verified against this target at four
                // tokens per step.  Its advertised maximum is 16, but using that
                // maximum as an implicit default substantially lowers acceptance and
                // regresses decode speed.  Callers can still explicitly choose any
                // compatible block size.
                let effectiveBlockSize = dflashBlockSize ?? min(maximum, 4)
                guard (2...maximum).contains(effectiveBlockSize) else {
                    throw LocalModelRunnerError.invalidDFlashBlockSize(effectiveBlockSize, maximum: maximum)
                }
                try await targetContainer.perform { context in
                    guard let target = context.model as? MuseGlimmer else {
                        throw LocalModelRunnerError.incompatibleDFlash("Muse assistant requires a MuseGlimmer target")
                    }
                    try target.configureDFlash(targetLayerIDs: layerIDs, captureWindow: captureWindow)
                }
                return (
                    targetContainer,
                    LoadedSpeculativeModel(
                        container: draftContainer,
                        model: reference, blockSize: effectiveBlockSize, modelPath: dflashURL.path)
                )
            }

            let draftContainer = try await MTPDrafterModelFactory.shared.loadContainer(
                from: dflashURL,
                using: #huggingFaceTokenizerLoader()
            )
            let loaded = await draftContainer.perform {
                context -> (SpeculativeModelReference, LagunaDFlashTargetDescriptor)? in
                guard let model = context.model as? LagunaDFlashModel else {
                    return nil
                }
                return (SpeculativeModelReference(model), model.targetDescriptor)
            }
            guard let (draftReference, descriptor) = loaded else {
                throw LocalModelRunnerError.incompatibleDFlash(
                    "checkpoint is not DFlashLagunaForCausalLM")
            }
            // The checkpoint maximum (16) is a capability limit, not a useful
            // default for the local quantized Laguna target. A paired Metal run
            // accepted only 7.8% of proposals at 16 and lost 56.6% of throughput;
            // three-row verification reached 60.1% acceptance and nearly matched
            // target-only throughput (without establishing output equivalence).
            // Keep explicit overrides for workload/backend-specific experiments.
            let effectiveBlockSize = dflashBlockSize ?? min(descriptor.blockSize, 3)
            guard effectiveBlockSize >= 2, effectiveBlockSize <= descriptor.blockSize else {
                throw LocalModelRunnerError.invalidDFlashBlockSize(
                    effectiveBlockSize, maximum: descriptor.blockSize)
            }
            try await targetContainer.perform(values: descriptor) { context, descriptor in
                guard let target = context.model as? LagunaModel else {
                    throw LocalModelRunnerError.incompatibleDFlash(
                        "DFlash requires a native Laguna target, got \(type(of: context.model))")
                }
                try target.configureDFlash(descriptor)
            }
            return (
                targetContainer,
                LoadedSpeculativeModel(
                    container: draftContainer,
                    model: draftReference,
                    blockSize: effectiveBlockSize,
                    modelPath: dflashURL.path
                )
            )
        }
        let (container, loadedSpeculativeModel): (ModelContainer, LoadedSpeculativeModel?)
        #if os(macOS) && MODEL_RUNNER_PINNED_MLX
            (container, loadedSpeculativeModel) = try await withPinnedMLXRuntime(
                device: device,
                stream: mlxStream,
                operation: loadModels
            )
        #else
            (container, loadedSpeculativeModel) = try await Device.withDefaultDevice(
                usesCUDAOffload ? .cpu : device,
                loadModels
            )
        #endif
        let cudaWeightPlacement: CUDAWeightPlacement?
        if usesCUDAOffload {
            cudaWeightPlacement = try await container.perform { context in
                try CUDAWeightPlacement(
                    model: context.model, mode: offloadMode,
                    gpuBudget: gpuResourceLimits.memoryLimitBytes, hostBudget: hostBudget)
            }
            let finalLimits = try gpuResourceLimits.addingHostAllowance(cudaWeightPlacement!.plan.hostWeightBytes)
            try MLXResourceGuard.apply(finalLimits)
        } else {
            cudaWeightPlacement = nil
        }
        if let adapterPath {
            // Adapter safetensors are loaded lazily on CPU. Pinned mode constructs
            // them on its CPU stream; the default path retains the fresh-stream
            // boundary before the container crosses executors.
            let adapter: LoRAContainer
            #if os(macOS) && MODEL_RUNNER_PINNED_MLX
                let adapterStream = try await makePinnedMLXStream(device: .cpu)
                adapter = try await withPinnedMLXRuntime(device: .cpu, stream: adapterStream) {
                    try Self.loadAdapter(
                        directory: adapterPath,
                        scaleOverride: adapterScale
                    )
                }
                try await withPinnedMLXRuntime(device: device, stream: mlxStream) {
                    try await container.perform { context in
                        try adapter.load(into: context.model)
                    }
                }
            #else
                adapter = try Device.withDefaultDevice(.cpu) {
                    try Stream.withNewDefaultStream(device: .cpu) {
                        try Self.loadAdapter(
                            directory: adapterPath,
                            scaleOverride: adapterScale
                        )
                    }
                }
                try await Device.withDefaultDevice(device) {
                    try await container.perform { context in
                        try adapter.load(into: context.model)
                    }
                }
            #endif
            print(
                "Loaded LoRA adapter \(adapterPath)"
                    + (adapterScale.map { " at scale \($0)" } ?? "")
            )
        } else if adapterScale != nil {
            throw LocalModelRunnerError.adapterScaleWithoutAdapter
        }
        let supportsLagunaPromptCache: Bool
        #if os(macOS) && MODEL_RUNNER_PINNED_MLX
            supportsLagunaPromptCache = try await withPinnedMLXRuntime(
                device: device,
                stream: mlxStream
            ) {
                await container.perform { context in
                    context.model is LagunaModel
                }
            }
        #else
            supportsLagunaPromptCache = await container.perform { context in
                context.model is LagunaModel
            }
        #endif
        var cacheParameters = GenerateParameters(maxTokens: 1, temperature: 0)
        try longContext.apply(to: &cacheParameters)
        if longContext.compression != .none {
            guard supportsLagunaPromptCache || runtimeCapabilities.mistralFamily != nil else {
                throw RequestAdmissionError.configuration(
                    "KV compression is currently supported only for Laguna and Mistral-family text models")
            }
            let status = try await container.cacheStatus(parameters: cacheParameters)
            guard status.pendingLayerCount + status.compressedLayerCount > 0 else {
                throw RequestAdmissionError.configuration("no attention layer supports the requested KV compression")
            }
            print(
                "Experimental KV compression: \(longContext.compression.rawValue), eligible=\(status.pendingLayerCount + status.compressedLayerCount), skipped=\(status.skippedLayerCount)"
            )
        }
        let tuningParameters = cacheParameters
        var wiredMemoryPlan: MLXWiredMemoryPlan?
        if engine == .metal, MLXWiredMemoryPlan.isEnabled(environment: environment) {
            let tuningTokens = min(
                memoryProfile.contextLength, MLXWiredMemoryPlan.tuningTokenCount(environment: environment))
            let measure: @Sendable () async throws -> WiredMemoryMeasurement = {
                try await container.perform { context in
                    try await WiredMemoryUtils.tune(
                        context: context,
                        tokenCount: tuningTokens,
                        parameters: tuningParameters
                    )
                }
            }
            do {
                let measurement: WiredMemoryMeasurement
                #if os(macOS) && MODEL_RUNNER_PINNED_MLX
                    measurement = try await withPinnedMLXRuntime(
                        device: device,
                        stream: mlxStream,
                        operation: measure
                    )
                #else
                    measurement = try await Device.withDefaultDevice(device, measure)
                #endif
                #if os(macOS)
                    let recommendedWorkingSetBytes = GPU.maxRecommendedWorkingSetBytes()
                #else
                    let recommendedWorkingSetBytes: Int? = nil
                #endif
                wiredMemoryPlan = MLXWiredMemoryPlan.measured(
                    peakActiveBytes: measurement.peakActiveBytes,
                    cacheReserveBytes: resourceLimits.cacheLimitBytes,
                    allocatorLimitBytes: resourceLimits.memoryLimitBytes,
                    recommendedWorkingSetBytes: recommendedWorkingSetBytes,
                    minimumHeadroomBytes: loadedSpeculativeModel == nil
                        ? MLXWiredMemoryPlan.defaultMinimumHeadroomBytes
                        : MLXWiredMemoryPlan.dflashMinimumHeadroomBytes
                )
                Memory.clearCache()
                if let wiredMemoryPlan {
                    print(
                        "Measured MLX wired-memory plan: peak=\(measurement.peakActiveBytes) "
                            + "kv=\(measurement.kvBytes) workspace=\(measurement.workspaceBytes) "
                            + "limit=\(wiredMemoryPlan.limitBytes) cap=\(wiredMemoryPlan.capBytes)"
                    )
                }
            } catch {
                if error is CancellationError {
                    throw error
                }
                print("MLX wired-memory measurement unavailable; continuing unwired: \(error)")
                wiredMemoryPlan = nil
            }
        }
        let residentModelBytes = await container.perform { context in
            eval(context.model.parameters().flattened().map { $0.1 })
            return Memory.activeMemory
        }
        self.longContext = longContext
        self.memoryProfile = memoryProfile
        self.residentModelBytes = residentModelBytes
        self.contextLength = memoryProfile.contextLength
        self.prefillStepSize = longContext.prefillStepSize
        self.kvCompression = longContext.compression.rawValue
        self.cudaWeightPlacement = cudaWeightPlacement
        self.cudaDeviceBudgetBytes = engine == .cuda ? gpuResourceLimits.memoryLimitBytes : nil
        self.memoryLimitBytes = gpuResourceLimits.memoryLimitBytes + (cudaWeightPlacement?.plan.hostWeightBytes ?? 0)
        print(
            "Context policy: limit=\(memoryProfile.contextLength), prefill=\(longContext.prefillStepSize), KV=\(longContext.compression.rawValue), geometry=\(memoryProfile.hasKnownGeometry ? "known" : "conservative fallback")"
        )
        let decisionDirectory = URL(fileURLWithPath: adapterPath ?? modelURL.path)
        let decisionURL = decisionDirectory.appendingPathComponent(DecisionModelContract.filename)
        if FileManager.default.fileExists(atPath: decisionURL.path) {
            let bytes = try Data(contentsOf: decisionURL)
            guard bytes.count <= 256 * 1024 else {
                throw DecisionError.invalidContract("Decision contract exceeds 256 KiB.")
            }
            let contract = try JSONDecoder().decode(DecisionModelContract.self, from: bytes)
            try contract.validate()
            try ModelDecisionArtifacts.validate(
                contract: contract, base: modelURL, adapter: adapterPath.map { URL(fileURLWithPath: $0) })
            guard adapterScale == nil, loadedSpeculativeModel == nil else {
                throw DecisionError.invalidContract(
                    "Decision scoring requires the contracted adapter scale and no speculative assistant.")
            }
            self.decisionContract = contract
        } else {
            self.decisionContract = nil
        }
        self.container = container
        self.modelPath = modelURL.path
        self.servedModelName =
            servedModelName?.trimmingCharacters(in: .whitespacesAndNewlines)
            .nilIfBlank ?? modelURL.lastPathComponent
        self.engine = engine
        self.tokenLimit = tokenLimit
        self.device = device
        self.normalizesGemma4Prompt = normalizesGemma4Prompt
        self.runtimeCapabilities = runtimeCapabilities
        self.supportsMistralHotConversationCache =
            runtimeCapabilities.supportsMistralConversationPrefixCache
        self.supportsHotConversationCache = runtimeCapabilities.supportsHotConversationCache
        self.supportsLagunaPromptCache = supportsLagunaPromptCache
        self.wiredMemoryPlan = wiredMemoryPlan
        let prefixCacheLimits = ConversationPrefixCacheLimits.resolve(environment: environment)
        self.sharedPromptCache = SharedPromptCache(
            // Gemma's many KV heads exceed the ordinary 64 MiB checkpoint cap
            // even at 128 tokens. Use the existing configured retention budget;
            // every snapshot still checks available memory before allocation.
            maximumBytes: prefixCacheLimits.maximumEntries > 0
                ? min(prefixCacheLimits.maximumBytes, normalizesGemma4Prompt ? Int.max : 64 * 1_048_576) : 0,
            maximumEntries: prefixCacheLimits.maximumEntries
        )
        self.conversationCache = CompletedMessagePrefixLRU(
            maximumEntries: prefixCacheLimits.maximumEntries,
            maximumBytes: prefixCacheLimits.maximumBytes
        )
        self.dflash = loadedSpeculativeModel
        self.modelImplementation = await container.perform { context in
            String(reflecting: type(of: context.model))
        }
        #if os(macOS) && MODEL_RUNNER_PINNED_MLX
            self.mlxStream = mlxStream
        #endif
        self.dflashModelPath = gemmaAssistantModelPath == nil ? loadedSpeculativeModel?.modelPath : nil
        self.dflashBlockSize = gemmaAssistantModelPath == nil ? loadedSpeculativeModel?.blockSize : nil
        self.gemmaAssistantModelPath = gemmaAssistantModelPath == nil ? nil : loadedSpeculativeModel?.modelPath
        self.gemmaAssistantBlockSize = gemmaAssistantModelPath == nil ? nil : loadedSpeculativeModel?.blockSize
        self.gemmaAssistantQuantizationBits = gemmaAssistantQuantizationBits
        self.gemmaAssistantQuantizedModuleCount =
            gemmaAssistantQuantizationBits == nil ? nil : loadedSpeculativeModel?.quantizedModuleCount
        if supportsLagunaPromptCache {
            print(
                "Conversation prefix cache: entries=\(prefixCacheLimits.maximumEntries) "
                    + "memory=\(prefixCacheLimits.maximumBytes) bytes"
            )
        } else if let family = runtimeCapabilities.mistralFamily {
            print(
                "Mistral hot conversation cache (\(family.rawValue)): "
                    + "zero-copy linear reuse enabled; branch snapshots disabled"
            )
        } else if runtimeCapabilities.supportsGPTOSSConversationPrefixCache {
            print(
                "GPT-OSS tool-continuation cache available: zero-copy reuse after tool calls; "
                    + "completed text replies release KV state"
            )
        }
        if let loadedSpeculativeModel {
            let samplingSupport =
                loadedSpeculativeModel.model.model.requiresGreedySampling
                ? "greedy only; sampled requests use target-only decoding"
                : "target-verified sampling"
            print(
                "Loaded speculative assistant \(loadedSpeculativeModel.modelPath) "
                    + "(block size \(loadedSpeculativeModel.blockSize), \(samplingSupport))"
            )
        }
    }

    deinit {
        // An arm belongs to this loaded instance. Replacing or unloading the
        // runtime must not leave the listener's single recording slot occupied.
        if let pending = pendingInspectorRecording { pending.store.cancel(id: pending.id) }
    }

    private static func modelType(at directory: URL) throws -> String? {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        return (try JSONSerialization.jsonObject(with: data) as? [String: Any])?["model_type"] as? String
    }

    package var benchmarkWiredMemoryPlan: BenchmarkWiredMemoryPlan? {
        wiredMemoryPlan.map(BenchmarkWiredMemoryPlan.init)
    }

    package func beginBenchmarkWiredMemoryDiagnostics() {
        benchmarkWiredMemoryRequests = []
    }

    package func takeBenchmarkWiredMemoryDiagnostics() -> [BenchmarkWiredMemoryRequest] {
        let records = benchmarkWiredMemoryRequests ?? []
        benchmarkWiredMemoryRequests = nil
        return records
    }

    package func makeBenchmarkWiredMemorySession() throws -> BenchmarkWiredMemorySession {
        try BenchmarkWiredMemorySession(engine: engine, plan: wiredMemoryPlan)
    }

    /// Wait for stream producers, including cancelled producers, to finish their
    /// cleanup. Call only after new operations have been stopped and direct calls
    /// such as prompt preparation and inspection have returned.
    public func waitUntilIdle() async {
        await producerLifetime.waitUntilIdle()
        #if os(macOS) && MODEL_RUNNER_PINNED_MLX
            let stream = mlxStream
            await MLXPinnedRuntime.shared.runCleanup { stream.synchronize() }
        #else
            StreamOrDevice.device(device).stream.synchronize()
        #endif
    }

    private func acquireExecution() async throws {
        do { try await generationAdmission.acquire() } catch GenerationAdmission.Failure.full {
            throw LocalModelRunnerError.busy
        }
    }

    /// Score independent typed fields with release-specific candidate tokens and calibration.
    public func scoreDecisions(request: DecisionRequest) async throws -> DecisionResponse {
        guard let contract = decisionContract else {
            throw DecisionError.unsupported("The loaded model has no verified decision contract.")
        }
        guard request.model == servedModelName else {
            throw DecisionError.invalidRequest("Request model does not match the loaded model.")
        }
        try await acquireExecution()
        defer { generationAdmission.release() }
        try Task.checkCancellation()
        let prepared = try await container.perform { context in
            try ModelDecisionScoring.prepare(request: request, contract: contract, tokenizer: context.tokenizer)
        }
        let longest = prepared.map { $0.tokens.count }.max() ?? 0
        try validateAdmission(prompt: longest, output: 1, resident: residentModelBytes + hotCacheBytes)
        #if os(macOS) && MODEL_RUNNER_PINNED_MLX
            return try await withPinnedMLXRuntime(device: device, stream: mlxStream) {
                try await self.runDecisionScoring(request: request, contract: contract)
            }
        #else
            return try await runDecisionScoring(request: request, contract: contract)
        #endif
    }

    private func runDecisionScoring(request: DecisionRequest, contract: DecisionModelContract) async throws
        -> DecisionResponse
    {
        let decisionContainer = container
        return try await withGenerationWiredResidency {
            try await Device.withDefaultDevice(device) { @Sendable in
                try await decisionContainer.perform { context in
                    try ModelDecisionScoring.score(context: context, request: request, contract: contract)
                }
            }
        }
    }

    /// Clear retained KV state across clients, serialized with generation.
    /// Model weights and conversation records are unaffected.
    public func resetKVCache() async throws {
        try await acquireExecution()
        defer { generationAdmission.release() }
        try Task.checkCancellation()
        #if os(macOS) && MODEL_RUNNER_PINNED_MLX
            let stream = mlxStream
            await MLXPinnedRuntime.shared.runCleanup { stream.synchronize() }
        #else
            StreamOrDevice.device(device).stream.synchronize()
        #endif
        hotConversation = nil
        hotCacheBytes = 0
        conversationCache.removeAll()
        sharedPromptCache.clear()
        Memory.clearCache()
    }

    /// Validate exact rendered prompt tokens before HTTP response headers or GPU prefill.
    public func preparePrompt(
        messages: [OpenAIMessage], maximumTokens: Int?,
        tools: [OpenAIToolDefinition]? = nil,
        toolChoice: ToolChoicePlan.Constraint = .automatic,
        reasoningEffort: ChatCompletionRequest.ReasoningEffort? = nil,
        responseFormat: OpenAIResponseFormat? = nil, thinkingEnabled: Bool? = nil,
        nativeProtocol: String? = nil
    ) async throws -> PreparedModelPrompt {
        try await acquireExecution()
        defer { generationAdmission.release() }
        try Task.checkCancellation()
        var prepared = try await renderPrompt(
            messages: messages, tools: tools,
            toolChoice: toolChoice, reasoningEffort: reasoningEffort, responseFormat: responseFormat,
            thinkingEnabled: thinkingEnabled)
        if let nativeProtocol {
            let supported = await container.perform { context in
                (nativeProtocol == "laguna" && context.model is LagunaModel)
                    || (nativeProtocol == "muse" && context.model is MuseGlimmer)
            }
            guard supported, !StructuredOutputRequest.isStructured(responseFormat) else {
                throw RequestAdmissionError.configuration(
                    "native_protocol requires a matching Muse or Laguna model and unstructured output.")
            }
            prepared.nativeProtocol = nativeProtocol
        }
        try validateAdmission(
            prompt: prepared.promptTokenCount,
            output: tokenLimit.resolve(
                requested: maximumTokens, promptTokens: prepared.promptTokenCount,
                contextLength: contextLength), resident: residentModelBytes)
        return prepared
    }

    private func renderPrompt(
        messages: [OpenAIMessage], tools: [OpenAIToolDefinition]?,
        toolChoice: ToolChoicePlan.Constraint = .automatic,
        reasoningEffort: ChatCompletionRequest.ReasoningEffort?,
        responseFormat: OpenAIResponseFormat? = nil, thinkingEnabled: Bool? = nil
    ) async throws -> PreparedModelPrompt {
        guard let last = messages.last, last.role == "user" || last.role == "tool" else {
            throw LocalModelRunnerError.lastMessageMustBeUserOrTool
        }
        try StructuredOutputRequest.validate(format: responseFormat, tools: tools, stop: [])
        var promptMessages = messages
        if StructuredOutputRequest.isStructured(responseFormat), let responseFormat {
            let instruction = try StructuredOutputRequest.instruction(for: responseFormat)
            if let first = promptMessages.first, first.role == "system" || first.role == "developer" {
                promptMessages[0] = OpenAIMessage(
                    role: first.role, content: (first.content ?? "") + "\n\n" + instruction)
            } else {
                promptMessages.insert(OpenAIMessage(role: "system", content: instruction), at: 0)
            }
        }
        let chatMessages = try promptMessages.map(Self.chatMessage)
        let toolSpecs = try Self.toolSpecs(tools)
        let effectiveReasoningEffort = Self.effectiveReasoningEffort(
            reasoningEffort, capabilities: runtimeCapabilities)
        let input = try await container.prepare(
            input: UserInput(
                chat: chatMessages, tools: toolSpecs,
                additionalContext: Self.promptAdditionalContext(
                    reasoningEffort: effectiveReasoningEffort,
                    thinkingEnabled: supportsLagunaPromptCache
                        ? (toolChoice.requiresToolCall ? false : thinkingEnabled)
                        : (normalizesGemma4Prompt
                            ? (StructuredOutputRequest.isStructured(responseFormat) ? false : thinkingEnabled) : nil),
                    preserveThinking: supportsLagunaPromptCache)))
        let toolChoicePlan = ToolChoicePlan(tools: tools, constraint: toolChoice)
        let forcedToolPrefixTokenIDs: [[Int]] = try await container.perform { context in
            let prefixes = try toolChoicePlan.forcedToolCallPrefixes(
                for: context.configuration.toolCallFormat ?? .json)
            guard !prefixes.isEmpty else {
                return [[Int]]()
            }
            let vocabularySize: Int
            if let laguna = context.model as? LagunaModel {
                vocabularySize = laguna.vocabularySize
            } else if let muse = context.model as? MuseGlimmer {
                vocabularySize = muse.vocabularySize
            } else {
                throw LocalModelRunnerError.unsupportedForcedToolChoiceModel(
                    String(describing: type(of: context.model)))
            }
            return try prefixes.map { prefix in
                let tokenIDs = context.tokenizer.encode(text: prefix, addSpecialTokens: false)
                guard !tokenIDs.isEmpty,
                    tokenIDs.allSatisfy({ $0 >= 0 && $0 < vocabularySize })
                else {
                    throw LocalModelRunnerError.invalidForcedToolChoiceTokens
                }
                return tokenIDs
            }
        }

        if toolChoice.requiresToolCall && forcedToolPrefixTokenIDs.isEmpty {
            throw LocalModelRunnerError.emptyForcedToolChoicePrefix
        }
        // Tokenizers normally produce int32 IDs. Widen on the CPU instead of
        // submitting an int32-to-int64 GPU cast merely to read a Swift [Int].
        // Preserve the original conversion for processors returning another dtype.
        var tokens =
            input.text.tokens.dtype == .int32
            ? input.text.tokens.asArray(Int32.self).map(Int.init)
            : input.text.tokens.asArray(Int.self)
        if normalizesGemma4Prompt, tokens.count >= 3, Array(tokens.prefix(3)) == [2, 107, 105] {
            tokens.remove(at: 1)
        }
        let structuredOutput: StructuredOutputLogitProcessor?
        if StructuredOutputRequest.isStructured(responseFormat), let responseFormat {
            let grammar = try StructuredOutputGrammar(responseFormat: responseFormat)
            let directory = URL(fileURLWithPath: modelPath)
            // Harmony's ordinary assistant prefix leaves the channel undecided. For
            // structured replies prefill the final channel before enforcing JSON bytes.
            if runtimeCapabilities.isGPTOSS {
                let finalChannel = await container.perform { context in
                    context.tokenizer.encode(text: "<|channel|>final<|message|>", addSpecialTokens: false)
                }
                tokens.append(contentsOf: finalChannel)
            }
            do {
                structuredOutput = try await container.perform { context in
                    var eos = context.configuration.eosTokenIds
                    if let token = context.tokenizer.eosTokenId {
                        eos.insert(token)
                    }
                    for token in context.configuration.extraEOSTokens {
                        if let id = context.tokenizer.convertTokenToId(token) {
                            eos.insert(id)
                        }
                    }
                    return try StructuredOutputLogitProcessor(
                        grammar: grammar, modelDirectory: directory,
                        tokenizer: context.tokenizer, eosTokenIDs: eos)
                }
            } catch {
                throw StructuredOutputRequestError.unsupportedTokenizer(error.localizedDescription)
            }
        } else {
            structuredOutput = nil
        }
        return PreparedModelPrompt(
            modelPath: modelPath, messages: messages, tools: tools,
            toolChoice: toolChoice, thinkingEnabled: thinkingEnabled, reasoningEffort: effectiveReasoningEffort,
            forcedToolPrefixTokenIDs: forcedToolPrefixTokenIDs, tokenIDs: tokens,
            responseFormat: responseFormat, structuredOutput: structuredOutput)
    }

    static func effectiveReasoningEffort(
        _ effort: ChatCompletionRequest.ReasoningEffort?, capabilities: ModelRuntimeCapabilities
    ) -> ChatCompletionRequest.ReasoningEffort? {
        capabilities.isGPTOSS ? effort : nil
    }

    static func promptAdditionalContext(
        reasoningEffort: ChatCompletionRequest.ReasoningEffort?, thinkingEnabled: Bool? = nil,
        preserveThinking: Bool = false
    ) -> [String: any Sendable]? {
        var context: [String: any Sendable] = [:]
        if let reasoningEffort {
            context["reasoning_effort"] = reasoningEffort.rawValue
        }
        if let thinkingEnabled {
            context["enable_thinking"] = thinkingEnabled
        }
        if preserveThinking {
            context["preserve_thinking"] = true
        }
        return context.isEmpty ? nil : context
    }

    private func validateAdmission(prompt: Int, output: Int, resident: Int) throws {
        try memoryProfile.validateContext(prompt: prompt, output: output)
        let required = memoryProfile.requestBytes(
            prompt: prompt, output: output,
            prefillStepSize: min(longContext.prefillStepSize, max(1, prompt)), residentBytes: resident)
        guard required <= memoryLimitBytes else {
            throw RequestAdmissionError.memoryExceeded(required: required, available: memoryLimitBytes)
        }
        if let cudaDeviceBudgetBytes {
            let hostWeights = cudaWeightPlacement?.plan.hostWeightBytes ?? 0
            let snapshot = try CUDAMemoryRuntime.snapshot()
            let deviceActive = max(0, Memory.activeMemory - hostWeights)
            let available = min(cudaDeviceBudgetBytes, ModelMemoryProfile.add(deviceActive, snapshot.usableBytes))
            let deviceRequired = max(0, required - hostWeights)
            guard deviceRequired <= available else {
                throw RequestAdmissionError.memoryExceeded(required: deviceRequired, available: available)
            }
        }
    }

    /// Metadata-only capability check for the opt-in same-loaded kernel benchmark.
    public func lagunaFusedGateUpSiluCoverage() async -> (eligible: Int, sparse: Int, traces: Int) {
        await container.perform { context in
            guard let model = context.model as? LagunaModel else {
                return (0, 0, 0)
            }
            return (
                model.fusedGateUpSiluEligibleLayerCount, model.fusedGateUpSiluSparseLayerCount,
                model.fusedGateUpSiluTraceCount
            )
        }
    }

    /// Architecture metadata comes from the loaded module graph and tensor shapes.
    public func inspectorModel() async throws -> InspectorModel {
        try await acquireExecution()
        defer { generationAdmission.release() }
        try Task.checkCancellation()
        return await loadedInspectorDescriptor()
    }

    private func loadedInspectorDescriptor() async -> InspectorModel {
        let metadata = ModelInspectionMetadata.load(modelPath: modelPath)
        let id = servedModelName
        let contextLength = contextLength
        return await container.perform { context in
            ModelInspection.descriptor(
                model: context.model, id: id, metadata: metadata,
                contextLength: contextLength)
        }
    }

    /// Arms capture of the next ordinary text generation, without submitting a prompt.
    public func armInspectorRecording(
        request: InspectorRecordingRequest, store: InspectorRecordingStore
    ) async throws -> InspectorRecordingSession {
        try await acquireExecution()
        defer { generationAdmission.release() }
        try Task.checkCancellation()
        var descriptor = await loadedInspectorDescriptor()
        descriptor.runtimeGeneration = request.runtimeGeneration
        let estimate = try PendingInspectorRecording.validate(request: request, descriptor: descriptor)
        let session = try store.arm(model: descriptor)
        pendingInspectorRecording = PendingInspectorRecording(
            id: session.id, request: request, descriptor: descriptor, store: store, estimatedBytes: estimate)
        return session
    }

    /// Opt-in bounded inspection owns the runner for its complete lifetime. Its
    /// fresh cache and temporary observers do not become a retained chat session.
    public func inspectorTrace(request: InspectorTraceRequest) async throws -> InspectorTrace {
        try await acquireExecution()
        defer { generationAdmission.release() }
        try Task.checkCancellation()
        let requestedMaximum = try ModelInspection.maximumTokens(request)
        let maximumTokens =
            request.maxTokens == nil
            ? min(requestedMaximum, tokenLimit.configuredMaximum) : requestedMaximum
        guard maximumTokens <= tokenLimit.configuredMaximum else {
            throw ModelInspectionError.invalidRequest(
                "Inspector maxTokens exceeds this server's configured maximum of \(tokenLimit.configuredMaximum).")
        }
        let descriptor = await loadedInspectorDescriptor()
        guard descriptor.traceSupported else {
            throw ModelInspectionError.unsupported(descriptor.traceReason ?? "Activation capture is unavailable.")
        }
        let prepared = try await renderPrompt(
            messages: [OpenAIMessage(role: "user", content: request.question)],
            tools: nil, reasoningEffort: runtimeCapabilities.isGPTOSS ? .low : nil)
        guard prepared.promptTokenCount <= ModelInspection.promptTokenLimit else {
            throw ModelInspectionError.invalidRequest(
                "The rendered inspector prompt exceeds \(ModelInspection.promptTokenLimit) tokens. Shorten the question."
            )
        }
        let capturePlan = try ModelInspection.capturePlan(
            request: request, descriptor: descriptor,
            promptTokenCount: prepared.promptTokenCount, maximumTokens: maximumTokens)
        // Capture/serialization allocations are bounded before observers are installed.
        // Include them alongside retained chat caches in normal memory admission.
        // Prompt-only inspection still reserves one output slot for its final logits.
        try validateAdmission(
            prompt: prepared.promptTokenCount, output: max(1, maximumTokens),
            resident: max(residentModelBytes, Memory.activeMemory) + capturePlan.estimatedMemoryBytes)
        #if os(macOS) && MODEL_RUNNER_PINNED_MLX
            return try await withPinnedMLXRuntime(device: device, stream: mlxStream) {
                try await self.runInspectorTrace(
                    request: request, descriptor: descriptor,
                    prepared: prepared, maximumTokens: maximumTokens, capturePlan: capturePlan)
            }
        #else
            return try await runInspectorTrace(
                request: request, descriptor: descriptor,
                prepared: prepared, maximumTokens: maximumTokens, capturePlan: capturePlan)
        #endif
    }

    private func runInspectorTrace(
        request: InspectorTraceRequest, descriptor: InspectorModel,
        prepared: PreparedModelPrompt, maximumTokens: Int,
        capturePlan: InspectionCapturePlan
    ) async throws -> InspectorTrace {
        let inspectionContainer = container
        return try await withGenerationWiredResidency {
            try await Device.withDefaultDevice(device) { @Sendable in
                try await inspectionContainer.perform { context in
                    try LagunaRuntimeTuning.$useCompiledBlockTail.withValue(false) {
                        try ModelInspection.trace(
                            context: context, descriptor: descriptor, question: request.question,
                            promptTokens: prepared.promptTokenIDs, maximumTokens: maximumTokens,
                            capturePlan: capturePlan)
                    }
                }
            }
        }
    }

    /// Generates ordered local model events from a prepared request.
    public func stream(
        messages: [OpenAIMessage],
        maximumTokens: Int?,
        temperature: Double? = nil,
        topP: Double? = nil,
        stop: [String] = [],
        tools: [OpenAIToolDefinition]? = nil,
        toolChoice: ToolChoicePlan.Constraint = .automatic,
        reasoningEffort: ChatCompletionRequest.ReasoningEffort? = nil,
        thinkingEnabled: Bool? = nil,
        enablePromptCache: Bool = true,
        enableSpeculativeDecoding: Bool = true,
        preparedPrompt: PreparedModelPrompt? = nil,
        responseFormat: OpenAIResponseFormat? = nil
    ) -> AsyncThrowingStream<LocalModelRunnerEvent, Error> {
        let lagunaFastPaths = LagunaDecodeFastPathSelection.resolve(
            engine: engine,
            compiledBlockTailOverride: LagunaRuntimeTuning.useCompiledBlockTail,
            fusedRouterTopKOverride: LagunaRuntimeTuning.useFusedRouterTopK)
        let fusedGateUpSilu =
            engine == .metal
            && (LagunaRuntimeTuning.useFusedGateUpSilu
                ?? (ProcessInfo.processInfo.environment["MODEL_RUNNER_LAGUNA_FUSED_GATHER_SILU"] == "1"))
        return AsyncThrowingStream { continuation in
            let generationTask = LagunaRuntimeTuning.$useFusedGateUpSilu.withValue(fusedGateUpSilu) {
                LagunaRuntimeTuning.$useCompiledBlockTail.withValue(
                    lagunaFastPaths.useCompiledBlockTail
                ) {
                    LagunaRuntimeTuning.$useFusedRouterTopK.withValue(
                        lagunaFastPaths.useFusedRouterTopK
                    ) {
                        Task {
                            do {
                                try await generate(
                                    messages: messages,
                                    maximumTokens: maximumTokens,
                                    temperature: temperature,
                                    topP: topP,
                                    stop: stop,
                                    tools: tools,
                                    toolChoice: toolChoice,
                                    reasoningEffort: reasoningEffort,
                                    thinkingEnabled: thinkingEnabled,
                                    enablePromptCache: enablePromptCache,
                                    enableSpeculativeDecoding: enableSpeculativeDecoding,
                                    preparedPrompt: preparedPrompt,
                                    responseFormat: responseFormat
                                ) {
                                    continuation.yield($0)
                                }
                                continuation.finish()
                            } catch {
                                continuation.finish(throwing: error)
                            }
                        }
                    }
                }
            }
            producerLifetime.track(generationTask)
            continuation.onTermination = { _ in generationTask.cancel() }
        }
    }

    private func generate(
        messages: [OpenAIMessage],
        maximumTokens: Int?,
        temperature: Double?,
        topP: Double?,
        stop: [String],
        tools: [OpenAIToolDefinition]?,
        toolChoice: ToolChoicePlan.Constraint,
        reasoningEffort: ChatCompletionRequest.ReasoningEffort?,
        thinkingEnabled: Bool?,
        enablePromptCache: Bool,
        enableSpeculativeDecoding: Bool,
        preparedPrompt: PreparedModelPrompt?,
        responseFormat: OpenAIResponseFormat?,
        onEvent: @escaping @Sendable (LocalModelRunnerEvent) throws -> Void
    ) async throws {
        try await acquireExecution()
        defer { generationAdmission.release() }
        // Claim after admission so concurrent queued requests cannot consume the same arm.
        let recording: InspectorRecordingCapture?
        if let pending = pendingInspectorRecording, pending.store.claim(id: pending.id) {
            recording = InspectorRecordingCapture(recording: pending)
        } else {
            recording = nil
        }
        pendingInspectorRecording = nil
        let monitoring = generationProgress.begin()
        var succeeded = false
        defer {
            if monitoring {
                generationProgress.finish(succeeded: succeeded, cancelled: Task.isCancelled)
            }
            if !succeeded, let recording {
                recording.recording.store.finish(
                    id: recording.recording.id, status: Task.isCancelled ? "cancelled" : "failed",
                    message: "The conversation ended before recording could complete.")
            }
        }
        try Task.checkCancellation()
        try StructuredOutputRequest.validate(format: responseFormat, tools: tools, stop: stop)

        let prepared: PreparedModelPrompt
        let effectiveReasoningEffort = Self.effectiveReasoningEffort(
            reasoningEffort, capabilities: runtimeCapabilities)
        if let preparedPrompt, preparedPrompt.modelPath == modelPath,
            preparedPrompt.messages == messages, preparedPrompt.tools == tools,
            preparedPrompt.toolChoice == toolChoice,
            preparedPrompt.reasoningEffort == effectiveReasoningEffort,
            preparedPrompt.thinkingEnabled == thinkingEnabled,
            preparedPrompt.responseFormat == responseFormat
        {
            prepared = preparedPrompt
        } else {
            prepared = try await renderPrompt(
                messages: messages, tools: tools,
                toolChoice: toolChoice,
                reasoningEffort: effectiveReasoningEffort, responseFormat: responseFormat,
                thinkingEnabled: thinkingEnabled)
        }
        let effectiveMaximumTokens = try tokenLimit.resolve(
            requested: maximumTokens,
            promptTokens: prepared.promptTokenCount, contextLength: contextLength)
        if monitoring {
            generationProgress.prepare(promptTokens: prepared.promptTokenCount, maximumTokens: effectiveMaximumTokens)
        }
        if prepared.structuredOutput != nil || !prepared.forcedToolPrefixTokenIDs.isEmpty {
            // A forced prefix changes generation semantics independently of the
            // rendered transcript. Keep it out of every retained ChatSession path:
            // branch checkpoints are keyed by messages and cannot express that
            // request-local constraint as part of their identity.
            hotConversation = nil
            hotCacheBytes = 0
            conversationCache.removeAll()
        }
        if runtimeCapabilities.isGPTOSS, let hotConversation {
            // Harmony tool continuations can retain private analysis omitted by a
            // cold template render. Bound the actual live timeline plus the entire
            // new render before permitting a suffix-only prefill. Rebuilding drops
            // that unrendered history when the conservative bound cannot fit.
            let processedTokens = try await hotConversation.session.processedTokenCount()
            if !Self.hotConversationFitsContext(
                processedTokenCount: processedTokens,
                renderedPromptTokenCount: prepared.promptTokenCount,
                maximumTokens: effectiveMaximumTokens,
                contextLength: contextLength
            ) {
                self.hotConversation = nil
                hotCacheBytes = 0
            }
        }
        try validateAdmission(
            prompt: prepared.promptTokenCount, output: effectiveMaximumTokens,
            resident: residentModelBytes)
        let required = memoryProfile.requestBytes(
            prompt: prepared.promptTokenCount,
            output: effectiveMaximumTokens,
            prefillStepSize: min(longContext.prefillStepSize, max(1, prepared.promptTokenCount)),
            residentBytes: residentModelBytes)
        // Reserve for both the requested cache and existing retained caches. Evict first;
        // never keep branch snapshots at the expense of admitting a fitting request.
        let spare = max(0, memoryLimitBytes - required)
        if sharedPromptCache.bytes > spare / 2 {
            sharedPromptCache.clear()
        }
        conversationCache.trim(toBytes: max(0, spare - hotCacheBytes - sharedPromptCache.bytes))
        if hotCacheBytes > spare {
            hotConversation = nil
            hotCacheBytes = 0
        }
        // Account for live allocations beyond the model baseline before starting prefill.
        let captureMemory =
            recording.map {
                $0.recording.estimatedBytes * 3
                    + ($0.recording.descriptor.hiddenSize ?? 0)
                    * $0.recording.request.layers.count * $0.recording.request.sites.count * 32
            } ?? 0
        do {
            try validateAdmission(
                prompt: prepared.promptTokenCount, output: effectiveMaximumTokens,
                resident: max(residentModelBytes, Memory.activeMemory) + captureMemory)
        } catch {
            // An armed observer must not turn an otherwise admissible chat into
            // an inference failure. Reject capture and recheck ordinary admission.
            recording?.fail(error)
            try validateAdmission(
                prompt: prepared.promptTokenCount, output: effectiveMaximumTokens,
                resident: max(residentModelBytes, Memory.activeMemory))
        }
        var settings = try generationRequestSettings(
            maximumTokens: effectiveMaximumTokens,
            temperature: temperature,
            topP: topP,
            normalizesGemma4Prompt: normalizesGemma4Prompt,
            forcedTokenPrefix: prepared.forcedToolPrefixTokenIDs
        )
        try longContext.apply(to: &settings.parameters)
        let requestSettings = settings
        let outputValidator = ToolChoiceOutputValidator(constraint: toolChoice, tools: tools)
        let progress = generationProgress
        let validatedOnEvent: @Sendable (LocalModelRunnerEvent) throws -> Void = { event in
            if case .metrics(let metrics) = event {
                self.reportedGeneration.withLock {
                    $0 =
                        metrics.tokensPerSecond.isFinite && metrics.promptTokensPerSecond.isFinite
                            && metrics.tokensPerSecond >= 0 && metrics.promptTokensPerSecond >= 0 ? metrics : nil
                }
                recording?.record(metrics: metrics)
                if monitoring { progress.record(metrics: metrics) }
            }
            try outputValidator.observe(event)
            try onEvent(event)
        }
        let observer: (@Sendable (Int) -> Void)?
        if monitoring {
            observer = { progress.record(tokenCount: $0) }
        } else {
            observer = nil
        }
        let run: @Sendable () async throws -> Void = {
            try await self.withGenerationWiredResidency {
                try await self.withInspectorRecording(recording) { activeRecording in
                    try await self.generateOnSelectedRuntime(
                        messages: messages,
                        effectiveMaximumTokens: effectiveMaximumTokens,
                        temperature: temperature,
                        topP: topP,
                        stop: stop,
                        tools: tools,
                        enablePromptCache: enablePromptCache,
                        // Accepted-token coordinates require target-only decoding.
                        // Recording preserves the caller's sampler, parser, and cache policy.
                        enableSpeculativeDecoding: activeRecording == nil && enableSpeculativeDecoding,
                        settings: requestSettings,
                        prepared: prepared,
                        recording: activeRecording,
                        onEvent: validatedOnEvent)
                    if prepared.nativeProtocol == nil {
                        try outputValidator.validateCompletion()
                    }
                }
            }
        }
        try await GenerationTokenObserver.$onToken.withValue(observer) {
            #if os(macOS) && MODEL_RUNNER_PINNED_MLX
                try await withPinnedMLXRuntime(device: device, stream: mlxStream, operation: run)
            #else
                try await run()
            #endif
        }
        succeeded = true
    }

    private func withInspectorRecording(
        _ recording: InspectorRecordingCapture?,
        operation: @Sendable (InspectorRecordingCapture?) async throws -> Void
    ) async throws {
        guard let recording, recording.recording.store.isRecording(id: recording.recording.id) else {
            try await operation(nil)
            return
        }
        do {
            try await container.perform { context in try recording.install(model: context.model) }
        } catch {
            recording.fail(error)
            try await operation(nil)
            return
        }
        do {
            // Compiled Laguna tails bypass the Swift normalization observers.
            // Disable them only for this explicitly recorded request.
            try await LagunaRuntimeTuning.$useCompiledBlockTail.withValue(false) {
                try await operation(recording)
            }
            await container.perform { context in
                recording.restore()
                recording.finish(tokenizer: context.tokenizer)
            }
        } catch {
            // All generation paths join their producer before returning or throwing.
            await container.perform { context in
                recording.restore()
                recording.finish(tokenizer: context.tokenizer, error: error)
            }
            throw error
        }
    }

    /// Keep residency elevated until the request's producer has synchronized and
    /// learn from the completed request's real process-wide peak. Manual awaited
    /// teardown is intentional: the generic cancellation helper may end its ticket
    /// before this runner has cancelled and joined the GPU producer.
    private func withGenerationWiredResidency<Result: Sendable>(
        _ operation: () async throws -> Result
    ) async throws -> Result {
        guard let plan = wiredMemoryPlan else {
            return try await operation()
        }

        let ticket = plan.makeTicket()
        let appliedLimit = await ticket.start()
        if benchmarkWiredMemoryRequests != nil {
            let state = await BenchmarkWiredMemoryState.capture()
            benchmarkWiredMemoryRequests?.append(
                BenchmarkWiredMemoryRequest(
                    requestedLimitBytes: plan.limitBytes,
                    startReturnedLimitBytes: appliedLimit, activeState: state))
        }
        guard appliedLimit >= plan.limitBytes else {
            _ = await ticket.end()
            print(
                "MLX wired-memory request was not applied "
                    + "(requested=\(plan.limitBytes), applied=\(appliedLimit)); continuing unwired"
            )
            return try await operation()
        }

        Memory.peakMemory = 0
        do {
            let result = try await operation()
            let observedPeak = Memory.peakMemory
            _ = await ticket.end()
            wiredMemoryPlan?.observe(peakActiveBytes: observedPeak)
            return result
        } catch {
            let observedPeak = Memory.peakMemory
            _ = await ticket.end()
            wiredMemoryPlan?.observe(peakActiveBytes: observedPeak)
            throw error
        }
    }

    private func generateOnSelectedRuntime(
        messages: [OpenAIMessage],
        effectiveMaximumTokens: Int,
        temperature: Double?,
        topP: Double?,
        stop: [String],
        tools: [OpenAIToolDefinition]?,
        enablePromptCache: Bool,
        enableSpeculativeDecoding: Bool,
        settings: GenerationRequestSettings,
        prepared: PreparedModelPrompt,
        recording: InspectorRecordingCapture?,
        onEvent: @Sendable (LocalModelRunnerEvent) throws -> Void
    ) async throws {
        var settings = settings
        if let recording { settings.components = recording.components(settings.components) }
        if let nativeProtocol = prepared.nativeProtocol {
            hotConversation = nil
            hotCacheBytes = 0
            try await generateNativeOnDevice(
                container: container, device: device,
                settings: settings, preparedTokenIDs: prepared.tokenIDs,
                protocolName: nativeProtocol,
                thinkingEnabled: prepared.toolChoice.requiresToolCall ? false : prepared.thinkingEnabled,
                onEvent: onEvent)
            return
        }
        if let structuredOutput = prepared.structuredOutput {
            try await generateStructuredOnDevice(
                container: container, device: device,
                settings: settings, preparedTokenIDs: prepared.tokenIDs,
                processor: structuredOutput.copy(), onEvent: onEvent)
            return
        }
        let hasForcedToolPrefix = !prepared.forcedToolPrefixTokenIDs.isEmpty
        let usesDFlash =
            enableSpeculativeDecoding && dflash != nil
            && !hasForcedToolPrefix
        let cacheRoute = Self.promptCacheRoute(
            supportsLagunaPromptCache: supportsLagunaPromptCache,
            capabilities: runtimeCapabilities,
            enablePromptCache: enablePromptCache,
            normalizesGemma4Prompt: normalizesGemma4Prompt,
            hasCustomStopStrings: !stop.isEmpty,
            usesDFlash: usesDFlash,
            hasTools: tools?.isEmpty == false,
            hasForcedToolPrefix: hasForcedToolPrefix,
            hasSharedPrefix: sharedPromptCache.maximumBytes > 0
                && kvCompression == "none" && SharedPromptCache.prefixLength(prepared.tokenIDs) > 0
        )
        if cacheRoute == .lagunaConversation {
            try await generateWithLagunaPromptCache(
                messages: messages,
                settings: settings,
                tools: tools,
                allowsReuse: enablePromptCache,
                thinkingEnabled: prepared.thinkingEnabled,
                onEvent: onEvent
            )
            return
        }
        if cacheRoute == .hotConversation {
            try await generateWithHotConversationCache(
                messages: messages,
                settings: settings,
                tools: tools,
                reasoningEffort: prepared.reasoningEffort,
                allowsReuse: enablePromptCache,
                onEvent: onEvent
            )
            return
        }

        // The one-shot paths can mutate cache state independently (custom stop
        // strings, Gemma prompt normalization, or Laguna's custom DFlash iterator).
        // Do not retain a session across either side of that boundary.
        hotConversation = nil
        hotCacheBytes = 0
        try await generateOnDevice(
            container: container,
            device: device,
            messages: messages,
            maximumTokens: effectiveMaximumTokens,
            temperature: temperature,
            topP: topP,
            stop: stop,
            tools: tools,
            normalizesGemma4Prompt: normalizesGemma4Prompt,
            dflash: dflash,
            enableSpeculativeDecoding: enableSpeculativeDecoding
                && prepared.forcedToolPrefixTokenIDs.isEmpty,
            preparedTokenIDs: prepared.tokenIDs,
            forcedToolPrefixTokenIDs: prepared.forcedToolPrefixTokenIDs,
            longContext: longContext,
            sharedPromptCache: cacheRoute == .sharedPrefix ? sharedPromptCache : nil,
            memoryLimitBytes: memoryLimitBytes,
            recording: recording,
            onEvent: onEvent
        )
    }

    enum PromptCacheRoute: Equatable, Sendable {
        case lagunaConversation, hotConversation, sharedPrefix, uncached
    }

    /// Establish the full conversation session on the first eligible turn. A
    /// long prompt must not divert it into partial shared checkpoints instead.
    /// This planner also preserves the model-specific one-shot escape hatches.
    static func promptCacheRoute(
        supportsLagunaPromptCache: Bool,
        capabilities: ModelRuntimeCapabilities,
        enablePromptCache: Bool,
        normalizesGemma4Prompt: Bool,
        hasCustomStopStrings: Bool,
        usesDFlash: Bool,
        hasTools: Bool,
        hasForcedToolPrefix: Bool,
        hasSharedPrefix: Bool
    ) -> PromptCacheRoute {
        // Laguna uses ChatSession even with reuse disabled, retaining its existing
        // model-specific execution path without retaining a reusable conversation.
        if supportsLagunaPromptCache && !normalizesGemma4Prompt
            && !hasCustomStopStrings && !usesDFlash && !hasForcedToolPrefix
        {
            return .lagunaConversation
        }
        if shouldUseHotConversationCache(
            capabilities: capabilities, enablePromptCache: enablePromptCache,
            normalizesGemma4Prompt: normalizesGemma4Prompt,
            hasCustomStopStrings: hasCustomStopStrings, usesDFlash: usesDFlash,
            hasTools: hasTools, hasForcedToolPrefix: hasForcedToolPrefix
        ) {
            return .hotConversation
        }
        if enablePromptCache && hasSharedPrefix && !usesDFlash
            && !hasForcedToolPrefix
        {
            // This cache consumes preparedTokenIDs after normalization. Unlike
            // ChatSession it never re-renders Gemma's thinking/tool template.
            return .sharedPrefix
        }
        return .uncached
    }

    /// Returns the first uncached message only when `incoming` is a strict
    /// extension of the transcript committed after the previous successful turn.
    /// Keeping this planner pure makes edited/branched conversation behavior
    /// independently testable without loading a model.
    static func shouldUseMistralPromptCache(
        capabilities: ModelRuntimeCapabilities,
        enablePromptCache: Bool,
        normalizesGemma4Prompt: Bool,
        hasCustomStopStrings: Bool,
        usesDFlash: Bool
    ) -> Bool {
        capabilities.supportsMistralConversationPrefixCache
            && shouldUseHotConversationCache(
                capabilities: capabilities,
                enablePromptCache: enablePromptCache,
                normalizesGemma4Prompt: normalizesGemma4Prompt,
                hasCustomStopStrings: hasCustomStopStrings,
                usesDFlash: usesDFlash
            )
    }

    /// The session validates actual rendered tokens before reusing KV state.
    /// GPT-OSS Harmony tool restarts can retain generated analysis. Ordinary
    /// GPT-OSS text follows the one-shot path because removing that analysis
    /// from completed-turn history prevents useful rotating-cache reuse.
    static func shouldUseHotConversationCache(
        capabilities: ModelRuntimeCapabilities,
        enablePromptCache: Bool,
        normalizesGemma4Prompt: Bool,
        hasCustomStopStrings: Bool,
        usesDFlash: Bool,
        hasTools: Bool = false,
        hasForcedToolPrefix: Bool = false
    ) -> Bool {
        enablePromptCache
            && capabilities.supportsHotConversationCache
            && (!capabilities.isGPTOSS || hasTools)
            && !normalizesGemma4Prompt
            && !hasCustomStopStrings
            && !usesDFlash
            && !hasForcedToolPrefix
    }

    static func cachedConversationSuffixStart(
        committed: [OpenAIMessage],
        incoming: [OpenAIMessage],
        committedTools: [OpenAIToolDefinition]? = nil,
        tools: [OpenAIToolDefinition]? = nil,
        committedReasoningEffort: ChatCompletionRequest.ReasoningEffort? = nil,
        reasoningEffort: ChatCompletionRequest.ReasoningEffort? = nil
    ) -> Int? {
        guard committedTools == tools, committedReasoningEffort == reasoningEffort,
            incoming.count > committed.count, incoming.starts(with: committed)
        else {
            return nil
        }
        return committed.count
    }

    static func hotConversationFitsContext(
        processedTokenCount: Int?, renderedPromptTokenCount: Int,
        maximumTokens: Int, contextLength: Int
    ) -> Bool {
        guard let processedTokenCount, processedTokenCount >= 0,
            renderedPromptTokenCount > 0, maximumTokens > 0
        else {
            return false
        }
        let upperBound = ModelMemoryProfile.add(
            processedTokenCount,
            ModelMemoryProfile.add(renderedPromptTokenCount, maximumTokens))
        return upperBound <= contextLength
    }

    /// Select the deepest immutable conversation checkpoint that is a strict
    /// prefix of the incoming transcript. Keeping older checkpoints enables
    /// correct branching after a later turn has already been cached.
    static func longestCachedConversationPrefixIndex(
        committed candidates: [[OpenAIMessage]],
        incoming: [OpenAIMessage]
    ) -> Int? {
        candidates.indices
            .filter {
                cachedConversationSuffixStart(
                    committed: candidates[$0],
                    incoming: incoming
                ) != nil
            }
            .max { candidates[$0].count < candidates[$1].count }
    }

    private func generateWithLagunaPromptCache(
        messages: [OpenAIMessage],
        settings: GenerationRequestSettings,
        tools: [OpenAIToolDefinition]?,
        allowsReuse: Bool,
        thinkingEnabled: Bool?,
        onEvent: @Sendable (LocalModelRunnerEvent) throws -> Void
    ) async throws {
        try await generateWithConversationPrefixCache(
            messages: messages,
            settings: settings,
            tools: tools,
            thinkingEnabled: thinkingEnabled,
            allowsReuse: allowsReuse && thinkingEnabled == nil,
            retainsBranchSnapshots: true,
            onEvent: onEvent
        )
    }

    private func generateWithHotConversationCache(
        messages: [OpenAIMessage],
        settings: GenerationRequestSettings,
        tools: [OpenAIToolDefinition]?,
        reasoningEffort: ChatCompletionRequest.ReasoningEffort?,
        allowsReuse: Bool,
        onEvent: @Sendable (LocalModelRunnerEvent) throws -> Void
    ) async throws {
        try await generateWithConversationPrefixCache(
            messages: messages,
            settings: settings,
            tools: tools,
            reasoningEffort: reasoningEffort,
            allowsReuse: allowsReuse,
            retainsBranchSnapshots: false,
            onEvent: onEvent
        )
    }

    private func generateWithConversationPrefixCache(
        messages: [OpenAIMessage],
        settings: GenerationRequestSettings,
        tools: [OpenAIToolDefinition]?,
        reasoningEffort: ChatCompletionRequest.ReasoningEffort? = nil,
        thinkingEnabled: Bool? = nil,
        allowsReuse: Bool,
        retainsBranchSnapshots: Bool,
        onEvent: @Sendable (LocalModelRunnerEvent) throws -> Void
    ) async throws {
        guard let final = messages.last, final.role == "user" || final.role == "tool" else {
            throw LocalModelRunnerError.lastMessageMustBeUserOrTool
        }

        let chatMessages = try messages.map(Self.chatMessage)
        let toolSpecs = try Self.toolSpecs(tools)
        let additionalContext = Self.promptAdditionalContext(
            reasoningEffort: reasoningEffort, thinkingEnabled: thinkingEnabled,
            preserveThinking: supportsLagunaPromptCache)
        let session: ChatSessionReference
        let pendingMessages: [Chat.Message]
        let reusedHotConversation: Bool
        if allowsReuse, let hotConversation,
            let suffixStart = Self.cachedConversationSuffixStart(
                committed: hotConversation.committedMessages,
                incoming: messages,
                committedTools: hotConversation.tools,
                tools: tools,
                committedReasoningEffort: hotConversation.reasoningEffort,
                reasoningEffort: reasoningEffort)
        {
            session = hotConversation.session
            pendingMessages = Array(chatMessages.dropFirst(suffixStart))
            session.configure(
                parameters: settings.parameters,
                components: settings.components,
                additionalContext: additionalContext,
                tools: toolSpecs
            )
            reusedHotConversation = true
        } else if allowsReuse, retainsBranchSnapshots,
            let cached = conversationCache.longestPrefix(of: messages)
        {
            session = ChatSessionReference(
                cached.value.restore(
                    container: container,
                    parameters: settings.parameters,
                    components: settings.components,
                    additionalContext: additionalContext,
                    tools: toolSpecs
                )
            )
            pendingMessages = Array(chatMessages.dropFirst(cached.suffixStart))
            reusedHotConversation = false
        } else {
            session = ChatSessionReference(
                ChatSession(
                    container,
                    history: Array(chatMessages.dropLast()),
                    generateParameters: settings.parameters,
                    components: settings.components,
                    additionalContext: additionalContext,
                    tools: toolSpecs
                )
            )
            pendingMessages = [chatMessages[chatMessages.index(before: chatMessages.endIndex)]]
            reusedHotConversation = false
        }

        var content = ""
        var reasoning = ""
        var generatedToolCalls: [OpenAIToolCall] = []
        var canRetainSession = true
        var producedOutput = false
        do {
            // ChatSession creates its producer task synchronously here. Creating it
            // inside the device scope makes that task inherit the selected backend.
            let stream = Device.withDefaultDevice(device) {
                session.streamDetails(to: pendingMessages)
            }
            for try await event in stream {
                try Task.checkCancellation()
                switch event {
                case .reasoning(let text):
                    reasoning += text
                    // Muse Glimmer deliberately begins every turn in its private
                    // `to=self` channel.  A completed reasoning-only turn is still a
                    // model response and must not be rewritten as an empty-generation
                    // server failure before the HTTP layer can return it.
                    producedOutput = true
                    try onEvent(.reasoning(text))
                case .chunk(let chunk):
                    producedOutput = true
                    content += chunk
                    try onEvent(.content(chunk))
                case .info(let info):
                    try onEvent(.metrics(Self.metrics(info)))
                case .toolCall(let call):
                    producedOutput = true
                    // The public OpenAI surface synthesizes an ID when MLX omits one.
                    // Such a transcript would no longer exactly match ChatSession's
                    // internal history, so serve it but rebuild on the next turn.
                    if call.id?.nilIfBlank == nil {
                        canRetainSession = false
                    }
                    let converted = try Self.openAIToolCall(call)
                    generatedToolCalls.append(converted)
                    try onEvent(.toolCall(converted))
                case .rejectedToolCall(let rejection):
                    throw RejectedToolCallError(rejection)
                }
            }
            try Task.checkCancellation()
            await session.synchronize()
        } catch {
            await session.synchronize()
            if reusedHotConversation {
                hotConversation = nil
                hotCacheBytes = 0
            }
            throw error
        }

        guard producedOutput else {
            if reusedHotConversation {
                hotConversation = nil
                hotCacheBytes = 0
            }
            throw LocalModelRunnerError.emptyResponse
        }
        if runtimeCapabilities.isGPTOSS && (generatedToolCalls.isEmpty || !canRetainSession) {
            // Only a live tool restart can preserve Harmony's hidden analysis.
            // Retaining a final text reply would keep KV allocations that the next
            // ordinary turn must rebuild from its cold transcript anyway.
            hotConversation = nil
            hotCacheBytes = 0
            return
        }
        guard canRetainSession else {
            if reusedHotConversation {
                hotConversation = nil
                hotCacheBytes = 0
            }
            return
        }
        guard allowsReuse else {
            return
        }
        let assistant = OpenAIMessage(
            role: "assistant",
            content: content.isEmpty ? nil : content,
            toolCalls: generatedToolCalls.isEmpty ? nil : generatedToolCalls,
            reasoningContent: reasoning.isEmpty ? nil : reasoning
        )
        hotConversation = HotConversation(
            session: session,
            committedMessages: messages + [assistant],
            tools: tools,
            reasoningEffort: reasoningEffort
        )
        // A snapshot deep-copies and evaluates every KV array. Keep Laguna's
        // established branchable LRU unchanged, but make the shared text fast
        // path zero-copy: the hot session handles compatible append-only chat
        // without adding a potentially GiB-scale copy after each response.
        hotCacheBytes = await session.cacheMemoryBytes()
        guard hotCacheBytes <= conversationCache.maximumBytes else {
            hotConversation = nil
            hotCacheBytes = 0
            conversationCache.removeAll()
            return
        }
        conversationCache.trim(toBytes: conversationCache.maximumBytes - hotCacheBytes)
        guard retainsBranchSnapshots else {
            return
        }
        guard conversationCache.canStore(costBytes: hotCacheBytes, reservedBytes: hotCacheBytes),
            hotCacheBytes <= max(0, memoryLimitBytes - Memory.activeMemory - 512 * 1_048_576)
        else {
            return
        }
        // Evict BEFORE allocating the independent snapshot, accounting for the hot cache too.
        conversationCache.trim(toBytes: conversationCache.maximumBytes - 2 * hotCacheBytes)
        let snapshot = try await session.snapshot()
        conversationCache.insert(
            snapshot,
            committedMessages: messages + [assistant],
            costBytes: snapshot.estimatedBytes
        )
    }

    fileprivate static func metrics(_ info: GenerateCompletionInfo) -> LocalModelRunnerMetrics {
        LocalModelRunnerMetrics(
            promptTokenCount: info.totalPromptTokenCount,
            prefilledPromptTokenCount: info.promptTokenCount,
            cachedPromptTokenCount: info.cachedPromptTokenCount,
            generationTokenCount: info.generationTokenCount,
            promptTokensPerSecond: info.promptTokensPerSecond,
            tokensPerSecond: info.tokensPerSecond,
            stopReason: String(describing: info.stopReason),
            proposedDraftTokens: info.proposedDraftTokens,
            acceptedDraftTokens: info.acceptedDraftTokens,
            speculativePassthroughReason: info.passthroughReason
        )
    }

    static func chatMessage(_ message: OpenAIMessage) throws -> Chat.Message {
        switch message.role {
        case "user":
            guard let content = message.content else {
                throw LocalModelRunnerError.missingMessageContent("user")
            }
            return .user(content)
        case "assistant":
            let toolCalls = try message.toolCalls?.map(mlxToolCall)
            guard message.content != nil || toolCalls?.isEmpty == false else {
                throw LocalModelRunnerError.missingMessageContent("assistant")
            }
            var result = Chat.Message.assistant(message.content ?? "", toolCalls: toolCalls)
            result.reasoningContent = message.reasoningContent
            return result
        case "system", "developer":
            guard let content = message.content else {
                throw LocalModelRunnerError.missingMessageContent(message.role)
            }
            return .system(content)
        case "tool":
            guard let content = message.content else {
                throw LocalModelRunnerError.missingMessageContent("tool")
            }
            guard let callID = message.toolCallID?.nilIfBlank else {
                throw LocalModelRunnerError.missingToolCallID
            }
            return .tool(content, id: callID, name: message.name?.nilIfBlank)
        default: throw LocalModelRunnerError.unsupportedRole(message.role)
        }
    }

    fileprivate static func toolSpecs(
        _ definitions: [OpenAIToolDefinition]?
    ) throws -> [ToolSpec]? {
        guard let definitions else {
            return nil
        }
        guard !definitions.isEmpty else {
            return []
        }
        var names = Set<String>()
        return try definitions.map { definition in
            guard definition.type == "function" else {
                throw LocalModelRunnerError.unsupportedToolType(definition.type)
            }
            let name = definition.function.name.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else {
                throw LocalModelRunnerError.missingToolName
            }
            guard names.insert(name).inserted else {
                throw LocalModelRunnerError.duplicateToolName(name)
            }
            guard case .object = definition.function.parameters else {
                throw LocalModelRunnerError.invalidToolParameters(name)
            }

            var function: [String: any Sendable] = [
                "name": name,
                "parameters": definition.function.parameters.sendableValue,
            ]
            if let description = definition.function.description?.nilIfBlank {
                function["description"] = description
            }
            return [
                "type": "function",
                "function": function,
            ]
        }
    }

    private static func mlxToolCall(_ call: OpenAIToolCall) throws -> MLXLMCommon.ToolCall {
        guard call.type == "function" else {
            throw LocalModelRunnerError.unsupportedToolType(call.type)
        }
        let arguments: [String: MLXLMCommon.JSONValue]
        do {
            arguments = try JSONDecoder().decode(
                [String: MLXLMCommon.JSONValue].self,
                from: Data(call.function.arguments.utf8)
            )
        } catch {
            throw LocalModelRunnerError.invalidToolCallArguments(call.id)
        }
        return MLXLMCommon.ToolCall(
            function: .init(name: call.function.name, arguments: arguments),
            id: call.id
        )
    }

    fileprivate static func openAIToolCall(_ call: MLXLMCommon.ToolCall) throws -> OpenAIToolCall {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let arguments = String(
            decoding: try encoder.encode(call.function.arguments),
            as: UTF8.self
        )
        return OpenAIToolCall(
            id: call.id?.nilIfBlank
                ?? "call_\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))",
            function: .init(name: call.function.name, arguments: arguments)
        )
    }

    private static func validateModelFolder(_ folder: URL) throws {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: folder.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw LocalModelRunnerError.missingDirectory(folder.path)
        }

        guard
            FileManager.default.fileExists(
                atPath: folder.appendingPathComponent("config.json").path
            )
        else {
            throw LocalModelRunnerError.missingConfig(folder.path)
        }

        let files = try FileManager.default.contentsOfDirectory(
            at: folder,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        guard files.contains(where: { $0.pathExtension == "safetensors" }) else {
            throw LocalModelRunnerError.missingWeights(folder.path)
        }
    }

    private static func isGemma4Model(_ folder: URL) throws -> Bool {
        let data = try Data(contentsOf: folder.appendingPathComponent("config.json"))
        guard
            let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let modelType = root["model_type"] as? String
        else {
            return false
        }
        return modelType == "gemma4" || modelType == "gemma4_text"
    }

    private static func loadAdapter(
        directory: String,
        scaleOverride: Float?
    ) throws -> LoRAContainer {
        let expandedPath = NSString(string: directory).expandingTildeInPath
        let url = URL(fileURLWithPath: expandedPath, isDirectory: true)
            .standardizedFileURL
            .resolvingSymlinksInPath()
        let original = try LoRAContainer.from(directory: url)
        guard let scaleOverride else {
            return original
        }
        guard scaleOverride.isFinite, scaleOverride >= 0 else {
            throw LocalModelRunnerError.invalidAdapterScale(scaleOverride)
        }
        let source = original.configuration
        let parameters = source.loraParameters
        let configuration = LoRAConfiguration(
            numLayers: source.numLayers,
            fineTuneType: source.fineTuneType,
            loraParameters: LoRAConfiguration.LoRAParameters(
                rank: parameters.rank,
                scale: scaleOverride,
                dropout: parameters.dropout,
                keys: parameters.keys
            )
        )
        return LoRAContainer(
            configuration: configuration,
            parameters: original.parameters
        )
    }
}

private struct GenerationRequestSettings: Sendable {
    let temperature: Double
    var parameters: GenerateParameters
    var components: GenerationComponents
}

/// Native clients own response framing, reasoning separation and call parsing.
private func generateNativeOnDevice(
    container: ModelContainer, device: Device, settings: GenerationRequestSettings,
    preparedTokenIDs: [Int], protocolName: String, thinkingEnabled: Bool?,
    onEvent: @Sendable (LocalModelRunnerEvent) throws -> Void
) async throws {
    try await Device.withDefaultDevice(device) {
        let (stream, producer, tokenizer) = try await container.perform(
            nonSendable: LMInput(tokens: MLXArray(preparedTokenIDs))
        ) { context, input in
            let result = try MLXLMCommon.generateTokensTask(
                input: input,
                parameters: settings.parameters, context: context, includeStopToken: true,
                components: settings.components)
            return (result.0, result.1, context.tokenizer)
        }
        var decoder = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        do {
            try onEvent(
                .content(
                    protocolName == "muse"
                        ? "<|start|>assistant"
                        : "<assistant>" + (thinkingEnabled == false ? "</think>" : "<think>")))
            for await event in stream {
                try Task.checkCancellation()
                switch event {
                case .token(let token):
                    decoder.append(token: token)
                    while let text = decoder.next() {
                        if !text.isEmpty {
                            try onEvent(.content(text))
                        }
                    }
                case .info(let info): try onEvent(.metrics(LocalModelRunner.metrics(info)))
                }
            }
            await producer.value
            try Task.checkCancellation()
        } catch {
            producer.cancel()
            await producer.value
            throw error
        }
    }
}

/// Raw-token delivery keeps JSON strings away from model-specific tool/reasoning
/// parsers. The token mask's byte map is also used for output, so split UTF-8 and
/// tokenizer whitespace cleanup cannot make the wire text differ from the grammar.
private func generateStructuredOnDevice(
    container: ModelContainer,
    device: Device,
    settings: GenerationRequestSettings,
    preparedTokenIDs: [Int],
    processor: StructuredOutputLogitProcessor,
    onEvent: @Sendable (LocalModelRunnerEvent) throws -> Void
) async throws {
    try await Device.withDefaultDevice(device) {
        let components = settings.components.appendingLogitProcessor { processor }
        let (stream, producerTask) = try await container.perform(
            nonSendable: LMInput(tokens: MLXArray(preparedTokenIDs))
        ) { context, input in
            try MLXLMCommon.generateTokensTask(
                input: input, parameters: settings.parameters,
                context: context, components: components)
        }
        var pendingBytes: [UInt8] = []
        var text = ""
        var completionInfo: GenerateCompletionInfo?
        do {
            for await event in stream {
                try Task.checkCancellation()
                switch event {
                case .token(let token):
                    guard let bytes = processor.bytes(for: token) else {
                        throw StructuredOutputRequestError.generation("No decoded bytes for token \(token).")
                    }
                    pendingBytes.append(contentsOf: bytes)
                    if let chunk = String(bytes: pendingBytes, encoding: .utf8) {
                        text += chunk
                        pendingBytes.removeAll(keepingCapacity: true)
                        if !chunk.isEmpty {
                            try onEvent(.content(chunk))
                        }
                    }
                case .info(let info):
                    completionInfo = info
                }
            }
            await producerTask.value
            try Task.checkCancellation()
            try processor.throwIfFailed()
            guard let completionInfo else {
                throw StructuredOutputRequestError.generation("Missing completion status.")
            }
            // OpenAI permits incomplete JSON when the output budget is exhausted;
            // preserve finish_reason=length, never report that as a completed object.
            if completionInfo.stopReason != .length {
                guard pendingBytes.isEmpty, processor.isComplete,
                    let format = try? JSONSerialization.jsonObject(with: Data(text.utf8)),
                    format is [String: Any]
                else {
                    throw StructuredOutputRequestError.generation(
                        "The model stopped before completing its JSON object.")
                }
            }
            try onEvent(.metrics(LocalModelRunner.metrics(completionInfo)))
        } catch {
            producerTask.cancel()
            await producerTask.value
            throw error
        }
    }
}

private func generationRequestSettings(
    maximumTokens: Int,
    temperature requestedTemperature: Double?,
    topP requestedTopP: Double?,
    normalizesGemma4Prompt: Bool,
    forcedTokenPrefix: [[Int]] = []
) throws -> GenerationRequestSettings {
    let temperature = requestedTemperature ?? 1.0
    guard temperature.isFinite, temperature >= 0 else {
        throw LocalModelRunnerError.invalidTemperature(temperature)
    }
    let topP = requestedTopP ?? 0.95
    guard topP.isFinite, topP > 0, topP <= 1 else {
        throw LocalModelRunnerError.invalidTopP(topP)
    }
    let parameters = GenerateParameters(
        maxTokens: maximumTokens,
        // Gemma 4's generation_config.json specifies sampling. Greedy decoding
        // can select <pad> mid-response even when useful continuations remain.
        temperature: Float(temperature),
        topP: Float(topP),
        topK: 64
    )
    var components =
        normalizesGemma4Prompt
        ? GenerationComponents(
            logitProcessorFactory: { SuppressTokenLogitProcessor(tokenID: 0) }
        )
        : GenerationComponents()
    if !forcedTokenPrefix.isEmpty {
        components = components.appendingLogitProcessor {
            ForcedTokenPrefixLogitProcessor(tokenSequences: forcedTokenPrefix)
        }
    }
    return GenerationRequestSettings(
        temperature: temperature,
        parameters: parameters,
        components: components
    )
}

private func generateOnDevice(
    container: ModelContainer,
    device: Device,
    messages: [OpenAIMessage],
    maximumTokens: Int,
    temperature requestedTemperature: Double?,
    topP requestedTopP: Double?,
    stop: [String],
    tools: [OpenAIToolDefinition]?,
    normalizesGemma4Prompt: Bool,
    dflash: LoadedSpeculativeModel?,
    enableSpeculativeDecoding: Bool,
    preparedTokenIDs: [Int],
    forcedToolPrefixTokenIDs: [[Int]],
    longContext: LongContextOptions,
    sharedPromptCache: SharedPromptCache?,
    memoryLimitBytes: Int,
    recording: InspectorRecordingCapture?,
    onEvent: @Sendable (LocalModelRunnerEvent) throws -> Void
) async throws {
    try await Device.withDefaultDevice(device) {
        // Generation is actor-serialized, so the device's persistent default stream
        // is sufficient. Creating a fresh MLX stream for every request leaves a
        // busy CUDA worker behind on Linux and progressively steals CPU from decode.
        guard let final = messages.last, final.role == "user" || final.role == "tool" else {
            throw LocalModelRunnerError.lastMessageMustBeUserOrTool
        }

        let toolSpecs = try LocalModelRunner.toolSpecs(tools)
        let promptTokenIDs = preparedTokenIDs
        if ProcessInfo.processInfo.environment["MODEL_RUNNER_DEBUG_PROMPT_TOKENS"] == "1" {
            print("Prompt token IDs (\(promptTokenIDs.count)): \(promptTokenIDs)")
        }
        var settings = try generationRequestSettings(
            maximumTokens: maximumTokens,
            temperature: requestedTemperature,
            topP: requestedTopP,
            normalizesGemma4Prompt: normalizesGemma4Prompt,
            forcedTokenPrefix: forcedToolPrefixTokenIDs
        )
        try longContext.apply(to: &settings.parameters)
        if let recording { settings.components = recording.components(settings.components) }
        let requestSettings = settings
        // The MTP iterator samples every verifier position from the target in
        // sequence when temperature is nonzero. A draft is accepted only when it
        // matches that target sample, so sampling remains target-defined without
        // requiring clients to force temperature to zero.
        let useDFlash =
            enableSpeculativeDecoding && dflash != nil
        let promptTokenCount = promptTokenIDs.count
        let (stream, producerTask, cachedCount, prefixTime) = try await container.perform(
            nonSendable: LMInput(tokens: MLXArray(promptTokenIDs))
        ) { context, input in
            var requestContext = context
            if !stop.isEmpty {
                requestContext.configuration.stopStrings =
                    context.configuration.effectiveStopStrings.union(stop)
            }
            if useDFlash, let dflash {
                let iterator = try MTPSpeculativeTokenIterator(
                    input: input,
                    mainModel: requestContext.model,
                    drafter: dflash.model.model,
                    parameters: requestSettings.parameters,
                    blockSize: dflash.blockSize,
                    components: requestSettings.components
                )
                let result = MLXLMCommon.generateTask(
                    promptTokenCount: promptTokenCount,
                    modelConfiguration: requestContext.configuration,
                    tokenizer: requestContext.tokenizer,
                    iterator: iterator,
                    tools: toolSpecs
                )
                return (result.0, result.1, 0, 0.0)
            }
            let prefixStarted = Date.timeIntervalSinceReferenceDate
            let prefix = try sharedPromptCache?.prepare(
                tokens: promptTokenIDs, model: requestContext.model, parameters: requestSettings.parameters,
                memoryLimitBytes: memoryLimitBytes)
            let prefixTime = Date.timeIntervalSinceReferenceDate - prefixStarted
            let skipped = prefix?.count ?? 0
            let iterator = try TokenIterator(
                input: skipped > 0 ? LMInput(tokens: MLXArray(Array(promptTokenIDs.dropFirst(skipped)))) : input,
                model: requestContext.model,
                cache: prefix?.cache,
                parameters: requestSettings.parameters,
                components: requestSettings.components
            )
            let result = MLXLMCommon.generateTask(
                promptTokenCount: promptTokenCount,
                modelConfiguration: requestContext.configuration,
                tokenizer: requestContext.tokenizer,
                iterator: iterator,
                tools: toolSpecs
            )
            return (result.0, result.1, prefix?.cachedCount ?? 0, prefixTime)
        }

        var producedOutput = false
        do {
            for await event in stream {
                try Task.checkCancellation()
                switch event {
                case .reasoning(let text):
                    // Muse begins responses in its `to=self` reasoning channel.  That
                    // is still a valid model response even if it contains no final
                    // user-channel chunk.
                    producedOutput = true
                    try onEvent(.reasoning(text))
                case .chunk(let chunk):
                    producedOutput = true
                    try onEvent(.content(chunk))
                case .info(let info):
                    try onEvent(
                        .metrics(
                            LocalModelRunnerMetrics(
                                promptTokenCount: promptTokenCount,
                                prefilledPromptTokenCount: promptTokenCount - cachedCount,
                                cachedPromptTokenCount: cachedCount, generationTokenCount: info.generationTokenCount,
                                promptTokensPerSecond: Double(promptTokenCount - cachedCount)
                                    / max(0.000001, info.promptTime + prefixTime),
                                tokensPerSecond: info.tokensPerSecond,
                                stopReason: String(describing: info.stopReason),
                                proposedDraftTokens: info.proposedDraftTokens,
                                acceptedDraftTokens: info.acceptedDraftTokens,
                                speculativePassthroughReason: info.passthroughReason)
                        )
                    )
                case .toolCall(let call):
                    producedOutput = true
                    try onEvent(.toolCall(try LocalModelRunner.openAIToolCall(call)))
                case .rejectedToolCall(let rejection):
                    throw RejectedToolCallError(rejection)
                }
            }
        } catch {
            // The producer owns speculative iterator finalization and the final GPU
            // synchronization. Do not release actor serialization until both finish.
            producerTask.cancel()
            await producerTask.value
            throw error
        }
        await producerTask.value
        guard producedOutput else {
            throw LocalModelRunnerError.emptyResponse
        }
    }
}

struct ForcedTokenPrefixLogitProcessor: LogitProcessor {
    let tokenSequences: [[Int]]
    private var candidates: [[Int]]
    private var nextIndex = 0

    init(tokenIDs: [Int]) {
        self.init(tokenSequences: [tokenIDs])
    }

    init(tokenSequences: [[Int]]) {
        self.tokenSequences = tokenSequences
        self.candidates = tokenSequences
    }

    mutating func prompt(_ prompt: MLXArray) {
        candidates = tokenSequences
        nextIndex = 0
    }

    func process(logits: MLXArray) -> MLXArray {
        guard !candidates.contains(where: { $0.count == nextIndex }) else {
            return logits
        }
        let vocabularyIDs = arange(logits.dim(-1), dtype: .int32)
        var allowed = vocabularyIDs .== Int32(-1)
        for token in Set(candidates.compactMap({ nextIndex < $0.count ? $0[nextIndex] : nil })) {
            allowed = logicalOr(allowed, vocabularyIDs .== Int32(token))
        }
        let selectedTokenMask = broadcast(allowed, to: logits.shape)
        let suppressed = MLXArray(-Float.infinity).asType(logits.dtype)
        // Preserve the selected model logits, both to let the model choose among
        // declared tools and to evaluate the lazy model/KV-cache graph normally.
        return which(selectedTokenMask, logits, suppressed)
    }

    mutating func didSample(token: MLXArray) {
        guard !candidates.contains(where: { $0.count == nextIndex }) else {
            return
        }
        let sampled = token.item(Int.self)
        candidates = candidates.filter { nextIndex < $0.count && $0[nextIndex] == sampled }
        nextIndex += 1
    }
}

struct SuppressTokenLogitProcessor: MTPStatelessLogitProcessor {
    let tokenID: Int
    private let eosTokenID = 1

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        let vocabularyIDs = arange(logits.dim(-1), dtype: .int32)
        let suppressed = MLXArray(-Float.infinity).asType(logits.dtype)
        let finite = isFinite(logits)
        let sanitized = which(finite, logits, suppressed)
        let withoutPad = which(vocabularyIDs .== Int32(tokenID), suppressed, sanitized)
        // If the backend produces an entirely non-finite logit row, terminate at
        // the checkpoint's real EOS rather than letting argmax emit token 0 forever.
        let eosOnly = which(
            vocabularyIDs .== Int32(eosTokenID),
            MLXArray(0 as Float).asType(logits.dtype),
            suppressed
        )
        return which(any(finite), withoutPad, eosOnly)
    }

    mutating func didSample(token: MLXArray) {}
}

/// A local model load or generation request cannot be completed.
public enum LocalModelRunnerError: LocalizedError, Equatable {
    case busy
    case emptyResponse
    case lastMessageMustBeUserOrTool
    case unsupportedRole(String)
    case missingMessageContent(String)
    case missingToolCallID
    case unsupportedToolType(String)
    case missingToolName
    case duplicateToolName(String)
    case invalidToolParameters(String)
    case invalidToolCallArguments(String)
    case unsupportedForcedToolChoiceFormat(String)
    case unsupportedForcedToolChoiceModel(String)
    case invalidForcedToolChoiceTokens
    case emptyForcedToolChoicePrefix
    case prohibitedToolCall(String)
    case undeclaredToolCall(String)
    case wrongNamedToolCall(expected: String, actual: String)
    case requiredToolCallMissing
    case missingDirectory(String)
    case missingConfig(String)
    case missingWeights(String)
    case adapterScaleWithoutAdapter
    case invalidAdapterScale(Float)
    case invalidTemperature(Double)
    case invalidTopP(Double)
    case incompatibleDFlash(String)
    case invalidDFlashBlockSize(Int, maximum: Int)

    /// User-facing explanation for this error.
    public var errorDescription: String? {
        switch self {
        case .busy:
            "Midnight Runner request queue is full (64 waiting operations). Retry after current requests finish."
        case .emptyResponse: "The model ended the turn without producing text."
        case .lastMessageMustBeUserOrTool:
            "The final chat message must have role 'user' or 'tool'."
        case .unsupportedRole(let role): "Unsupported chat role: \(role)"
        case .missingMessageContent(let role):
            "A chat message with role '\(role)' requires content."
        case .missingToolCallID: "A chat message with role 'tool' requires tool_call_id."
        case .unsupportedToolType(let type): "Unsupported tool type: \(type)"
        case .missingToolName: "A function tool requires a non-empty name."
        case .duplicateToolName(let name): "Tool names must be unique: \(name)"
        case .invalidToolParameters(let name):
            "Function tool '\(name)' requires an object JSON schema for parameters."
        case .invalidToolCallArguments(let id):
            "Tool call '\(id)' arguments must be a JSON object."
        case .unsupportedForcedToolChoiceFormat(let format):
            "The loaded model's '\(format)' tool-call format cannot enforce tool_choice."
        case .unsupportedForcedToolChoiceModel(let model):
            "The loaded model type '\(model)' does not support enforced tool_choice."
        case .invalidForcedToolChoiceTokens:
            "The tokenizer encoded a forced tool-call prefix outside the model vocabulary."
        case .emptyForcedToolChoicePrefix:
            "The loaded tokenizer cannot encode the required tool-call prefix."
        case .prohibitedToolCall(let name):
            "The model emitted prohibited tool call '\(name)' while tool_choice was 'none'."
        case .undeclaredToolCall(let name):
            "The model emitted undeclared tool call '\(name)'."
        case .wrongNamedToolCall(let expected, let actual):
            "The model emitted tool call '\(actual)' when tool_choice required '\(expected)'."
        case .requiredToolCallMissing:
            "The model completed without the tool call required by tool_choice."
        case .missingDirectory(let path): "Model folder does not exist: \(path)"
        case .missingConfig(let path): "Model folder is missing config.json: \(path)"
        case .missingWeights(let path): "Model folder contains no .safetensors weights: \(path)"
        case .adapterScaleWithoutAdapter: "--adapter-scale requires --adapter."
        case .invalidAdapterScale(let value):
            "Adapter scale must be finite and non-negative, not \(value)."
        case .invalidTemperature(let value):
            "Temperature must be finite and non-negative, not \(value)."
        case .invalidTopP(let value):
            "top_p must be finite, greater than zero, and at most one, not \(value)."
        case .incompatibleDFlash(let detail):
            "Incompatible Laguna DFlash checkpoint: \(detail)."
        case .invalidDFlashBlockSize(let value, let maximum):
            "DFlash block size must be in 2...\(maximum), not \(value)."
        }
    }
}

extension String {
    fileprivate var nilIfBlank: String? { isEmpty ? nil : self }
}
