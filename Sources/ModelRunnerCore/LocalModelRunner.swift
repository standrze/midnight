import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import ModelRunnerProtocol
import Tokenizers

public enum LocalModelRunnerEvent: Equatable, Sendable {
  case content(String)
  case toolCall(OpenAIToolCall)
  case metrics(LocalModelRunnerMetrics)
}

public struct LocalModelRunnerMetrics: Equatable, Sendable {
  public let promptTokenCount: Int
  public let prefilledPromptTokenCount: Int
  public let cachedPromptTokenCount: Int
  public let generationTokenCount: Int
  public let promptTokensPerSecond: Double
  public let tokensPerSecond: Double
  public let stopReason: String
  public let proposedDraftTokens: Int?
  public let acceptedDraftTokens: Int?
  public let speculativePassthroughReason: String?

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

private final class DFlashModelReference: @unchecked Sendable {
  let model: any MTPDrafterModel

  init(_ model: any MTPDrafterModel) {
    self.model = model
  }
}

private struct LoadedDFlash: Sendable {
  // Retain the factory-owned context for the lifetime of the shared model.
  let container: MTPDrafterContainer
  let model: DFlashModelReference
  let blockSize: Int
  let modelPath: String
}

/// `ChatSession` is deliberately single-consumer rather than `Sendable`.
/// LocalModelRunner's actor and `isGenerating` guard provide that serialization;
/// this reference keeps the assertion at one explicit boundary for Swift 6.
private final class ChatSessionReference: @unchecked Sendable {
  let session: ChatSession

  init(_ session: ChatSession) {
    self.session = session
  }

  func configure(
    parameters: GenerateParameters,
    components: GenerationComponents,
    tools: [ToolSpec]?
  ) {
    session.generateParameters = parameters
    session.components = components
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
    tools: [ToolSpec]?
  ) -> ChatSession {
    ChatSession(
      container,
      restoring: snapshot,
      generateParameters: parameters,
      components: components,
      tools: tools
    )
  }
}

private struct HotConversation: Sendable {
  let session: ChatSessionReference
  let committedMessages: [OpenAIMessage]
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

public actor LocalModelRunner {
  #if os(macOS) && MODEL_RUNNER_PINNED_MLX
    /// Keep actor-isolated orchestration on the same pthread as MLX evaluation.
    public nonisolated var unownedExecutor: UnownedSerialExecutor {
      MLXPinnedRuntime.shared.unownedSerialExecutor
    }
  #endif

  public nonisolated let modelPath: String
  public nonisolated let servedModelName: String
  public nonisolated let engine: ModelEngine
  public nonisolated let dflashModelPath: String?
  public nonisolated let dflashBlockSize: Int?
  public nonisolated let supportsMistralHotConversationCache: Bool

  private let container: ModelContainer
  private let tokenLimit: GenerationTokenLimit
  private let device: Device
  private let normalizesGemma4Prompt: Bool
  private let runtimeCapabilities: ModelRuntimeCapabilities
  private let supportsLagunaPromptCache: Bool
  private let dflash: LoadedDFlash?
  private var wiredMemoryPlan: MLXWiredMemoryPlan?
  #if os(macOS) && MODEL_RUNNER_PINNED_MLX
    private nonisolated let mlxStream: MLX.Stream
  #endif
  private var conversationCache: CompletedMessagePrefixLRU<ChatSessionSnapshotReference>
  private var hotConversation: HotConversation?
  private var isGenerating = false

  public init(
    modelPath: String,
    servedModelName: String? = nil,
    engine requestedEngine: ModelEngine = .auto,
    maximumTokens: Int = 512,
    adapterPath: String? = nil,
    adapterScale: Float? = nil,
    dflashModelPath: String? = nil,
    dflashBlockSize: Int? = nil
  ) async throws {
    let tokenLimit = try GenerationTokenLimit(configuredMaximum: maximumTokens)

    let engine = try requestedEngine.resolve()
    let device: Device = engine == .cpu ? .cpu : .gpu
    let resourceLimits = try MLXResourceLimits.resolve(
      for: engine,
      physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory
    )

    let expandedPath = NSString(string: modelPath).expandingTildeInPath
    let modelURL = URL(fileURLWithPath: expandedPath, isDirectory: true)
      .standardizedFileURL
      .resolvingSymlinksInPath()
    try Self.validateModelFolder(modelURL)
    let normalizesGemma4Prompt = try Self.isGemma4Model(modelURL)
    let runtimeCapabilities = try ModelRuntimeCapabilities.load(from: modelURL)

    let dflashURL: URL? = try dflashModelPath.map { path in
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

    let loadModels: @Sendable () async throws -> (ModelContainer, LoadedDFlash?) = {
      // Laguna is implemented by this target so the runner can use Poolside
      // checkpoints natively without a Python sidecar or an mlx-swift-lm fork.
      // Register and construct it on the same permanent pthread used for eval.
      await LagunaModelRegistration.register()
      await LagunaDFlashRegistration.register()

      print(
        "MLX resource guard: memory=\(resourceLimits.memoryLimitBytes) bytes "
          + "cache=\(resourceLimits.cacheLimitBytes) bytes"
      )
      try MLXResourceGuard.apply(resourceLimits)
      let targetContainer = try await #huggingFaceLoadModelContainer(
        configuration: ModelConfiguration(
          directory: modelURL,
          extraEOSTokens: ["<end_of_turn>", "<turn|>"]
        )
      )
      guard let dflashURL else { return (targetContainer, nil) }

      let draftContainer = try await MTPDrafterModelFactory.shared.loadContainer(
        from: dflashURL,
        using: #huggingFaceTokenizerLoader()
      )
      let loaded = await draftContainer.perform {
        context -> (DFlashModelReference, LagunaDFlashTargetDescriptor)? in
        guard let model = context.model as? LagunaDFlashModel else { return nil }
        return (DFlashModelReference(model), model.targetDescriptor)
      }
      guard let (draftReference, descriptor) = loaded else {
        throw LocalModelRunnerError.incompatibleDFlash(
          "checkpoint is not DFlashLagunaForCausalLM")
      }
      let effectiveBlockSize = dflashBlockSize ?? descriptor.blockSize
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
        LoadedDFlash(
          container: draftContainer,
          model: draftReference,
          blockSize: effectiveBlockSize,
          modelPath: dflashURL.path
        )
      )
    }
    let (container, loadedDFlash): (ModelContainer, LoadedDFlash?)
    #if os(macOS) && MODEL_RUNNER_PINNED_MLX
      (container, loadedDFlash) = try await withPinnedMLXRuntime(
        device: device,
        stream: mlxStream,
        operation: loadModels
      )
    #else
      (container, loadedDFlash) = try await Device.withDefaultDevice(
        device,
        loadModels
      )
    #endif
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
    let environment = ProcessInfo.processInfo.environment
    var wiredMemoryPlan: MLXWiredMemoryPlan?
    if engine == .metal, MLXWiredMemoryPlan.isEnabled(environment: environment) {
      let tuningTokens = MLXWiredMemoryPlan.tuningTokenCount(environment: environment)
      let measure: @Sendable () async throws -> WiredMemoryMeasurement = {
        try await container.perform { context in
          try await WiredMemoryUtils.tune(
            context: context,
            tokenCount: tuningTokens,
            parameters: GenerateParameters(maxTokens: 1, temperature: 0)
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
          minimumHeadroomBytes: loadedDFlash == nil
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
        if error is CancellationError { throw error }
        print("MLX wired-memory measurement unavailable; continuing unwired: \(error)")
        wiredMemoryPlan = nil
      }
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
    self.supportsLagunaPromptCache = supportsLagunaPromptCache
    self.wiredMemoryPlan = wiredMemoryPlan
    let prefixCacheLimits = ConversationPrefixCacheLimits.resolve(environment: environment)
    self.conversationCache = CompletedMessagePrefixLRU(
      maximumEntries: prefixCacheLimits.maximumEntries,
      maximumBytes: prefixCacheLimits.maximumBytes
    )
    self.dflash = loadedDFlash
    #if os(macOS) && MODEL_RUNNER_PINNED_MLX
      self.mlxStream = mlxStream
    #endif
    self.dflashModelPath = loadedDFlash?.modelPath
    self.dflashBlockSize = loadedDFlash?.blockSize
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
    }
    if let loadedDFlash {
      print(
        "Loaded Laguna DFlash \(loadedDFlash.modelPath) "
          + "(block size \(loadedDFlash.blockSize), greedy decoding)"
      )
    }
  }

  public func stream(
    messages: [OpenAIMessage],
    maximumTokens: Int?,
    temperature: Double? = nil,
    topP: Double? = nil,
    stop: [String] = [],
    tools: [OpenAIToolDefinition]? = nil,
    enablePromptCache: Bool = true,
    enableSpeculativeDecoding: Bool = true
  ) -> AsyncThrowingStream<LocalModelRunnerEvent, Error> {
    let lagunaFastPaths = LagunaDecodeFastPathSelection.resolve(
      engine: engine,
      compiledBlockTailOverride: LagunaRuntimeTuning.useCompiledBlockTail,
      fusedRouterTopKOverride: LagunaRuntimeTuning.useFusedRouterTopK)
    return AsyncThrowingStream { continuation in
      let generationTask = LagunaRuntimeTuning.$useCompiledBlockTail.withValue(
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
                enablePromptCache: enablePromptCache,
                enableSpeculativeDecoding: enableSpeculativeDecoding
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
    enablePromptCache: Bool,
    enableSpeculativeDecoding: Bool,
    onEvent: @escaping @Sendable (LocalModelRunnerEvent) throws -> Void
  ) async throws {
    guard !isGenerating else { throw LocalModelRunnerError.busy }
    let effectiveMaximumTokens = try tokenLimit.resolve(requested: maximumTokens)
    isGenerating = true
    defer { isGenerating = false }

    let settings = try generationRequestSettings(
      maximumTokens: effectiveMaximumTokens,
      temperature: temperature,
      topP: topP,
      normalizesGemma4Prompt: normalizesGemma4Prompt
    )
    #if os(macOS) && MODEL_RUNNER_PINNED_MLX
      try await withPinnedMLXRuntime(device: device, stream: mlxStream) {
        try await self.withGenerationWiredResidency {
          try await self.generateOnSelectedRuntime(
            messages: messages,
            effectiveMaximumTokens: effectiveMaximumTokens,
            temperature: temperature,
            topP: topP,
            stop: stop,
            tools: tools,
            enablePromptCache: enablePromptCache,
            enableSpeculativeDecoding: enableSpeculativeDecoding,
            settings: settings,
            onEvent: onEvent
          )
        }
      }
    #else
      try await withGenerationWiredResidency {
        try await generateOnSelectedRuntime(
          messages: messages,
          effectiveMaximumTokens: effectiveMaximumTokens,
          temperature: temperature,
          topP: topP,
          stop: stop,
          tools: tools,
          enablePromptCache: enablePromptCache,
          enableSpeculativeDecoding: enableSpeculativeDecoding,
          settings: settings,
          onEvent: onEvent
        )
      }
    #endif
  }

  /// Keep residency elevated until the request's producer has synchronized and
  /// learn from the completed request's real process-wide peak. Manual awaited
  /// teardown is intentional: the generic cancellation helper may end its ticket
  /// before this runner has cancelled and joined the GPU producer.
  private func withGenerationWiredResidency(
    _ operation: () async throws -> Void
  ) async throws {
    guard let plan = wiredMemoryPlan else {
      try await operation()
      return
    }

    let ticket = plan.makeTicket()
    let appliedLimit = await ticket.start()
    guard appliedLimit >= plan.limitBytes else {
      _ = await ticket.end()
      print(
        "MLX wired-memory request was not applied "
          + "(requested=\(plan.limitBytes), applied=\(appliedLimit)); continuing unwired"
      )
      try await operation()
      return
    }

    Memory.peakMemory = 0
    do {
      try await operation()
      let observedPeak = Memory.peakMemory
      _ = await ticket.end()
      wiredMemoryPlan?.observe(peakActiveBytes: observedPeak)
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
    onEvent: @Sendable (LocalModelRunnerEvent) throws -> Void
  ) async throws {
    let usesDFlash =
      enableSpeculativeDecoding && dflash != nil && settings.temperature == 0
    if supportsLagunaPromptCache && !normalizesGemma4Prompt && stop.isEmpty && !usesDFlash {
      try await generateWithLagunaPromptCache(
        messages: messages,
        settings: settings,
        tools: tools,
        allowsReuse: enablePromptCache,
        onEvent: onEvent
      )
      return
    }
    if Self.shouldUseMistralPromptCache(
      capabilities: runtimeCapabilities,
      enablePromptCache: enablePromptCache,
      normalizesGemma4Prompt: normalizesGemma4Prompt,
      hasCustomStopStrings: !stop.isEmpty,
      usesDFlash: usesDFlash
    ) {
      try await generateWithMistralPromptCache(
        messages: messages,
        settings: settings,
        tools: tools,
        allowsReuse: enablePromptCache,
        onEvent: onEvent
      )
      return
    }

    // The one-shot paths can mutate cache state independently (custom stop
    // strings, Gemma prompt normalization, or Laguna's custom DFlash iterator).
    // Do not retain a session across either side of that boundary.
    hotConversation = nil
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
      enableSpeculativeDecoding: enableSpeculativeDecoding,
      onEvent: onEvent
    )
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
    enablePromptCache
      && capabilities.supportsMistralConversationPrefixCache
      && !normalizesGemma4Prompt
      && !hasCustomStopStrings
      && !usesDFlash
  }

  static func cachedConversationSuffixStart(
    committed: [OpenAIMessage],
    incoming: [OpenAIMessage]
  ) -> Int? {
    guard incoming.count > committed.count, incoming.starts(with: committed) else {
      return nil
    }
    return committed.count
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
    onEvent: @Sendable (LocalModelRunnerEvent) throws -> Void
  ) async throws {
    try await generateWithConversationPrefixCache(
      messages: messages,
      settings: settings,
      tools: tools,
      allowsReuse: allowsReuse,
      retainsBranchSnapshots: true,
      onEvent: onEvent
    )
  }

  private func generateWithMistralPromptCache(
    messages: [OpenAIMessage],
    settings: GenerationRequestSettings,
    tools: [OpenAIToolDefinition]?,
    allowsReuse: Bool,
    onEvent: @Sendable (LocalModelRunnerEvent) throws -> Void
  ) async throws {
    try await generateWithConversationPrefixCache(
      messages: messages,
      settings: settings,
      tools: tools,
      allowsReuse: allowsReuse,
      retainsBranchSnapshots: false,
      onEvent: onEvent
    )
  }

  private func generateWithConversationPrefixCache(
    messages: [OpenAIMessage],
    settings: GenerationRequestSettings,
    tools: [OpenAIToolDefinition]?,
    allowsReuse: Bool,
    retainsBranchSnapshots: Bool,
    onEvent: @Sendable (LocalModelRunnerEvent) throws -> Void
  ) async throws {
    guard let final = messages.last, final.role == "user" || final.role == "tool" else {
      throw LocalModelRunnerError.lastMessageMustBeUserOrTool
    }

    let chatMessages = try messages.map(Self.chatMessage)
    let toolSpecs = try Self.toolSpecs(tools)
    let session: ChatSessionReference
    let pendingMessages: [Chat.Message]
    let reusedHotConversation: Bool
    if allowsReuse, let hotConversation,
      let suffixStart = Self.cachedConversationSuffixStart(
        committed: hotConversation.committedMessages,
        incoming: messages)
    {
      session = hotConversation.session
      pendingMessages = Array(chatMessages.dropFirst(suffixStart))
      session.configure(
        parameters: settings.parameters,
        components: settings.components,
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
          tools: toolSpecs
        )
      )
      pendingMessages = [chatMessages[chatMessages.index(before: chatMessages.endIndex)]]
      reusedHotConversation = false
    }

    var content = ""
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
      if reusedHotConversation { hotConversation = nil }
      throw error
    }

    guard producedOutput else {
      if reusedHotConversation { hotConversation = nil }
      throw LocalModelRunnerError.emptyResponse
    }
    guard canRetainSession else {
      if reusedHotConversation { hotConversation = nil }
      return
    }
    guard allowsReuse else { return }
    let assistant = OpenAIMessage(
      role: "assistant",
      content: content.isEmpty ? nil : content,
      toolCalls: generatedToolCalls.isEmpty ? nil : generatedToolCalls
    )
    hotConversation = HotConversation(
      session: session,
      committedMessages: messages + [assistant]
    )
    // A snapshot deep-copies and evaluates every KV array. Keep Laguna's
    // established branchable LRU unchanged, but make the Mistral-family fast
    // path zero-copy: the hot session handles the common append-only chat case
    // without adding a potentially GiB-scale copy after each response.
    guard retainsBranchSnapshots else { return }
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

  fileprivate static func chatMessage(_ message: OpenAIMessage) throws -> Chat.Message {
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
      return .assistant(message.content ?? "", toolCalls: toolCalls)
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
    guard let definitions, !definitions.isEmpty else { return nil }
    var names = Set<String>()
    return try definitions.map { definition in
      guard definition.type == "function" else {
        throw LocalModelRunnerError.unsupportedToolType(definition.type)
      }
      let name = definition.function.name.trimmingCharacters(in: .whitespacesAndNewlines)
      guard !name.isEmpty else { throw LocalModelRunnerError.missingToolName }
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
    else { throw LocalModelRunnerError.missingDirectory(folder.path) }

    guard
      FileManager.default.fileExists(
        atPath: folder.appendingPathComponent("config.json").path
      )
    else { throw LocalModelRunnerError.missingConfig(folder.path) }

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
    else { return false }
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
    guard let scaleOverride else { return original }
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
  let parameters: GenerateParameters
  let components: GenerationComponents
}

private func generationRequestSettings(
  maximumTokens: Int,
  temperature requestedTemperature: Double?,
  topP requestedTopP: Double?,
  normalizesGemma4Prompt: Bool
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
  let components =
    normalizesGemma4Prompt
    ? GenerationComponents(
      logitProcessorFactory: { SuppressTokenLogitProcessor(tokenID: 0) }
    )
    : GenerationComponents()
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
  dflash: LoadedDFlash?,
  enableSpeculativeDecoding: Bool,
  onEvent: @Sendable (LocalModelRunnerEvent) throws -> Void
) async throws {
  try await Device.withDefaultDevice(device) {
    // Generation is actor-serialized, so the device's persistent default stream
    // is sufficient. Creating a fresh MLX stream for every request leaves a
    // busy CUDA worker behind on Linux and progressively steals CPU from decode.
    guard let final = messages.last, final.role == "user" || final.role == "tool" else {
      throw LocalModelRunnerError.lastMessageMustBeUserOrTool
    }

    let chatMessages = try messages.map(LocalModelRunner.chatMessage)
    let toolSpecs = try LocalModelRunner.toolSpecs(tools)
    let preparedInput = try await container.prepare(
      input: UserInput(chat: chatMessages, tools: toolSpecs)
    )
    var promptTokenIDs = preparedInput.text.tokens.asArray(Int.self)
    // The pinned Swift Jinja renderer inserts a newline between Gemma 4's BOS
    // and first turn token. Python mlx-vlm/Transformers renders `<bos><|turn>`
    // directly. Remove only that model-specific, verified token sequence so
    // both runtimes feed the checkpoint the same prompt.
    if normalizesGemma4Prompt,
      promptTokenIDs.count >= 3,
      promptTokenIDs[0] == 2,
      promptTokenIDs[1] == 107,
      promptTokenIDs[2] == 105
    {
      promptTokenIDs.remove(at: 1)
    }
    if ProcessInfo.processInfo.environment["MODEL_RUNNER_DEBUG_PROMPT_TOKENS"] == "1" {
      print("Prompt token IDs (\(promptTokenIDs.count)): \(promptTokenIDs)")
    }
    let settings = try generationRequestSettings(
      maximumTokens: maximumTokens,
      temperature: requestedTemperature,
      topP: requestedTopP,
      normalizesGemma4Prompt: normalizesGemma4Prompt
    )
    // The pinned MTP verifier is lossless for greedy decoding. Explicitly
    // sampled requests stay on the ordinary target path until probability-
    // ratio rejection sampling is available; this also avoids paying DFlash's
    // specialized prefill cost only to enter passthrough mode.
    let useDFlash =
      enableSpeculativeDecoding && dflash != nil && settings.temperature == 0
    let promptTokenCount = promptTokenIDs.count
    let (stream, producerTask) = try await container.perform(
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
          parameters: settings.parameters,
          blockSize: dflash.blockSize,
          components: settings.components
        )
        return MLXLMCommon.generateTask(
          promptTokenCount: promptTokenCount,
          modelConfiguration: requestContext.configuration,
          tokenizer: requestContext.tokenizer,
          iterator: iterator,
          tools: toolSpecs
        )
      }
      let iterator = try TokenIterator(
        input: input,
        model: requestContext.model,
        parameters: settings.parameters,
        components: settings.components
      )
      return MLXLMCommon.generateTask(
        promptTokenCount: promptTokenCount,
        modelConfiguration: requestContext.configuration,
        tokenizer: requestContext.tokenizer,
        iterator: iterator,
        tools: toolSpecs
      )
    }

    var producedOutput = false
    do {
      for await event in stream {
        try Task.checkCancellation()
        switch event {
        case .chunk(let chunk):
          producedOutput = true
          try onEvent(.content(chunk))
        case .info(let info):
          try onEvent(
            .metrics(
              LocalModelRunner.metrics(info)
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
    guard producedOutput else { throw LocalModelRunnerError.emptyResponse }
  }
}

private struct SuppressTokenLogitProcessor: LogitProcessor {
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
  case missingDirectory(String)
  case missingConfig(String)
  case missingWeights(String)
  case adapterScaleWithoutAdapter
  case invalidAdapterScale(Float)
  case invalidTemperature(Double)
  case invalidTopP(Double)
  case incompatibleDFlash(String)
  case invalidDFlashBlockSize(Int, maximum: Int)

  public var errorDescription: String? {
    switch self {
    case .busy: "Midnight Runner is already generating a response."
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
