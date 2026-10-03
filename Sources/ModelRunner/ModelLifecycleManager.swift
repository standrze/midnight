import Foundation
import ModelRunnerCore
import ModelRunnerProtocol

/// All model-specific state travels together for the full lifetime of a request.
final class LoadedModel: Sendable {
    let runner: LocalModelRunner?
    let servedModelName: String
    /// Canonical paths retained only for explicit model-management operations.
    let protectedDirectories: [URL]
    /// Held across backend loading, requests, cancellation, and drain.
    let fileUsage: ManagedModelUsage?
    let tokenLimit: GenerationTokenLimit
    let speechSynthesizer: (any LocalSpeechSynthesizing)?
    let vision: (any VisionModelServing)?
    let visionContextLength: Int?
    let visionMemoryBytes: Int?
    let loadRequest: ModelLoadRequest?
    let modelCard: ModelCard?
    let created = Int(Date().timeIntervalSince1970)

    init(
        runner: LocalModelRunner? = nil, servedModelName: String,
        tokenLimit: GenerationTokenLimit,
        speechSynthesizer: (any LocalSpeechSynthesizing)? = nil,
        protectedDirectories: [URL] = [], fileUsage: ManagedModelUsage? = nil,
        vision: (any VisionModelServing)? = nil, visionContextLength: Int? = nil,
        visionMemoryBytes: Int? = nil, loadRequest: ModelLoadRequest? = nil, modelCard: ModelCard? = nil
    ) {
        self.runner = runner
        self.fileUsage = fileUsage
        self.protectedDirectories = protectedDirectories.map { $0.standardizedFileURL.resolvingSymlinksInPath() }
        self.servedModelName = servedModelName
        self.tokenLimit = tokenLimit
        self.speechSynthesizer = speechSynthesizer
        self.vision = vision
        self.visionContextLength = visionContextLength
        self.visionMemoryBytes = visionMemoryBytes
        self.loadRequest = loadRequest
        self.modelCard = (modelCard ?? ModelCard(name: servedModelName)).withCapabilities(
            .init(
                vision: vision != nil, audioInput: speechSynthesizer?.supportsReferenceAudio ?? false,
                audioOutput: speechSynthesizer != nil, decisions: runner?.supportsDecisions == true ? true : nil)
        ).withVoices(speechSynthesizer?.voiceCatalog.modelCardVoices)
    }

    func waitUntilIdle() async {
        await runner?.waitUntilIdle()
        await speechSynthesizer?.waitUntilIdle()
        await vision?.waitUntilIdle()
    }

    var descriptor: ModelLifecycleDescriptor {
        ModelLifecycleDescriptor(
            id: servedModelName, created: created,
            contextLength: runner?.contextLength ?? visionContextLength, prefillStepSize: runner?.prefillStepSize,
            kvCompression: runner?.kvCompression, memoryLimitBytes: runner?.memoryLimitBytes ?? visionMemoryBytes,
            maximumOutputTokens: runner == nil && vision == nil ? nil : tokenLimit.configuredMaximum,
            defaultOutputTokens: runner == nil && vision == nil ? nil : tokenLimit.defaultTokens,
            modality: vision != nil ? "vision" : speechSynthesizer != nil ? "voice" : "text",
            loadRequest: loadRequest, modelCard: modelCard, nativeProtocol: nativeModelProtocol(at: runner?.modelPath))
    }
}

struct ModelLifecycleDescriptor: Encodable, Sendable {
    let id: String
    let created: Int
    let object = "model"
    let ownedBy = "midnight"
    let contextLength: Int?
    let prefillStepSize: Int?
    let kvCompression: String?
    let memoryLimitBytes: Int?
    let maximumOutputTokens: Int?
    let defaultOutputTokens: Int?
    let modality: String
    let loadRequest: ModelLoadRequest?

    var modelCard: ModelCard? = nil
    var nativeProtocol: String? = nil
    var loaded: Bool? = nil

    enum CodingKeys: String, CodingKey {
        case id, created, object
        case modelCard = "model_card"
        case nativeProtocol = "native_protocol"
        case ownedBy = "owned_by"
        case contextLength = "context_length"
        case prefillStepSize = "prefill_step_size"
        case kvCompression = "kv_compression"
        case memoryLimitBytes = "memory_limit_bytes"
        case maximumOutputTokens = "max_output_tokens"
        case defaultOutputTokens = "default_output_tokens"
        case modality, loadRequest, loaded
    }
}

struct ModelLifecycleState: Encodable, Sendable {
    enum Phase: String, Encodable, Sendable {
        case empty, draining, unloading, loading, ready
    }
    let instanceID: String?
    let processID: Int32
    let phase: Phase
    /// Retained after completion so clients can correlate an asynchronous load.
    let operationID: String?
    let modelGeneration: UInt64
    let loadedModel: ModelLifecycleDescriptor?
    let targetModel: String?
    let lastError: String?
    let memory: LocalModelMemorySnapshot
    /// Latest text backend report for the currently resident model, including headless serving.
    var generationMetrics: LocalModelRunnerMetrics? = nil
}

struct ModelLoadOperation: Sendable {
    let name: String
    var prepare: (@Sendable () async throws -> ModelLoadOperation)? = nil
    let load: @Sendable () async throws -> LoadedModel
}

enum ModelLifecycleError: LocalizedError {
    case busy
    case unavailable
    case staleSelection
    case unknownModel
    case disabledModel
    case loadFailed(String)

    var errorDescription: String? {
        switch self {
        case .busy: "A model load or unload is already in progress."
        case .unavailable: "No model is ready. Check /v1/runtime for load status."
        case .staleSelection: "The active Midnight model changed. Refresh runtime state before retrying."
        case .unknownModel: "The requested model is not in the installed catalog. Use an ID from /v1/models."
        case .disabledModel:
            "The model is marked unavailable. Make it available in the Midnight console before using it."
        case .loadFailed(let detail): "The requested model could not be loaded: \(detail)"
        }
    }
}

/// The listener owns this actor; only this actor owns an installed backend.
/// Admission closes synchronously before a transition can suspend.
actor ModelLifecycleManager {
    private let validate: @Sendable (ModelLoadRequest) throws -> ModelLoadOperation
    private let cleanup: @Sendable () async -> Void
    private let downloads: ManagedModelDownloads
    private let catalog: @Sendable () -> [InstalledModelEntry]
    private var remembered: [String: InstalledModelEntry] = [:]
    private var availability: ModelAvailabilityStore
    private var admissionBusy = false
    private var stopping = false
    private var admissionWaiters: [(UUID, CheckedContinuation<Void, Error>)] = []
    private let instanceID = ProcessInfo.processInfo.environment["MIDNIGHT_CONTROL_INSTANCE"] ?? UUID().uuidString
    private var loaded: LoadedModel?
    private var phase: ModelLifecycleState.Phase = .empty
    private var operationID: String?
    private var modelGeneration: UInt64 = 0
    private var targetModel: String?
    private var lastError: String?
    private var transition: Task<Void, Never>?
    private var generationMonitoringEnabled = false
    private var activeRequests = 0
    private var drainWaiter: CheckedContinuation<Void, Never>?

    init(
        loader: ModelLoader, downloads: ManagedModelDownloads = ManagedModelDownloads(),
        availability: ModelAvailabilityStore = ModelAvailabilityStore()
    ) {
        self.availability = availability
        self.downloads = downloads
        catalog = { loader.installedModels() }
        validate = { request in
            try loader.operation(for: request)
        }
        cleanup = { await LocalModelRuntime.releaseUnusedMemory() }
    }

    init(
        validate: @escaping @Sendable (ModelLoadRequest) throws -> ModelLoadOperation,
        cleanup: @escaping @Sendable () async -> Void = {},
        catalog: @escaping @Sendable () -> [InstalledModelEntry] = { [] },
        downloads: ManagedModelDownloads = ManagedModelDownloads(),
        availability: ModelAvailabilityStore = ModelAvailabilityStore()
    ) {
        self.availability = availability
        self.downloads = downloads
        self.validate = validate
        self.cleanup = cleanup
        self.catalog = catalog
    }

    func state() -> ModelLifecycleState {
        ModelLifecycleState(
            instanceID: instanceID,
            processID: ProcessInfo.processInfo.processIdentifier,
            phase: phase, operationID: operationID, modelGeneration: modelGeneration,
            loadedModel: phase == .ready ? loaded?.descriptor : nil,
            targetModel: targetModel, lastError: lastError,
            memory: LocalModelRuntime.memorySnapshot(),
            generationMetrics: phase == .ready ? loaded?.runner?.latestGenerationMetrics : nil)
    }

    func setGenerationMonitoringEnabled(_ enabled: Bool) {
        generationMonitoringEnabled = enabled
        loaded?.runner?.setGenerationMonitoringEnabled(enabled)
    }

    func dashboardSnapshot() -> ServerDashboardSnapshot {
        let waitingForGeneration = loaded?.runner?.queuedGenerationCount ?? 0
        return ServerDashboardSnapshot(
            generation: loaded?.runner?.generationProgressSnapshot,
            activeRequests: max(0, activeRequests - waitingForGeneration),
            queuedRequests: admissionWaiters.count + waitingForGeneration)
    }

    /// Closes admission, joins an in-progress load, then drains and releases the resident backend.
    func shutdown() async {
        stopping = true
        if let transition {
            await transition.value
        }
        if loaded != nil {
            phase = .draining
            await replace(with: nil)
        }
    }

    func load(_ request: ModelLoadRequest) throws -> ModelLifecycleState {
        guard !admissionBusy else {
            throw ModelLifecycleError.busy
        }
        return try beginLoad(request)
    }

    private func beginLoad(_ request: ModelLoadRequest) throws -> ModelLifecycleState {
        try requireSelection(generation: request.expectedGeneration, instance: request.expectedInstanceID)
        try requireNoTransition()
        guard availability.isAvailable(request) else {
            throw ModelLifecycleError.disabledModel
        }
        let operation: ModelLoadOperation
        do { operation = try validate(request) } catch {
            // A malformed replacement must leave the existing model usable.
            if loaded == nil {
                lastError = error.localizedDescription
            }
            throw error
        }
        operationID = UUID().uuidString
        targetModel = operation.name
        lastError = nil
        phase = loaded == nil ? .loading : .draining
        transition = Task { await self.replace(with: operation) }
        return state()
    }

    func unload(expectedGeneration: UInt64? = nil, expectedInstanceID: String? = nil) throws -> ModelLifecycleState {
        guard !admissionBusy else {
            throw ModelLifecycleError.busy
        }
        try requireSelection(generation: expectedGeneration, instance: expectedInstanceID)
        try requireNoTransition()
        operationID = UUID().uuidString
        targetModel = nil
        lastError = nil
        guard loaded != nil else {
            return state()
        }
        phase = .draining
        transition = Task { await self.replace(with: nil) }
        return state()
    }

    func downloadedModels() throws -> ManagedModelDownloadList {
        try downloads.list(protectedDirectories: loaded?.protectedDirectories ?? [])
    }

    func availableModels() -> [ModelLifecycleDescriptor] {
        modelAvailability().filter(\.available).map(\.model)
    }

    func modelAvailability() -> [ModelAvailabilityEntry] {
        var result = entries().mapValues(\.descriptor)
        if phase == .ready, let loaded {
            result[loaded.servedModelName] = loaded.descriptor
        }
        return result.values.sorted { $0.id < $1.id }.map { descriptor in
            var descriptor = descriptor
            descriptor.loaded = phase == .ready && loaded?.servedModelName == descriptor.id
            return ModelAvailabilityEntry(
                model: descriptor, available: descriptor.loadRequest.map { availability.isAvailable($0) } ?? true)
        }
    }

    /// Changing availability never deletes weights or loads an enabled model.
    /// A disabled resident model closes admission immediately and drains through the normal unload path.
    func setModelAvailable(_ available: Bool, id: String) throws -> ModelLifecycleState {
        try requireNoTransition()
        guard !admissionBusy else {
            throw ModelLifecycleError.busy
        }
        guard let request = modelAvailability().first(where: { $0.model.id == id })?.model.loadRequest else {
            throw ModelLifecycleError.unknownModel
        }
        try availability.setAvailable(available, for: request)
        if !available, let request = loaded?.loadRequest, !availability.isAvailable(request) {
            return try unload()
        }
        return state()
    }

    private func entries() -> [String: InstalledModelEntry] {
        var result = Dictionary(uniqueKeysWithValues: catalog().map { ($0.descriptor.id, $0) })
        for (id, entry) in remembered {
            guard (try? validate(entry.request)) != nil else {
                continue
            }
            // Preserve an explicitly configured alias instead of listing its path twice.
            result = result.filter { $0.value.request.model != entry.request.model }
            result[id] = entry
        }
        return result
    }

    /// FIFO admission serializes switches, not inference on an already loaded model.
    /// The model lease is acquired before another requester can begin a switch.
    func withRequestedModel<Result: Sendable>(
        _ id: String, _ operation: @Sendable (LoadedModel) async throws -> Result
    ) async throws -> Result {
        guard !stopping else {
            throw ModelLifecycleError.unavailable
        }
        try await acquireAdmission()
        do {
            try Task.checkCancellation()
            guard !stopping else {
                throw ModelLifecycleError.unavailable
            }
            let request = loaded?.servedModelName == id ? loaded?.loadRequest : entries()[id]?.request
            guard request != nil || (phase == .ready && loaded?.servedModelName == id) else {
                throw ModelLifecycleError.unknownModel
            }
            if let request, !availability.isAvailable(request) {
                throw ModelLifecycleError.disabledModel
            }
            if let transition {
                await transition.value
            }
            try Task.checkCancellation()
            if phase != .ready || loaded?.servedModelName != id {
                guard let request else {
                    throw ModelLifecycleError.unknownModel
                }
                do { _ = try beginLoad(request) } catch {
                    throw ModelLifecycleError.loadFailed(error.localizedDescription)
                }
                if let transition {
                    await transition.value
                }
                try Task.checkCancellation()
                guard phase == .ready, loaded?.servedModelName == id else {
                    throw ModelLifecycleError.loadFailed(lastError ?? "Model did not become ready")
                }
            }
            activeRequests += 1
        } catch {
            releaseAdmission()
            throw error
        }
        releaseAdmission()
        do {
            let result = try await invoke(operation)
            finishRequest()
            return result
        } catch {
            finishRequest()
            throw error
        }
    }

    private func acquireAdmission() async throws {
        try Task.checkCancellation()
        if !admissionBusy {
            admissionBusy = true
            return
        }
        guard admissionWaiters.count < 64 else {
            throw ModelLifecycleError.busy
        }
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (waiter: CheckedContinuation<Void, Error>) in
                if Task.isCancelled {
                    waiter.resume(throwing: CancellationError())
                } else {
                    admissionWaiters.append((id, waiter))
                }
            }
        } onCancel: {
            Task { await self.cancelAdmission(id) }
        }
    }

    private func cancelAdmission(_ id: UUID) {
        if let index = admissionWaiters.firstIndex(where: { $0.0 == id }) {
            admissionWaiters.remove(at: index).1.resume(throwing: CancellationError())
        }
    }

    private func releaseAdmission() {
        if admissionWaiters.isEmpty {
            admissionBusy = false
        } else {
            admissionWaiters.removeFirst().1.resume()
        }
    }

    func removeDownload(named name: String) async throws -> ManagedModelRemoval {
        try requireNoTransition()
        // The file lease excludes loads in other cooperating processes as well
        // as this listener. Deleting the detached tree runs off the lifecycle
        // actor and adds no inference-path work.
        let removal = try downloads.detach(
            named: name,
            protectedDirectories: loaded?.protectedDirectories ?? [])
        return try await Task.detached(priority: .utility) {
            try removal.finish()
        }.value
    }

    func withModel<Result: Sendable>(
        expectedGeneration: UInt64? = nil,
        _ operation: @Sendable (LoadedModel) async throws -> Result
    ) async throws -> Result {
        guard !stopping else {
            throw ModelLifecycleError.unavailable
        }
        try requireSelection(generation: expectedGeneration, instance: nil)
        guard phase == .ready, loaded != nil else {
            throw ModelLifecycleError.unavailable
        }
        if let request = loaded?.loadRequest, !availability.isAvailable(request) {
            throw ModelLifecycleError.disabledModel
        }
        activeRequests += 1
        do {
            // Keep the model reference in a separate async frame. It must be
            // gone before resuming an unload waiting for the final request.
            let result = try await invoke(operation)
            finishRequest()
            return result
        } catch {
            finishRequest()
            throw error
        }
    }

    private func invoke<Result: Sendable>(
        _ operation: @Sendable (LoadedModel) async throws -> Result
    ) async throws -> Result {
        try await operation(loaded!)
    }

    private func finishRequest() {
        activeRequests -= 1
        if activeRequests == 0 {
            let waiter = drainWaiter
            drainWaiter = nil
            waiter?.resume()
        }
    }

    private func requireNoTransition() throws {
        guard !stopping else {
            throw ModelLifecycleError.unavailable
        }
        guard transition == nil else {
            throw ModelLifecycleError.busy
        }
    }

    private func requireSelection(generation: UInt64?, instance: String?) throws {
        if let generation, generation != modelGeneration {
            throw ModelLifecycleError.staleSelection
        }
        if let instance, instance != instanceID {
            throw ModelLifecycleError.staleSelection
        }
    }

    private func drainBackend() async {
        // Stream consumers may have returned on disconnect while their producer
        // is still cancelling. Join that producer before releasing the backend.
        await loaded?.waitUntilIdle()
    }

    private func replace(with replacement: ModelLoadOperation?) async {
        var replacement = replacement
        if let prepare = replacement?.prepare {
            do { replacement = try await prepare() } catch {
                lastError = error.localizedDescription
                phase = loaded == nil ? .empty : .ready
                targetModel = nil
                transition = nil
                print("Assistant preparation failed: \(error.localizedDescription)")
                return
            }
        }
        if activeRequests > 0 {
            await withCheckedContinuation { drainWaiter = $0 }
        }
        if loaded != nil {
            phase = .unloading
            await drainBackend()
            await loaded?.vision?.shutdown()
            loaded = nil
            modelGeneration &+= 1
            await cleanup()
        }
        if let replacement {
            phase = .loading
            do {
                loaded = try await replacement.load()
                loaded?.runner?.setGenerationMonitoringEnabled(generationMonitoringEnabled)
                if let loaded, let request = loaded.loadRequest {
                    remembered = remembered.filter { $0.value.request.model != request.model }
                    remembered[loaded.servedModelName] = InstalledModelEntry(
                        request: request, descriptor: loaded.descriptor)
                }
                modelGeneration &+= 1
                phase = .ready
                print("Model ready: \(replacement.name)")
            } catch {
                // Failed initializers may have allocated weights. Their async
                // frame has returned before allocator cleanup runs here.
                await cleanup()
                phase = .empty
                lastError = error.localizedDescription
                print("Model load failed: \(error.localizedDescription)")
            }
        } else {
            phase = .empty
        }
        targetModel = nil
        transition = nil
    }
}
