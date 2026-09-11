import Foundation
import ModelRunnerCore
import ModelRunnerProtocol

/// All model-specific state travels together for the full lifetime of a request.
final class LoadedModel: Sendable {
    let runner: LocalModelRunner?
    let servedModelName: String
    let tokenLimit: GenerationTokenLimit
    let speechSynthesizer: (any LocalSpeechSynthesizing)?
    let created = Int(Date().timeIntervalSince1970)

    init(runner: LocalModelRunner? = nil, servedModelName: String,
         tokenLimit: GenerationTokenLimit,
         speechSynthesizer: (any LocalSpeechSynthesizing)? = nil) {
        self.runner = runner
        self.servedModelName = servedModelName
        self.tokenLimit = tokenLimit
        self.speechSynthesizer = speechSynthesizer
    }

    func waitUntilIdle() async {
        await runner?.waitUntilIdle()
        await speechSynthesizer?.waitUntilIdle()
    }

    var descriptor: ModelLifecycleDescriptor {
        ModelLifecycleDescriptor(id: servedModelName, created: created,
            contextLength: runner?.contextLength, prefillStepSize: runner?.prefillStepSize,
            kvCompression: runner?.kvCompression, memoryLimitBytes: runner?.memoryLimitBytes)
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

    enum CodingKeys: String, CodingKey {
        case id, created, object
        case ownedBy = "owned_by"
        case contextLength = "context_length"
        case prefillStepSize = "prefill_step_size"
        case kvCompression = "kv_compression"
        case memoryLimitBytes = "memory_limit_bytes"
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
}

struct ModelLoadOperation: Sendable {
    let name: String
    let load: @Sendable () async throws -> LoadedModel
}

enum ModelLifecycleError: LocalizedError {
    case busy
    case unavailable

    var errorDescription: String? {
        switch self {
        case .busy: "A model load or unload is already in progress."
        case .unavailable: "No model is ready. Check /v1/runtime for load status."
        }
    }
}

/// The listener owns this actor; only this actor owns an installed backend.
/// Admission closes synchronously before a transition can suspend.
actor ModelLifecycleManager {
    private let validate: @Sendable (ModelLoadRequest) throws -> ModelLoadOperation
    private let cleanup: @Sendable () async -> Void
    private var loaded: LoadedModel?
    private var phase: ModelLifecycleState.Phase = .empty
    private var operationID: String?
    private var modelGeneration: UInt64 = 0
    private var targetModel: String?
    private var lastError: String?
    private var transition: Task<Void, Never>?
    private var activeRequests = 0
    private var drainWaiter: CheckedContinuation<Void, Never>?

    init(loader: ModelLoader) {
        validate = { request in
            let configuration = try loader.validate(request)
            return ModelLoadOperation(name: configuration.servedModelName) {
                try await loader.load(configuration)
            }
        }
        cleanup = { await LocalModelRuntime.releaseUnusedMemory() }
    }

    init(validate: @escaping @Sendable (ModelLoadRequest) throws -> ModelLoadOperation,
         cleanup: @escaping @Sendable () async -> Void = {}) {
        self.validate = validate
        self.cleanup = cleanup
    }

    func state() -> ModelLifecycleState {
        ModelLifecycleState(
            instanceID: ProcessInfo.processInfo.environment["MIDNIGHT_CONTROL_INSTANCE"],
            processID: ProcessInfo.processInfo.processIdentifier,
            phase: phase, operationID: operationID, modelGeneration: modelGeneration,
            loadedModel: phase == .ready ? loaded?.descriptor : nil,
            targetModel: targetModel, lastError: lastError,
            memory: LocalModelRuntime.memorySnapshot())
    }

    func load(_ request: ModelLoadRequest) throws -> ModelLifecycleState {
        try requireNoTransition()
        let operation: ModelLoadOperation
        do { operation = try validate(request) }
        catch {
            // A malformed replacement must leave the existing model usable.
            if loaded == nil { lastError = error.localizedDescription }
            throw error
        }
        operationID = UUID().uuidString
        targetModel = operation.name
        lastError = nil
        phase = loaded == nil ? .loading : .draining
        transition = Task { await self.replace(with: operation) }
        return state()
    }

    func unload() throws -> ModelLifecycleState {
        try requireNoTransition()
        operationID = UUID().uuidString
        targetModel = nil
        lastError = nil
        guard loaded != nil else { return state() }
        phase = .draining
        transition = Task { await self.replace(with: nil) }
        return state()
    }

    func withModel<Result: Sendable>(
        _ operation: @Sendable (LoadedModel) async throws -> Result
    ) async throws -> Result {
        guard phase == .ready, loaded != nil else { throw ModelLifecycleError.unavailable }
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
        guard transition == nil else { throw ModelLifecycleError.busy }
    }

    private func drainBackend() async {
        // Stream consumers may have returned on disconnect while their producer
        // is still cancelling. Join that producer before releasing the backend.
        await loaded?.waitUntilIdle()
    }

    private func replace(with replacement: ModelLoadOperation?) async {
        if activeRequests > 0 {
            await withCheckedContinuation { drainWaiter = $0 }
        }
        if loaded != nil {
            phase = .unloading
            await drainBackend()
            loaded = nil
            modelGeneration &+= 1
            await cleanup()
        }
        if let replacement {
            phase = .loading
            do {
                loaded = try await replacement.load()
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
