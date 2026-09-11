import Foundation
import ModelRunnerCore
import ModelRunnerProtocol
import Testing

@testable import Midnight

@Suite("Persistent model lifecycle")
struct ModelLifecycleManagerTests {
    @Test("Switching drains the admitted request and releases A before loading B")
    func drainsAndReleasesBeforeReplacement() async throws {
        let trace = LifecycleTrace()
        let requestStarted = LifecycleGate()
        let finishRequest = LifecycleGate()
        let manager = makeManager(trace: trace)
        _ = try await manager.load(ModelLoadRequest(model: "A"))
        #expect(await eventually { await manager.state().phase == .ready })

        let request = Task {
            try await manager.withModel { model in
                await requestStarted.open()
                await finishRequest.wait()
                return model.servedModelName
            }
        }
        await requestStarted.wait()
        let accepted = try await manager.load(ModelLoadRequest(model: "B"))
        #expect(accepted.phase == .draining)
        #expect(accepted.targetModel == "B")
        #expect(accepted.loadedModel == nil)
        #expect(!trace.events.contains("release-A"))
        #expect(!trace.events.contains("load-B"))
        await #expect(throws: ModelLifecycleError.self) {
            try await manager.withModel { $0.servedModelName }
        }
        await #expect(throws: ModelLifecycleError.self) {
            try await manager.load(ModelLoadRequest(model: "C"))
        }
        await #expect(throws: ModelLifecycleError.self) {
            try await manager.unload()
        }

        await finishRequest.open()
        #expect(try await request.value == "A")
        #expect(await eventually { await manager.state().phase == .ready })
        #expect(await manager.state().loadedModel?.id == "B")
        let events = trace.events
        let released = try #require(events.firstIndex(of: "release-A"))
        let cleanup = try #require(events.firstIndex(of: "cleanup"))
        let loaded = try #require(events.firstIndex(of: "load-B"))
        #expect(released < cleanup)
        #expect(cleanup < loaded)
        _ = try await manager.unload()
        #expect(await eventually { await manager.state().phase == .empty })
    }

    @Test("A cancelled consumer releases its lease but unloading still joins its producer")
    func joinsProducerAfterConsumerCancellation() async throws {
        let trace = LifecycleTrace()
        let requestStarted = LifecycleGate()
        let finishProducer = LifecycleGate()
        let manager = makeManager(trace: trace, producerGate: finishProducer)
        _ = try await manager.load(ModelLoadRequest(model: "A"))
        #expect(await eventually { await manager.state().phase == .ready })
        let request = Task {
            try await manager.withModel { model in
                #expect(model.servedModelName == "A")
                await requestStarted.open()
                try await Task.sleep(for: .seconds(30))
            }
        }
        await requestStarted.wait()
        _ = try await manager.load(ModelLoadRequest(model: "B"))
        request.cancel()
        do {
            try await request.value
            Issue.record("The consumer should have observed cancellation")
        } catch is CancellationError {
        }
        #expect(await eventually { trace.events.contains("join-A") })
        #expect(await manager.state().phase == .unloading)
        #expect(!trace.events.contains("release-A"))
        #expect(!trace.events.contains("load-B"))
        await #expect(throws: ModelLifecycleError.self) {
            try await manager.withModel { $0.servedModelName }
        }

        await finishProducer.open()
        #expect(await eventually { await manager.state().phase == .ready })
        let events = trace.events
        let idle = try #require(events.firstIndex(of: "idle-A"))
        let released = try #require(events.firstIndex(of: "release-A"))
        let loaded = try #require(events.firstIndex(of: "load-B"))
        #expect(idle < released)
        #expect(released < loaded)
        _ = try await manager.unload()
        #expect(await eventually { await manager.state().phase == .empty })
    }

    @Test("Load failure leaves an empty server with an error and permits recovery")
    func failureAndRetry() async throws {
        let trace = LifecycleTrace()
        let manager = makeManager(trace: trace)
        _ = try await manager.load(ModelLoadRequest(model: "A"))
        #expect(await eventually { await manager.state().phase == .ready })
        let initial = await manager.state()
        let accepted = try await manager.load(ModelLoadRequest(model: "load-failure"))
        #expect(await eventually { await manager.state().phase == .empty })
        let failed = await manager.state()
        #expect(failed.loadedModel == nil)
        #expect(failed.targetModel == nil)
        #expect(failed.lastError == "Synthetic weight load failure")
        #expect(failed.operationID == accepted.operationID)
        #expect(failed.modelGeneration > initial.modelGeneration)
        #expect(failed.processID == initial.processID)
        #expect(trace.events.contains("release-A"))
        #expect(trace.events.filter { $0 == "cleanup" }.count == 2)
        await #expect(throws: ModelLifecycleError.self) {
            try await manager.withModel { $0.servedModelName }
        }

        _ = try await manager.load(ModelLoadRequest(model: "B"))
        #expect(await eventually { await manager.state().phase == .ready })
        let recovered = await manager.state()
        #expect(recovered.loadedModel?.id == "B")
        #expect(recovered.lastError == nil)
        #expect(recovered.processID == initial.processID)
        #expect(try await manager.withModel { $0.tokenLimit.configuredMaximum } == 128)
        _ = try await manager.unload()
        #expect(await eventually { await manager.state().phase == .empty })
    }

    @Test("Invalid preflight preserves the active model and records initial selection errors")
    func preflightDoesNotDisturbLoadedModel() async throws {
        let trace = LifecycleTrace()
        let manager = makeManager(trace: trace)
        await #expect(throws: LifecycleTestFailure.self) {
            try await manager.load(ModelLoadRequest(model: "invalid"))
        }
        #expect(await manager.state().phase == .empty)
        #expect(await manager.state().lastError == "Synthetic preflight failure")
        _ = try await manager.load(ModelLoadRequest(model: "A"))
        #expect(await eventually { await manager.state().phase == .ready })
        let before = await manager.state()
        await #expect(throws: LifecycleTestFailure.self) {
            try await manager.load(ModelLoadRequest(model: "invalid"))
        }
        let after = await manager.state()
        #expect(after.phase == .ready)
        #expect(after.loadedModel?.id == "A")
        #expect(after.modelGeneration == before.modelGeneration)
        #expect(after.operationID == before.operationID)
        #expect(after.lastError == nil)
        #expect(try await manager.withModel { $0.servedModelName } == "A")
        #expect(!trace.events.contains("release-A"))
        #expect(!trace.events.contains("cleanup"))
        _ = try await manager.unload()
        #expect(await eventually { await manager.state().phase == .empty })
    }

    @Test("A to B to A advances model identity while preserving process identity")
    func repeatedLoadsPublishCurrentMetadata() async throws {
        let trace = LifecycleTrace()
        let manager = makeManager(trace: trace)
        let empty = await manager.state()
        #expect(empty.phase == .empty)
        #expect(empty.modelGeneration == 0)
        #expect(empty.loadedModel == nil)
        var previousGeneration = empty.modelGeneration
        var previousOperation: String?
        for (name, maximumTokens) in [("A", 64), ("B", 128), ("A", 256)] {
            let accepted = try await manager.load(ModelLoadRequest(model: name, maxTokens: maximumTokens))
            #expect(accepted.operationID != nil)
            #expect(accepted.operationID != previousOperation)
            #expect(await eventually { await manager.state().phase == .ready })
            let ready = await manager.state()
            #expect(ready.loadedModel?.id == name)
            #expect(ready.targetModel == nil)
            #expect(ready.modelGeneration > previousGeneration)
            #expect(ready.operationID == accepted.operationID)
            #expect(ready.processID == empty.processID)
            #expect(ready.instanceID == empty.instanceID)
            #expect(try await manager.withModel { $0.tokenLimit.configuredMaximum } == maximumTokens)
            previousGeneration = ready.modelGeneration
            previousOperation = ready.operationID
        }
        _ = try await manager.unload()
        #expect(await eventually { await manager.state().phase == .empty })
        let unloaded = await manager.state()
        #expect(unloaded.loadedModel == nil)
        #expect(unloaded.modelGeneration > previousGeneration)
        #expect(unloaded.processID == empty.processID)
        #expect(unloaded.instanceID == empty.instanceID)
        #expect(trace.events.filter { $0 == "release-A" }.count == 2)
        #expect(trace.events.filter { $0 == "release-B" }.count == 1)
        #expect(trace.events.filter { $0 == "cleanup" }.count == 3)
        let alreadyEmpty = try await manager.unload()
        #expect(alreadyEmpty.phase == .empty)
        #expect(alreadyEmpty.modelGeneration == unloaded.modelGeneration)
    }

    private func makeManager(trace: LifecycleTrace, producerGate: LifecycleGate? = nil) -> ModelLifecycleManager {
        ModelLifecycleManager(validate: { request in
            if request.model == "invalid" { throw LifecycleTestFailure.preflight }
            return ModelLoadOperation(name: request.model) {
                trace.record("load-\(request.model)")
                if request.model == "load-failure" { throw LifecycleTestFailure.load }
                let backend = try LifecycleSpeechBackend(
                    name: request.model, trace: trace,
                    producerGate: request.model == "A" ? producerGate : nil)
                return LoadedModel(servedModelName: request.model,
                    tokenLimit: try GenerationTokenLimit(configuredMaximum: request.maxTokens ?? 128),
                    speechSynthesizer: backend)
            }
        }, cleanup: { trace.record("cleanup") })
    }

    private func eventually(_ predicate: @escaping @Sendable () async -> Bool) async -> Bool {
        let deadline = ContinuousClock.now.advanced(by: .seconds(2))
        while ContinuousClock.now < deadline {
            if await predicate() { return true }
            try? await Task.sleep(for: .milliseconds(5))
        }
        return await predicate()
    }
}

private enum LifecycleTestFailure: LocalizedError {
    case preflight, load
    var errorDescription: String? {
        switch self {
        case .preflight: "Synthetic preflight failure"
        case .load: "Synthetic weight load failure"
        }
    }
}

private actor LifecycleGate {
    private var isOpen = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        guard !isOpen else { return }
        await withCheckedContinuation { waiters.append($0) }
    }

    func open() {
        isOpen = true
        let pending = waiters
        waiters.removeAll()
        pending.forEach { $0.resume() }
    }
}

private final class LifecycleTrace: @unchecked Sendable {
    private let lock = NSLock()
    private var recordedEvents: [String] = []
    var events: [String] { lock.withLock { recordedEvents } }
    func record(_ event: String) { lock.withLock { recordedEvents.append(event) } }
}

private final class LifecycleSpeechBackend: LocalSpeechSynthesizing {
    let servedModelName: String
    let voiceCatalog: VoxtralVoiceCatalog
    private let trace: LifecycleTrace
    private let producerGate: LifecycleGate?

    init(name: String, trace: LifecycleTrace, producerGate: LifecycleGate?) throws {
        servedModelName = name
        self.trace = trace
        self.producerGate = producerGate
        let fixture = FileManager.default.temporaryDirectory
            .appendingPathComponent("midnight-lifecycle-voice-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: fixture, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: fixture) }
        try Data(#"{"model_type":"voxtral_tts","multimodal":{"audio_tokenizer_args":{"voice":{"en_female":0}}}}"#.utf8)
            .write(to: fixture.appendingPathComponent("config.json"))
        voiceCatalog = try #require(try VoxtralVoiceCatalog(modelDirectory: fixture.path))
    }

    func stream(request: LocalSpeechSynthesisRequest) async -> AsyncThrowingStream<LocalSpeechSynthesisEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }

    func waitUntilIdle() async {
        trace.record("join-\(servedModelName)")
        await producerGate?.wait()
        trace.record("idle-\(servedModelName)")
    }

    deinit { trace.record("release-\(servedModelName)") }
}
