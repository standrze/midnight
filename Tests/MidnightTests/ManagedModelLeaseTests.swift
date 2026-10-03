import Foundation
import ModelRunnerCore
import ModelRunnerProtocol
import Testing

@testable import Midnight

#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif

@Suite("Cross-process managed model leases")
struct ManagedModelLeaseTests {
    @Test("A different process's reader prevents deletion without blocking unrelated models")
    func foreignReader() throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ManagedModelDownloads(root: root)
        let process = try ForeignModelLease(root: root, name: "weights", shared: true)
        defer { process.stop() }
        // Prove the foreign reader alone blocks deletion before taking a local
        // lease, which would otherwise mask a broken cross-process check.
        do {
            _ = try store.detach(named: "weights", protectedDirectories: [])
            Issue.record("Expected the other process's shared lease to block removal")
        } catch let error as ModelRemovalError {
            #expect(error.code == "model_in_use")
        }
        #expect(try store.list(protectedDirectories: []).data.first { $0.id == "weights" }?.inUse == true)
        let usage = try store.acquireUsage(protectedDirectories: [root.appendingPathComponent("weights")])
        #expect(throws: ModelRemovalError.self) { try store.detach(named: "weights", protectedDirectories: []) }
        #expect(try store.list(protectedDirectories: []).data.first { $0.id == "weights" }?.inUse == true)
        #expect(try store.detach(named: "other", protectedDirectories: []).finish().deleted)
        process.stop()
        // A second reader in this process still owns the first model.
        #expect(throws: ModelRemovalError.self) { try store.detach(named: "weights", protectedDirectories: []) }
        withExtendedLifetime(usage) {}
    }

    @Test("Removal's exclusive lease is checked before provenance and backend weights")
    func foreignWriterBeforeValidation() async throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ManagedModelDownloads(root: root)
        let loader = ModelLoader(defaultEngine: "cpu", downloads: store)
        let configuration = try loader.validate(ModelLoadRequest(model: root.appendingPathComponent("weights").path))
        let process = try ForeignModelLease(root: root, name: "weights", shared: false)
        defer { process.stop() }
        try FileManager.default.removeItem(at: root.appendingPathComponent("weights/midnight-download.json"))
        do {
            _ = try store.detach(named: "weights", protectedDirectories: [])
            Issue.record("Expected an in-use lease")
        } catch let error as ModelRemovalError {
            #expect(error.code == "model_in_use")
        }
        // Invalid dummy weights must never reach the backend while removal owns
        // the exclusive lease. This check performs no inference or GPU work.
        do {
            _ = try await loader.load(configuration)
            Issue.record("Expected load to fail before opening weights")
        } catch let error as ModelRemovalError {
            #expect(error.code == "model_in_use")
        }
        process.stop()
        do {
            _ = try store.detach(named: "weights", protectedDirectories: [])
            Issue.record("Expected the missing provenance record to be rejected")
        } catch let error as ModelRemovalError {
            #expect(error.code == "model_not_managed")
        }
    }

    @Test("Crash/exit releases a foreign lease and persistent lock names survive re-download")
    func processExitAndReuse() throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ManagedModelDownloads(root: root)
        let process = try ForeignModelLease(root: root, name: "weights", shared: true)
        process.stop()
        #expect(try store.detach(named: "weights", protectedDirectories: []).finish().deleted)
        let lockPath = root.appendingPathComponent(".leases/weights")
        let before = try FileManager.default.attributesOfItem(atPath: lockPath.path)[.systemFileNumber] as? NSNumber
        try checkpoint("weights", in: root)
        var usage: ManagedModelUsage? = try store.acquireUsage(protectedDirectories: [
            root.appendingPathComponent("weights")
        ])
        #expect(usage != nil)
        #expect(throws: ModelRemovalError.self) { try store.detach(named: "weights", protectedDirectories: []) }
        usage = nil
        #expect(try store.detach(named: "weights", protectedDirectories: []).finish().deleted)
        let after = try FileManager.default.attributesOfItem(atPath: lockPath.path)[.systemFileNumber] as? NSNumber
        #expect(before != nil && before == after)
    }

    @Test("Owner teardown releases a lease even while an inherited descriptor remains open", arguments: [false, true])
    func releasesDuplicatedDescriptor(aggregate: Bool) throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let locks = try ManagedModelLeaseDirectory(root: root)
        var usage: ManagedModelFileLease? =
            aggregate
            ? try locks.root(shared: false) : try locks.model("weights", shared: true)
        weak let lifetime = usage
        // dup reproduces the shared open-file-description lifetime of a child
        // paused between fork and exec, without forking the multithreaded tests.
        let duplicate = try duplicateLeaseDescriptor(
            at: root.appendingPathComponent(aggregate ? ".leases/.root.lock" : ".leases/weights"))
        defer { close(duplicate) }
        withExtendedLifetime(usage) {}
        usage = nil
        #expect(lifetime == nil)
        let replacement =
            aggregate
            ? try locks.root(shared: true) : try locks.model("weights", shared: false)
        withExtendedLifetime(replacement) {}
    }

    @Test("Aliases, child files, root ancestors and dependencies acquire the matching leases")
    func dependenciesAndAncestors() throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ManagedModelDownloads(root: root)
        let alias = root.appendingPathComponent("alias")
        try FileManager.default.createSymbolicLink(
            at: alias, withDestinationURL: root.appendingPathComponent("weights"))
        let paths = [alias, root.appendingPathComponent("other/config.json")]
        var usage: ManagedModelUsage? = try store.acquireUsage(protectedDirectories: paths)
        #expect(usage?.leases.count == 2)
        for name in ["weights", "other"] {
            #expect(throws: ModelRemovalError.self) { try store.detach(named: name, protectedDirectories: []) }
        }
        usage = nil
        for ancestor in [root, root.deletingLastPathComponent()] {
            let rootUsage = try store.acquireUsage(protectedDirectories: [ancestor])
            #expect(rootUsage.leases.count == 1)
            #expect(throws: ModelRemovalError.self) { try store.detach(named: "weights", protectedDirectories: []) }
            withExtendedLifetime(rootUsage) {}
        }
        #expect(try store.detach(named: "weights", protectedDirectories: []).finish().deleted)
    }

    @Test("Resolved backend configuration protects adapter, draft and speech assets")
    func configuredDependencies() throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        try checkpoint("draft", in: root)
        let store = ManagedModelDownloads(root: root)
        let model = root.appendingPathComponent("weights")
        let other = root.appendingPathComponent("other")
        let draft = root.appendingPathComponent("draft")
        let text = ValidatedModelConfiguration(
            selection: ResolvedModelSelection(modelPath: model.path, adapterPath: other.path, servedModelName: "text"),
            engine: .cpu, tokenLimit: try GenerationTokenLimit(configuredMaximum: 64),
            adapterScale: nil, dflashModel: draft.path, dflashBlockSize: nil,
            longContext: try LongContextOptions(), backend: .text)
        let usage = try store.acquireUsage(protectedDirectories: text.protectedDirectories)
        #expect(usage.leases.count == 3)
        for name in ["weights", "other", "draft"] {
            #expect(throws: ModelRemovalError.self) { try store.detach(named: name, protectedDirectories: []) }
        }
        withExtendedLifetime(usage) {}
    }

    @Test("Missing files release partially acquired leases and unmanaged paths need no root scan")
    func missingAndUnrelated() throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ManagedModelDownloads(root: root)
        #expect(throws: ModelRemovalError.self) {
            try store.acquireUsage(protectedDirectories: [
                root.appendingPathComponent("weights"), root.appendingPathComponent("other/missing"),
            ])
        }
        #expect(try store.detach(named: "weights", protectedDirectories: []).finish().deleted)
        let missingRoot = root.appendingPathComponent("absent-model-root")
        let unrelated = try ManagedModelDownloads(root: missingRoot)
            .acquireUsage(protectedDirectories: [root.appendingPathComponent("other")])
        #expect(unrelated.leases.isEmpty)
        #expect(!FileManager.default.fileExists(atPath: missingRoot.path))
    }

    @Test("Symbolic-link lease stores and lock files fail closed")
    func symlinkLocks() throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ManagedModelDownloads(root: root)
        let outside = root.appendingPathComponent("outside")
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: false)
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent(".leases"), withDestinationURL: outside)
        #expect(throws: ModelRemovalError.self) { try store.detach(named: "weights", protectedDirectories: []) }
        #expect(throws: ModelRemovalError.self) {
            try store.acquireUsage(protectedDirectories: [root.appendingPathComponent("weights")])
        }
        try FileManager.default.removeItem(at: root.appendingPathComponent(".leases"))
        let directory = try ManagedModelLeaseDirectory(root: root)
        let target = outside.appendingPathComponent("target")
        try Data("do not alter".utf8).write(to: target)
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent(".leases/weights"), withDestinationURL: target)
        #expect(throws: ModelRemovalError.self) { try directory.model("weights", shared: true) }
        #expect(throws: ModelRemovalError.self) { try store.detach(named: "weights", protectedDirectories: []) }
        #expect(try Data(contentsOf: target) == Data("do not alter".utf8))
        #expect(FileManager.default.fileExists(atPath: root.appendingPathComponent("weights/config.json").path))
    }

    @Test("Loading, request drain and backend producer drain retain leases; unload releases them")
    func lifecycleOwnership() async throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ManagedModelDownloads(root: root)
        let loading = LeaseGate()
        let loaded = LeaseGate()
        let entered = LeaseGate()
        let requestFinish = LeaseGate()
        let draining = LeaseGate()
        let producerFinish = LeaseGate()
        let manager = ModelLifecycleManager(
            validate: { _ in
                ModelLoadOperation(name: "weights") {
                    let usage = try store.acquireUsage(protectedDirectories: [root.appendingPathComponent("weights")])
                    await loading.open()
                    await loaded.wait()
                    return LoadedModel(
                        servedModelName: "weights", tokenLimit: try GenerationTokenLimit(configuredMaximum: 64),
                        speechSynthesizer: try LeaseSpeechBackend(
                            draining: draining, finish: producerFinish, fixture: root),
                        fileUsage: usage)
                }
            }, downloads: store)
        _ = try await manager.load(ModelLoadRequest(model: "weights"))
        await loading.wait()
        #expect(throws: ModelRemovalError.self) { try store.detach(named: "weights", protectedDirectories: []) }
        await loaded.open()
        try await wait(.ready, manager)
        let request = Task {
            try await manager.withModel { _ in
                await entered.open()
                await requestFinish.wait()
            }
        }
        await entered.wait()
        _ = try await manager.unload()
        #expect(throws: ModelRemovalError.self) { try store.detach(named: "weights", protectedDirectories: []) }
        await requestFinish.open()
        try await request.value
        await draining.wait()
        #expect(await manager.state().phase == .unloading)
        #expect(throws: ModelRemovalError.self) { try store.detach(named: "weights", protectedDirectories: []) }
        await producerFinish.open()
        try await wait(.empty, manager)
        #expect(try store.detach(named: "weights", protectedDirectories: []).finish().deleted)
    }

    @Test("A failed initializer releases every dependency lease")
    func failedInitializer() async throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ManagedModelDownloads(root: root)
        let acquired = LeaseGate()
        let fail = LeaseGate()
        let manager = ModelLifecycleManager(
            validate: { _ in
                ModelLoadOperation(name: "weights") {
                    let usage = try store.acquireUsage(protectedDirectories: [
                        root.appendingPathComponent("weights"), root.appendingPathComponent("other"),
                    ])
                    defer { withExtendedLifetime(usage) {} }
                    await acquired.open()
                    await fail.wait()
                    throw LeaseTestError.expected
                }
            }, downloads: store)
        _ = try await manager.load(ModelLoadRequest(model: "weights"))
        await acquired.wait()
        for name in ["weights", "other"] {
            #expect(throws: ModelRemovalError.self) { try store.detach(named: name, protectedDirectories: []) }
        }
        await fail.open()
        try await wait(.empty, manager)
        for name in ["weights", "other"] {
            #expect(try store.detach(named: name, protectedDirectories: []).finish().deleted)
        }
    }

    /// Locate the test fixture's owned descriptor by filesystem identity rather
    /// than exposing the production lease's private descriptor to callers.
    private func duplicateLeaseDescriptor(at path: URL) throws -> Int32 {
        let reference = open(path.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard reference >= 0 else {
            throw LeaseTestError.childFailed
        }
        defer { close(reference) }
        var expected = stat()
        guard fstat(reference, &expected) == 0 else {
            throw LeaseTestError.childFailed
        }
        let limit = Int32(clamping: sysconf(Int32(_SC_OPEN_MAX)))
        for candidate in 0..<limit where candidate != reference {
            var metadata = stat()
            if fstat(candidate, &metadata) == 0,
                metadata.st_dev == expected.st_dev, metadata.st_ino == expected.st_ino
            {
                let duplicate = fcntl(candidate, F_DUPFD_CLOEXEC, 0)
                guard duplicate >= 0 else {
                    throw LeaseTestError.childFailed
                }
                return duplicate
            }
        }
        throw LeaseTestError.childFailed
    }

    private func fixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory.resolvingSymlinksInPath()
            .appendingPathComponent("midnight-lease-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try checkpoint("weights", in: root)
        try checkpoint("other", in: root)
        return root
    }

    private func checkpoint(_ name: String, in root: URL) throws {
        let model = root.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try Data(#"{"model_type":"llama","max_position_embeddings":32768}"#.utf8).write(
            to: model.appendingPathComponent("config.json"))
        try Data().write(to: model.appendingPathComponent("model.safetensors"))
        try JSONEncoder().encode(["repository": "owner/model", "revision": String(repeating: "a", count: 40)])
            .write(to: model.appendingPathComponent("midnight-download.json"))
    }

    private func wait(_ phase: ModelLifecycleState.Phase, _ manager: ModelLifecycleManager) async throws {
        for _ in 0..<300 {
            if await manager.state().phase == phase {
                return
            }
            try await Task.sleep(for: .milliseconds(10))
        }
        throw LeaseTestError.timeout
    }
}

private enum LeaseTestError: Error { case expected, timeout, childFailed }

/// A separate Python process uses the same OS advisory locking primitive. This
/// catches same-process-only implementations and closes the lease on process exit.
private final class ForeignModelLease {
    private let process = Process()
    private let input = Pipe()
    init(root: URL, name: String, shared: Bool) throws {
        _ = try ManagedModelLeaseDirectory(root: root)
        let output = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "python3", "-c",
            """
            import fcntl, os, sys
            fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(fd, fcntl.LOCK_SH if sys.argv[2] == 'shared' else fcntl.LOCK_EX)
            print('R', flush=True)
            sys.stdin.readline()
            os.close(fd)
            """, root.appendingPathComponent(".leases/\(name)").path, shared ? "shared" : "exclusive",
        ]
        process.standardInput = input
        process.standardOutput = output
        try process.run()
        guard output.fileHandleForReading.readData(ofLength: 1) == Data("R".utf8) else {
            stop()
            throw LeaseTestError.childFailed
        }
    }
    func stop() {
        guard process.isRunning else {
            return
        }
        try? input.fileHandleForWriting.close()
        process.waitUntilExit()
    }
    deinit { stop() }
}

private actor LeaseGate {
    private var ready = false
    private var continuation: CheckedContinuation<Void, Never>?
    func wait() async {
        guard !ready else {
            return
        }
        await withCheckedContinuation { continuation = $0 }
    }
    func open() {
        ready = true
        continuation?.resume()
        continuation = nil
    }
}

private final class LeaseSpeechBackend: LocalSpeechSynthesizing {
    let servedModelName = "weights"
    let voiceCatalog: VoxtralVoiceCatalog
    let draining: LeaseGate
    let finish: LeaseGate
    init(draining: LeaseGate, finish: LeaseGate, fixture: URL) throws {
        self.draining = draining
        self.finish = finish
        let speech = fixture.appendingPathComponent("speech")
        try FileManager.default.createDirectory(at: speech, withIntermediateDirectories: true)
        try Data(#"{"model_type":"voxtral_tts","multimodal":{"audio_tokenizer_args":{"voice":{"en_female":0}}}}"#.utf8)
            .write(to: speech.appendingPathComponent("config.json"))
        voiceCatalog = try #require(try VoxtralVoiceCatalog(modelDirectory: speech.path))
    }
    func stream(request: LocalSpeechSynthesisRequest) async -> AsyncThrowingStream<LocalSpeechSynthesisEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }
    func waitUntilIdle() async {
        await draining.open()
        await finish.wait()
    }
}
