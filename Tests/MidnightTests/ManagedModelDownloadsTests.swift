import Foundation
import ModelRunnerProtocol
import Testing

@testable import Midnight

@Suite("Managed model deletion")
struct ManagedModelDownloadsTests {
    @Test("Removal deletes only the chosen managed download")
    func selectedDownload() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try model("first", in: root)
        try model("second", in: root)
        let store = ManagedModelDownloads(root: root)
        #expect(try store.list(protectedDirectories: []).data.map(\.id) == ["first", "second"])
        let detached = try store.detach(named: "first", protectedDirectories: [])
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("first").path))
        let result = try detached.finish()
        #expect(result.deleted && result.id == "first")
        #expect(FileManager.default.fileExists(atPath: root.appendingPathComponent("second/config.json").path))
        #expect(try store.list(protectedDirectories: []).data.map(\.id) == ["second"])
    }

    @Test("Directory URL hints do not change managed-root identity", arguments: [true, false])
    func directoryURLHints(isDirectory: Bool) throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try model("download", in: root)
        let store = ManagedModelDownloads(root: URL(fileURLWithPath: root.path, isDirectory: isDirectory))
        #expect(try store.list(protectedDirectories: []).data.map(\.id) == ["download"])
        #expect(try store.detach(named: "download", protectedDirectories: []).finish().deleted)
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("download").path))
    }

    @Test("Unmanaged folders, malformed records, and unsafe names are never deleted")
    func rejectsUnmanaged() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let unmanaged = root.appendingPathComponent("manual")
        try FileManager.default.createDirectory(at: unmanaged, withIntermediateDirectories: true)
        let store = ManagedModelDownloads(root: root)
        for name in ["../manual", "/tmp/manual", ".", "..", "", "-first", "a/b", "a\\b", "a%2fb"] {
            #expect(throws: ModelRemovalError.self) { try store.detach(named: name, protectedDirectories: []) }
        }
        for name in ["manual", "missing"] {
            #expect(throws: ModelRemovalError.self) { try store.detach(named: name, protectedDirectories: []) }
        }
        try Data(#"{"repository":"owner/model","revision":"main"}"#.utf8)
            .write(to: unmanaged.appendingPathComponent("midnight-download.json"))
        #expect(throws: ModelRemovalError.self) { try store.detach(named: "manual", protectedDirectories: []) }
        #expect(try store.list(protectedDirectories: []).data.isEmpty)
        #expect(FileManager.default.fileExists(atPath: unmanaged.path))
    }

    @Test("Symbolic link roots, model folders, and provenance markers are rejected")
    func rejectsSymlinks() throws {
        let parent = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: parent) }
        let root = parent.appendingPathComponent("models")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try model("outside", in: parent)
        let outside = parent.appendingPathComponent("outside")
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("linked"), withDestinationURL: outside)
        let store = ManagedModelDownloads(root: root)
        #expect(throws: ModelRemovalError.self) { try store.detach(named: "linked", protectedDirectories: []) }
        try model("marker-link", in: root)
        let marker = root.appendingPathComponent("marker-link/midnight-download.json")
        try FileManager.default.removeItem(at: marker)
        try FileManager.default.createSymbolicLink(
            at: marker,
            withDestinationURL: outside.appendingPathComponent("midnight-download.json"))
        #expect(throws: ModelRemovalError.self) { try store.detach(named: "marker-link", protectedDirectories: []) }
        let alias = parent.appendingPathComponent("alias")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: root)
        #expect(throws: ModelRemovalError.self) {
            try ManagedModelDownloads(root: alias).detach(named: "marker-link", protectedDirectories: [])
        }
        #expect(FileManager.default.fileExists(atPath: outside.appendingPathComponent("config.json").path))
    }

    @Test("Loaded model, adapter or draft paths protect matching parents and children")
    func protectsOverlappingPaths() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try model("active", in: root)
        try model("active-other", in: root)
        let store = ManagedModelDownloads(root: root)
        let active = root.appendingPathComponent("active")
        for path in [active, active.appendingPathComponent("adapter"), root] {
            #expect(throws: ModelRemovalError.self) { try store.detach(named: "active", protectedDirectories: [path]) }
        }
        let alias = root.appendingPathComponent("alias")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: active)
        #expect(throws: ModelRemovalError.self) { try store.detach(named: "active", protectedDirectories: [alias]) }
        let listed = try store.list(protectedDirectories: [active]).data
        #expect(listed.first { $0.id == "active" }?.inUse == true)
        #expect(listed.first { $0.id == "active-other" }?.inUse == false)
        _ = try store.detach(named: "active-other", protectedDirectories: [active]).finish()
        #expect(FileManager.default.fileExists(atPath: active.path))
    }

    @Test("A loaded alias cannot bypass protection; unload permits removal")
    func lifecycleProtection() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try model("weights", in: root)
        let gate = RemovalGate()
        let manager = ModelLifecycleManager(
            validate: { _ in
                ModelLoadOperation(name: "public-alias") {
                    await gate.wait()
                    return LoadedModel(
                        servedModelName: "public-alias",
                        tokenLimit: try GenerationTokenLimit(configuredMaximum: 64),
                        protectedDirectories: [root.appendingPathComponent("weights")])
                }
            }, downloads: ManagedModelDownloads(root: root))
        _ = try await manager.load(ModelLoadRequest(model: "weights"))
        await #expect(throws: ModelLifecycleError.self) { try await manager.removeDownload(named: "weights") }
        await gate.open()
        try await wait(.ready, manager)
        await #expect(throws: ModelRemovalError.self) { try await manager.removeDownload(named: "weights") }
        _ = try await manager.unload()
        try await wait(.empty, manager)
        #expect(try await manager.removeDownload(named: "weights").deleted)
    }

    private func temporaryRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory.resolvingSymlinksInPath()
            .appendingPathComponent("midnight-removal-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func model(_ name: String, in root: URL) throws {
        let directory = root.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: directory.appendingPathComponent("config.json"))
        try JSONEncoder().encode(["repository": "owner/model", "revision": String(repeating: "a", count: 40)])
            .write(to: directory.appendingPathComponent("midnight-download.json"))
    }

    private func wait(_ phase: ModelLifecycleState.Phase, _ manager: ModelLifecycleManager) async throws {
        for _ in 0..<300 {
            if await manager.state().phase == phase {
                return
            }
            try await Task.sleep(for: .milliseconds(10))
        }
        throw TestTimeout.timeout
    }
    private enum TestTimeout: Error { case timeout }
}

private actor RemovalGate {
    private var ready = false
    private var waiter: CheckedContinuation<Void, Never>?
    func wait() async {
        guard !ready else {
            return
        }
        await withCheckedContinuation { waiter = $0 }
    }
    func open() {
        ready = true
        waiter?.resume()
        waiter = nil
    }
}
