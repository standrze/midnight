import Foundation
import ModelRunnerProtocol
import Testing

@testable import Midnight

@Suite("Remembered model availability")
struct ModelAvailabilityStoreTests {
    @Test func persistenceFollowsCanonicalPathsAcrossAliases() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let checkpoint = directory.appendingPathComponent("model")
        let alias = directory.appendingPathComponent("alias")
        try FileManager.default.createDirectory(at: checkpoint, withIntermediateDirectories: true)
        let weights = checkpoint.appendingPathComponent("weights.safetensors")
        try Data("keep weights".utf8).write(to: weights)
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: checkpoint)
        let file = directory.appendingPathComponent("config/availability.json")
        let request = ModelLoadRequest(model: checkpoint.path)
        var store = try ModelAvailabilityStore(file: file)
        #expect(store.isAvailable(request))
        try store.setAvailable(false, for: request)
        let restarted = try ModelAvailabilityStore(file: file)
        let aliased = ModelLoadRequest(model: alias.path, name: "another-served-name")
        #expect(!restarted.isAvailable(aliased))
        #expect(try Data(contentsOf: weights) == Data("keep weights".utf8))
        try store.setAvailable(true, for: aliased)
        #expect(try ModelAvailabilityStore(file: file).isAvailable(request))
    }

    @Test func failedWritesLeaveModelsAvailableAndMalformedSettingsFail() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let parentFile = directory.appendingPathComponent("not-a-directory")
        try Data().write(to: parentFile)
        var store = try ModelAvailabilityStore(file: parentFile.appendingPathComponent("settings.json"))
        let request = ModelLoadRequest(model: directory.appendingPathComponent("model").path)
        #expect(throws: (any Error).self) { try store.setAvailable(false, for: request) }
        #expect(store.isAvailable(request))
        let malformed = directory.appendingPathComponent("malformed.json")
        try Data("{}".utf8).write(to: malformed)
        #expect(throws: (any Error).self) { try ModelAvailabilityStore(file: malformed) }
    }
}
