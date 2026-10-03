import Foundation
import Testing

@testable import Midnight

@Suite("Native protocol discovery")
struct NativeProtocolDiscoveryTests {
    @Test(arguments: [("laguna", "laguna"), ("muse_glimmer", "muse"), ("gemma4", "")])
    func checkpointMetadata(type: String, expected: String) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try JSONSerialization.data(withJSONObject: ["model_type": type]).write(
            to: directory.appendingPathComponent("config.json"))
        #expect(nativeModelProtocol(at: directory.path) == (expected.isEmpty ? nil : expected))
    }
}
