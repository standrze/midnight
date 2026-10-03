import Foundation
import ModelFiles
import Testing
import VisionProtocol

@Suite("FastVLM expanded image admission")
struct FastVLMAdmissionTests {
    func configuration(_ modify: (inout [String: Any]) -> Void = { _ in }) throws -> Data {
        var object: [String: Any] = [
            "model_type": "llava_qwen2", "tokenizer_model_max_length": 8192,
            "vision_config": [
                "layers": [2, 12, 24, 4, 2], "downsamples": [true, true, true, true, true], "down_stride": 2,
                "down_patch_size": 7, "image_size": 1024,
            ],
        ]
        modify(&object)
        return try JSONSerialization.data(withJSONObject: object)
    }
    let processor = Data(#"{"processor_class":"FastVLMProcessor","crop_size":{"width":1024,"height":1024}}"#.utf8)
    @Test func includesImageExpansionAtContextBoundary() throws {
        let admission = try FastVLMAdmission(configuration: configuration(), processor: processor)
        let fitting = [-200] + Array(repeating: 123, count: 7935)
        #expect(try admission.promptPositions(tokenIDs: fitting, maximumTokens: 1) == 8191)
        #expect(throws: VisionError.self) { try admission.promptPositions(tokenIDs: fitting + [123], maximumTokens: 1) }
        // A placeholder-only budget would have accepted both above incorrectly.
        #expect(try admission.promptPositions(tokenIDs: [42, -200, 71], maximumTokens: 10) == 258)
    }
    @Test func rejectsUnverifiedImageLayoutsAndExtraMarkers() throws {
        #expect(throws: VisionError.self) {
            try FastVLMAdmission(configuration: configuration { $0["model_type"] = "qwen2_vl" }, processor: processor)
        }
        let admission = try FastVLMAdmission(configuration: configuration(), processor: processor)
        #expect(throws: VisionError.self) {
            try admission.promptPositions(tokenIDs: [-200, 42, -200], maximumTokens: 1)
        }
        #expect(throws: VisionError.self) { try admission.promptPositions(tokenIDs: [42, 71], maximumTokens: 1) }
    }
    @Test func honorsSmallerModelContext() throws {
        let admission = try FastVLMAdmission(
            configuration: configuration { $0["tokenizer_model_max_length"] = 512 }, processor: processor)
        #expect(throws: VisionError.self) { try admission.promptPositions(tokenIDs: [-200, 42], maximumTokens: 256) }
    }
    @Test func managedModelsAndAliasesHoldIndependentDeletionLeases() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "vision-leases-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("model/child"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("model", isDirectory: true)
        try Data("{}".utf8).write(to: model.appendingPathComponent("midnight-download.json"))
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("alias"), withDestinationURL: model)
        let locks = try ModelFileLeaseDirectory(root: root)
        for path in [model, model.appendingPathComponent("child"), root.appendingPathComponent("alias")] {
            var parent: ModelFileUsage? = try .acquire(protectedDirectories: [model], root: root)
            let worker = try ModelFileUsage.acquire(protectedDirectories: [path], root: root)
            #expect(parent != nil)
            parent = nil
            #expect(throws: ModelFileLeaseError.self) { try locks.model("model", shared: false) }
            withExtendedLifetime(worker) {}
        }
        let removal = try locks.model("model", shared: false)
        withExtendedLifetime(removal) {}
    }
}
