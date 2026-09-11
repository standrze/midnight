import Foundation
import ModelRunnerProtocol
import Testing

@testable import Midnight

@Suite("Persistent model loader preflight")
struct ModelLoaderTests {
    @Test("A new load resolves selected checkpoint settings without retaining earlier overrides")
    func freshSettingsOnEveryLoad() throws {
        try withFixture { directory in
            let first = try checkpoint("first", in: directory)
            let second = try checkpoint("second", in: directory)
            let settingsData = try JSONSerialization.data(withJSONObject: ["mlxRunner": [
                "modelPath": first.path,
                "servedModelName": "configured-first",
                "maximumTokens": 1024,
                "contextLength": 8192,
                "prefillStepSize": 256,
                "engine": "cpu",
            ]])
            let settings = try JSONDecoder().decode(ModelStackSettings.self, from: settingsData)
            try Data(#"{"maximumTokens":96,"contextLength":2048}"#.utf8)
                .write(to: second.appendingPathComponent("midnight.json"))
            let loader = ModelLoader(settings: settings.mlxRunner)
            let firstLoad = try loader.validate(ModelLoadRequest(
                model: first.path, name: "cli-first", maxTokens: 2048, prefillStepSize: 1024))
            #expect(firstLoad.servedModelName == "cli-first")
            #expect(firstLoad.tokenLimit.configuredMaximum == 2048)
            #expect(firstLoad.longContext.prefillStepSize == 1024)

            let secondLoad = try loader.validate(ModelLoadRequest(model: second.path))
            #expect(secondLoad.servedModelName == "second")
            #expect(secondLoad.tokenLimit.configuredMaximum == 96)
            #expect(secondLoad.longContext.contextLength == 2048)
            #expect(secondLoad.longContext.prefillStepSize == 512)
            #expect(secondLoad.engine == .cpu)
            #expect(secondLoad.selection.adapterPath == nil)
            #expect(secondLoad.dflashModel == nil)

            let firstAgain = try loader.validate(ModelLoadRequest(model: first.path))
            #expect(firstAgain.servedModelName == "configured-first")
            #expect(firstAgain.tokenLimit.configuredMaximum == 1024)
            #expect(firstAgain.longContext.prefillStepSize == 256)
        }
    }

    @Test("Missing model and adapter files and invalid limits fail before weights are loaded")
    func rejectsInvalidSelection() throws {
        try withFixture { directory in
            let model = try checkpoint("model", in: directory)
            let loader = ModelLoader(defaultEngine: "cpu")
            for request in [
                ModelLoadRequest(model: "   "),
                ModelLoadRequest(model: directory.appendingPathComponent("missing").path),
                ModelLoadRequest(model: model.path, adapter: directory.appendingPathComponent("missing-adapter").path),
                ModelLoadRequest(model: model.path, adapterScale: 1),
                ModelLoadRequest(model: model.path, maxTokens: 0),
                ModelLoadRequest(model: model.path, contextLength: 32769),
                ModelLoadRequest(model: model.path, prefillStepSize: 0),
                ModelLoadRequest(model: model.path, dflashBlockSize: 4),
                ModelLoadRequest(model: model.path, engine: "unknown"),
            ] {
                #expect(throws: (any Error).self) { try loader.validate(request) }
            }
            try FileManager.default.removeItem(at: model.appendingPathComponent("model.safetensors"))
            #expect(throws: (any Error).self) {
                try loader.validate(ModelLoadRequest(model: model.path))
            }
        }
    }

    @Test("Malformed checkpoint policy and incompatible drafter configuration are rejected")
    func rejectsPolicyAndDrafter() throws {
        try withFixture { directory in
            let model = try checkpoint("model", in: directory)
            let draft = try checkpoint("draft", in: directory)
            let loader = ModelLoader(defaultEngine: "cpu")
            #expect(throws: (any Error).self) {
                try loader.validate(ModelLoadRequest(model: model.path, dflashModel: draft.path))
            }
            try Data(#"{"maximumToken":512}"#.utf8)
                .write(to: model.appendingPathComponent("midnight.json"))
            #expect(throws: ModelStackSettingsError.self) {
                try loader.validate(ModelLoadRequest(model: model.path))
            }
        }
    }

    @Test("Load requests use camelCase and require a model")
    func wireFormat() throws {
        let request = try JSONDecoder().decode(ModelLoadRequest.self, from: Data(
            #"{"model":"local-model","maxTokens":128,"contextLength":4096,"dflashBlockSize":8}"#.utf8))
        #expect(request.model == "local-model")
        #expect(request.maxTokens == 128)
        #expect(request.contextLength == 4096)
        #expect(request.dflashBlockSize == 8)
        #expect(request.adapter == nil)
        #expect(try JSONDecoder().decode(ModelLoadRequest.self, from: JSONEncoder().encode(request)) == request)
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(ModelLoadRequest.self, from: Data("{}".utf8))
        }
    }

#if os(macOS)
    @Test("Explicit Chatterbox paths retain the backend's default-config support")
    func chatterboxDefaultConfiguration() throws {
        try withFixture { directory in
            let model = try checkpoint("speech", in: directory)
            try FileManager.default.removeItem(at: model.appendingPathComponent("config.json"))
            try Data(#"{"variant":"turbo"}"#.utf8)
                .write(to: model.appendingPathComponent("chatterbox.json"))
            let configuration = try ModelLoader(defaultEngine: "metal")
                .validate(ModelLoadRequest(model: model.path))
            guard case .chatterbox = configuration.backend else {
                Issue.record("chatterbox.json should select the native speech backend")
                return
            }
            // Conditioning alone is not a usable set of model weights.
            try FileManager.default.moveItem(at: model.appendingPathComponent("model.safetensors"),
                                            to: model.appendingPathComponent("conds.safetensors"))
            #expect(throws: (any Error).self) {
                try ModelLoader(defaultEngine: "metal").validate(ModelLoadRequest(model: model.path))
            }
        }
    }
#endif

    private func withFixture(_ body: (URL) throws -> Void) throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("midnight-loader-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try body(directory)
    }

    private func checkpoint(_ name: String, in directory: URL) throws -> URL {
        let model = directory.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try Data(#"{"model_type":"llama","max_position_embeddings":32768}"#.utf8)
            .write(to: model.appendingPathComponent("config.json"))
        // Preflight reads metadata only. Deliberately unusable weights ensure
        // these tests never accidentally depend on model/GPU initialization.
        try Data().write(to: model.appendingPathComponent("model.safetensors"))
        return model
    }
}
