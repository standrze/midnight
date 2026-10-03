import Foundation
import ModelRunnerCore
import Testing

@Suite("Voxtral preset voice catalog wiring")
struct VoxtralVoiceCatalogWiringTests {
    @Test("Loads and orders preset voices from a Voxtral configuration")
    func loadsCatalog() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let configuration = """
            {
              "model_type": "voxtral_tts",
              "multimodal": {
                "audio_tokenizer_args": {
                  "voice": {
                    "fr_female": 2,
                    "casual_male": 0,
                    "pt_male": 1
                  }
                }
              }
            }
            """
        try Data(configuration.utf8).write(to: directory.appendingPathComponent("config.json"))

        let loadedCatalog = try VoxtralVoiceCatalog(modelDirectory: directory.path)
        let catalog = try #require(loadedCatalog)
        #expect(catalog.voices.map(\.id) == ["casual_male", "pt_male", "fr_female"])
        #expect(catalog.voices.map(\.languages) == [["en"], ["pt"], ["fr"]])
        #expect(catalog.voice(id: "fr_female")?.name == "French Female")
        let frenchAPIID = try #require(catalog.voice(id: "fr_female")?.apiID)
        #expect(UUID(uuidString: frenchAPIID) != nil)
        #expect(catalog.voice(id: frenchAPIID)?.id == "fr_female")
        #expect(catalog.voice(id: "missing") == nil)
    }

    @Test("Ignores non-Voxtral model configurations")
    func ignoresOtherModels() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("{\"model_type\":\"gemma4\"}".utf8)
            .write(to: directory.appendingPathComponent("config.json"))

        #expect(try VoxtralVoiceCatalog(modelDirectory: directory.path) == nil)
    }

    @Test("Core reads preset voices from the checkpoint instead of hard-coding them")
    func sourceWiring() throws {
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: packageRoot
                .appendingPathComponent("Sources/ModelRunnerCore/VoxtralVoiceCatalog.swift"),
            encoding: .utf8
        )

        #expect(source.contains("root[\"model_type\"] as? String == \"voxtral_tts\""))
        #expect(source.contains("tokenizer[\"voice\"]"))
        #expect(source.contains("sorted { lhs, rhs"))
        #expect(!source.contains("casual_female"))
    }
}
