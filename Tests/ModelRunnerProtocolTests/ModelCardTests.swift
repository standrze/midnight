import Foundation
import ModelRunnerProtocol
import Testing

@Suite("Editable model cards")
struct ModelCardTests {
    @Test("Assistant references are optional and preserve the snake-case wire field")
    func assistantWireFormat() throws {
        let legacy = try JSONDecoder().decode(ModelCard.self, from: Data(#"{"name":"Legacy"}"#.utf8))
        #expect(legacy.assistantModel == nil)
        let card = ModelCard(name: "Gemma", assistantModel: "google/gemma-4-26B-A4B-it-assistant")
        let data = try JSONEncoder().encode(card.withCapabilities(.init()))
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["assistant_model"] as? String == card.assistantModel)
        #expect(try JSONDecoder().decode(ModelCard.self, from: data).assistantModel == card.assistantModel)
    }

    @Test("Voice metadata is optional, round-trips, and describes cloning requirements")
    func voiceWireFormat() throws {
        let legacy = try JSONDecoder().decode(ModelCard.self, from: Data(#"{"name":"Legacy"}"#.utf8))
        #expect(legacy.voices == nil)
        let card = legacy.withVoices([
            .init(
                id: "clone", slug: "clone", name: "Clone",
                languages: ["en"], requiresReferenceAudio: true)
        ])
        let data = try JSONEncoder().encode(card)
        #expect(try JSONDecoder().decode(ModelCard.self, from: data) == card)
        #expect(String(decoding: data, as: UTF8.self).contains("requires_reference_audio"))
        #expect(card.withVoices(nil).voices == nil)
    }

    @Test("Legacy cards leave support unknown; capability fields retain explicit false values")
    func capabilityWireFormat() throws {
        let decoder = JSONDecoder()
        let legacy = try decoder.decode(ModelCard.self, from: Data(#"{"name":"Legacy"}"#.utf8))
        #expect(legacy.capabilities == nil)
        let card = legacy.withCapabilities(.init(vision: true, audioOutput: true))
        let encoded = try JSONEncoder().encode(card)
        let object = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        let capabilities = try #require(object["capabilities"] as? [String: Bool])
        #expect(capabilities == ["vision": true, "audio_input": false, "audio_output": true])
        #expect(try decoder.decode(ModelCard.self, from: encoded) == card)
        #expect(legacy.capabilities == nil)
        let textCard = card.withCapabilities(.init())
        #expect(textCard.capabilities == .init(vision: false, audioInput: false, audioOutput: false))
        #expect(textCard.name == legacy.name)
    }

    @Test("Missing cards, editable names, and publisher cards are preserved")
    func roundTrip() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(try ModelCard.load(directory: root.path) == nil)
        try ModelCard.createIfMissing(directory: root, name: "Talkie 1930")
        try ModelCard.createIfMissing(directory: root, name: "Do not overwrite")
        #expect(try ModelCard.load(directory: root.path)?.name == "Talkie 1930")
        let file = root.appendingPathComponent(ModelCard.filename)
        try Data(#"{"name":"  Polski model  ","description":"Książki","futureField":true}"#.utf8).write(to: file)
        #expect(try ModelCard.load(directory: root.path) == ModelCard(name: "Polski model", description: "Książki"))
        for invalid in [
            "{", #"{"description":"missing name"}"#, #"{"name":"\n"}"#,
            #"{"name":"bad\u0000name"}"#, String(repeating: " ", count: 65_537),
        ] {
            try Data(invalid.utf8).write(to: file)
            #expect(throws: (any Error).self) { try ModelCard.load(directory: root.path) }
        }
    }
}
