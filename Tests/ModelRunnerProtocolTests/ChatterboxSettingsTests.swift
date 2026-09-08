import Foundation
import ModelRunnerProtocol
import Testing

@Suite("Chatterbox model-local settings")
struct ChatterboxSettingsTests {
    private func decode(_ json: String) throws -> ChatterboxSettings {
        try JSONDecoder().decode(ChatterboxSettings.self, from: Data(json.utf8))
    }

    @Test func variantsAndTextConditioning() throws {
        let turbo = try decode(#"{"variant":"turbo"}"#)
        #expect(turbo.preparedText("Hello [laugh]") == "Hello [laugh]")
        #expect(turbo.topK == 1000)
        let multilingual = try decode(#"{"variant":"multilingual","language":"pl","top_p":0.9,"max_tokens":200}"#)
        #expect(multilingual.preparedText("Cześć świecie") == "[pl]cześć[SPACE]świecie")
        #expect(multilingual.maxTokens == 200)
        #expect(multilingual.topP == 0.9)
    }

    @Test(arguments: [
        #"{"variant":"turbo","language":"pl"}"#,
        #"{"variant":"turbo","cfg_weight":0.5}"#,
        #"{"variant":"turbo","min_p":0.2}"#,
        #"{"variant":"multilingual","top_k":10}"#,
        #"{"variant":"multilingual","language":"ja"}"#,
        #"{"variant":"turbo","temperature":0}"#,
        #"{"variant":"turbo","max_tokens":0}"#,
        #"{"variant":"turbo","typo":1}"#,
        #"{"variant":"turbo","voices":{"speaker":"voice.wav"}}"#
    ])
    func rejectsUnsupportedSettings(_ json: String) {
        #expect(throws: (any Error).self) { try decode(json) }
    }

    @Test func configuredReferenceVoice() throws {
        let settings = try decode(#"{"variant":"multilingual","language":"pl","voices":{"speaker":"reference.wav"},"speech_tokenizer":"../s3-tokenizer"}"#)
        #expect(settings.voices["speaker"] == "reference.wav")
    }
}
