import ModelRunnerProtocol
import Testing

@Suite("Audio API routing")
struct AudioAPIRouteTests {
    @Test("Recognizes every official speech and voice route")
    func routeRecognition() {
        #expect(AudioAPIRoute.parse(uri: "/v1/audio/speech") == .speech)
        #expect(AudioAPIRoute.parse(uri: "/v1/audio/voices?limit=10&offset=2&type=preset") == .voices)
        #expect(AudioAPIRoute.parse(uri: "/v1/audio/voices/casual_female") == .voice(id: "casual_female"))
        #expect(
            AudioAPIRoute.parse(uri: "/v1/audio/voices/casual%20female/sample")
                == .voiceSample(id: "casual female")
        )
    }

    @Test("Enforces the official method matrix")
    func methodMatrix() {
        #expect(AudioAPIRoute.speech.allows(method: "POST"))
        #expect(!AudioAPIRoute.speech.allows(method: "GET"))
        #expect(AudioAPIRoute.voices.allows(method: "GET"))
        #expect(AudioAPIRoute.voices.allows(method: "POST"))
        #expect(AudioAPIRoute.voice(id: "voice").allows(method: "PATCH"))
        #expect(AudioAPIRoute.voice(id: "voice").allows(method: "DELETE"))
        #expect(!AudioAPIRoute.voiceSample(id: "voice").allows(method: "POST"))
    }

    @Test("Rejects extra segments and encoded path traversal")
    func unsafePaths() {
        #expect(AudioAPIRoute.parse(uri: "/v1/audio/voices/") == nil)
        #expect(AudioAPIRoute.parse(uri: "/v1/audio/voices/a/sample/extra") == nil)
        #expect(AudioAPIRoute.parse(uri: "/v1/audio/voices/%2Fetc") == nil)
        #expect(AudioAPIRoute.parse(uri: "/v1/audio/voices/%2E%2E") == nil)
        #expect(AudioAPIRoute.parse(uri: "/v1/audio/other") == nil)
    }

    @Test("Returns list query items without changing route matching")
    func queryItems() {
        let items = AudioAPIRoute.queryItems(
            uri: "/v1/audio/voices?limit=25&offset=5&type=preset"
        )
        #expect(items.first(where: { $0.name == "limit" })?.value == "25")
        #expect(items.first(where: { $0.name == "offset" })?.value == "5")
        #expect(items.first(where: { $0.name == "type" })?.value == "preset")
    }
}
