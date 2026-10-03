import Foundation
import Testing

@Suite("OpenAI HTTP compatibility wiring")
struct HTTPCompatibilityWiringTests {
    @Test("Server supports both completion modes and loaded-model discovery")
    func serverWiring() throws {
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf:
                packageRoot
                .appendingPathComponent("Sources/ModelRunner/ModelHTTPServer.swift"),
            encoding: .utf8
        )

        #expect(source.contains("handleStreamingChat("))
        #expect(source.contains("handleNonStreamingChat("))
        #expect(source.contains("maxCompletionTokens ?? completion.maxTokens"))
        #expect(source.contains("/v1/models/"))
        #expect(source.contains("ModelListResponse(models: [modelDescriptor()])"))
        #expect(source.contains("code: \"model_not_found\""))
        #expect(source.contains("try await channel.writeAndFlush(\n            HTTPServerResponsePart.head"))
    }

    @Test("Mistral audio routes coexist without an API mode switch")
    func audioWiring() throws {
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf:
                packageRoot
                .appendingPathComponent("Sources/ModelRunner/ModelHTTPServer.swift"),
            encoding: .utf8
        )

        let audio = try String(
            contentsOf: packageRoot.appendingPathComponent("Sources/ModelRunner/ModelHTTPAudio.swift"), encoding: .utf8)

        #expect(source.contains("AudioAPIRoute.parse(uri: head.uri)"))
        #expect(audio.contains("handleMistralSpeech("))
        #expect(audio.contains("handleOpenAISpeech("))
        #expect(audio.contains("streamOpenAISSE("))
        #expect(audio.contains("speech.audio.delta"))
        #expect(audio.contains("speech.audio.done"))
        #expect(audio.contains("handleVoiceList("))
        #expect(audio.contains("handleCreateVoice("))
        #expect(audio.contains("handleUpdateVoice("))
        #expect(audio.contains("handleDeleteVoice("))
        #expect(audio.contains("handleVoiceSample("))
        #expect(source.contains("/v1/chat/completions"))
        #expect(!source.contains("apiMode"))
    }
}
