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
      contentsOf: packageRoot
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
      contentsOf: packageRoot
        .appendingPathComponent("Sources/ModelRunner/ModelHTTPServer.swift"),
      encoding: .utf8
    )

    #expect(source.contains("AudioAPIRoute.parse(uri: head.uri)"))
    #expect(source.contains("handleMistralSpeech("))
    #expect(source.contains("handleOpenAISpeech("))
    #expect(source.contains("streamOpenAISSE("))
    #expect(source.contains("speech.audio.delta"))
    #expect(source.contains("speech.audio.done"))
    #expect(source.contains("handleVoiceList("))
    #expect(source.contains("handleCreateVoice("))
    #expect(source.contains("handleUpdateVoice("))
    #expect(source.contains("handleDeleteVoice("))
    #expect(source.contains("handleVoiceSample("))
    #expect(source.contains("/v1/chat/completions"))
    #expect(!source.contains("apiMode"))
  }
}
