import Foundation
import Testing

@Suite("Generation stop-token configuration")
struct GenerationStopTokenTests {
  @Test("The model loader stops on padding and common turn terminators")
  func loaderIncludesTerminalTokens() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
    let source = try String(
      contentsOf: packageRoot
        .appendingPathComponent("Sources/ModelRunnerCore/LocalModelRunner.swift"),
      encoding: .utf8
    )

    #expect(source.contains("extraEOSTokens:"))
    #expect(source.contains("\"<end_of_turn>\""))
    #expect(source.contains("\"<turn|>\""))
    #expect(source.contains("SuppressTokenLogitProcessor(tokenID: 0)"))
    #expect(source.contains("let temperature = requestedTemperature ?? 1.0"))
    #expect(source.contains("temperature: Float(temperature)"))
    #expect(source.contains("let topP = requestedTopP ?? 0.95"))
    #expect(source.contains("topK: 64"))
    #expect(source.contains("normalizesGemma4Prompt"))
    #expect(source.contains("case \"system\", \"developer\""))
    #expect(source.contains("requestContext.configuration.stopStrings"))
    #expect(source.contains("topP: Float(topP)"))
    #expect(source.contains("promptTokenIDs.remove(at: 1)"))
    #expect(source.contains("MLXLMCommon.generateTask("))
    #expect(source.contains("info.totalPromptTokenCount"))
    #expect(source.contains("info.generationTokenCount"))
    #expect(source.contains("info.promptTokensPerSecond"))
    #expect(source.contains("info.tokensPerSecond"))
    #expect(source.contains("info.stopReason"))
  }
}
