import Foundation
import ModelRunnerProtocol
import Testing

@testable import ModelRunnerCore

@Suite("GPT-OSS reasoning runtime")
struct GPTOSSReasoningRuntimeTests {
  @Test("OpenAI reasoning controls reach GPT-OSS prompt context only")
  func scopesReasoningToGPTOSS() throws {
    let gptOSS = try ModelRuntimeCapabilities.decode(Data(#"{"model_type":"gpt_oss"}"#.utf8))
    let otherModels: [ModelRuntimeCapabilities] = [
      .none, .init(mistralFamily: .mistral), .init(mistralFamily: .mixtral),
      .init(mistralFamily: .mistral3),
    ]
    for effort in ["low", "medium", "high"] {
      let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data("""
        {"model":"gpt-oss-20b","messages":[{"role":"user","content":"Hello"}],
         "reasoning_effort":"\(effort)"}
        """.utf8))
      let effective = LocalModelRunner.effectiveReasoningEffort(
        request.reasoningEffort, capabilities: gptOSS)
      let context = try #require(LocalModelRunner.promptAdditionalContext(reasoningEffort: effective))

      #expect(context.count == 1)
      #expect(context["reasoning_effort"] as? String == effort)
      for capabilities in otherModels {
        let ignored = LocalModelRunner.effectiveReasoningEffort(
          request.reasoningEffort, capabilities: capabilities)
        #expect(ignored == nil)
        #expect(LocalModelRunner.promptAdditionalContext(reasoningEffort: ignored) == nil)
      }
    }
  }

  @Test("No request override leaves the template's own default intact")
  func preservesTemplateDefault() {
    for capabilities in [ModelRuntimeCapabilities.none, .init(mistralFamily: nil, isGPTOSS: true)] {
      let effective = LocalModelRunner.effectiveReasoningEffort(nil, capabilities: capabilities)
      #expect(effective == nil)
      #expect(LocalModelRunner.promptAdditionalContext(reasoningEffort: effective) == nil)
    }
  }

  @Test("Changing or removing reasoning effort invalidates an otherwise reusable conversation")
  func reasoningCacheIdentity() {
    let committed: [OpenAIMessage] = [
      .init(role: "user", content: "Name a prime."),
      .init(role: "assistant", content: "Two."),
    ]
    let incoming = committed + [.init(role: "user", content: "Another.")]
    let efforts: [ChatCompletionRequest.ReasoningEffort?] = [nil, .low, .medium, .high]

    for previousEffort in efforts {
      for requestedEffort in efforts {
        let suffix = LocalModelRunner.cachedConversationSuffixStart(
          committed: committed,
          incoming: incoming,
          committedReasoningEffort: previousEffort,
          reasoningEffort: requestedEffort)
        #expect(suffix == (previousEffort == requestedEffort ? committed.count : nil))
      }
    }
  }

  @Test("Tool schema changes cannot reuse a GPT-OSS reasoning session")
  func toolCacheIdentity() {
    let committed: [OpenAIMessage] = [
      .init(role: "user", content: "Check the weather."),
      .init(role: "assistant", content: "Sunny."),
    ]
    let incoming = committed + [.init(role: "user", content: "And tomorrow?")]
    let original = [OpenAIToolDefinition(function: .init(
      name: "weather", parameters: .object(["type": .string("object")])))]
    let revised = [OpenAIToolDefinition(function: .init(
      name: "weather", parameters: .object([
        "type": .string("object"), "required": .array([.string("city")]),
      ])))]

    for tools: [OpenAIToolDefinition]? in [original, revised, nil] {
      let suffix = LocalModelRunner.cachedConversationSuffixStart(
        committed: committed,
        incoming: incoming,
        committedTools: original,
        tools: tools,
        committedReasoningEffort: .low,
        reasoningEffort: .low)
      #expect(suffix == (tools == original ? committed.count : nil))
    }
  }
}
