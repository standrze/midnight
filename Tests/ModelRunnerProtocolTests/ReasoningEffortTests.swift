import Foundation
import ModelRunnerProtocol
import Testing

@Suite("OpenAI reasoning effort")
struct ReasoningEffortTests {
    @Test("GPT-OSS accepts each supported reasoning effort", arguments: ["low", "medium", "high"])
    func decodesSupportedEffort(_ effort: String) throws {
        let data = Data(
            """
            {"model":"gpt-oss-20b","messages":[{"role":"user","content":"Hello"}],
             "reasoning_effort":"\(effort)"}
            """.utf8)
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: data)

        #expect(request.reasoningEffort?.rawValue == effort)
        #expect(request.stream == nil)
        #expect(
            try JSONDecoder().decode(
                ChatCompletionRequest.self, from: JSONEncoder().encode(request)) == request)
    }

    @Test("Omitted and null effort preserve the model default", arguments: ["", ",\"reasoning_effort\":null"])
    func preservesDefault(_ optionalField: String) throws {
        let data = Data(
            """
            {"model":"local-model","messages":[{"role":"user","content":"Hello"}]\(optionalField)}
            """.utf8)
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: data)

        #expect(request.reasoningEffort == nil)
        let encoded = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any])
        #expect(encoded["reasoning_effort"] == nil)
    }

    @Test("Explicit low effort uses the standard OpenAI wire key")
    func encodesLowEffort() throws {
        let request = ChatCompletionRequest(
            model: "gpt-oss-20b",
            messages: [.init(role: "user", content: "Hello")],
            stream: true,
            reasoningEffort: .low)
        let encoded = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any])

        #expect(encoded["reasoning_effort"] as? String == "low")
        #expect(encoded["reasoningEffort"] == nil)
    }

    @Test("Unsupported effort values are rejected", arguments: ["none", "minimal", "xhigh", "LOW", ""])
    func rejectsUnsupportedEffort(_ effort: String) throws {
        let data = Data(
            """
            {"model":"gpt-oss-20b","messages":[{"role":"user","content":"Hello"}],
             "reasoning_effort":"\(effort)"}
            """.utf8)

        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(ChatCompletionRequest.self, from: data)
        }
    }
}
