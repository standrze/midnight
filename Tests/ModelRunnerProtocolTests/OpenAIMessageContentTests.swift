import Foundation
import ModelRunnerProtocol
import Testing

@Suite("OpenAI text content compatibility")
struct OpenAIMessageContentTests {
    @Test("Pool 1.0.16's sanitized summarizer request decodes")
    func decodesPoolSummarizer() throws {
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/Pool/pool-1.0.16-summarizer.json")
        let request = try JSONDecoder().decode(
            ChatCompletionRequest.self, from: Data(contentsOf: fixture))

        #expect(request.messages.map(\.role) == ["system", "user", "assistant", "user"])
        #expect(request.messages.map(\.content) == ["hi", "hi", "hi", "hi"])
        #expect(request.maxCompletionTokens == 8192)
        #expect(request.reasoningEffort == nil)
    }

    @Test(
        "Text parts preserve order and whitespace for each text role",
        arguments: ["system", "developer", "user", "assistant", "tool"])
    func joinsTextParts(_ role: String) throws {
        let message = try decode(
            """
            {"role":"\(role)","content":[
              {"type":"text","text":"Hello ","cache_control":{"type":"ephemeral"}},
              {"type":"text","text":"world!\\n"},
              {"type":"text","text":"Next line."}
            ],"name":"lookup","tool_call_id":"call_1"}
            """)
        #expect(message.content == "Hello world!\nNext line.")
        #expect(message.role == role)
        #expect(message.name == "lookup")
        #expect(message.toolCallID == "call_1")
        let encoded = try JSONEncoder().encode(message)
        let object = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(object["content"] as? String == message.content)
        #expect(try JSONDecoder().decode(OpenAIMessage.self, from: encoded) == message)
    }

    @Test(
        "String, null, omitted, and empty array content remain distinct",
        arguments: [#""plain text""#, "null", "[]", ""])
    func handlesBasicContent(_ content: String) throws {
        let field = content.isEmpty ? "" : ",\"content\":\(content)"
        let message = try decode("{\"role\":\"assistant\"\(field)}")
        let expected: String? = content == "[]" ? "" : (content == #""plain text""# ? "plain text" : nil)
        #expect(message.content == expected)
    }

    @Test(
        "Non-text parts are rejected even when mixed with text",
        arguments: ["image_url", "input_audio", "file", "refusal", "unknown"])
    func rejectsUnsupportedParts(_ type: String) throws {
        let data = Data(
            """
            {"model":"local","messages":[{"role":"user","content":[
              {"type":"text","text":"Keep this"}, {"type":"\(type)","text":"Do not silently drop"}
            ]}]}
            """.utf8)
        do {
            _ = try JSONDecoder().decode(ChatCompletionRequest.self, from: data)
            Issue.record("Unsupported content part was accepted")
        } catch DecodingError.dataCorrupted(let context) {
            #expect(context.codingPath.map(\.stringValue) == ["messages", "Index 0", "content", "Index 1", "type"])
            #expect(context.debugDescription == "Only text content parts are supported by chat completions.")
        }
    }

    @Test(
        "Malformed content parts and scalar values are rejected",
        arguments: [
            #"[{"type":"text"}]"#, #"[{"type":"text","text":null}]"#,
            #"[{"type":"text","text":42}]"#, #"[{"text":"missing type"}]"#,
            #"["plain string"]"#, #"{"type":"text","text":"not an array"}"#,
            "42", "true",
        ])
    func rejectsMalformedContent(_ content: String) throws {
        #expect(throws: DecodingError.self) {
            try decode("{\"role\":\"user\",\"content\":\(content)}")
        }
    }

    private func decode(_ json: String) throws -> OpenAIMessage {
        try JSONDecoder().decode(OpenAIMessage.self, from: Data(json.utf8))
    }
}
