import Foundation
import ModelRunnerProtocol
import Testing

@Suite("Reasoning streaming wire contract")
struct ReasoningStreamingWireTests {
    @Test(
        "Reasoning exposure is opt in", arguments: ["", ",\"include_reasoning\":false", ",\"include_reasoning\":true"])
    func exposure(_ field: String) throws {
        let json = "{\"model\":\"gemma\",\"messages\":[],\"stream\":true\(field)}"
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(json.utf8))
        #expect((request.includeReasoning == true) == field.contains("true"))
        #expect(try JSONDecoder().decode(ChatCompletionRequest.self, from: JSONEncoder().encode(request)) == request)
    }

    @Test("Reasoning is separate from answer content")
    func separateDelta() throws {
        let delta = ChatCompletionChunk.Choice.Delta(reasoningContent: "Considering the question")
        let encoded = try JSONEncoder().encode(delta)
        let object = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(object["reasoning_content"] as? String == "Considering the question")
        #expect(object["content"] == nil)
        #expect(try JSONDecoder().decode(ChatCompletionChunk.Choice.Delta.self, from: encoded) == delta)
    }

    @Test("Ordinary answer deltas retain their existing shape")
    func ordinaryDelta() throws {
        let encoded = try JSONEncoder().encode(ChatCompletionChunk.Choice.Delta(content: "Answer"))
        let object = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(object["content"] as? String == "Answer")
        #expect(object["reasoning_content"] == nil)
    }
}
