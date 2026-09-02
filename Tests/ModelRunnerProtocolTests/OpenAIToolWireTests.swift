import Foundation
import ModelRunnerProtocol
import Testing

@Suite("OpenAI tool-calling wire format")
struct OpenAIToolWireTests {
    @Test("Decodes tool declarations and a complete tool exchange")
    func decodesToolRequest() throws {
        let json = #"""
        {
          "model": "local-model",
          "stream": true,
          "stream_options": {"include_usage": true},
          "max_tokens": 256,
          "tools": [{
            "type": "function",
            "function": {
              "name": "get_weather",
              "description": "Read the weather",
              "parameters": {
                "type": "object",
                "properties": {
                  "city": {"type": "string"},
                  "days": {"type": "integer", "minimum": 1},
                  "metric": {"type": "boolean", "default": true}
                },
                "required": ["city"]
              }
            }
          }],
          "messages": [
            {"role": "user", "content": "Weather in Boston?"},
            {
              "role": "assistant",
              "content": null,
              "tool_calls": [{
                "id": "call_weather_1",
                "type": "function",
                "function": {
                  "name": "get_weather",
                  "arguments": "{\"city\":\"Boston\",\"days\":1}"
                }
              }]
            },
            {
              "role": "tool",
              "content": "{\"temperature\":72}",
              "name": "get_weather",
              "tool_call_id": "call_weather_1"
            }
          ]
        }
        """#

        let request = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data(json.utf8)
        )

        #expect(request.tools?.count == 1)
        #expect(request.streamOptions?.includeUsage == true)
        #expect(request.tools?.first?.function.name == "get_weather")
        #expect(request.messages[1].content == nil)
        #expect(request.messages[1].toolCalls?.first?.id == "call_weather_1")
        #expect(request.messages[2].role == "tool")
        #expect(request.messages[2].toolCallID == "call_weather_1")
        #expect(request.messages[2].name == "get_weather")

        guard
            case .object(let schema)? = request.tools?.first?.function.parameters,
            case .object(let properties)? = schema["properties"],
            case .object(let days)? = properties["days"],
            case .integer(1)? = days["minimum"],
            case .object(let metric)? = properties["metric"],
            case .bool(true)? = metric["default"]
        else {
            Issue.record("Tool parameters lost their recursive JSON types")
            return
        }
    }

    @Test("Encodes usage in the final OpenAI streaming chunk")
    func encodesStreamingUsage() throws {
        let chunk = ChatCompletionChunk(
            id: "chatcmpl-test",
            model: "local-model",
            choices: [],
            usage: ChatCompletionUsage(promptTokens: 12, completionTokens: 8)
        )

        let encoded = try JSONEncoder().encode(chunk)
        let object = try #require(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        )
        let usage = try #require(object["usage"] as? [String: Any])
        #expect((object["choices"] as? [Any])?.isEmpty == true)
        #expect(usage["prompt_tokens"] as? Int == 12)
        #expect(usage["completion_tokens"] as? Int == 8)
        #expect(usage["total_tokens"] as? Int == 20)
        #expect(object["model_runner"] == nil)
    }

    @Test("Legacy requests remain valid when tools are absent")
    func decodesLegacyRequest() throws {
        let json = #"""
        {
          "model": "local-model",
          "stream": true,
          "messages": [{"role": "user", "content": "Hello"}]
        }
        """#

        let request = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data(json.utf8)
        )

        #expect(request.tools == nil)
        #expect(request.stream == true)
        #expect(request.messages == [OpenAIMessage(role: "user", content: "Hello")])
    }

    @Test("Non-streaming defaults and modern generation controls decode")
    func decodesGenerationControls() throws {
        let json = #"""
        {
          "model": "local-model",
          "max_completion_tokens": 64,
          "top_p": 0.8,
          "stop": ["END", "DONE"],
          "messages": [
            {"role": "developer", "content": "Be concise."},
            {"role": "user", "content": "Hello"}
          ]
        }
        """#

        let request = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data(json.utf8)
        )

        #expect(request.stream == nil)
        #expect(request.maxCompletionTokens == 64)
        #expect(request.maxTokens == nil)
        #expect(request.topP == 0.8)
        #expect(request.stop?.values == ["END", "DONE"])
        #expect(request.messages.first?.role == "developer")
    }

    @Test("Encodes a complete non-streaming completion with usage")
    func encodesNonStreamingCompletion() throws {
        let response = ChatCompletionResponse(
            id: "chatcmpl-test",
            created: 123,
            model: "local-model",
            choices: [
                .init(
                    message: OpenAIMessage(role: "assistant", content: "Hello"),
                    finishReason: "stop"
                )
            ],
            usage: ChatCompletionUsage(promptTokens: 5, completionTokens: 1)
        )
        let data = try JSONEncoder().encode(response)
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try #require(object["choices"] as? [[String: Any]])
        let usage = try #require(object["usage"] as? [String: Any])
        #expect(object["object"] as? String == "chat.completion")
        #expect((choices.first?["message"] as? [String: Any])?["content"] as? String == "Hello")
        #expect(choices.first?["finish_reason"] as? String == "stop")
        #expect(usage["total_tokens"] as? Int == 6)
        #expect(object["model_runner"] == nil)
    }

    @Test("Encodes standard OpenAI error details")
    func encodesErrorDetails() throws {
        let envelope = OpenAIErrorEnvelope(
            message: "Unknown model",
            param: "model",
            code: "model_not_found"
        )
        let data = try JSONEncoder().encode(envelope)
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let error = try #require(object["error"] as? [String: Any])

        #expect(error["type"] as? String == "invalid_request_error")
        #expect(error["param"] as? String == "model")
        #expect(error["code"] as? String == "model_not_found")
    }

    @Test("Encodes OpenAI streaming tool-call deltas and finish reason")
    func encodesToolCallDelta() throws {
        let chunk = ChatCompletionChunk(
            id: "chatcmpl-test",
            model: "local-model",
            choices: [
                .init(
                    delta: .init(
                        toolCalls: [
                            OpenAIToolCallDelta(
                                index: 0,
                                id: "call_weather_1",
                                type: "function",
                                function: .init(
                                    name: "get_weather",
                                    arguments: #"{"city":"Boston"}"#
                                )
                            )
                        ]
                    ),
                    finishReason: "tool_calls"
                )
            ]
        )

        let encoded = try JSONEncoder().encode(chunk)
        let object = try #require(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        )
        let choices = try #require(object["choices"] as? [[String: Any]])
        let choice = try #require(choices.first)
        let delta = try #require(choice["delta"] as? [String: Any])
        let calls = try #require(delta["tool_calls"] as? [[String: Any]])
        let call = try #require(calls.first)
        let function = try #require(call["function"] as? [String: Any])

        #expect(choice["finish_reason"] as? String == "tool_calls")
        #expect(call["index"] as? Int == 0)
        #expect(call["id"] as? String == "call_weather_1")
        #expect(function["name"] as? String == "get_weather")
        #expect(function["arguments"] as? String == #"{"city":"Boston"}"#)
    }
}
