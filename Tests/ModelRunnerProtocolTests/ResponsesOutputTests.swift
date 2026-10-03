import Foundation
import ModelRunnerProtocol
import Testing

@Suite("OpenAI Responses output and streaming")
struct ResponsesOutputTests {
    @Test("Reasoning streams before answers and is not replayed as answer text")
    func reasoningLifecycle() throws {
        var output = makeOutput()
        var events = output.appendReasoning("First")
        events += output.appendReasoning(" step")
        events += output.appendText("Answer")
        events += output.finish(inputTokens: 3, cachedInputTokens: 0, outputTokens: 5, stopReason: "stop")
        let types = try eventTypes(events)
        #expect(types.filter { $0 == "response.reasoning_text.delta" }.count == 2)
        #expect(
            try #require(types.firstIndex(of: "response.reasoning_text.done")) < #require(
                types.firstIndex(of: "response.output_text.delta")))
        #expect(output.outputItems.count == 2)
        let reasoning = try object(output.outputItems[0])
        #expect(reasoning["type"] == .string("reasoning"))
        #expect(reasoning["status"] == .string("completed"))
        #expect(
            reasoning["content"]
                == .array([.object(["type": .string("reasoning_text"), "text": .string("First step")])]))
        #expect(output.assistantMessages == [.init(role: "assistant", content: "Answer")])
        #expect(output.appendReasoning("late").isEmpty)
    }

    @Test("Interrupted reasoning closes as incomplete")
    func reasoningFailure() throws {
        var output = makeOutput()
        _ = output.appendReasoning("Partial")
        let events = output.fail(message: "Cancelled", code: "cancelled")
        #expect(
            try eventTypes(events) == ["response.reasoning_text.done", "response.output_item.done", "response.failed"])
        #expect(try object(output.outputItems[0])["status"] == .string("incomplete"))
        #expect(output.assistantMessages.isEmpty)
    }

    @Test("Text stream has a complete semantic lifecycle and immutable event snapshots")
    func textLifecycle() throws {
        var output = makeOutput()
        let start = output.started()
        var events = start
        events += output.appendText("Hello")
        events += output.appendText(" 世界\n")
        events += output.finish(inputTokens: 12, cachedInputTokens: 4, outputTokens: 5, stopReason: "stop")

        #expect(
            try eventTypes(events) == [
                "response.created", "response.in_progress", "response.output_item.added",
                "response.content_part.added", "response.output_text.delta", "response.output_text.delta",
                "response.output_text.done", "response.content_part.done", "response.output_item.done",
                "response.completed",
            ])
        #expect(try events.map { try object($0)["sequence_number"] } == (0..<10).map(OpenAIJSONValue.integer))
        let initialResponse = try object(try #require(object(start[0])["response"]))
        #expect(initialResponse["status"] == .string("in_progress"))
        #expect(initialResponse["output"] == .array([]))
        #expect(initialResponse["usage"] == .null)
        let added = try object(try #require(object(events[2])["item"]))
        let completed = try object(output.outputItems[0])
        #expect(added["id"] == completed["id"])
        #expect(added["content"] == .array([]))
        #expect(added["status"] == .string("in_progress"))
        #expect(completed["status"] == .string("completed"))
        for event in events[3...7] {
            let fields = try object(event)
            #expect(fields["item_id"] == completed["id"])
            #expect(fields["output_index"] == .integer(0))
            #expect(fields["content_index"] == .integer(0))
        }
        let content = try array(try #require(completed["content"]))
        let part = try object(try #require(content.first))
        #expect(part["text"] == .string("Hello 世界\n"))
        #expect(part["annotations"] == .array([]))
        #expect(try object(events.last!)["response"] == output.response)
        #expect(output.assistantMessages == [.init(role: "assistant", content: "Hello 世界\n")])
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let encoded = try encoder.encode(output.response)
        let decoded = try JSONDecoder().decode(OpenAIJSONValue.self, from: encoded)
        #expect(try encoder.encode(decoded) == encoded)
    }

    @Test("Final response includes SDK fields and accurate usage")
    func envelopeAndUsage() throws {
        var output = ResponsesOutput(
            id: "resp_test", model: "local", createdAt: 123,
            fields: [
                "metadata": .object(["task": .string("test")]),
                "previous_response_id": .string("resp_previous"), "store": .bool(false),
                "status": .string("completed"), "id": .string("wrong"),
            ])
        _ = output.finish(inputTokens: 7, cachedInputTokens: 2, outputTokens: 3, stopReason: "stop")
        let response = try object(output.response)
        #expect(response["id"] == .string("resp_test"))
        #expect(response["model"] == .string("local"))
        #expect(response["object"] == .string("response"))
        #expect(response["created_at"] == .integer(123))
        #expect(response["status"] == .string("completed"))
        #expect(response["completed_at"] != .null)
        #expect(response["error"] == .null)
        #expect(response["incomplete_details"] == .null)
        #expect(response["instructions"] == .null)
        #expect(response["output"] == .array([]))
        #expect(response["tools"] == .array([]))
        #expect(response["parallel_tool_calls"] == .bool(true))
        #expect(response["tool_choice"] == .string("auto"))
        #expect(response["text"] == .object(["format": .object(["type": .string("text")])]))
        #expect(response["metadata"] == .object(["task": .string("test")]))
        #expect(response["previous_response_id"] == .string("resp_previous"))
        #expect(response["store"] == .bool(false))
        #expect(
            response["usage"]
                == .object([
                    "input_tokens": .integer(7),
                    "input_tokens_details": .object([
                        "cached_tokens": .integer(2), "cache_write_tokens": .integer(0),
                    ]),
                    "output_tokens": .integer(3), "output_tokens_details": .object(["reasoning_tokens": .integer(0)]),
                    "total_tokens": .integer(10),
                ]))
    }

    @Test("Tool calls use distinct item and call IDs with argument events")
    func functionLifecycle() throws {
        var output = makeOutput()
        let call = tool("call_1", name: "weather", arguments: #"{"city":"Paris"}"#)
        var events = output.appendToolCall(call)
        events += output.finish(inputTokens: 9, cachedInputTokens: 0, outputTokens: 8, stopReason: "tool_calls")
        #expect(
            try eventTypes(events) == [
                "response.created", "response.in_progress", "response.output_item.added",
                "response.function_call_arguments.delta", "response.function_call_arguments.done",
                "response.output_item.done", "response.completed",
            ])
        let item = try object(output.outputItems[0])
        #expect(item["id"] != item["call_id"])
        #expect(item["call_id"] == .string("call_1"))
        #expect(item["type"] == .string("function_call"))
        #expect(item["name"] == .string("weather"))
        #expect(item["arguments"] == .string(call.function.arguments))
        #expect(item["status"] == .string("completed"))
        let added = try object(try #require(object(events[2])["item"]))
        #expect(added["arguments"] == .string(""))
        #expect(added["status"] == .string("in_progress"))
        #expect(try object(events[3])["delta"] == item["arguments"])
        #expect(try object(events[4])["arguments"] == item["arguments"])
        #expect(try object(events[4])["name"] == item["name"])
        #expect(try object(events[3])["item_id"] == item["id"])
        #expect(output.assistantMessages == [.init(role: "assistant", content: nil, toolCalls: [call])])
    }

    @Test("Interleaved output retains indices, stable IDs, and tool history grouping")
    func mixedOutput() throws {
        var output = makeOutput()
        let first = tool("call_a", name: "first", arguments: "{}")
        let second = tool("call_b", name: "second", arguments: "{}")
        var events = output.appendText("Checking.")
        events += output.appendToolCall(first)
        events += output.appendToolCall(second)
        events += output.appendText("Done.")
        events += output.finish(inputTokens: 10, cachedInputTokens: 0, outputTokens: 12, stopReason: "stop")
        #expect(
            try output.outputItems.map { try object($0)["type"] } == [
                .string("message"), .string("function_call"), .string("function_call"), .string("message"),
            ])
        let added = try events.filter { try object($0)["type"] == .string("response.output_item.added") }
        let done = try events.filter { try object($0)["type"] == .string("response.output_item.done") }
        #expect(added.count == 4)
        #expect(done.count == 4)
        for index in 0..<4 {
            #expect(try object(added[index])["output_index"] == .integer(index))
            #expect(try object(done[index])["output_index"] == .integer(index))
            #expect(try object(done[index])["item"] == output.outputItems[index])
        }
        #expect(
            output.assistantMessages == [
                .init(role: "assistant", content: "Checking.", toolCalls: [first, second]),
                .init(role: "assistant", content: "Done."),
            ])
        #expect(try events.map { try object($0)["sequence_number"] } == (0..<events.count).map(OpenAIJSONValue.integer))
    }

    @Test("Token limit preserves partial text and marks response and open item incomplete")
    func tokenLimit() throws {
        var output = makeOutput()
        _ = output.appendText(#"{"answer":"part"#)
        let events = output.finish(inputTokens: 5, cachedInputTokens: 0, outputTokens: 4, stopReason: "length")
        let response = try object(output.response)
        #expect(try eventTypes(events).last == "response.incomplete")
        #expect(response["status"] == .string("incomplete"))
        #expect(response["completed_at"] == .null)
        #expect(response["incomplete_details"] == .object(["reason": .string("max_output_tokens")]))
        #expect(response["error"] == .null)
        #expect(try object(output.outputItems[0])["status"] == .string("incomplete"))
        #expect(try object(events[0])["text"] == .string(#"{"answer":"part"#))
        #expect(!((try eventTypes(events)).contains("response.completed")))
    }

    @Test("Failure is terminal, preserves partial output, and never emits completion")
    func failure() throws {
        var output = makeOutput()
        _ = output.appendText("partial")
        let events = output.fail(message: "Generation failed", code: "server_error")
        #expect(try eventTypes(events).last == "response.failed")
        #expect(!((try eventTypes(events)).contains("response.completed")))
        let response = try object(output.response)
        #expect(response["status"] == .string("failed"))
        #expect(response["completed_at"] == .null)
        #expect(response["usage"] == .null)
        #expect(response["incomplete_details"] == .null)
        #expect(
            response["error"]
                == .object([
                    "code": .string("server_error"), "message": .string("Generation failed"),
                ]))
        #expect(try object(output.outputItems[0])["status"] == .string("incomplete"))
        #expect(output.appendText("ignored").isEmpty)
        #expect(output.appendToolCall(tool("later", name: "later", arguments: "{}")).isEmpty)
        #expect(output.finish(inputTokens: 1, cachedInputTokens: 0, outputTokens: 1, stopReason: "stop").isEmpty)
        #expect(output.fail(message: "ignored", code: "server_error").isEmpty)
        #expect(output.started().isEmpty)
        #expect(output.response == .object(response))
    }

    @Test("Empty text and repeated lifecycle calls do not create duplicate events")
    func emptyAndRepeatedCalls() throws {
        var output = makeOutput()
        #expect(output.appendText("").isEmpty)
        #expect(output.outputItems.isEmpty)
        #expect(output.started().count == 2)
        #expect(output.started().isEmpty)
        let events = output.finish(inputTokens: 0, cachedInputTokens: 0, outputTokens: 0, stopReason: "stop")
        #expect(try eventTypes(events) == ["response.completed"])
        #expect(output.finish(inputTokens: 1, cachedInputTokens: 0, outputTokens: 1, stopReason: "length").isEmpty)
        #expect(output.assistantMessages.isEmpty)
    }

    private func makeOutput() -> ResponsesOutput {
        .init(id: "resp_test", model: "local", createdAt: 123)
    }

    private func tool(_ id: String, name: String, arguments: String) -> OpenAIToolCall {
        .init(id: id, function: .init(name: name, arguments: arguments))
    }

    private func object(_ value: OpenAIJSONValue) throws -> [String: OpenAIJSONValue] {
        guard case .object(let fields) = value else {
            throw TestValueError.expectedObject
        }
        return fields
    }

    private func array(_ value: OpenAIJSONValue) throws -> [OpenAIJSONValue] {
        guard case .array(let elements) = value else {
            throw TestValueError.expectedArray
        }
        return elements
    }

    private func eventTypes(_ events: [OpenAIJSONValue]) throws -> [String] {
        try events.map {
            guard case .string(let name) = try object($0)["type"] else {
                throw TestValueError.expectedEventType
            }
            return name
        }
    }

    private enum TestValueError: Error {
        case expectedObject, expectedArray, expectedEventType
    }
}
