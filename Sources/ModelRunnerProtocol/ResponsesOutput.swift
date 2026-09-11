import Foundation

/// Builds the Responses object and its semantic stream from local generation events.
/// The transport writes each returned object as an SSE event whose name is its `type`.
public struct ResponsesOutput: Sendable {
    private enum Kind: Sendable {
        case text(String)
        case function(OpenAIToolCall)
    }

    private struct Item: Sendable {
        let id: String
        var kind: Kind
        var status = "in_progress"

        var value: OpenAIJSONValue {
            switch kind {
            case .text(let text):
                return .object([
                    "id": .string(id), "type": .string("message"),
                    "role": .string("assistant"), "status": .string(status),
                    "content": .array([ResponsesOutput.textPart(text)]),
                ])
            case .function(let call):
                return .object([
                    "id": .string(id), "type": .string("function_call"),
                    "call_id": .string(call.id), "name": .string(call.function.name),
                    "arguments": .string(call.function.arguments), "status": .string(status),
                ])
            }
        }
    }

    private var fields: [String: OpenAIJSONValue]
    private var items: [Item] = []
    private var activeTextIndex: Int?
    private var sequenceNumber = 0
    private var hasStarted = false
    private var isTerminal = false

    public init(
        id: String,
        model: String,
        createdAt: Int,
        fields: [String: OpenAIJSONValue] = [:]
    ) {
        var values: [String: OpenAIJSONValue] = [
            "background": .bool(false), "instructions": .null,
            "max_output_tokens": .null, "max_tool_calls": .null,
            "metadata": .object([:]), "parallel_tool_calls": .bool(true),
            "previous_response_id": .null, "prompt_cache_key": .null,
            "reasoning": .object(["effort": .null, "summary": .null]),
            "safety_identifier": .null, "service_tier": .string("default"),
            "store": .bool(true), "temperature": .number(1),
            "text": .object(["format": .object(["type": .string("text")])]),
            "tool_choice": .string("auto"), "tools": .array([]),
            "top_logprobs": .integer(0), "top_p": .number(1),
            "truncation": .string("disabled"), "user": .null,
        ]
        values.merge(fields) { _, supplied in supplied }
        // Generation-owned values cannot be replaced by echoed request options.
        values["id"] = .string(id)
        values["object"] = .string("response")
        values["model"] = .string(model)
        values["created_at"] = .integer(createdAt)
        values["completed_at"] = .null
        values["status"] = .string("in_progress")
        values["error"] = .null
        values["incomplete_details"] = .null
        values["usage"] = .null
        values.removeValue(forKey: "output")
        self.fields = values
    }

    public var response: OpenAIJSONValue {
        var values = fields
        values["output"] = .array(outputItems)
        return .object(values)
    }

    public var outputItems: [OpenAIJSONValue] { items.map(\.value) }

    /// Preserves output order while grouping adjacent calls into one assistant turn.
    public var assistantMessages: [OpenAIMessage] {
        var result: [OpenAIMessage] = []
        var text: String?
        var calls: [OpenAIToolCall] = []
        for item in items {
            switch item.kind {
            case .text(let value):
                if !calls.isEmpty {
                    result.append(.init(role: "assistant", content: text, toolCalls: calls))
                    text = nil
                    calls = []
                }
                text = (text ?? "") + value
            case .function(let call):
                calls.append(call)
            }
        }
        if text != nil || !calls.isEmpty {
            result.append(.init(
                role: "assistant", content: text, toolCalls: calls.isEmpty ? nil : calls))
        }
        return result
    }

    /// Safe to call once at stream creation or implicitly through append/finish.
    public mutating func started() -> [OpenAIJSONValue] {
        guard !hasStarted, !isTerminal else { return [] }
        hasStarted = true
        return [
            event("response.created", ["response": response]),
            event("response.in_progress", ["response": response]),
        ]
    }

    public mutating func appendText(_ text: String) -> [OpenAIJSONValue] {
        guard !isTerminal, !text.isEmpty else { return [] }
        var events = started()
        let index: Int
        if let currentIndex = activeTextIndex {
            index = currentIndex
        } else {
            index = items.count
            let item = Item(id: Self.itemID(prefix: "msg"), kind: .text(""))
            items.append(item)
            activeTextIndex = index
            var addedItem = Self.object(item.value)
            addedItem["content"] = .array([])
            events.append(event("response.output_item.added", [
                "output_index": .integer(index), "item": .object(addedItem),
            ]))
            events.append(event("response.content_part.added", textCoordinates(index).merging([
                "part": Self.textPart(""),
            ]) { _, new in new }))
        }
        if case .text(let previous) = items[index].kind {
            items[index].kind = .text(previous + text)
        }
        events.append(event("response.output_text.delta", textCoordinates(index).merging([
            "delta": .string(text), "logprobs": .array([]),
        ]) { _, new in new }))
        return events
    }

    /// Local parsers yield complete calls, so arguments are sent as one semantic delta.
    public mutating func appendToolCall(_ call: OpenAIToolCall) -> [OpenAIJSONValue] {
        guard !isTerminal else { return [] }
        var events = started()
        events += closeText(status: "completed")
        let index = items.count
        let item = Item(id: Self.itemID(prefix: "fc"), kind: .function(call))
        items.append(item)
        var addedItem = Self.object(item.value)
        addedItem["arguments"] = .string("")
        events.append(event("response.output_item.added", [
            "output_index": .integer(index), "item": .object(addedItem),
        ]))
        let coordinates: [String: OpenAIJSONValue] = [
            "item_id": .string(item.id), "output_index": .integer(index),
        ]
        events.append(event("response.function_call_arguments.delta", coordinates.merging([
            "delta": .string(call.function.arguments),
        ]) { _, new in new }))
        events.append(event("response.function_call_arguments.done", coordinates.merging([
            "arguments": .string(call.function.arguments), "name": .string(call.function.name),
        ]) { _, new in new }))
        items[index].status = "completed"
        events.append(event("response.output_item.done", [
            "output_index": .integer(index), "item": items[index].value,
        ]))
        return events
    }

    public mutating func finish(
        inputTokens: Int,
        cachedInputTokens: Int,
        outputTokens: Int,
        stopReason: String
    ) -> [OpenAIJSONValue] {
        guard !isTerminal else { return [] }
        var events = started()
        let incompleteReason: String?
        switch stopReason {
        case "length", "max_output_tokens": incompleteReason = "max_output_tokens"
        case "content_filter": incompleteReason = "content_filter"
        default: incompleteReason = nil
        }
        let status = incompleteReason == nil ? "completed" : "incomplete"
        events += closeText(status: status)
        fields["status"] = .string(status)
        fields["completed_at"] = incompleteReason == nil
            ? .integer(Int(Date().timeIntervalSince1970)) : .null
        fields["incomplete_details"] = incompleteReason.map {
            .object(["reason": .string($0)])
        } ?? .null
        let inputCount = max(0, inputTokens)
        let outputCount = max(0, outputTokens)
        let (totalCount, overflow) = inputCount.addingReportingOverflow(outputCount)
        fields["usage"] = .object([
            "input_tokens": .integer(inputCount),
            "input_tokens_details": .object([
                "cached_tokens": .integer(min(inputCount, max(0, cachedInputTokens))),
                "cache_write_tokens": .integer(0),
            ]),
            "output_tokens": .integer(outputCount),
            "output_tokens_details": .object(["reasoning_tokens": .integer(0)]),
            "total_tokens": .integer(overflow ? Int.max : totalCount),
        ])
        isTerminal = true
        events.append(event("response.\(status)", ["response": response]))
        return events
    }

    public mutating func fail(message: String, code: String) -> [OpenAIJSONValue] {
        guard !isTerminal else { return [] }
        var events = started()
        events += closeText(status: "incomplete")
        fields["status"] = .string("failed")
        fields["error"] = .object(["code": .string(code), "message": .string(message)])
        isTerminal = true
        events.append(event("response.failed", ["response": response]))
        return events
    }

    private mutating func closeText(status: String) -> [OpenAIJSONValue] {
        guard let index = activeTextIndex, case .text(let text) = items[index].kind else {
            return []
        }
        activeTextIndex = nil
        items[index].status = status
        let coordinates = textCoordinates(index)
        return [
            event("response.output_text.done", coordinates.merging([
                "text": .string(text), "logprobs": .array([]),
            ]) { _, new in new }),
            event("response.content_part.done", coordinates.merging([
                "part": Self.textPart(text),
            ]) { _, new in new }),
            event("response.output_item.done", [
                "output_index": .integer(index), "item": items[index].value,
            ]),
        ]
    }

    private func textCoordinates(_ index: Int) -> [String: OpenAIJSONValue] {
        [
            "item_id": .string(items[index].id), "output_index": .integer(index),
            "content_index": .integer(0),
        ]
    }

    private mutating func event(
        _ name: String, _ fields: [String: OpenAIJSONValue]
    ) -> OpenAIJSONValue {
        var values = fields
        values["type"] = .string(name)
        values["sequence_number"] = .integer(sequenceNumber)
        sequenceNumber += 1
        return .object(values)
    }

    private static func textPart(_ text: String) -> OpenAIJSONValue {
        .object([
            "type": .string("output_text"), "text": .string(text),
            "annotations": .array([]), "logprobs": .array([]),
        ])
    }

    private static func itemID(prefix: String) -> String {
        "\(prefix)_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased())"
    }

    private static func object(_ value: OpenAIJSONValue) -> [String: OpenAIJSONValue] {
        guard case .object(let fields) = value else { return [:] }
        return fields
    }
}
