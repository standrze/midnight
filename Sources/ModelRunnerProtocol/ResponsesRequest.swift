import Foundation

/// An unsupported or invalid Responses API parameter. The HTTP layer reports this as a 400.
public struct ResponsesRequestError: Error, LocalizedError, Sendable {
    public let message: String
    public let param: String

    public init(message: String, param: String) {
        self.message = message
        self.param = param
    }

    public var errorDescription: String? { message }
}

/// The text and function-calling subset of OpenAI's Responses request format.
/// Instructions are kept separate so that previous_response_id does not carry them forward.
public struct ResponsesRequest: Decodable, Sendable {
    public let model: String
    public let inputMessages: [OpenAIMessage]
    public let inputItems: [OpenAIJSONValue]
    public let instructions: String?
    public let previousResponseID: String?
    public let store: Bool
    public let stream: Bool
    public let maxOutputTokens: Int?
    public let temperature: Double?
    public let topP: Double?
    public let tools: [OpenAIToolDefinition]?
    public let toolChoice: OpenAIToolChoice?
    public let parallelToolCalls: Bool
    public let reasoningEffort: ChatCompletionRequest.ReasoningEffort?
    public let responseFormat: OpenAIResponseFormat?
    public let metadata: [String: String]
    private let passiveFields: [String: OpenAIJSONValue]

    public init(from decoder: Decoder) throws {
        let fields = try decoder.singleValueContainer().decode([String: OpenAIJSONValue].self)
        try rejectUnknown(fields, allowed: [
            "model", "input", "instructions", "previous_response_id", "store", "stream",
            "max_output_tokens", "temperature", "top_p", "tools", "tool_choice",
            "parallel_tool_calls", "reasoning", "text", "metadata", "user", "safety_identifier",
            "prompt_cache_key", "background", "truncation", "service_tier", "include",
            "top_logprobs", "stream_options",
        ], at: "")
        model = try requiredString(fields["model"], at: "model", nonempty: true)
        instructions = try optionalString(fields["instructions"], at: "instructions")
        previousResponseID = try optionalString(
            fields["previous_response_id"], at: "previous_response_id", nonempty: true)
        store = try optionalBool(fields["store"], at: "store") ?? true
        stream = try optionalBool(fields["stream"], at: "stream") ?? false
        parallelToolCalls = try optionalBool(fields["parallel_tool_calls"], at: "parallel_tool_calls") ?? true
        maxOutputTokens = try optionalInteger(fields["max_output_tokens"], at: "max_output_tokens")
        if let maxOutputTokens, maxOutputTokens < 1 {
            throw invalid("max_output_tokens must be greater than zero", at: "max_output_tokens")
        }
        temperature = try optionalNumber(fields["temperature"], at: "temperature")
        if let temperature, !(0...2).contains(temperature) {
            throw invalid("temperature must be between 0 and 2", at: "temperature")
        }
        topP = try optionalNumber(fields["top_p"], at: "top_p")
        if let topP, topP <= 0 || topP > 1 {
            throw invalid("top_p must be greater than 0 and at most 1", at: "top_p")
        }

        let input = try Self.normalizeInput(fields["input"])
        inputMessages = input.messages
        inputItems = input.items
        tools = try Self.decodeTools(fields["tools"])
        toolChoice = try Self.decodeToolChoice(fields["tool_choice"])
        responseFormat = try Self.decodeText(fields["text"])
        reasoningEffort = try Self.decodeReasoning(fields["reasoning"])

        let metadataObject = try optionalObject(fields["metadata"], at: "metadata") ?? [:]
        guard metadataObject.count <= 16 else {
            throw invalid("metadata supports at most 16 entries", at: "metadata")
        }
        var metadata: [String: String] = [:]
        for key in metadataObject.keys.sorted() {
            let value = try requiredString(metadataObject[key], at: "metadata.\(key)")
            guard key.count <= 64, value.count <= 512 else {
                throw invalid("metadata keys and values are limited to 64 and 512 characters", at: "metadata.\(key)")
            }
            metadata[key] = value
        }
        self.metadata = metadata

        var passive: [String: OpenAIJSONValue] = [:]
        for key in ["user", "safety_identifier", "prompt_cache_key"] {
            if let value = try optionalString(fields[key], at: key) { passive[key] = .string(value) }
        }
        if try optionalBool(fields["background"], at: "background") == true {
            throw invalid("Background responses are not supported", at: "background")
        }
        if let truncation = try optionalString(fields["truncation"], at: "truncation"), truncation != "disabled" {
            throw invalid("Only truncation 'disabled' is supported", at: "truncation")
        }
        if let tier = try optionalString(fields["service_tier"], at: "service_tier") {
            guard ["auto", "default"].contains(tier) else {
                throw invalid("Only service_tier 'auto' or 'default' is supported", at: "service_tier")
            }
            passive["service_tier"] = .string(tier)
        }
        if let include = try optionalArray(fields["include"], at: "include"), !include.isEmpty {
            throw invalid("Additional include fields are not supported", at: "include")
        }
        if let count = try optionalInteger(fields["top_logprobs"], at: "top_logprobs"), count != 0 {
            throw invalid("Log probabilities are not supported", at: "top_logprobs")
        }
        if let options = try optionalObject(fields["stream_options"], at: "stream_options") {
            try rejectUnknown(options, allowed: ["include_obfuscation"], at: "stream_options")
            if try optionalBool(options["include_obfuscation"], at: "stream_options.include_obfuscation") == true {
                throw invalid("Stream obfuscation is not supported", at: "stream_options.include_obfuscation")
            }
            guard stream else {
                throw invalid("stream_options requires stream: true", at: "stream_options")
            }
        }
        passiveFields = passive
    }

    /// Accepted configuration in Responses wire format, for the response envelope.
    /// Omitted sampling parameters remain null; the HTTP layer may replace them with runtime defaults.
    public func responseFields() -> [String: OpenAIJSONValue] {
        var fields = passiveFields
        fields["model"] = .string(model)
        fields["instructions"] = instructions.map(OpenAIJSONValue.string) ?? .null
        fields["previous_response_id"] = previousResponseID.map(OpenAIJSONValue.string) ?? .null
        fields["store"] = .bool(store)
        fields["parallel_tool_calls"] = .bool(parallelToolCalls)
        fields["max_output_tokens"] = maxOutputTokens.map(OpenAIJSONValue.integer) ?? .null
        fields["temperature"] = temperature.map(OpenAIJSONValue.number) ?? .null
        fields["top_p"] = topP.map(OpenAIJSONValue.number) ?? .null
        fields["metadata"] = .object(metadata.mapValues(OpenAIJSONValue.string))
        fields["background"] = .bool(false)
        fields["truncation"] = .string("disabled")
        fields["service_tier"] = fields["service_tier"] ?? .string("default")
        fields["top_logprobs"] = .integer(0)
        fields["reasoning"] = .object([
            "effort": reasoningEffort.map { .string($0.rawValue) } ?? .null,
            "summary": .null,
        ])
        fields["text"] = .object(["format": Self.formatJSON(responseFormat ?? .text)])
        fields["tools"] = .array((tools ?? []).map { tool in
            var function: [String: OpenAIJSONValue] = [
                "type": .string("function"), "name": .string(tool.function.name),
                "parameters": tool.function.parameters, "strict": .bool(false),
            ]
            if let description = tool.function.description { function["description"] = .string(description) }
            return .object(function)
        })
        switch toolChoice ?? .auto {
        case .none: fields["tool_choice"] = .string("none")
        case .auto: fields["tool_choice"] = .string("auto")
        case .required: fields["tool_choice"] = .string("required")
        case .function(let name): fields["tool_choice"] = .object(["type": .string("function"), "name": .string(name)])
        }
        return fields
    }

    private static func normalizeInput(_ value: OpenAIJSONValue?) throws -> (messages: [OpenAIMessage], items: [OpenAIJSONValue]) {
        if absent(value) { return ([], []) }
        let values: [OpenAIJSONValue]
        if case .string(let text) = value {
            values = [.object(["role": .string("user"), "content": .string(text)])]
        } else {
            values = try requiredArray(value, at: "input")
        }
        guard values.count <= 4096 else {
            throw invalid("input supports at most 4096 items", at: "input")
        }
        var messages: [OpenAIMessage] = []
        var items: [OpenAIJSONValue] = []
        var precedingFunctionCall = false
        var seenIDs = Set<String>()
        var seenCalls = Set<String>()
        for (index, value) in values.enumerated() {
            let path = "input[\(index)]"
            let item = try requiredObject(value, at: path)
            let type = try optionalString(item["type"], at: "\(path).type") ?? "message"
            let identifier = try optionalString(item["id"], at: "\(path).id", nonempty: true)
                ?? "\(type == "message" ? "msg" : "fc")_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased())"
            guard seenIDs.insert(identifier).inserted else {
                throw invalid("Input item IDs must be unique", at: "\(path).id")
            }
            let status = try optionalString(item["status"], at: "\(path).status") ?? "completed"
            guard ["completed", "incomplete", "in_progress"].contains(status) else {
                throw invalid("Invalid input item status", at: "\(path).status")
            }
            var canonical = item
            canonical["id"] = .string(identifier)
            canonical["type"] = .string(type)
            canonical["status"] = .string(status)
            switch type {
            case "message":
                try rejectUnknown(item, allowed: ["id", "type", "role", "content", "status", "phase"], at: path)
                // SDK model_dump() may include its unset optional phase field.
                // Nonnull phases carry model-specific semantics we do not implement.
                if !absent(item["phase"]) {
                    throw invalid("Message phases are not supported", at: "\(path).phase")
                }
                canonical.removeValue(forKey: "phase")
                let role = try requiredString(item["role"], at: "\(path).role")
                guard ["user", "assistant", "system", "developer"].contains(role) else {
                    throw invalid("Message role must be user, assistant, system, or developer", at: "\(path).role")
                }
                let content = try textContent(item["content"], at: "\(path).content", assistant: role == "assistant")
                canonical["content"] = .array(content.parts)
                messages.append(.init(role: role, content: content.text))
                precedingFunctionCall = false
            case "function_call":
                try rejectUnknown(item, allowed: ["id", "type", "call_id", "name", "arguments", "status", "async", "async_", "caller", "namespace"], at: path)
                // Current SDK model_dump() includes these optional fields even
                // for ordinary synchronous, direct function calls.
                for key in ["async", "async_", "caller", "namespace"] {
                    if !absent(item[key]) && !(["async", "async_"].contains(key) && item[key] == .bool(false)) {
                        throw invalid("Function call \(key) is not supported", at: "\(path).\(key)")
                    }
                    canonical.removeValue(forKey: key)
                }
                let callID = try requiredString(item["call_id"], at: "\(path).call_id", nonempty: true)
                guard seenCalls.insert(callID).inserted else {
                    throw invalid("Function call IDs must be unique", at: "\(path).call_id")
                }
                let name = try functionName(item["name"], at: "\(path).name")
                let arguments = try requiredString(item["arguments"], at: "\(path).arguments")
                let call = OpenAIToolCall(id: callID, function: .init(name: name, arguments: arguments))
                if precedingFunctionCall, let preceding = messages.popLast() {
                    messages.append(.init(role: "assistant", content: preceding.content, toolCalls: (preceding.toolCalls ?? []) + [call]))
                } else {
                    messages.append(.init(role: "assistant", content: nil, toolCalls: [call]))
                }
                precedingFunctionCall = true
            case "function_call_output":
                try rejectUnknown(item, allowed: ["id", "type", "call_id", "output", "status"], at: path)
                let callID = try requiredString(item["call_id"], at: "\(path).call_id", nonempty: true)
                let content = try textContent(item["output"], at: "\(path).output", assistant: false)
                messages.append(.init(role: "tool", content: content.text, toolCallID: callID))
                precedingFunctionCall = false
            default:
                throw invalid("Input item type '\(type)' is not supported; only text messages, function_call, and function_call_output are supported", at: "\(path).type")
            }
            items.append(.object(canonical))
        }
        return (messages, items)
    }

    private static func textContent(_ value: OpenAIJSONValue?, at path: String, assistant: Bool) throws -> (text: String, parts: [OpenAIJSONValue]) {
        if case .string(let text) = value {
            var part: [String: OpenAIJSONValue] = ["type": .string(assistant ? "output_text" : "input_text"), "text": .string(text)]
            if assistant { part["annotations"] = .array([]) }
            return (text, [.object(part)])
        }
        let values = try requiredArray(value, at: path)
        guard values.count <= 4096 else {
            throw invalid("Text content supports at most 4096 parts", at: path)
        }
        var text = ""
        var parts: [OpenAIJSONValue] = []
        for (index, value) in values.enumerated() {
            let partPath = "\(path)[\(index)]"
            var part = try requiredObject(value, at: partPath)
            let type = try requiredString(part["type"], at: "\(partPath).type")
            guard type == "input_text" || (assistant && type == "output_text") else {
                throw invalid("Only text content is supported; '\(type)' is not supported here", at: "\(partPath).type")
            }
            try rejectUnknown(part, allowed: type == "output_text" ? ["type", "text", "annotations", "logprobs", "parsed"] : ["type", "text"], at: partPath)
            if type == "output_text" {
                // OpenAI Python's parse()/stream() helpers attach this local
                // convenience value when replaying output objects. Text remains
                // authoritative; parsed is not a Responses API content field.
                part.removeValue(forKey: "parsed")
                for key in ["annotations", "logprobs"] {
                    if let values = try optionalArray(part[key], at: "\(partPath).\(key)"), !values.isEmpty {
                        throw invalid("Nonempty output_text \(key) are not supported", at: "\(partPath).\(key)")
                    }
                }
                part["annotations"] = .array([])
            }
            text += try requiredString(part["text"], at: "\(partPath).text")
            parts.append(.object(part))
        }
        return (text, parts)
    }

    private static func decodeTools(_ value: OpenAIJSONValue?) throws -> [OpenAIToolDefinition]? {
        guard let values = try optionalArray(value, at: "tools") else { return nil }
        guard values.count <= 128 else { throw invalid("At most 128 tools are supported", at: "tools") }
        return try values.enumerated().map { index, value in
            let path = "tools[\(index)]"
            let tool = try requiredObject(value, at: path)
            let type = try requiredString(tool["type"], at: "\(path).type")
            guard type == "function" else {
                throw invalid("Only function tools are supported", at: "\(path).type")
            }
            try rejectUnknown(tool, allowed: ["type", "name", "description", "parameters", "strict"], at: path)
            if try optionalBool(tool["strict"], at: "\(path).strict") == true {
                throw invalid("Strict function tools are not supported; use strict: false", at: "\(path).strict")
            }
            let name = try functionName(tool["name"], at: "\(path).name")
            let description = try optionalString(tool["description"], at: "\(path).description")
            let parameters = try optionalObject(tool["parameters"], at: "\(path).parameters") ?? [:]
            return .init(function: .init(name: name, description: description, parameters: .object(parameters)))
        }
    }

    private static func decodeToolChoice(_ value: OpenAIJSONValue?) throws -> OpenAIToolChoice? {
        if absent(value) { return nil }
        if case .string(let choice) = value {
            switch choice {
            case "auto": return .auto
            case "none": return OpenAIToolChoice.none
            case "required": return .required
            default: throw invalid("Invalid tool_choice; expected auto, none, required, or a named function", at: "tool_choice")
            }
        }
        let choice = try requiredObject(value, at: "tool_choice")
        try rejectUnknown(choice, allowed: ["type", "name"], at: "tool_choice")
        guard try requiredString(choice["type"], at: "tool_choice.type") == "function" else {
            throw invalid("Only named function tool choices are supported", at: "tool_choice.type")
        }
        return .function(name: try functionName(choice["name"], at: "tool_choice.name"))
    }

    private static func decodeReasoning(_ value: OpenAIJSONValue?) throws -> ChatCompletionRequest.ReasoningEffort? {
        guard let reasoning = try optionalObject(value, at: "reasoning") else { return nil }
        try rejectUnknown(reasoning, allowed: ["effort", "summary"], at: "reasoning")
        if !absent(reasoning["summary"]) {
            throw invalid("Reasoning summaries are not supported", at: "reasoning.summary")
        }
        guard let effort = try optionalString(reasoning["effort"], at: "reasoning.effort") else { return nil }
        guard let result = ChatCompletionRequest.ReasoningEffort(rawValue: effort) else {
            throw invalid("Supported reasoning effort values are low, medium, and high", at: "reasoning.effort")
        }
        return result
    }

    private static func decodeText(_ value: OpenAIJSONValue?) throws -> OpenAIResponseFormat? {
        guard let text = try optionalObject(value, at: "text") else { return nil }
        try rejectUnknown(text, allowed: ["format"], at: "text")
        guard let format = try optionalObject(text["format"], at: "text.format") else { return nil }
        let type = try requiredString(format["type"], at: "text.format.type")
        switch type {
        case "text", "json_object":
            try rejectUnknown(format, allowed: ["type"], at: "text.format")
            return type == "text" ? .text : .jsonObject
        case "json_schema":
            try rejectUnknown(format, allowed: ["type", "name", "description", "schema", "strict"], at: "text.format")
            let name = try functionName(format["name"], at: "text.format.name")
            let description = try optionalString(format["description"], at: "text.format.description")
            let strict = try optionalBool(format["strict"], at: "text.format.strict")
            guard let schema = format["schema"], case .object = schema else {
                throw invalid("text.format.schema must be a JSON Schema object", at: "text.format.schema")
            }
            return .jsonSchema(.init(name: name, description: description, schema: schema, strict: strict))
        default:
            throw invalid("text.format.type must be text, json_object, or json_schema", at: "text.format.type")
        }
    }

    private static func formatJSON(_ format: OpenAIResponseFormat) -> OpenAIJSONValue {
        switch format {
        case .text: return .object(["type": .string("text")])
        case .jsonObject: return .object(["type": .string("json_object")])
        case .jsonSchema(let schema):
            var fields: [String: OpenAIJSONValue] = [
                "type": .string("json_schema"), "name": .string(schema.name), "schema": schema.schema,
            ]
            if let description = schema.description { fields["description"] = .string(description) }
            if let strict = schema.strict { fields["strict"] = .bool(strict) }
            return .object(fields)
        }
    }
}

private func invalid(_ message: String, at path: String) -> ResponsesRequestError {
    .init(message: message, param: path)
}

private func absent(_ value: OpenAIJSONValue?) -> Bool { value == nil || value == .null }

private func rejectUnknown(_ object: [String: OpenAIJSONValue], allowed: Set<String>, at path: String) throws {
    for key in object.keys.sorted() where !allowed.contains(key) {
        let member = path.isEmpty ? key : "\(path).\(key)"
        throw invalid("Unsupported Responses parameter '\(member)'", at: member)
    }
}

private func requiredString(_ value: OpenAIJSONValue?, at path: String, nonempty: Bool = false) throws -> String {
    guard case .string(let result) = value, !nonempty || !result.isEmpty else {
        throw invalid("\(path) must be \(nonempty ? "a nonempty string" : "a string")", at: path)
    }
    return result
}

private func optionalString(_ value: OpenAIJSONValue?, at path: String, nonempty: Bool = false) throws -> String? {
    if absent(value) { return nil }
    return try requiredString(value, at: path, nonempty: nonempty)
}

private func optionalBool(_ value: OpenAIJSONValue?, at path: String) throws -> Bool? {
    if absent(value) { return nil }
    guard case .bool(let result) = value else { throw invalid("\(path) must be a boolean", at: path) }
    return result
}

private func optionalInteger(_ value: OpenAIJSONValue?, at path: String) throws -> Int? {
    if absent(value) { return nil }
    guard case .integer(let result) = value else { throw invalid("\(path) must be an integer", at: path) }
    return result
}

private func optionalNumber(_ value: OpenAIJSONValue?, at path: String) throws -> Double? {
    if absent(value) { return nil }
    switch value {
    case .integer(let result): return Double(result)
    case .number(let result) where result.isFinite: return result
    default: throw invalid("\(path) must be a finite number", at: path)
    }
}

private func requiredObject(_ value: OpenAIJSONValue?, at path: String) throws -> [String: OpenAIJSONValue] {
    guard case .object(let result) = value else { throw invalid("\(path) must be an object", at: path) }
    return result
}

private func optionalObject(_ value: OpenAIJSONValue?, at path: String) throws -> [String: OpenAIJSONValue]? {
    if absent(value) { return nil }
    return try requiredObject(value, at: path)
}

private func requiredArray(_ value: OpenAIJSONValue?, at path: String) throws -> [OpenAIJSONValue] {
    guard case .array(let result) = value else { throw invalid("\(path) must be an array", at: path) }
    return result
}

private func optionalArray(_ value: OpenAIJSONValue?, at path: String) throws -> [OpenAIJSONValue]? {
    if absent(value) { return nil }
    return try requiredArray(value, at: path)
}

private func functionName(_ value: OpenAIJSONValue?, at path: String) throws -> String {
    let name = try requiredString(value, at: path, nonempty: true)
    guard (1...64).contains(name.utf8.count), name.utf8.allSatisfy({ byte in
        (65...90).contains(byte) || (97...122).contains(byte) || (48...57).contains(byte) || byte == 95 || byte == 45
    }) else {
        throw invalid("\(path) must contain 1 to 64 letters, digits, underscores, or hyphens", at: path)
    }
    return name
}
