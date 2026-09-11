import Foundation

public enum OpenAIJSONValue: Codable, Equatable, Sendable {
    case null
    case bool(Bool)
    case integer(Int)
    case number(Double)
    case string(String)
    case array([OpenAIJSONValue])
    case object([String: OpenAIJSONValue])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .integer(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([OpenAIJSONValue].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: OpenAIJSONValue].self) {
            self = .object(value)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported JSON value"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let value): try container.encode(value)
        case .integer(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        }
    }

    public var sendableValue: any Sendable {
        switch self {
        case .null: NSNull()
        case .bool(let value): value
        case .integer(let value): value
        case .number(let value): value
        case .string(let value): value
        case .array(let value): value.map(\.sendableValue)
        case .object(let value): value.mapValues(\.sendableValue)
        }
    }
}

/// OpenAI-compatible output controls for Chat Completions.
public enum OpenAIResponseFormat: Codable, Equatable, Sendable {
    case text
    case jsonObject
    case jsonSchema(OpenAIJSONSchemaResponseFormat)

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "text":
            try rejectUnknownResponseFormatKeys(in: decoder, allowed: ["type"])
            self = .text
        case "json_object":
            try rejectUnknownResponseFormatKeys(in: decoder, allowed: ["type"])
            self = .jsonObject
        case "json_schema":
            try rejectUnknownResponseFormatKeys(in: decoder, allowed: ["type", "json_schema"])
            self = .jsonSchema(try container.decode(OpenAIJSONSchemaResponseFormat.self, forKey: .jsonSchema))
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type,
                in: container,
                debugDescription: "response_format.type must be 'text', 'json_object', or 'json_schema'"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text:
            try container.encode("text", forKey: .type)
        case .jsonObject:
            try container.encode("json_object", forKey: .type)
        case .jsonSchema(let format):
            try container.encode("json_schema", forKey: .type)
            try container.encode(format, forKey: .jsonSchema)
        }
    }

    private enum CodingKeys: String, CodingKey {
        case type
        case jsonSchema = "json_schema"
    }
}

/// The schema wrapper in a `response_format` of type `json_schema`.
/// The generation layer validates which JSON Schema features are supported.
public struct OpenAIJSONSchemaResponseFormat: Codable, Equatable, Sendable {
    public let name: String
    public let description: String?
    public let schema: OpenAIJSONValue
    public let strict: Bool?

    public init(
        name: String,
        description: String? = nil,
        schema: OpenAIJSONValue,
        strict: Bool? = nil
    ) {
        self.name = name
        self.description = description
        self.schema = schema
        self.strict = strict
    }

    public init(from decoder: Decoder) throws {
        try rejectUnknownResponseFormatKeys(in: decoder, allowed: ["name", "description", "schema", "strict"])
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        guard (1...64).contains(name.utf8.count), name.utf8.allSatisfy({ byte in
            (65...90).contains(byte) || (97...122).contains(byte) || (48...57).contains(byte)
                || byte == 95 || byte == 45
        }) else {
            throw DecodingError.dataCorruptedError(
                forKey: .name,
                in: container,
                debugDescription: "response_format.json_schema.name must contain 1 to 64 letters, digits, underscores, or hyphens"
            )
        }
        description = try container.decodeIfPresent(String.self, forKey: .description)
        schema = try container.decode(OpenAIJSONValue.self, forKey: .schema)
        strict = try container.decodeIfPresent(Bool.self, forKey: .strict)
    }

    private enum CodingKeys: String, CodingKey {
        case name, description, schema, strict
    }
}

private struct ResponseFormatObjectKey: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }

    init?(stringValue: String) {
        self.stringValue = stringValue
    }

    init?(intValue: Int) { return nil }
}

private func rejectUnknownResponseFormatKeys(in decoder: Decoder, allowed: Set<String>) throws {
    let container = try decoder.container(keyedBy: ResponseFormatObjectKey.self)
    for key in container.allKeys where !allowed.contains(key.stringValue) {
        throw DecodingError.dataCorruptedError(
            forKey: key,
            in: container,
            debugDescription: "Unsupported response_format member '\(key.stringValue)'"
        )
    }
}

public struct OpenAIToolDefinition: Codable, Equatable, Sendable {
    public struct Function: Codable, Equatable, Sendable {
        public let name: String
        public let description: String?
        public let parameters: OpenAIJSONValue

        public init(
            name: String,
            description: String? = nil,
            parameters: OpenAIJSONValue = .object([:])
        ) {
            self.name = name
            self.description = description
            self.parameters = parameters
        }
    }

    public let type: String
    public let function: Function

    public init(type: String = "function", function: Function) {
        self.type = type
        self.function = function
    }
}

/// OpenAI-compatible tool selection for chat completions.
///
/// A named function choice is represented on the wire as
/// `{ "type": "function", "function": { "name": "..." } }`.
public enum OpenAIToolChoice: Codable, Equatable, Sendable {
    case none
    case auto
    case required
    case function(name: String)

    private struct NamedChoice: Codable {
        struct Function: Codable {
            let name: String
        }

        let type: String
        let function: Function
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) {
            switch value {
            case "none": self = .none
            case "auto": self = .auto
            case "required": self = .required
            default:
                throw DecodingError.dataCorruptedError(
                    in: container,
                    debugDescription: "tool_choice must be 'none', 'auto', 'required', or a named function"
                )
            }
            return
        }

        let choice = try container.decode(NamedChoice.self)
        guard choice.type == "function" else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Named tool_choice type must be 'function'"
            )
        }
        self = .function(name: choice.function.name)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .none: try container.encode("none")
        case .auto: try container.encode("auto")
        case .required: try container.encode("required")
        case .function(let name):
            try container.encode(
                NamedChoice(type: "function", function: .init(name: name))
            )
        }
    }
}

public struct OpenAIToolCall: Codable, Equatable, Sendable {
    public struct Function: Codable, Equatable, Sendable {
        public let name: String
        public let arguments: String

        public init(name: String, arguments: String) {
            self.name = name
            self.arguments = arguments
        }
    }

    public let id: String
    public let type: String
    public let function: Function

    public init(id: String, type: String = "function", function: Function) {
        self.id = id
        self.type = type
        self.function = function
    }
}

public struct OpenAIToolCallDelta: Codable, Equatable, Sendable {
    public struct Function: Codable, Equatable, Sendable {
        public let name: String?
        public let arguments: String?

        public init(name: String? = nil, arguments: String? = nil) {
            self.name = name
            self.arguments = arguments
        }
    }

    public let index: Int
    public let id: String?
    public let type: String?
    public let function: Function?

    public init(
        index: Int,
        id: String? = nil,
        type: String? = nil,
        function: Function? = nil
    ) {
        self.index = index
        self.id = id
        self.type = type
        self.function = function
    }
}

public struct OpenAIMessage: Codable, Equatable, Sendable {
    public let role: String
    public let content: String?
    public let name: String?
    public let toolCalls: [OpenAIToolCall]?
    public let toolCallID: String?

    public init(
        role: String,
        content: String?,
        name: String? = nil,
        toolCalls: [OpenAIToolCall]? = nil,
        toolCallID: String? = nil
    ) {
        self.role = role
        self.content = content
        self.name = name
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(String.self, forKey: .role)
        name = try container.decodeIfPresent(String.self, forKey: .name)
        toolCalls = try container.decodeIfPresent([OpenAIToolCall].self, forKey: .toolCalls)
        toolCallID = try container.decodeIfPresent(String.self, forKey: .toolCallID)

        if try !container.contains(.content) || container.decodeNil(forKey: .content) {
            content = nil
        } else if let text = try? container.decode(String.self, forKey: .content) {
            content = text
        } else {
            // Text-only OpenAI clients can send ordered content parts with optional
            // metadata (for example Pool's cache_control). Normalize at the wire
            // boundary so templates and response encoding keep their string format.
            content = try container.decode([TextPart].self, forKey: .content)
                .map(\.text).joined()
        }
    }

    private struct TextPart: Decodable {
        let text: String

        enum CodingKeys: String, CodingKey {
            case type, text
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let type = try container.decode(String.self, forKey: .type)
            guard type == "text" else {
                throw DecodingError.dataCorruptedError(
                    forKey: .type,
                    in: container,
                    debugDescription: "Only text content parts are supported by chat completions."
                )
            }
            text = try container.decode(String.self, forKey: .text)
        }
    }

    enum CodingKeys: String, CodingKey {
        case role, content, name
        case toolCalls = "tool_calls"
        case toolCallID = "tool_call_id"
    }
}

public struct ChatCompletionRequest: Codable, Equatable, Sendable {
    /// GPT-OSS controls reasoning length through its chat template.
    /// Leaving this unset preserves the loaded model's template default.
    public enum ReasoningEffort: String, Codable, Equatable, Sendable {
        case low
        case medium
        case high
    }

    public struct StreamOptions: Codable, Equatable, Sendable {
        public let includeUsage: Bool

        public init(includeUsage: Bool = false) {
            self.includeUsage = includeUsage
        }

        enum CodingKeys: String, CodingKey {
            case includeUsage = "include_usage"
        }
    }

    public let model: String
    public let messages: [OpenAIMessage]
    public let stream: Bool?
    public let maxTokens: Int?
    public let maxCompletionTokens: Int?
    public let temperature: Double?
    public let topP: Double?
    public let stop: OpenAIStop?
    public let tools: [OpenAIToolDefinition]?
    public let toolChoice: OpenAIToolChoice?
    public let streamOptions: StreamOptions?
    public let reasoningEffort: ReasoningEffort?
    public let responseFormat: OpenAIResponseFormat?

    public init(
        model: String,
        messages: [OpenAIMessage],
        stream: Bool,
        maxTokens: Int? = nil,
        maxCompletionTokens: Int? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        stop: OpenAIStop? = nil,
        tools: [OpenAIToolDefinition]? = nil,
        toolChoice: OpenAIToolChoice? = nil,
        streamOptions: StreamOptions? = nil,
        reasoningEffort: ReasoningEffort? = nil,
        responseFormat: OpenAIResponseFormat? = nil
    ) {
        self.model = model
        self.messages = messages
        self.stream = stream
        self.maxTokens = maxTokens
        self.maxCompletionTokens = maxCompletionTokens
        self.temperature = temperature
        self.topP = topP
        self.stop = stop
        self.tools = tools
        self.toolChoice = toolChoice
        self.streamOptions = streamOptions
        self.reasoningEffort = reasoningEffort
        self.responseFormat = responseFormat
    }

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, temperature, tools, stop
        case toolChoice = "tool_choice"
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case topP = "top_p"
        case streamOptions = "stream_options"
        case reasoningEffort = "reasoning_effort"
        case responseFormat = "response_format"
    }
}

public enum OpenAIStop: Codable, Equatable, Sendable {
    case string(String)
    case strings([String])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) {
            self = .string(value)
        } else {
            self = .strings(try container.decode([String].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .strings(let values): try container.encode(values)
        }
    }

    public var values: [String] {
        switch self {
        case .string(let value): [value]
        case .strings(let values): values
        }
    }
}

public struct ChatCompletionUsage: Codable, Equatable, Sendable {
    public struct PromptTokensDetails: Codable, Equatable, Sendable {
        public let cachedTokens: Int
        enum CodingKeys: String, CodingKey { case cachedTokens = "cached_tokens" }
    }
    public let promptTokensDetails: PromptTokensDetails?

    public let promptTokens: Int
    public let completionTokens: Int
    public let totalTokens: Int

    public init(promptTokens: Int, completionTokens: Int, cachedTokens: Int = 0) {
        self.promptTokensDetails = .init(cachedTokens: min(promptTokens, max(0, cachedTokens)))
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = promptTokens + completionTokens
    }

    enum CodingKeys: String, CodingKey {
        case promptTokensDetails = "prompt_tokens_details"
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }
}

public struct ChatCompletionChunk: Codable, Equatable, Sendable {
    public struct Choice: Codable, Equatable, Sendable {
        public struct Delta: Codable, Equatable, Sendable {
            public let role: String?
            public let content: String?
            public let toolCalls: [OpenAIToolCallDelta]?

            public init(
                role: String? = nil,
                content: String? = nil,
                toolCalls: [OpenAIToolCallDelta]? = nil
            ) {
                self.role = role
                self.content = content
                self.toolCalls = toolCalls
            }

            enum CodingKeys: String, CodingKey {
                case role, content
                case toolCalls = "tool_calls"
            }
        }

        public let index: Int
        public let delta: Delta
        public let finishReason: String?

        public init(index: Int = 0, delta: Delta, finishReason: String? = nil) {
            self.index = index
            self.delta = delta
            self.finishReason = finishReason
        }

        enum CodingKeys: String, CodingKey {
            case index, delta
            case finishReason = "finish_reason"
        }
    }

    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [Choice]
    public let usage: ChatCompletionUsage?

    public init(
        id: String,
        model: String,
        choices: [Choice],
        usage: ChatCompletionUsage? = nil
    ) {
        self.id = id
        self.object = "chat.completion.chunk"
        self.created = Int(Date().timeIntervalSince1970)
        self.model = model
        self.choices = choices
        self.usage = usage
    }

    enum CodingKeys: String, CodingKey {
        case id, object, created, model, choices, usage
    }
}

public struct ChatCompletionResponse: Encodable, Equatable, Sendable {
    public struct Choice: Codable, Equatable, Sendable {
        public let index: Int
        public let message: OpenAIMessage
        public let finishReason: String

        public init(index: Int = 0, message: OpenAIMessage, finishReason: String) {
            self.index = index
            self.message = message
            self.finishReason = finishReason
        }

        enum CodingKeys: String, CodingKey {
            case index, message
            case finishReason = "finish_reason"
        }
    }

    public let id: String
    public let object = "chat.completion"
    public let created: Int
    public let model: String
    public let choices: [Choice]
    public let usage: ChatCompletionUsage

    public init(
        id: String,
        created: Int = Int(Date().timeIntervalSince1970),
        model: String,
        choices: [Choice],
        usage: ChatCompletionUsage
    ) {
        self.id = id
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
    }


    enum CodingKeys: String, CodingKey {
        case id, object, created, model, choices, usage
    }
}

public struct OpenAIErrorEnvelope: Codable, Equatable, Sendable {
    public struct Detail: Codable, Equatable, Sendable {
        public let message: String
        public let type: String
        public let param: String?
        public let code: String?

        public init(
            message: String,
            type: String = "invalid_request_error",
            param: String? = nil,
            code: String? = nil
        ) {
            self.message = message
            self.type = type
            self.param = param
            self.code = code
        }
    }

    public let error: Detail

    public init(
        message: String,
        type: String = "invalid_request_error",
        param: String? = nil,
        code: String? = nil
    ) {
        self.error = Detail(message: message, type: type, param: param, code: code)
    }
}
