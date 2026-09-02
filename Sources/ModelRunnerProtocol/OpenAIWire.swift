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

    enum CodingKeys: String, CodingKey {
        case role, content, name
        case toolCalls = "tool_calls"
        case toolCallID = "tool_call_id"
    }
}

public struct ChatCompletionRequest: Codable, Equatable, Sendable {
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
    public let streamOptions: StreamOptions?

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
        streamOptions: StreamOptions? = nil
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
        self.streamOptions = streamOptions
    }

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, temperature, tools, stop
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case topP = "top_p"
        case streamOptions = "stream_options"
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
    public let promptTokens: Int
    public let completionTokens: Int
    public let totalTokens: Int

    public init(promptTokens: Int, completionTokens: Int) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = promptTokens + completionTokens
    }

    enum CodingKeys: String, CodingKey {
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
