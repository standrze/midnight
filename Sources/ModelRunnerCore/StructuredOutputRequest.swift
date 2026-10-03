import Foundation
import ModelRunnerProtocol

/// An unsupported combination or failure during constrained JSON generation.
public enum StructuredOutputRequestError: LocalizedError {
    case incompatibleTools
    case incompatibleStop
    case unsupportedTokenizer(String)
    case generation(String)

    /// User-facing explanation for this error.
    public var errorDescription: String? {
        switch self {
        case .incompatibleTools:
            "Structured response_format cannot be combined with enabled tools in this release. Use tool_choice: none or omit tools."
        case .incompatibleStop:
            "Structured response_format cannot be combined with stop strings, which could interrupt a JSON value."
        case .unsupportedTokenizer(let message):
            "Structured response_format is unavailable for this tokenizer: \(message)"
        case .generation(let message):
            "Structured output generation failed: \(message)"
        }
    }
}

/// Validates and prepares JSON response-format constraints for generation.
public enum StructuredOutputRequest {
    /// Returns whether a response format requires constrained JSON output.
    public static func isStructured(_ format: OpenAIResponseFormat?) -> Bool {
        switch format {
        case nil, .text: false
        case .jsonObject, .jsonSchema: true
        }
    }

    /// Validate before streaming headers and again for direct runtime clients.
    public static func validate(
        format: OpenAIResponseFormat?, tools: [OpenAIToolDefinition]?, stop: [String]
    ) throws {
        guard isStructured(format), let format else {
            return
        }
        guard tools?.isEmpty != false else {
            throw StructuredOutputRequestError.incompatibleTools
        }
        guard stop.isEmpty else {
            throw StructuredOutputRequestError.incompatibleStop
        }
        _ = try StructuredOutputGrammar(responseFormat: format)
    }

    static func instruction(for format: OpenAIResponseFormat) throws -> String {
        switch format {
        case .text: return ""
        case .jsonObject:
            return "Return a JSON object only. Do not include markdown fences or commentary outside the JSON."
        case .jsonSchema(let configuration):
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
            let schema = String(decoding: try encoder.encode(configuration.schema), as: UTF8.self)
            return
                "Return only a JSON object matching this JSON Schema. Do not include markdown fences or commentary outside the JSON. Schema: \(schema)"
        }
    }
}
