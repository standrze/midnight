import Foundation
@testable import ModelRunnerCore
import ModelRunnerProtocol
import Testing

@Suite("Structured output request validation")
struct StructuredOutputRequestTests {
    private let tools = [OpenAIToolDefinition(function: .init(name: "lookup"))]
    private let schema: OpenAIJSONValue = .object([
        "type": .string("object"),
        "properties": .object([
            "answer": .object(["type": .string("string")]),
        ]),
        "required": .array([.string("answer")]),
        "additionalProperties": .bool(false),
    ])

    @Test("Absent and text formats retain ordinary tool and stop behavior", arguments: [
        Optional<OpenAIResponseFormat>.none, .some(.text),
    ])
    func ordinaryFormats(_ format: OpenAIResponseFormat?) throws {
        #expect(!StructuredOutputRequest.isStructured(format))
        try StructuredOutputRequest.validate(format: format, tools: tools, stop: ["END"])
    }

    @Test("JSON object and JSON schema accept no enabled tools or stop strings")
    func validStructuredRequests() throws {
        for format in formats {
            #expect(StructuredOutputRequest.isStructured(format))
            try StructuredOutputRequest.validate(format: format, tools: nil, stop: [])
            // The tool-choice plan passes an empty declaration list for tool_choice: none.
            try StructuredOutputRequest.validate(format: format, tools: [], stop: [])
        }
    }

    @Test("Enabled tools produce a specific structured output validation error")
    func incompatibleTools() throws {
        for format in formats {
            do {
                try StructuredOutputRequest.validate(format: format, tools: tools, stop: [])
                Issue.record("Expected enabled tools to be rejected")
            } catch StructuredOutputRequestError.incompatibleTools {
                // Expected: this cannot quietly choose between JSON and a tool call.
            }
        }
    }

    @Test("Custom stops produce a specific structured output validation error")
    func incompatibleStops() throws {
        for format in formats {
            do {
                try StructuredOutputRequest.validate(format: format, tools: nil, stop: ["}"])
                Issue.record("Expected a stop string to be rejected before JSON can be truncated")
            } catch StructuredOutputRequestError.incompatibleStop {
                // Expected: schema validation must not silently discard this option.
            }
        }
    }

    @Test("Unsupported schema constraints fail request validation")
    func unsupportedSchema() {
        let unsupported: OpenAIJSONValue = .object([
            "type": .string("object"),
            "properties": .object([
                "answer": .object([
                    "type": .string("string"),
                    "pattern": .string("^[a-z]+$"),
                ]),
            ]),
            "required": .array([.string("answer")]),
            "additionalProperties": .bool(false),
        ])
        #expect(throws: StructuredOutputGrammar.SchemaError.self) {
            try StructuredOutputRequest.validate(
                format: .jsonSchema(.init(name: "answer", schema: unsupported, strict: true)),
                tools: nil,
                stop: []
            )
        }
    }

    @Test("Prompt instruction carries the actual schema as valid JSON")
    func schemaInstruction() throws {
        let instruction = try StructuredOutputRequest.instruction(
            for: .jsonSchema(.init(name: "answer", schema: schema, strict: true))
        )
        let schemaStart = try #require(instruction.range(of: "Schema: "))
        let embedded = Data(instruction[schemaStart.upperBound...].utf8)
        #expect(try JSONDecoder().decode(OpenAIJSONValue.self, from: embedded) == schema)
        #expect(instruction.contains("JSON object"))
    }

    @Test("Object mode requests JSON while ordinary text adds no instruction")
    func basicInstructions() throws {
        #expect(try StructuredOutputRequest.instruction(for: .text) == "")
        let instruction = try StructuredOutputRequest.instruction(for: .jsonObject)
        #expect(instruction.contains("JSON object"))
        #expect(!instruction.contains("Schema:"))
    }

    private var formats: [OpenAIResponseFormat] {
        [.jsonObject, .jsonSchema(.init(name: "answer", schema: schema, strict: true))]
    }
}
