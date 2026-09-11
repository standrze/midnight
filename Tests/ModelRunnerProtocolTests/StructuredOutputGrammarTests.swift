import Foundation
import ModelRunnerProtocol
import Testing

@Suite("Structured output byte grammar")
struct StructuredOutputGrammarTests {
    private func schema(_ json: String, strict: Bool = false) throws -> StructuredOutputGrammar {
        let value = try JSONDecoder().decode(OpenAIJSONValue.self, from: Data(json.utf8))
        return try StructuredOutputGrammar(responseFormat: .jsonSchema(.init(
            name: "test", schema: value, strict: strict
        )))
    }

    @Test("JSON object accepts arbitrary nested JSON and whitespace")
    func jsonObject() throws {
        let grammar = try StructuredOutputGrammar(responseFormat: .jsonObject)
        for output in [
            "{}", " \n { \"items\" : [null, true, false, -12.5e+2, {}, []] } \r\t",
            #"{"s":"quote\" backslash\\ newline\n slash\/","unicode":"日本語 😀"}"#,
            #"{"s":"\u0000\ud83d\ude00\uD834\uDD1E\ud7ff\ue000"}"#
        ] {
            #expect(grammar.validate(output), "Rejected valid JSON: \(output)")
            var state = grammar.initialState
            for byte in output.utf8 {
                state = try #require(grammar.advancing(state, byte: byte))
            }
            #expect(grammar.isComplete(state))
        }
    }

    @Test("Malformed JSON and non-object roots are rejected")
    func invalidJSON() throws {
        let grammar = try StructuredOutputGrammar(responseFormat: .jsonObject)
        for output in [
            "", "[]", "null", "42", "\"str\"", "{} trailing", "{}{}", "{\"a\":}",
            "{\"a\":01}", "{\"a\":-01}", "{\"a\":+1}", "{\"a\":1.}", "{\"a\":.1}",
            "{\"a\":1e}", "{\"a\":NaN}", "{\"a\":Infinity}", "{\"a\":[1,]}",
            "{\"a\":true,}", "{a:1}", "{'a':1}", "{\"a\":\"line\nfeed\"}",
            #"{"a":"\x20"}"#, #"{"a":"\ud800"}"#, #"{"a":"\udc00"}"#,
            #"{"a":"\ud800\u1234"}"#, #"{"a":"\u00z0"}"#
        ] {
            #expect(!grammar.validate(output), "Accepted invalid JSON: \(output)")
        }
    }

    @Test("Malformed UTF8 fails before a token is accepted")
    func invalidUTF8() throws {
        let grammar = try StructuredOutputGrammar(responseFormat: .jsonObject)
        let start = try #require(grammar.advancing(grammar.initialState, bytes: Array(#"{"a":""#.utf8)))
        for bytes: [UInt8] in [[0x80], [0xc0], [0xc1], [0xf5], [0xe0, 0x80], [0xed, 0xa0], [0xf0, 0x80], [0xf4, 0x90]] {
            #expect(grammar.advancing(start, bytes: bytes) == nil)
        }
        let partial = try #require(grammar.advancing(start, bytes: [0xf0, 0x9f]))
        #expect(!grammar.isComplete(partial))
        #expect(grammar.advancing(partial, byte: 34) == nil)
        let completedScalar = try #require(grammar.advancing(partial, bytes: [0x98, 0x80, 34, 125]))
        #expect(grammar.isComplete(completedScalar))
    }

    @Test("Schema properties, enum, array items, and nullable union constrain every prefix")
    func schemaConstraints() throws {
        let grammar = try schema(#"""
        {
          "type":"object", "additionalProperties":false,
          "properties": {
            "answer":{"type":"string","enum":["yes","no"]},
            "details":{"type":["object","null"],"additionalProperties":false,
                       "properties":{"count":{"type":"integer"}},"required":["count"]},
            "scores":{"type":"array","items":{"type":"number"}}
          },
          "required":["answer","details","scores"]
        }
        """#, strict: true)
        #expect(grammar.validate(#"{"answer":"yes","details":{"count":2},"scores":[1,-2.4e3]}"#))
        #expect(grammar.validate(#"{"answer":"no","details":null,"scores":[]}"#))
        for output in [
            #"{"answer":"maybe","details":null,"scores":[]}"#,
            #"{"answer":"yes","details":{},"scores":[]}"#,
            #"{"answer":"yes","details":{"count":2.3},"scores":[]}"#,
            #"{"answer":"yes","details":null,"scores":["1"]}"#,
            #"{"answer":"yes","scores":[]}"#,
            #"{"answer":"yes","details":null,"scores":[],"extra":1}"#
        ] { #expect(!grammar.validate(output)) }
        let prefix = try #require(grammar.advancing(grammar.initialState, bytes: Array(#"{"answer":""#.utf8)))
        #expect(grammar.advancing(prefix, byte: Character("y").asciiValue!) != nil)
        #expect(grammar.advancing(prefix, byte: Character("n").asciiValue!) != nil)
        #expect(grammar.advancing(prefix, byte: Character("m").asciiValue!) == nil)
    }

    @Test("Optional properties may be omitted in non-strict schemas")
    func optionalProperties() throws {
        let grammar = try schema(#"""
        {"type":"object","additionalProperties":false,
         "properties":{"a":{"type":"boolean"},"b":{"type":"integer"},"c":{"type":"string"}},
         "required":["b"]}
        """#)
        for output in [#"{"b":1}"#, #"{"a":true,"b":1}"#, #"{"b":1,"c":"x"}"#, #"{"a":false,"b":1,"c":"x"}"#] {
            #expect(grammar.validate(output))
        }
        #expect(!grammar.validate("{}"))
        #expect(!grammar.validate(#"{"a":true,"c":"x"}"#))
        #expect(!grammar.validate(#"{"b":1,"a":true}"#), "Property order is deterministic")
    }

    @Test("Local definitions, nested anyOf, and const work together")
    func definitionsAndAlternatives() throws {
        let grammar = try schema(#"""
        {"type":"object","additionalProperties":false,
         "$defs":{"value":{"anyOf":[{"type":"integer"},{"const":"unknown"}]}},
         "properties":{"value":{"$ref":"#/$defs/value"}},"required":["value"]}
        """#, strict: true)
        #expect(grammar.validate(#"{"value":12}"#))
        #expect(grammar.validate(#"{"value":"unknown"}"#))
        #expect(!grammar.validate(#"{"value":"other"}"#))
        #expect(!grammar.validate(#"{"value":null}"#))
    }

    @Test("Object-valued enums enforce the complete literal")
    func objectEnum() throws {
        let grammar = try schema(#"""
        {"type":"object","additionalProperties":false,
         "properties":{"a":{"type":"integer"}},"required":["a"],
         "enum":[{"a":1},{"a":2}]}
        """#, strict: true)
        #expect(grammar.validate(#"{"a":1}"#))
        #expect(grammar.validate(#"{"a":2}"#))
        #expect(!grammar.validate(#"{"a":3}"#))
    }

    @Test("Completion is recognized only after required structure closes")
    func completion() throws {
        let grammar = try schema(#"""
        {"type":"object","additionalProperties":false,
         "properties":{"n":{"type":"number"}},"required":["n"]}
        """#, strict: true)
        var state = grammar.initialState
        for byte in #"{"n":-1.25e+10"#.utf8 {
            #expect(!grammar.isComplete(state))
            state = try #require(grammar.advancing(state, byte: byte))
        }
        #expect(!grammar.isComplete(state))
        state = try #require(grammar.advancing(state, byte: 125))
        #expect(grammar.isComplete(state))
        #expect(grammar.advancing(state, byte: 44) == nil)
        #expect(grammar.advancing(state, byte: 10) != nil)
    }

    @Test("Malformed and unsupported schemas fail at construction")
    func invalidSchemas() throws {
        for json in [
            "false", "[]", #"{"type":"string"}"#,
            #"{"type":"object"}"#,
            #"{"type":"object","additionalProperties":true}"#,
            #"{"type":"object","additionalProperties":false,"properties":[]}"#,
            #"{"type":"object","additionalProperties":false,"required":["missing"]}"#,
            #"{"type":"object","additionalProperties":false,"properties":{"a":{"type":"string","pattern":"x"}}}"#,
            #"{"type":"object","additionalProperties":false,"properties":{"a":{"type":"array"}}}"#,
            #"{"type":"object","additionalProperties":false,"properties":{"a":{"type":"integer","minimum":0}}}"#,
            #"{"type":"object","additionalProperties":false,"properties":{"a":{"type":"unknown"}}}"#,
            #"{"type":"object","additionalProperties":false,"properties":{"a":{"type":["string","string"]}}}"#,
            #"{"type":"object","additionalProperties":false,"properties":{"a":{"type":"integer","enum":["wrong"]}}}"#,
            #"{"type":"object","additionalProperties":false,"properties":{"a":{"anyOf":[]}}}"#,
            #"{"$ref":"https://example.com/schema"}"#,
            ##"{"$ref":"#/$defs/missing"}"##,
            ##"{"$defs":{"loop":{"$ref":"#/$defs/loop"}},"$ref":"#/$defs/loop"}"##,
            #"{"anyOf":[{"type":"object","additionalProperties":false}]}"#
        ] {
            #expect(throws: StructuredOutputGrammar.SchemaError.self) { try schema(json) }
        }
    }

    @Test("Strict schemas require every property at all nesting levels")
    func strictRequired() throws {
        for json in [
            #"{"type":"object","additionalProperties":false,"properties":{"a":{"type":"string"}}}"#,
            #"{"type":"object","additionalProperties":false,"properties":{"a":{"type":"object","additionalProperties":false,"properties":{"b":{"type":"integer"}}}},"required":["a"]}"#
        ] {
            #expect(throws: StructuredOutputGrammar.SchemaError.self) { try schema(json, strict: true) }
        }
    }
    @Test("String continuation states are canonical for token-mask caching")
    func equivalentStates() throws {
        let grammar = try StructuredOutputGrammar(responseFormat: .jsonObject)
        let first = try #require(grammar.advancing(grammar.initialState, bytes: Array(#"{"a":"first"#.utf8)))
        let second = try #require(grammar.advancing(grammar.initialState, bytes: Array(#"{"a":"second"#.utf8)))
        #expect(first == second)
        #expect(Set([first, second]).count == 1)
    }

    @Test("Long enum literals fit the bounded parser and total enum bytes are limited")
    func longEnums() throws {
        let text = String(repeating: "x", count: 5000)
        let format: OpenAIJSONValue = .object([
            "type": .string("object"), "additionalProperties": .bool(false),
            "properties": .object(["a": .object(["enum": .array([.string(text)])])]),
            "required": .array([.string("a")])
        ])
        let grammar = try StructuredOutputGrammar(responseFormat: .jsonSchema(.init(name: "long", schema: format, strict: true)))
        let encoder = JSONEncoder()
        let output = String(decoding: try encoder.encode(OpenAIJSONValue.object(["a": .string(text)])), as: UTF8.self)
        #expect(grammar.validate(output))
        let tooLarge: OpenAIJSONValue = .object(["enum": .array([.object(["a": .string(String(repeating: "x", count: 70_000))])])])
        #expect(throws: StructuredOutputGrammar.SchemaError.self) {
            try StructuredOutputGrammar(responseFormat: .jsonSchema(.init(name: "huge", schema: tooLarge)))
        }
    }

}
