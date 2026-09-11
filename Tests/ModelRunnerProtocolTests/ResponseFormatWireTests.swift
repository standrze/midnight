import Foundation
import ModelRunnerProtocol
import Testing

@Suite("OpenAI response format wire controls")
struct ResponseFormatWireTests {
    @Test("Omitted and null response format preserve default text", arguments: [
        "", #", "response_format": null"#,
    ])
    func defaultFormat(_ optionalField: String) throws {
        let request = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data("""
                {"model":"local-model","messages":[]\(optionalField)}
                """.utf8)
        )
        #expect(request.responseFormat == nil)
        let object = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any]
        )
        #expect(object["response_format"] == nil)
    }

    @Test("Text and JSON object modes use the standard discriminator", arguments: [
        ("text", OpenAIResponseFormat.text),
        ("json_object", OpenAIResponseFormat.jsonObject),
    ])
    func basicFormats(type: String, expected: OpenAIResponseFormat) throws {
        let request = try decode(format: #"{"type":"\#(type)"}"#)
        #expect(request.responseFormat == expected)
        let data = try JSONEncoder().encode(request)
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let format = try #require(object["response_format"] as? [String: Any])
        #expect(format.count == 1)
        #expect(format["type"] as? String == type)
        #expect(object["responseFormat"] == nil)
        #expect(try JSONDecoder().decode(ChatCompletionRequest.self, from: data) == request)
    }

    @Test("JSON schema decodes recursive schema values and metadata")
    func schemaFormat() throws {
        let request = try decode(format: #"""
            {"type":"json_schema","json_schema":{
              "name":"answer_v1-2","description":"A structured answer","strict":true,
              "schema":{
                "type":"object",
                "properties":{
                  "answer":{"type":"string"},
                  "confidence":{"type":"number","minimum":0.25},
                  "count":{"type":"integer","enum":[1,2]},
                  "extra":{"type":["string","null"],"default":null}
                },
                "required":["answer","confidence","count","extra"],
                "additionalProperties":false
              }
            }}
            """#)
        guard case .jsonSchema(let format) = request.responseFormat else {
            Issue.record("Expected a JSON schema format")
            return
        }
        #expect(format.name == "answer_v1-2")
        #expect(format.description == "A structured answer")
        #expect(format.strict == true)
        guard case .object(let schema) = format.schema,
            case .object(let properties) = schema["properties"],
            case .object(let confidence) = properties["confidence"],
            case .object(let extra) = properties["extra"]
        else {
            Issue.record("Expected nested JSON schema objects")
            return
        }
        #expect(confidence["minimum"] == .number(0.25))
        #expect(extra["default"] == .null)
        #expect(schema["additionalProperties"] == .bool(false))
        #expect(try JSONDecoder().decode(
            ChatCompletionRequest.self, from: JSONEncoder().encode(request)) == request)
    }

    @Test("Schema initializer encodes the OpenAI wrapper and false strictness")
    func encodesSchemaFormat() throws {
        let request = ChatCompletionRequest(
            model: "local-model",
            messages: [.init(role: "user", content: "Return an answer.")],
            stream: false,
            responseFormat: .jsonSchema(.init(
                name: "answer",
                schema: .object(["type": .string("object")]),
                strict: false
            ))
        )
        let data = try JSONEncoder().encode(request)
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let format = try #require(object["response_format"] as? [String: Any])
        let wrapper = try #require(format["json_schema"] as? [String: Any])
        #expect(format["type"] as? String == "json_schema")
        #expect(wrapper["name"] as? String == "answer")
        #expect(wrapper["strict"] as? Bool == false)
        #expect(wrapper["description"] == nil)
        #expect((wrapper["schema"] as? [String: String]) == ["type": "object"])
        #expect(try JSONDecoder().decode(ChatCompletionRequest.self, from: data) == request)
    }

    @Test("Optional schema metadata accepts omitted or null values", arguments: [
        "", #", "description":null,"strict":null"#,
    ])
    func optionalSchemaMetadata(_ fields: String) throws {
        let request = try decode(format: """
            {"type":"json_schema","json_schema":{"name":"a","schema":{}\(fields)}}
            """)
        #expect(request.responseFormat == .jsonSchema(.init(name: "a", schema: .object([:]))))
    }

    @Test("Schema name accepts the maximum 64 ASCII characters")
    func maximumName() throws {
        let name = String(repeating: "A", count: 64)
        let request = try decode(format: """
            {"type":"json_schema","json_schema":{"name":"\(name)","schema":{}}}
            """)
        #expect(request.responseFormat == .jsonSchema(.init(name: name, schema: .object([:]))))
    }

    @Test("Malformed format controls fail decoding", arguments: [
        #""json_object""#,
        #"[]"#,
        #"{}"#,
        #"{"type":null}"#,
        #"{"type":1}"#,
        #"{"type":"JSON"}"#,
        #"{"type":"text","unknown":true}"#,
        #"{"type":"text","json_schema":{}}"#,
        #"{"type":"json_object","schema":{}}"#,
        #"{"type":"json_schema"}"#,
        #"{"type":"json_schema","json_schema":null}"#,
        #"{"type":"json_schema","json_schema":[]}"#,
        #"{"type":"json_schema","json_schema":{"schema":{}}}"#,
        #"{"type":"json_schema","json_schema":{"name":"answer"}}"#,
        #"{"type":"json_schema","json_schema":{"name":1,"schema":{}}}"#,
        #"{"type":"json_schema","json_schema":{"name":"answer","schema":{},"unknown":true}}"#,
        #"{"type":"json_schema","unknown":true,"json_schema":{"name":"answer","schema":{}}}"#,
        #"{"type":"json_schema","json_schema":{"name":"answer","schema":{},"strict":"true"}}"#,
        #"{"type":"json_schema","json_schema":{"name":"answer","schema":{},"strict":1}}"#,
        #"{"type":"json_schema","json_schema":{"name":"answer","schema":{},"description":7}}"#,
    ])
    func invalidFormat(_ format: String) {
        #expect(throws: DecodingError.self) {
            try decode(format: format)
        }
    }

    @Test("Invalid schema names fail decoding", arguments: [
        "", "answer with spaces", "answer.json", "é", "💡", String(repeating: "a", count: 65),
    ])
    func invalidSchemaName(_ name: String) throws {
        let encodedName = String(decoding: try JSONEncoder().encode(name), as: UTF8.self)
        #expect(throws: DecodingError.self) {
            try decode(format: """
                {"type":"json_schema","json_schema":{"name":\(encodedName),"schema":{}}}
                """)
        }
    }

    private func decode(format: String) throws -> ChatCompletionRequest {
        try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data("""
                {"model":"local-model","messages":[],"response_format":\(format)}
                """.utf8)
        )
    }
}
