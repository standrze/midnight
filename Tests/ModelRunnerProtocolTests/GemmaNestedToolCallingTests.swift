import MLXLMCommon
import Testing

@Suite("Gemma nested tool values")
struct GemmaNestedToolCallingTests {
    private let parser = GemmaFunctionParser(
        startTag: "<|tool_call>", endTag: "<tool_call|>", escapeMarker: #"<|"|>"#)

    @Test(
        "Retained artifact filter arrives as an object with typed array and boolean values", arguments: [false, true])
    func retainedFilter(withSchema: Bool) throws {
        let content =
            #"<|tool_call>call:collect_artifacts{filters:{extensions:[<|"|>swift<|"|>,<|"|>json<|"|>],include_hidden:false},limit:2,scope:<|"|>staging<|"|>}<tool_call|>"#
        let tools: [[String: any Sendable]]? =
            withSchema
            ? [
                [
                    "function": [
                        "name": "collect_artifacts",
                        "parameters": [
                            "properties": [
                                "filters": ["type": "object"], "limit": ["type": "integer"],
                                "scope": ["type": "string"],
                            ] as [String: any Sendable]
                        ] as [String: any Sendable],
                    ] as [String: any Sendable]
                ]
            ] : nil
        let call = try #require(parser.parse(content: content, tools: tools))
        #expect(
            call.function.arguments["filters"]
                == .object([
                    "extensions": .array([.string("swift"), .string("json")]), "include_hidden": .bool(false),
                ]))
        #expect(call.function.arguments["scope"] == .string("staging"))
    }

    @Test("Nested marker strings preserve punctuation, quotes, backslashes and empty values")
    func nestedStrings() throws {
        let content =
            #"<|tool_call>call:inspect{data:{items:[{text:<|"|>a,b:{c} "quoted" \path<|"|>},{text:<|"|><|"|>}],nothing:null,count:1}}<tool_call|>"#
        let call = try #require(parser.parse(content: content, tools: nil))
        #expect(
            call.function.arguments["data"]
                == .object([
                    "items": .array([
                        .object(["text": .string(#"a,b:{c} "quoted" \path"#)]),
                        .object(["text": .string("")]),
                    ]), "nothing": .null, "count": .int(1),
                ]))
    }

    @Test("Bare values are not invented and whole escaped object-shaped strings remain strings")
    func stringsAndInvalidValues() throws {
        let call = try #require(
            parser.parse(
                content: #"<|tool_call>call:inspect{bad:{value:unquoted},note:<|"|>{value:true}<|"|>}<tool_call|>"#,
                tools: nil))
        #expect(call.function.arguments["bad"] == .string("{value:unquoted}"))
        #expect(call.function.arguments["note"] == .string("{value:true}"))
    }

    @Test("Malformed marker spans cannot become a valid structured object")
    func malformedNestedStrings() {
        for content in [
            #"<|tool_call>call:inspect{data:{x:<|"|>missing}}<tool_call|>"#,
            #"<|tool_call>call:inspect{data:{x:<|"|>value<|"|>garbage}}<tool_call|>"#,
        ] {
            let value = parser.parse(content: content, tools: nil)?.function.arguments["data"]
            if case .object = value { Issue.record("Malformed marker span became an object") }
        }
    }
}
