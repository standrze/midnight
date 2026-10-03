import Foundation
import ModelRunnerProtocol
import Testing

@Suite("Responses request normalization")
struct ResponsesRequestTests {
    @Test("String input becomes a canonical user message and defaults remain explicit")
    func stringInput() throws {
        let request = try decode(#"{"model":"local","input":"Hello 世界"}"#)
        #expect(request.inputMessages == [.init(role: "user", content: "Hello 世界")])
        #expect(request.store)
        #expect(!request.stream)
        #expect(request.parallelToolCalls)
        #expect(request.instructions == nil)
        #expect(request.responseFormat == nil)
        let item = try object(#require(request.inputItems.first))
        #expect(item["type"] == .string("message"))
        #expect(item["role"] == .string("user"))
        #expect(item["status"] == .string("completed"))
        guard case .string(let id) = item["id"] else {
            Issue.record("Missing canonical ID")
            return
        }
        #expect(id.hasPrefix("msg_"))
        #expect(item["content"] == .array([.object(["type": .string("input_text"), "text": .string("Hello 世界")])]))
        let fields = request.responseFields()
        #expect(fields["text"] == .object(["format": .object(["type": .string("text")])]))
        #expect(fields["tool_choice"] == .string("auto"))
        #expect(fields["tools"] == .array([]))
        #expect(fields["metadata"] == .object([:]))
        #expect(fields["temperature"] == .null)
    }

    @Test("Instructions stay separate from the input transcript")
    func separatedInstructions() throws {
        let request = try decode(
            #"{"model":"local","instructions":"Be brief","previous_response_id":"resp_prior","input":"Next","store":false,"stream":true,"parallel_tool_calls":false}"#
        )
        #expect(request.instructions == "Be brief")
        #expect(request.previousResponseID == "resp_prior")
        #expect(request.inputMessages == [.init(role: "user", content: "Next")])
        #expect(!request.store)
        #expect(request.stream)
        #expect(!request.parallelToolCalls)
        #expect(request.responseFields()["instructions"] == .string("Be brief"))
        #expect(request.responseFields()["previous_response_id"] == .string("resp_prior"))
    }

    @Test(
        "Previous-response and instructions-only requests may omit input",
        arguments: [
            #"{"model":"local","previous_response_id":"resp_prior"}"#,
            #"{"model":"local","instructions":"Say hello","input":null}"#,
        ])
    func noInput(_ body: String) throws {
        let request = try decode(body)
        #expect(request.inputMessages.isEmpty)
        #expect(request.inputItems.isEmpty)
    }

    @Test("Typed and easy messages preserve role, order, IDs, and text part boundaries")
    func messages() throws {
        let request = try decode(
            #"""
            {"model":"local","input":[
              {"role":"system","content":"Policy"},
              {"type":"message","role":"developer","content":[{"type":"input_text","text":"Instruction"}]},
              {"role":"user","content":[{"type":"input_text","text":"Hello "},{"type":"input_text","text":"world"}]},
              {"id":"msg_saved","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Answer","annotations":[],"logprobs":[]}]}
            ]}
            """#)
        #expect(request.inputMessages.map(\.role) == ["system", "developer", "user", "assistant"])
        #expect(request.inputMessages.map(\.content) == ["Policy", "Instruction", "Hello world", "Answer"])
        #expect(try object(request.inputItems[3])["id"] == .string("msg_saved"))
        guard case .array(let parts) = try object(request.inputItems[2])["content"] else {
            Issue.record("Expected canonical content parts")
            return
        }
        #expect(parts.count == 2)
    }

    @Test("OpenAI Python parsed output can be replayed without client-side convenience fields")
    func parsedSDKReplay() throws {
        let request = try decode(
            #"""
            {"model":"local","input":[
              {"id":"msg_parsed","type":"message","role":"assistant","status":"completed","phase":null,
               "content":[{"type":"output_text","text":"{\"answer\":\"yes\"}","annotations":[],"logprobs":null,"parsed":{"answer":"yes"}}]},
              {"role":"user","content":"Explain that answer"}
            ]}
            """#)
        #expect(request.inputMessages[0] == .init(role: "assistant", content: #"{"answer":"yes"}"#))
        let message = try object(request.inputItems[0])
        #expect(message["id"] == .string("msg_parsed"))
        #expect(message["phase"] == nil)
        guard case .array(let parts) = message["content"] else {
            Issue.record("Expected content parts")
            return
        }
        #expect(try object(parts[0])["parsed"] == nil)
        #expect(try object(parts[0])["text"] == .string(#"{"answer":"yes"}"#))
    }

    @Test("Multiple calls become one assistant turn and results preserve their call IDs")
    func functionRoundTrip() throws {
        let request = try decode(
            #"""
            {"model":"local","input":[
              {"role":"user","content":"Check both places"},
              {"id":"fc_one","type":"function_call","call_id":"call_one","name":"weather","arguments":"{\"city\":\"Paris\"}"},
              {"id":"fc_two","type":"function_call","call_id":"call_two","name":"weather","arguments":"{\"city\":\"Rome\"}","status":"completed"},
              {"type":"function_call_output","call_id":"call_one","output":"sunny"},
              {"type":"function_call_output","call_id":"call_two","output":[{"type":"input_text","text":"partly "},{"type":"input_text","text":"cloudy"}]},
              {"role":"user","content":"Summarize"}
            ]}
            """#)
        #expect(request.inputItems.count == 6)
        #expect(request.inputMessages.count == 5)
        let calls = try #require(request.inputMessages[1].toolCalls)
        #expect(calls.map(\.id) == ["call_one", "call_two"])
        #expect(calls[0].function.arguments == #"{"city":"Paris"}"#)
        #expect(request.inputMessages[2] == .init(role: "tool", content: "sunny", toolCallID: "call_one"))
        #expect(request.inputMessages[3] == .init(role: "tool", content: "partly cloudy", toolCallID: "call_two"))
        #expect(try object(request.inputItems[1])["id"] == .string("fc_one"))
        #expect(try object(request.inputItems[1])["call_id"] == .string("call_one"))
    }

    @Test("Tool results may refer to a call supplied by previous_response_id")
    func previousCallResult() throws {
        let request = try decode(
            #"{"model":"local","previous_response_id":"resp_call","input":[{"type":"function_call_output","call_id":"call_previous","output":"done"}]}"#
        )
        #expect(request.inputMessages == [.init(role: "tool", content: "done", toolCallID: "call_previous")])
    }

    @Test(
        "SDK function call dumps normalize optional default fields",
        arguments: [
            #""async_":null"#, #""async":null"#, #""async":false"#,
        ])
    func functionSDKReplay(_ asyncField: String) throws {
        let call =
            #"{"id":"fc_saved","type":"function_call","call_id":"call_saved","name":"lookup","arguments":"{}","status":"completed"}"#
        let result = #"{"type":"function_call_output","call_id":"call_saved","output":"found"}"#
        let expected = try decode("{\"model\":\"local\",\"input\":[\(call),\(result)]}")
        let withDefaults = String(call.dropLast()) + ",\(asyncField),\"caller\":null,\"namespace\":null}"
        let replay = try decode("{\"model\":\"local\",\"input\":[\(withDefaults),\(result)]}")
        #expect(replay.inputMessages == expected.inputMessages)
        #expect(replay.inputItems[0] == expected.inputItems[0])
    }

    @Test("Flattened function definitions and named choices map to chat controls")
    func functionDefinitions() throws {
        let request = try decode(
            #"""
            {"model":"local","input":"Hello","tools":[
              {"type":"function","name":"lookup","description":"Lookup a record","parameters":{"type":"object","properties":{"id":{"type":"string"}}},"strict":false}
            ],"tool_choice":{"type":"function","name":"lookup"}}
            """#)
        #expect(request.toolChoice == .function(name: "lookup"))
        let tool = try #require(request.tools?.first)
        #expect(tool.function.name == "lookup")
        #expect(tool.function.description == "Lookup a record")
        guard case .array(let echoed) = request.responseFields()["tools"] else {
            Issue.record("Expected echoed tools")
            return
        }
        let definition = try object(#require(echoed.first))
        #expect(definition["name"] == .string("lookup"))
        #expect(definition["strict"] == .bool(false))
        #expect(definition["parameters"] == tool.function.parameters)
        #expect(definition["function"] == nil)
    }

    @Test(
        "Simple tool choice modes preserve none as an explicit choice",
        arguments: [
            ("auto", OpenAIToolChoice.auto), ("none", OpenAIToolChoice.none), ("required", OpenAIToolChoice.required),
        ])
    func toolChoices(choice: String, expected: OpenAIToolChoice) throws {
        let request = try decode("{\"model\":\"local\",\"tool_choice\":\"\(choice)\"}")
        #expect(request.toolChoice == expected)
        #expect(request.responseFields()["tool_choice"] == .string(choice))
    }

    @Test("Responses text.format schema is flattened on both input and response")
    func structuredFormat() throws {
        let request = try decode(
            #"""
            {"model":"local","input":"Return JSON","text":{"format":{
              "type":"json_schema","name":"answer","description":"The answer","strict":true,
              "schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}
            }}}
            """#)
        guard case .jsonSchema(let schema) = request.responseFormat else {
            Issue.record("Expected schema")
            return
        }
        #expect(schema.name == "answer")
        #expect(schema.description == "The answer")
        #expect(schema.strict == true)
        let text = try object(#require(request.responseFields()["text"]))
        let format = try object(#require(text["format"]))
        #expect(format["type"] == .string("json_schema"))
        #expect(format["schema"] == schema.schema)
        #expect(format["json_schema"] == nil)
    }

    @Test(
        "Simple response formats normalize correctly",
        arguments: [
            ("text", OpenAIResponseFormat.text), ("json_object", OpenAIResponseFormat.jsonObject),
        ])
    func simpleFormats(type: String, expected: OpenAIResponseFormat) throws {
        let request = try decode("{\"model\":\"local\",\"text\":{\"format\":{\"type\":\"\(type)\"}}}")
        #expect(request.responseFormat == expected)
    }

    @Test("Sampling, reasoning, metadata, and passive fields survive normalization")
    func acceptedConfiguration() throws {
        let request = try decode(
            #"""
            {"model":"local","input":"Hi","max_output_tokens":17,"temperature":0.25,"top_p":0.9,
             "reasoning":{"effort":"high","summary":null},"metadata":{"job":"42"},
             "user":"person","safety_identifier":"safe-id","prompt_cache_key":"prefix",
             "background":false,"truncation":"disabled","service_tier":"auto","include":[],
             "top_logprobs":0,"stream":true,"stream_options":{"include_obfuscation":false}}
            """#)
        #expect(request.maxOutputTokens == 17)
        #expect(request.temperature == 0.25)
        #expect(request.topP == 0.9)
        #expect(request.reasoningEffort == .high)
        #expect(request.metadata == ["job": "42"])
        #expect(request.responseFields()["user"] == .string("person"))
        #expect(request.responseFields()["safety_identifier"] == .string("safe-id"))
        #expect(request.responseFields()["prompt_cache_key"] == .string("prefix"))
        #expect(request.responseFields()["service_tier"] == .string("auto"))
    }

    @Test(
        "Unsupported and malformed controls report their exact parameter",
        arguments: [
            (#""background":true"#, "background"),
            (#""truncation":"auto""#, "truncation"),
            (#""service_tier":"priority""#, "service_tier"),
            (#""include":["reasoning.encrypted_content"]"#, "include"),
            (#""top_logprobs":1"#, "top_logprobs"),
            (#""stream_options":{"include_obfuscation":true}"#, "stream_options.include_obfuscation"),
            (#""stream_options":{"include_usage":true}"#, "stream_options.include_usage"),
            (#""reasoning":{"summary":"auto"}"#, "reasoning.summary"),
            (#""reasoning":{"effort":"xhigh"}"#, "reasoning.effort"),
            (#""text":{"verbosity":"low"}"#, "text.verbosity"),
            (#""response_format":{"type":"json_object"}"#, "response_format"),
            (#""conversation":"conv_1""#, "conversation"),
            (#""max_tool_calls":2"#, "max_tool_calls"),
            (#""tools":[{"type":"web_search"}]"#, "tools[0].type"),
            (#""tools":[{"type":"function","name":"run","strict":true}]"#, "tools[0].strict"),
            (#""tools":[{"type":"function","function":{"name":"run"}}]"#, "tools[0].function"),
            (#""tools":[{"type":"function","name":"bad name"}]"#, "tools[0].name"),
            (#""tool_choice":{"type":"function","function":{"name":"run"}}"#, "tool_choice.function"),
            (#""text":{"format":{"type":"json_schema","json_schema":{}}}"#, "text.format.json_schema"),
            (#""text":{"format":{"type":"json_schema","name":"answer","schema":true}}"#, "text.format.schema"),
            (#""text":{"format":{"type":"text","schema":{}}}"#, "text.format.schema"),
            (#""max_output_tokens":0"#, "max_output_tokens"),
            (#""temperature":2.1"#, "temperature"),
            (#""top_p":-0.1"#, "top_p"),
            (#""top_p":0"#, "top_p"),
            (#""stream":"yes""#, "stream"),
            (#""metadata":{"job":42}"#, "metadata.job"),
            (#""previous_response_id":"""#, "previous_response_id"),
        ])
    func rejectedControls(field: String, param: String) throws {
        try expectError("{\"model\":\"local\",\"input\":\"Hi\",\(field)}", param: param)
    }

    @Test(
        "Stream options require an active stream",
        arguments: [
            "", #", "stream":false"#, #", "stream":null"#,
        ])
    func streamOptionsRequireStreaming(_ streamField: String) throws {
        try expectError(
            "{\"model\":\"local\",\"stream_options\":{\"include_obfuscation\":false}\(streamField)}",
            param: "stream_options")
    }

    @Test(
        "Unsupported input is rejected instead of discarded",
        arguments: [
            (#"{"type":"reasoning","summary":[]}"#, "input[0].type"),
            (#"{"type":"item_reference","id":"msg_old"}"#, "input[0].type"),
            (#"{"type":"custom_tool_call","name":"run"}"#, "input[0].type"),
            (#"{"role":"tool","content":"result"}"#, "input[0].role"),
            (
                #"{"role":"user","content":[{"type":"input_image","image_url":"https://example.com/image.png"}]}"#,
                "input[0].content[0].type"
            ),
            (#"{"role":"user","content":[{"type":"input_file","file_id":"file_1"}]}"#, "input[0].content[0].type"),
            (#"{"role":"user","content":[{"type":"input_audio","data":"abc"}]}"#, "input[0].content[0].type"),
            (#"{"role":"user","content":[{"type":"output_text","text":"wrong role"}]}"#, "input[0].content[0].type"),
            (#"{"role":"assistant","phase":"analysis","content":"thoughts"}"#, "input[0].phase"),
            (
                #"{"role":"assistant","content":[{"type":"output_text","text":"cited","annotations":[{}]}]}"#,
                "input[0].content[0].annotations"
            ),
            (#"{"type":"function_call","name":"lookup","arguments":"{}"}"#, "input[0].call_id"),
            (
                #"{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}","async":true}"#,
                "input[0].async"
            ),
            (
                #"{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}","caller":{"type":"direct"}}"#,
                "input[0].caller"
            ),
            (
                #"{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}","namespace":"remote"}"#,
                "input[0].namespace"
            ),
            (#"{"type":"function_call_output","call_id":"call_1","output":{"result":true}}"#, "input[0].output"),
            (#"{"role":"user","content":"Hello","status":"failed"}"#, "input[0].status"),
        ])
    func rejectedInputs(item: String, param: String) throws {
        try expectError("{\"model\":\"local\",\"input\":[\(item)]}", param: param)
    }

    @Test("Repeated item and function call identifiers fail explicitly")
    func repeatedIdentifiers() throws {
        try expectError(
            #"{"model":"local","input":[{"id":"msg_x","role":"user","content":"a"},{"id":"msg_x","role":"user","content":"b"}]}"#,
            param: "input[1].id")
        try expectError(
            #"{"model":"local","input":[{"type":"function_call","call_id":"call_x","name":"f","arguments":"{}"},{"type":"function_call","call_id":"call_x","name":"g","arguments":"{}"}]}"#,
            param: "input[1].call_id")
    }

    @Test("Input, tools, and metadata collections have explicit bounds")
    func bounds() throws {
        let inputs = Array(repeating: #"{"role":"user","content":"hi"}"#, count: 4097).joined(separator: ",")
        try expectError("{\"model\":\"local\",\"input\":[\(inputs)]}", param: "input")
        let tools = Array(repeating: #"{"type":"function","name":"f"}"#, count: 129).joined(separator: ",")
        try expectError("{\"model\":\"local\",\"tools\":[\(tools)]}", param: "tools")
        let metadata = (0..<17).map { "\"key\($0)\":\"value\"" }.joined(separator: ",")
        try expectError("{\"model\":\"local\",\"metadata\":{\(metadata)}}", param: "metadata")
    }

    private func decode(_ body: String) throws -> ResponsesRequest {
        try JSONDecoder().decode(ResponsesRequest.self, from: Data(body.utf8))
    }

    private func object(_ value: OpenAIJSONValue) throws -> [String: OpenAIJSONValue] {
        guard case .object(let object) = value else {
            throw ResponsesRequestError(message: "Expected an object in test", param: "test")
        }
        return object
    }

    private func expectError(_ body: String, param: String) throws {
        do {
            _ = try decode(body)
            Issue.record("Expected an error for \(param)")
        } catch let error as ResponsesRequestError {
            #expect(error.param == param)
            #expect(!error.message.isEmpty)
        }
    }
}
