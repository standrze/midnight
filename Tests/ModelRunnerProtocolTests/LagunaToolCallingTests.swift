import Foundation
import MLXLMCommon
import ModelRunnerProtocol
import Testing
import Tokenizers

@testable import ModelRunnerCore

@Suite("Laguna tool protocol")
struct LagunaToolCallingTests {
    let tools: [[String: any Sendable]] = [
        [
            "type": "function",
            "function": [
                "name": "inspect",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "text": ["type": "string"], "count": ["type": "integer"],
                        "enabled": ["type": "boolean"], "items": ["type": "array"],
                        "data": ["type": "object"], "empty": ["type": "null"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    ]

    @Test func configuration() {
        let config = LocalModelChatConfiguration.make(
            directory: URL(fileURLWithPath: "/tmp/renamed-checkpoint"), modelType: "laguna")
        #expect(config.toolCallFormat == .glm4)
        #expect(config.reasoningConfig == .thinkTagsWithEnableThinking)
        #expect(config.extraEOSTokens.contains("</assistant>"))
        #expect(!config.extraEOSTokens.contains("</tool_call>"))
        #expect(!config.extraEOSTokens.contains("</think>"))
    }

    @Test func emptyArguments() throws {
        let call = try #require(GLM4ToolCallParser().parse(content: "<tool_call>inspect</tool_call>", tools: tools))
        #expect(call.function.name == "inspect")
        #expect(call.function.arguments.isEmpty)
    }

    @Test func typedArguments() throws {
        let body =
            "inspect<arg_key>text</arg_key><arg_value>  hello\n世界  </arg_value>"
            + "<arg_key>count</arg_key><arg_value>3</arg_value>"
            + "<arg_key>enabled</arg_key><arg_value>true</arg_value>"
            + "<arg_key>items</arg_key><arg_value>[1,2]</arg_value>"
            + "<arg_key>data</arg_key><arg_value>{\"x\":1}</arg_value>"
            + "<arg_key>empty</arg_key><arg_value>null</arg_value>"
        let call = try #require(GLM4ToolCallParser().parse(content: body, tools: tools))
        let data = try JSONEncoder().encode(call.function.arguments)
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["text"] as? String == "  hello\n世界  ")
        #expect(object["count"] as? Int == 3)
        #expect(object["enabled"] as? Bool == true)
        #expect(object["items"] as? [Int] == [1, 2])
        #expect(object["data"] as? [String: Int] == ["x": 1])
        #expect(object["empty"] is NSNull)
    }

    @Test(arguments: [
        "", "inspect<arg_key>x</arg_key>",
        "inspect<arg_key>x</arg_key><arg_value>unfinished", "inspect garbage",
        "inspect<arg_key>x</arg_key><arg_value>1</arg_value>junk",
        "inspect<arg_key>x</arg_key><arg_value>1</arg_value><arg_key>x</arg_key><arg_value>2</arg_value>",
    ])
    func malformed(body: String) {
        #expect(GLM4ToolCallParser().parse(content: body, tools: tools) == nil)
    }

    @Test func streamEveryBoundary() {
        let text =
            "Before<tool_call>inspect</tool_call><tool_call>inspect<arg_key>text</arg_key><arg_value>42</arg_value></tool_call>After"
        for split in 0...text.count {
            let p = ToolCallProcessor(format: .glm4, tools: tools)
            let index = text.index(text.startIndex, offsetBy: split)
            let outputs =
                p.processChunkOutputs(String(text[..<index]))
                + p.processChunkOutputs(String(text[index...])) + p.processEOSOutputs()
            var calls: [ToolCall] = []
            var content = ""
            for output in outputs {
                switch output {
                case .response(let value): content += value
                case .toolCall(let call): calls.append(call)
                case .rejectedToolCall: Issue.record("Rejected valid call at split \(split)")
                }
            }
            #expect(calls.count == 2)
            #expect(content == "BeforeAfter")
            #expect(calls.last?.function.arguments["text"] == .string("42"))
        }
    }

    @Test func unauthorizedAndTruncated() {
        for (body, allowed) in [
            ("<tool_call>other</tool_call>", tools),
            ("<tool_call>inspect</tool_call>", []), ("<tool_call>inspect", tools),
        ] {
            let p = ToolCallProcessor(format: .glm4, tools: allowed)
            let outputs = p.processChunkOutputs(body) + p.processEOSOutputs()
            #expect(
                !outputs.contains {
                    if case .toolCall = $0 {
                        true
                    } else {
                        false
                    }
                })
            #expect(
                outputs.contains {
                    if case .rejectedToolCall = $0 {
                        true
                    } else {
                        false
                    }
                })
        }
    }

    @Test func reasoningHistory() throws {
        let message = OpenAIMessage(role: "assistant", content: "", reasoningContent: "Retain this context.")
        let chat = try LocalModelRunner.chatMessage(message)
        let raw = DefaultMessageGenerator().generate(message: chat)
        #expect(raw["reasoning_content"] as? String == "Retain this context.")
        let context = LocalModelRunner.promptAdditionalContext(
            reasoningEffort: nil, thinkingEnabled: false, preserveThinking: true)
        #expect(context?["enable_thinking"] as? Bool == false)
        #expect(context?["preserve_thinking"] as? Bool == true)
    }

    @Test func forcedChoiceCandidates() throws {
        let declarations = ["inspect", "clock"].map {
            OpenAIToolDefinition(function: .init(name: $0, parameters: .object(["type": .string("object")])))
        }
        let required = try ToolChoicePlan.resolve(choice: .required, tools: declarations)
        #expect(try required.forcedToolCallPrefixes(for: .glm4) == ["<tool_call>inspect", "<tool_call>clock"])
        let named = try ToolChoicePlan.resolve(choice: .function(name: "clock"), tools: declarations)
        #expect(try named.forcedToolCallPrefixes(for: .glm4) == ["<tool_call>clock"])
    }

    @Test func officialTemplateRoundTrip() async throws {
        let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Fixtures")
        let tokenizer = try await AutoTokenizer.from(modelFolder: fixtures.appendingPathComponent("TalkieTokenizer"))
        let template = try String(
            contentsOf: fixtures.appendingPathComponent("LagunaToolProtocol/chat_template.jinja"), encoding: .utf8)
        let call = OpenAIToolCall(id: "call_1", function: .init(name: "inspect", arguments: "{}"))
        let messages = [
            OpenAIMessage(role: "user", content: "Inspect."),
            OpenAIMessage(role: "assistant", content: nil, toolCalls: [call], reasoningContent: "Need the tool."),
            OpenAIMessage(role: "tool", content: "done", toolCallID: "call_1"),
        ]
        let raw = try messages.map { DefaultMessageGenerator().generate(message: try LocalModelRunner.chatMessage($0)) }
        for thinking in [true, false] {
            let tokens = try tokenizer.applyChatTemplate(
                messages: raw, chatTemplate: .literal(template), addGenerationPrompt: true,
                truncation: false, maxLength: nil, tools: tools,
                additionalContext: ["enable_thinking": thinking, "preserve_thinking": true])
            let prompt = tokenizer.decode(tokens: tokens, skipSpecialTokens: false)
            #expect(prompt.contains("<think>Need the tool.</think>") == thinking)
            #expect(prompt.contains("<tool_call>inspect</tool_call>"))
            #expect(prompt.contains("<tool_response>done</tool_response>"))
            #expect(prompt.hasSuffix(thinking ? "<assistant><think>" : "<assistant></think>"))
        }
    }

    @Test func reasoningThenCalls() {
        let text = "Plan first.</think><tool_call>inspect</tool_call>Done."
        for split in 0...text.count {
            var reasoner = ReasoningEventEmitter(config: .thinkTagsWithEnableThinking, primedInside: true)
            let parser = ToolCallProcessor(format: .glm4, tools: tools)
            let index = text.index(text.startIndex, offsetBy: split)
            let segments =
                reasoner.process(String(text[..<index])) + reasoner.process(String(text[index...]))
                + reasoner.finalize()
            var reasoning = ""
            var output: [ToolCallProcessor.Output] = []
            for segment in segments {
                switch segment {
                case .reasoning(let value): reasoning += value
                case .response(let value): output += parser.processChunkOutputs(value)
                }
            }
            output += parser.processEOSOutputs()
            #expect(reasoning == "Plan first.")
            #expect(
                output.filter {
                    if case .toolCall = $0 {
                        true
                    } else {
                        false
                    }
                }.count == 1)
        }
    }
    @Test func nativeRequestWire() throws {
        let request = ChatCompletionRequest(
            model: "renamed", messages: [.init(role: "user", content: "Hi")], stream: true, nativeProtocol: "laguna")
        let data = try JSONEncoder().encode(request)
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["native_protocol"] as? String == "laguna")
        #expect(try JSONDecoder().decode(ChatCompletionRequest.self, from: data) == request)
    }
}
