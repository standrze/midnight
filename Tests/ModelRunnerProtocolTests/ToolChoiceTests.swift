import Foundation
import MLX
import MLXLMCommon
import ModelRunnerProtocol
import Testing

@testable import ModelRunnerCore

@Suite("OpenAI tool choice")
struct ToolChoiceTests {
    private let submit = OpenAIToolDefinition(
        function: .init(
            name: "submit_assessment",
            parameters: .object([
                "type": .string("object"),
                "properties": .object([
                    "verdict": .object(["type": .string("string")])
                ]),
                "required": .array([.string("verdict")]),
            ])
        )
    )
    private let httpGet = OpenAIToolDefinition(
        function: .init(name: "http_get", parameters: .object(["type": .string("object")]))
    )

    @Test(
        "String choices decode, encode, and resolve",
        arguments: [
            ("auto", OpenAIToolChoice.auto),
            ("none", OpenAIToolChoice.none),
            ("required", OpenAIToolChoice.required),
        ])
    func stringChoices(value: String, expected: OpenAIToolChoice) throws {
        let request = try decode(toolChoiceJSON: "\"\(value)\"", toolsJSON: toolJSON([submit]))
        #expect(request.toolChoice == expected)
        let roundTrip = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: JSONEncoder().encode(request)
        )
        #expect(roundTrip == request)
    }

    @Test("Named function choice uses the OpenAI object shape")
    func namedChoiceWireShape() throws {
        let request = try decode(
            toolChoiceJSON: #"{"type":"function","function":{"name":"submit_assessment"}}"#,
            toolsJSON: toolJSON([submit, httpGet])
        )
        #expect(request.toolChoice == .function(name: "submit_assessment"))

        let object = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any]
        )
        let choice = try #require(object["tool_choice"] as? [String: Any])
        let function = try #require(choice["function"] as? [String: Any])
        #expect(choice["type"] as? String == "function")
        #expect(function["name"] as? String == "submit_assessment")
    }

    @Test("Auto preserves declared tools but denies calls when none are declared")
    func automaticPlan() throws {
        let withTools = try ToolChoicePlan.resolve(choice: .auto, tools: [submit, httpGet])
        #expect(withTools.constraint == .automatic)
        #expect(withTools.tools == [submit, httpGet])

        let withoutTools = try ToolChoicePlan.resolve(choice: nil, tools: nil)
        #expect(withoutTools.constraint == .automatic)
        #expect(withoutTools.tools == [])
    }

    @Test("None removes prompt tools and installs an empty parser allow-list")
    func nonePlan() throws {
        let plan = try ToolChoicePlan.resolve(
            choice: OpenAIToolChoice.none, tools: [submit, httpGet]
        )
        #expect(plan.constraint == .prohibited)
        #expect(plan.tools == [])
        let validator = ToolChoiceOutputValidator(constraint: plan.constraint, tools: plan.tools)
        #expect(throws: LocalModelRunnerError.prohibitedToolCall("submit_assessment")) {
            try validator.observe(toolCall(named: "submit_assessment"))
        }
    }

    @Test("Required exposes declarations, forces tool protocol, and rejects a text-only turn")
    func requiredPlan() throws {
        let plan = try ToolChoicePlan.resolve(choice: .required, tools: [submit, httpGet])
        #expect(plan.constraint == .required)
        #expect(plan.tools == [submit, httpGet])
        #expect(try plan.forcedToolCallPrefix(for: .glm4) == "<tool_call>")

        let validator = ToolChoiceOutputValidator(constraint: plan.constraint, tools: plan.tools)
        #expect(throws: LocalModelRunnerError.requiredToolCallMissing) {
            try validator.validateCompletion()
        }
        try validator.observe(toolCall(named: "http_get"))
        try validator.validateCompletion()
    }

    @Test("Named choice filters the allow-list and forces the exact GLM4 function")
    func namedPlan() throws {
        let plan = try ToolChoicePlan.resolve(
            choice: .function(name: "submit_assessment"),
            tools: [httpGet, submit]
        )
        #expect(plan.constraint == .named("submit_assessment"))
        #expect(plan.tools == [submit])
        #expect(
            try plan.forcedToolCallPrefix(for: .glm4)
                == "<tool_call>submit_assessment"
        )

        let validator = ToolChoiceOutputValidator(constraint: plan.constraint, tools: plan.tools)
        #expect(
            throws: LocalModelRunnerError.wrongNamedToolCall(
                expected: "submit_assessment", actual: "http_get")
        ) {
            try validator.observe(toolCall(named: "http_get"))
        }
    }

    @Test("Unknown named functions and required-without-tools fail before generation")
    func invalidPlans() {
        #expect(throws: ToolChoiceValidationError.unknownFunction("missing")) {
            try ToolChoicePlan.resolve(choice: .function(name: "missing"), tools: [submit])
        }
        #expect(throws: ToolChoiceValidationError.requiredWithoutTools) {
            try ToolChoicePlan.resolve(choice: .required, tools: nil)
        }
    }

    @Test("ATEM required choice offers every declared function and named choice offers one")
    func atemChoices() throws {
        let required = try ToolChoicePlan.resolve(choice: .required, tools: [submit, httpGet])
        let prefixes = try required.forcedToolCallPrefixes(for: .atem)
        #expect(
            prefixes
                == ["submit_assessment", "http_get"].map {
                    " to=\($0)<|message|><atem:function_calls>\n<atem:invoke name=\"\($0)\">\n"
                })
        let named = try ToolChoicePlan.resolve(choice: .function(name: "http_get"), tools: [submit, httpGet])
        #expect(try named.forcedToolCallPrefixes(for: .atem) == [prefixes[1]])
        for choice in [OpenAIToolChoice.auto, .none] {
            let plan = try ToolChoicePlan.resolve(choice: choice, tools: [submit])
            #expect(try plan.forcedToolCallPrefixes(for: .atem).isEmpty)
        }
    }

    @Test("ATEM forced headers reject names containing protocol syntax")
    func invalidATEMName() throws {
        let tool = OpenAIToolDefinition(function: .init(name: "bad\nname", parameters: .object([:])))
        let plan = try ToolChoicePlan.resolve(choice: .required, tools: [tool])
        #expect(throws: ToolChoiceValidationError.invalidFunctionName("bad\nname")) {
            try plan.forcedToolCallPrefixes(for: .atem)
        }
    }

    @Test("Prefix alternatives retain model scores, commit to a branch, and reset per prompt")
    func alternativePrefixes() {
        var processor = ForcedTokenPrefixLogitProcessor(tokenSequences: [[2, 1, 0], [2, 3, 4]])
        let logits = MLXArray([Float(10), 20, 30, 40, 50], [1, 5])
        let firstMask: [Float] = [-.infinity, -.infinity, 30, -.infinity, -.infinity]
        let branchMask: [Float] = [-.infinity, 20, -.infinity, 40, -.infinity]
        let lastMask: [Float] = [-.infinity, -.infinity, -.infinity, -.infinity, 50]
        processor.prompt(MLXArray([9]))
        #expect(processor.process(logits: logits).asArray(Float.self) == firstMask)
        processor.didSample(token: MLXArray(2))
        #expect(processor.process(logits: logits).asArray(Float.self) == branchMask)
        processor.didSample(token: MLXArray(3))
        #expect(processor.process(logits: logits).asArray(Float.self) == lastMask)
        processor.didSample(token: MLXArray(4))
        #expect(processor.process(logits: logits).asArray(Float.self) == logits.asArray(Float.self))
        processor.prompt(MLXArray([9]))
        #expect(processor.process(logits: logits).asArray(Float.self) == firstMask)
    }

    @Test("Unsupported model protocols are rejected instead of pretending to force")
    func unsupportedForcedProtocol() throws {
        let plan = try ToolChoicePlan.resolve(
            choice: .function(name: "submit_assessment"), tools: [submit]
        )
        #expect(throws: LocalModelRunnerError.unsupportedForcedToolChoiceFormat("mistral")) {
            try plan.forcedToolCallPrefix(for: .mistral)
        }
    }

    @Test("The logits processor emits every forced token before releasing generation")
    func tokenPrefixConstraint() {
        var processor = ForcedTokenPrefixLogitProcessor(tokenIDs: [2, 1])
        processor.prompt(MLXArray([9]))
        // TokenIterator presents [batch, vocabulary] logits. The batch dimension
        // must survive masking or the sampled token is scalar and Laguna mistakes
        // its hidden width for the next sequence length.
        let logits = MLXArray([Float(10), 20, 30], [1, 3])

        let first = processor.process(logits: logits)
        #expect(first.shape == [1, 3])
        #expect(
            first.asArray(Float.self) == [
                -.infinity, -.infinity, 30,
            ])
        processor.didSample(token: MLXArray(2))
        let second = processor.process(logits: logits)
        #expect(second.shape == [1, 3])
        #expect(
            second.asArray(Float.self) == [
                -.infinity, 20, -.infinity,
            ])
        processor.didSample(token: MLXArray(1))
        let released = processor.process(logits: logits)
        #expect(released.shape == [1, 3])
        #expect(released.asArray(Float.self) == [10, 20, 30])
    }

    @Test("A forced prefix excludes the retained conversation cache path")
    func forcedPrefixDisablesConversationCache() {
        let capabilities = ModelRuntimeCapabilities(mistralFamily: .mistral)
        #expect(
            !LocalModelRunner.shouldUseHotConversationCache(
                capabilities: capabilities,
                enablePromptCache: true,
                normalizesGemma4Prompt: false,
                hasCustomStopStrings: false,
                usesDFlash: false,
                hasForcedToolPrefix: true
            )
        )
    }

    private func decode(toolChoiceJSON: String, toolsJSON: String) throws
        -> ChatCompletionRequest
    {
        try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data(
                """
                {"model":"local-model","messages":[{"role":"user","content":"finish"}],
                 "tools":\(toolsJSON),"tool_choice":\(toolChoiceJSON)}
                """.utf8
            )
        )
    }

    private func toolJSON(_ tools: [OpenAIToolDefinition]) throws -> String {
        String(decoding: try JSONEncoder().encode(tools), as: UTF8.self)
    }

    private func toolCall(named name: String) -> LocalModelRunnerEvent {
        .toolCall(
            OpenAIToolCall(
                id: "call_test",
                function: .init(name: name, arguments: #"{"verdict":"safe"}"#)
            )
        )
    }
}
