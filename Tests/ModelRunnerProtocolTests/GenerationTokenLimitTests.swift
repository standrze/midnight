import ModelRunnerProtocol
import Testing

@Suite("Generation token limit")
struct GenerationTokenLimitTests {
    @Test("Automatic output defaults scale down for small contexts without imposing a 512 ceiling")
    func automaticPolicy() throws {
        let large = try GenerationTokenLimit.forTextModel(contextLength: 262144)
        #expect(large.defaultTokens == 4096)
        #expect(large.configuredMaximum == 262143)
        #expect(try large.resolve(requested: 8192) == 8192)
        let small = try GenerationTokenLimit.forTextModel(contextLength: 2048)
        #expect(small.defaultTokens == 512)
        #expect(small.configuredMaximum == 2047)
    }

    @Test("Operator and model-specific output ceilings take precedence")
    func explicitPolicies() throws {
        let operatorLimit = try GenerationTokenLimit.forTextModel(contextLength: 32768, configuredMaximum: 512)
        #expect(operatorLimit.defaultTokens == 512)
        #expect(operatorLimit.configuredMaximum == 512)
        let modelLimit = try GenerationTokenLimit.forTextModel(contextLength: 32768, modelMaximum: 2048)
        #expect(modelLimit.defaultTokens == 2048)
        #expect(modelLimit.configuredMaximum == 2048)
        #expect(
            try GenerationTokenLimit.forTextModel(contextLength: 2048, configuredMaximum: 8192).configuredMaximum
                == 2047)
        #expect(throws: GenerationTokenLimitError.self) {
            try GenerationTokenLimit.forTextModel(contextLength: 32768, configuredMaximum: 0)
        }
    }

    @Test("Exact prompt length bounds generation, including omitted output limits")
    func remainingContext() throws {
        let limit = try GenerationTokenLimit.forTextModel(contextLength: 8192)
        #expect(try limit.resolve(requested: 4096, promptTokens: 8000, contextLength: 8192) == 191)
        #expect(try limit.resolve(requested: nil, promptTokens: 8000, contextLength: 8192) == 191)
        #expect(throws: GenerationTokenLimitError.noOutputSpace) {
            try limit.resolve(requested: 1, promptTokens: 8191, contextLength: 8192)
        }
        #expect(throws: GenerationTokenLimitError.noOutputSpace) {
            try limit.resolve(requested: 1, promptTokens: Int.max, contextLength: 8192)
        }
    }

    @Test("The configured maximum is also the request default")
    func configuredDefault() throws {
        let limit = try GenerationTokenLimit(configuredMaximum: 512)

        #expect(try limit.resolve(requested: nil) == 512)
    }

    @Test("Positive requests through the configured maximum are accepted")
    func acceptedRequests() throws {
        let limit = try GenerationTokenLimit(configuredMaximum: 512)

        #expect(try limit.resolve(requested: 1) == 1)
        #expect(try limit.resolve(requested: 512) == 512)
    }

    @Test("Non-positive request values are rejected")
    func invalidRequests() throws {
        let limit = try GenerationTokenLimit(configuredMaximum: 512)

        #expect(throws: GenerationTokenLimitError.invalidRequest(0)) {
            try limit.resolve(requested: 0)
        }
        #expect(throws: GenerationTokenLimitError.invalidRequest(-1)) {
            try limit.resolve(requested: -1)
        }
    }

    @Test("A request cannot exceed the configured maximum")
    func hardCeiling() throws {
        let limit = try GenerationTokenLimit(configuredMaximum: 512)

        #expect(
            throws: GenerationTokenLimitError.exceedsConfiguredMaximum(
                requested: 513,
                configuredMaximum: 512
            )
        ) {
            try limit.resolve(requested: 513)
        }
    }

    @Test("The configured maximum must be positive")
    func invalidConfiguration() {
        #expect(throws: GenerationTokenLimitError.invalidConfiguredMaximum(0)) {
            try GenerationTokenLimit(configuredMaximum: 0)
        }
        #expect(throws: GenerationTokenLimitError.invalidConfiguredMaximum(-1)) {
            try GenerationTokenLimit(configuredMaximum: -1)
        }
    }
}
