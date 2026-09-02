import ModelRunnerProtocol
import Testing

@Suite("Generation token limit")
struct GenerationTokenLimitTests {
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
