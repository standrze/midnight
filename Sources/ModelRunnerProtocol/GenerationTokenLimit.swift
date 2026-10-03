import Foundation

/// Resolves an optional OpenAI `max_tokens` request against the runner's configured limit.
///
/// Explicit operator limits remain hard ceilings. Automatic text-model policies use
/// a separate default, and exact prompt tokenization bounds each generation.
public struct GenerationTokenLimit: Equatable, Sendable {
    /// Hard ceiling on emitted token IDs for one response.
    public let configuredMaximum: Int
    /// Emitted token IDs allowed when the request omits a limit.
    public let defaultTokens: Int

    /// Creates a positive token ceiling and an optional default no larger than it.
    public init(configuredMaximum: Int, defaultTokens: Int? = nil) throws {
        guard configuredMaximum > 0 else {
            throw GenerationTokenLimitError.invalidConfiguredMaximum(configuredMaximum)
        }
        self.configuredMaximum = configuredMaximum
        let replyDefault = defaultTokens ?? configuredMaximum
        guard replyDefault > 0, replyDefault <= configuredMaximum else {
            throw GenerationTokenLimitError.invalidConfiguredMaximum(replyDefault)
        }
        self.defaultTokens = replyDefault
    }

    /// The full context is a capacity, not a sensible default reply length.
    public static func forTextModel(
        contextLength: Int, configuredMaximum: Int? = nil,
        modelMaximum: Int? = nil
    ) throws -> Self {
        guard contextLength > 1 else {
            throw GenerationTokenLimitError.noOutputSpace
        }
        if let configuredMaximum, configuredMaximum <= 0 {
            throw GenerationTokenLimitError.invalidConfiguredMaximum(configuredMaximum)
        }
        if let modelMaximum, modelMaximum <= 0 {
            throw GenerationTokenLimitError.invalidConfiguredMaximum(modelMaximum)
        }
        let ceiling = min(contextLength - 1, configuredMaximum ?? Int.max, modelMaximum ?? Int.max)
        let replyDefault = configuredMaximum == nil ? min(4096, max(1, contextLength / 4), ceiling) : ceiling
        return try Self(configuredMaximum: ceiling, defaultTokens: replyDefault)
    }

    /// Tokenization includes the chat template; leave one slot for a terminal token.
    public func resolve(requested: Int?, promptTokens: Int, contextLength: Int) throws -> Int {
        let desired = try resolve(requested: requested)
        guard promptTokens >= 0, contextLength > 1, promptTokens < contextLength - 1 else {
            throw GenerationTokenLimitError.noOutputSpace
        }
        return min(desired, contextLength - promptTokens - 1)
    }

    /// Validates an explicit requested token count or returns the configured default.
    public func resolve(requested: Int?) throws -> Int {
        guard let requested else {
            return defaultTokens
        }
        guard requested > 0 else {
            throw GenerationTokenLimitError.invalidRequest(requested)
        }
        guard requested <= configuredMaximum else {
            throw GenerationTokenLimitError.exceedsConfiguredMaximum(
                requested: requested,
                configuredMaximum: configuredMaximum
            )
        }
        return requested
    }
}

/// A token limit is invalid or the rendered prompt leaves no output space.
public enum GenerationTokenLimitError: LocalizedError, Equatable, Sendable {
    case noOutputSpace
    case invalidConfiguredMaximum(Int)
    case invalidRequest(Int)
    case exceedsConfiguredMaximum(requested: Int, configuredMaximum: Int)

    /// User-facing explanation for this error.
    public var errorDescription: String? {
        switch self {
        case .noOutputSpace:
            "The rendered prompt fills the model context; shorten the conversation to leave room for a reply."
        case .invalidConfiguredMaximum(let value):
            "Configured maximum tokens must be greater than zero (received \(value))."
        case .invalidRequest(let value):
            "max_tokens must be greater than zero (received \(value))."
        case .exceedsConfiguredMaximum(let requested, let configuredMaximum):
            "max_tokens \(requested) exceeds the configured maximum of \(configuredMaximum)."
        }
    }
}
