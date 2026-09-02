import Foundation

/// Resolves an optional OpenAI `max_tokens` request against the runner's configured limit.
///
/// The configured value is both the default for requests that omit `max_tokens` and the
/// hard ceiling for requests that provide it. This prevents an HTTP client from allocating
/// an unexpectedly large generation/KV cache while preserving the configured default.
public struct GenerationTokenLimit: Equatable, Sendable {
    public let configuredMaximum: Int

    public init(configuredMaximum: Int) throws {
        guard configuredMaximum > 0 else {
            throw GenerationTokenLimitError.invalidConfiguredMaximum(configuredMaximum)
        }
        self.configuredMaximum = configuredMaximum
    }

    public func resolve(requested: Int?) throws -> Int {
        guard let requested else { return configuredMaximum }
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

public enum GenerationTokenLimitError: LocalizedError, Equatable, Sendable {
    case invalidConfiguredMaximum(Int)
    case invalidRequest(Int)
    case exceedsConfiguredMaximum(requested: Int, configuredMaximum: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguredMaximum(let value):
            "Configured maximum tokens must be greater than zero (received \(value))."
        case .invalidRequest(let value):
            "max_tokens must be greater than zero (received \(value))."
        case .exceedsConfiguredMaximum(let requested, let configuredMaximum):
            "max_tokens \(requested) exceeds the configured maximum of \(configuredMaximum)."
        }
    }
}
