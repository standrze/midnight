import Foundation
import NIOHTTP1

/// An optional process-wide bearer secret protecting every route when configured.
struct APIKeyAuthentication: Sendable {
    static let environmentKey = "MIDNIGHT_API_KEY"

    private let secret: [UInt8]
    private let secretLength: Int

    init(key: String) throws {
        let bytes = Array(key.utf8)
        let validCharacters = bytes.allSatisfy { byte in
            (48...57).contains(byte) || (65...90).contains(byte) || (97...122).contains(byte)
                || [45, 46, 95, 126].contains(byte)
        }
        guard (32...256).contains(bytes.count), validCharacters else {
            throw APIKeyAuthenticationError.invalidConfiguration
        }
        secret = bytes + Array(repeating: 0, count: 256 - bytes.count)
        secretLength = bytes.count
    }

    static func fromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> Self? {
        guard let key = environment[environmentKey] else {
            return nil
        }
        return try Self(key: key)
    }

    func accepts(_ headers: HTTPHeaders) -> Bool {
        let values = headers["authorization"]
        guard values.count == 1 else {
            return false
        }
        let parts = values[0].split(separator: " ", omittingEmptySubsequences: true)
        guard parts.count == 2, parts[0].lowercased() == "bearer" else {
            return false
        }
        let supplied = Array(parts[1].utf8)
        guard supplied.count <= 256 else {
            return false
        }
        let padded = supplied + Array(repeating: 0, count: 256 - supplied.count)
        var difference = secretLength ^ supplied.count
        for index in 0..<256 {
            difference |= Int(secret[index] ^ padded[index])
        }
        return difference == 0
    }
}

enum APIKeyAuthenticationError: LocalizedError {
    case invalidConfiguration

    var errorDescription: String? {
        switch self {
        case .invalidConfiguration:
            "MIDNIGHT_API_KEY must contain 32 to 256 ASCII letters, digits, hyphens, periods, underscores, or tildes."
        }
    }
}
