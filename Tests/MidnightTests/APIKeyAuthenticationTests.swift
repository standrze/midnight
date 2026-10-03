import ModelRunnerProtocol
import NIOHTTP1
import Testing

@testable import Midnight

@Suite("API key authentication")
struct APIKeyAuthenticationTests {
    private let key = "0123456789abcdef0123456789abcdef"

    @Test("Generated keys are unique and accepted by the listener")
    func generatedKeys() throws {
        let first = APIKeyCommand.Generate.generate()
        let second = APIKeyCommand.Generate.generate()
        #expect(first.count == 64)
        #expect(first.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) })
        #expect(first != second)
        let authentication = try APIKeyAuthentication(key: first)
        #expect(authentication.accepts(HTTPHeaders([("authorization", "Bearer \(first)")])))
    }

    @Test("Authentication is optional, but configured keys must be valid")
    func configuration() throws {
        #expect(try APIKeyAuthentication.fromEnvironment([:]) == nil)
        #expect(throws: APIKeyAuthenticationError.self) {
            try APIKeyAuthentication.fromEnvironment([APIKeyAuthentication.environmentKey: "short"])
        }
        let configured = try #require(
            try APIKeyAuthentication.fromEnvironment([APIKeyAuthentication.environmentKey: key]))
        #expect(configured.accepts(HTTPHeaders([("authorization", "Bearer \(key)")])))
        #expect(throws: APIKeyAuthenticationError.self) {
            try APIKeyAuthentication.fromEnvironment([APIKeyAuthentication.environmentKey: ""])
        }
    }

    @Test("A listener without an API key starts and shuts down on cancellation")
    func listenerWithoutKey() async throws {
        let server = ModelHTTPServer(
            servedModelName: "test", tokenLimit: try GenerationTokenLimit(configuredMaximum: 64))
        let (events, continuation) = AsyncStream<Void>.makeStream()
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                defer { continuation.finish() }
                try await server.run(host: "127.0.0.1", port: 0) {
                    continuation.yield(())
                }
            }
            var iterator = events.makeAsyncIterator()
            let _ = try #require(await iterator.next())
            group.cancelAll()
            do {
                try await group.waitForAll()
            } catch is CancellationError {
                // Cancellation closes the listener after its startup callback.
            }
        }
    }

    @Test("Only one matching Bearer authorization value is accepted")
    func bearerHeader() throws {
        let authentication = try APIKeyAuthentication(key: key)
        #expect(authentication.accepts(HTTPHeaders([("authorization", "Bearer \(key)")])))
        #expect(authentication.accepts(HTTPHeaders([("Authorization", "bearer \(key)")])))
        for headers in [
            HTTPHeaders(),
            HTTPHeaders([("authorization", key)]),
            HTTPHeaders([("authorization", "Basic \(key)")]),
            HTTPHeaders([("authorization", "Bearer \(key)X")]),
            HTTPHeaders([("authorization", "Bearer \(key.dropLast())")]),
            HTTPHeaders([("authorization", "Bearer \(key)"), ("authorization", "Bearer \(key)")]),
        ] {
            #expect(!authentication.accepts(headers))
        }
    }
}
