import Foundation
import ModelRunnerCore
import ModelRunnerProtocol
import NIOCore
import NIOEmbedded
import NIOHTTP1
import Testing

@testable import Midnight

@Suite("Decision HTTP", .serialized)
struct DecisionHTTPTests {
    @Test func bodyLimitBeforeManagedSelection() throws {
        #expect(throws: ModelHTTPError.self) {
            try ModelHTTPServer.requestedModel(path: "/v1/decisions", body: Data(repeating: 32, count: 65537))
        }
    }

    @Test func unsupportedModel() async throws {
        let server = ModelHTTPServer(
            servedModelName: "fixture", tokenLimit: try GenerationTokenLimit(configuredMaximum: 16))
        let response = try await request(server, .POST, "/v1/decisions", body: "{}")
        #expect(response.status == .unprocessableEntity)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_RUN_NIMBLE_SMOKE"] == "1"))
    func realNimbleRepeatedRequests() async throws {
        let root = "/Users/stephen/.midnight/models/nimble-9b"
        let runner = try await LocalModelRunner(
            modelPath: root + "/base", servedModelName: "nimble-9b", maximumTokens: 8192,
            adapterPath: root + "/adapter-native")
        let server = ModelHTTPServer(
            runner: runner, servedModelName: "nimble-9b", tokenLimit: try GenerationTokenLimit(configuredMaximum: 8192))
        let body = try String(
            contentsOfFile: "/Users/stephen/Documents/ChatGPT/midnight/benchmark-results/nimble-20261002/request.json",
            encoding: .utf8)
        let first = try await request(server, .POST, "/v1/decisions", body: body)
        #expect(first.status == .ok)
        let scored = try first.decode(DecisionResponse.self)
        #expect(scored.fields.count == 3)
        let second = try await request(server, .POST, "/v1/decisions", body: body)
        #expect(second.status == .ok)
        let repeated = try second.decode(DecisionResponse.self)
        #expect(scored.output == repeated.output)
        let discovery = try await request(server, .GET, "/v1/models")
        #expect(String(decoding: discovery.body, as: UTF8.self).contains("\"decisions\":true"))
        let malformed = try await request(server, .POST, "/v1/decisions", body: "{")
        #expect(malformed.status == .badRequest)
        let duplicate = try await request(
            server, .POST, "/v1/decisions", body: "{\"model\":\"nimble-9b\",\"model\":\"nimble-9b\"}")
        #expect(duplicate.status == .badRequest)
        let path = URL(
            fileURLWithPath:
                "/Users/stephen/Documents/ChatGPT/midnight/benchmark-results/nimble-20261002/midnight-http.json")
        try first.body.write(to: path, options: .atomic)
    }
    private func request(
        _ server: ModelHTTPServer, _ method: HTTPMethod, _ uri: String,
        body: String = "", extraHeaders: [(String, String)] = [],
        remoteIP: String = "127.0.0.1"
    ) async throws -> Response {
        let channel = await NIOAsyncTestingChannel(handler: ModelHTTPRequestHandler(server: server))
        try await channel.connect(to: SocketAddress(ipAddress: remoteIP, port: 9999)).get()
        do {
            let headers = HTTPHeaders([("content-type", "application/json")] + extraHeaders)
            _ = try await channel.writeInbound(
                HTTPServerRequestPart.head(
                    HTTPRequestHead(version: .http1_1, method: method, uri: uri, headers: headers)))
            if !body.isEmpty {
                var buffer = channel.allocator.buffer(capacity: body.utf8.count)
                buffer.writeString(body)
                _ = try await channel.writeInbound(HTTPServerRequestPart.body(buffer))
            }
            _ = try await channel.writeInbound(HTTPServerRequestPart.end(nil))
            var head: HTTPResponseHead?
            var data = Data()
            let deadline = ContinuousClock.now.advanced(by: .seconds(180))
            while ContinuousClock.now < deadline {
                guard let part = try await channel.readOutbound(as: HTTPServerResponsePart.self) else {
                    try await Task.sleep(for: .milliseconds(5))
                    continue
                }
                switch part {
                case .head(let value): head = value
                case .body(.byteBuffer(var buffer)):
                    if let bytes = buffer.readBytes(length: buffer.readableBytes) {
                        data.append(contentsOf: bytes)
                    }
                case .body(.fileRegion): throw DecisionHTTPFailure.unexpectedBody
                case .end:
                    let head = try #require(head)
                    _ = try await channel.finish(acceptAlreadyClosed: true)
                    return Response(status: head.status, headers: head.headers, body: data)
                }
            }
            throw DecisionHTTPFailure.timedOut
        } catch {
            _ = try? await channel.finish(acceptAlreadyClosed: true)
            throw error
        }
    }

    private struct Response {
        let status: HTTPResponseStatus
        let headers: HTTPHeaders
        let body: Data
        func decode<Value: Decodable>(_ type: Value.Type) throws -> Value {
            try JSONDecoder().decode(type, from: body)
        }
    }
}
private enum DecisionHTTPFailure: Error { case unexpectedBody, timedOut }
