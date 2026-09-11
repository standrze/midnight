import Foundation
import NIOCore
import NIOEmbedded
import NIOHTTP1
import NIOPosix
import Testing

@testable import Midnight

@Suite("Model HTTP request cancellation")
struct ModelHTTPRequestHandlerTests {
    @Test("Disconnect cancels generation and permits the next request")
    func disconnectCancellationReleasesGeneration() async throws {
        let probe = GenerationProbe()
        let firstChannel = try await makeChannel(probe: probe)
        try await writeRequest(uri: "/first", to: firstChannel)

        #expect(await eventually { await probe.snapshot().accepted == 1 })
        #expect(await probe.snapshot().isGenerating)

        try await firstChannel.close().get()

        #expect(await eventually {
            let snapshot = await probe.snapshot()
            return snapshot.cancelled == 1 && !snapshot.isGenerating
        })

        let secondChannel = try await makeChannel(probe: probe)
        try await writeRequest(uri: "/second", to: secondChannel)

        #expect(await eventually {
            let snapshot = await probe.snapshot()
            return snapshot.accepted == 2 && !snapshot.isGenerating
        })
        let final = await probe.snapshot()
        #expect(final.busy == 0)
        try await secondChannel.close().get()
        _ = try await firstChannel.finish(acceptAlreadyClosed: true)
        _ = try await secondChannel.finish(acceptAlreadyClosed: true)
    }

    @Test("Channel errors cancel the active response")
    func channelErrorCancelsResponse() async throws {
        let probe = GenerationProbe()
        let channel = try await makeChannel(probe: probe)
        try await writeRequest(uri: "/first", to: channel)
        #expect(await eventually { await probe.snapshot().isGenerating })

        channel.pipeline.fireErrorCaught(TestChannelError.disconnected)

        #expect(await eventually {
            let snapshot = await probe.snapshot()
            return snapshot.cancelled == 1 && !snapshot.isGenerating
        })
        _ = try await channel.finish(acceptAlreadyClosed: true)
    }

    @Test("A real TCP disconnect during silent streaming cancels the response task")
    func silentStreamingTCPDisconnectCancelsResponse() async throws {
        let probe = GenerationProbe()
        let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        let accepted = group.next().makePromise(of: Channel.self)
        let server = try await ServerBootstrap(group: group)
            .childChannelInitializer { channel in
                ModelHTTPServer.configureHTTPPipeline(on: channel).flatMap {
                    channel.pipeline.addHandler(ModelHTTPRequestHandler { head, _, _, channel in
                        do {
                            let headers = HTTPHeaders([
                                ("content-type", "text/event-stream"),
                                ("transfer-encoding", "chunked"),
                                ("connection", "close"),
                            ])
                            try await channel.writeAndFlush(HTTPServerResponsePart.head(
                                HTTPResponseHead(version: .http1_1, status: .ok, headers: headers)
                            )).get()
                            var body = channel.allocator.buffer(capacity: 64)
                            body.writeString("data: {\"role\":\"assistant\"}\n\n")
                            try await channel.writeAndFlush(
                                HTTPServerResponsePart.body(.byteBuffer(body))
                            ).get()
                            // No more writes occur until cancellation, matching a model
                            // that is prefilling or generating hidden reasoning.
                            await probe.respond(uri: head.uri)
                        } catch {
                            Issue.record("Unexpected streaming setup error: \(error)")
                        }
                    })
                }.map { accepted.succeed(channel) }
            }
            .bind(host: "127.0.0.1", port: 0).get()
        var client: Channel?
        var peer: Channel?
        do {
            let address = try #require(server.localAddress)
            let connected = try await ClientBootstrap(group: group).connect(to: address).get()
            client = connected
            peer = try await accepted.futureResult.get()
            var request = connected.allocator.buffer(capacity: 256)
            // A second pipelined request must remain ignored because responses
            // use Connection: close, even without NIO's pipelining assistance.
            request.writeString("POST /first HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n")
            request.writeString("POST /second HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n")
            try await connected.writeAndFlush(request).get()
            try #require(await eventually { await probe.snapshot().isGenerating })

            // Close only the client TCP socket. Directly closing the server's
            // channel would bypass the socket-read behavior under regression.
            let disconnected = ContinuousClock.now
            try await connected.close(mode: .all).get()
            let didCancel = await eventually {
                let snapshot = await probe.snapshot()
                return snapshot.cancelled == 1 && !snapshot.isGenerating
            }
            #expect(didCancel)
            if didCancel { #expect(disconnected.duration(to: .now) < .seconds(2)) }
            let snapshot = await probe.snapshot()
            #expect(snapshot.accepted == 1)
            #expect(snapshot.busy == 0)

            try? await peer?.close().get()
            try? await server.close().get()
            try await group.shutdownGracefully()
        } catch {
            try? await client?.close().get()
            try? await peer?.close().get()
            try? await server.close().get()
            try? await group.shutdownGracefully()
            throw error
        }
    }

    private func makeChannel(probe: GenerationProbe) async throws -> NIOAsyncTestingChannel {
        let handler = ModelHTTPRequestHandler { head, _, _, _ in
            await probe.respond(uri: head.uri)
        }
        let channel = await NIOAsyncTestingChannel(handler: handler)
        try await channel.connect(
            to: SocketAddress(ipAddress: "127.0.0.1", port: 9_999)
        ).get()
        return channel
    }

    private func writeRequest(
        uri: String,
        to channel: NIOAsyncTestingChannel
    ) async throws {
        _ = try await channel.writeInbound(
            HTTPServerRequestPart.head(
                HTTPRequestHead(version: .http1_1, method: .POST, uri: uri)
            )
        )
        _ = try await channel.writeInbound(HTTPServerRequestPart.end(nil))
    }

    private func eventually(
        timeout: Duration = .seconds(2),
        _ condition: @escaping @Sendable () async -> Bool
    ) async -> Bool {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: timeout)
        while clock.now < deadline {
            if await condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return await condition()
    }
}

private enum TestChannelError: Error {
    case disconnected
}

private actor GenerationProbe {
    struct Snapshot: Sendable {
        let accepted: Int
        let busy: Int
        let cancelled: Int
        let isGenerating: Bool
    }

    private var accepted = 0
    private var busy = 0
    private var cancelled = 0
    private var isGenerating = false

    func respond(uri: String) async {
        guard !isGenerating else {
            busy += 1
            return
        }
        isGenerating = true
        accepted += 1
        defer { isGenerating = false }

        guard uri == "/first" else { return }
        do {
            try await Task.sleep(for: .seconds(30))
        } catch is CancellationError {
            cancelled += 1
        } catch {
            Issue.record("Unexpected response task error: \(error)")
        }
    }

    func snapshot() -> Snapshot {
        Snapshot(
            accepted: accepted,
            busy: busy,
            cancelled: cancelled,
            isGenerating: isGenerating
        )
    }
}
