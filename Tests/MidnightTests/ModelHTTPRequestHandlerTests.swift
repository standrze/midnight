import Foundation
import NIOCore
import NIOEmbedded
import NIOHTTP1
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
