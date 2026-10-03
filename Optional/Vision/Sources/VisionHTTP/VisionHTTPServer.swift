import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import VisionProtocol

/// Generated vision text with token counts and completion reason.
public struct VisionAnswer: Codable, Sendable {
    public let text: String
    public let promptTokens: Int
    public let completionTokens: Int
    public let finishReason: String
    /// Creates an answer with text, token counts, and finish reason.
    public init(text: String, promptTokens: Int, completionTokens: Int, finishReason: String) {
        self.text = text
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.finishReason = finishReason
    }
}

/// Generates answers for validated vision requests and reports runtime state.
public protocol VisionGenerating: Sendable {
    /// Generates an answer for a validated request, throwing on failure, cancellation, or busy admission.
    func answer(_ request: VisionRequest) async throws -> VisionAnswer
    /// Waits until active generation has finished and its producer has released request-owned resources.
    func waitUntilIdle() async
    /// Reports whether the backend currently has an active generation request.
    func isBusy() async -> Bool
    /// Returns backend memory counters in bytes, keyed by their measurement names.
    func memorySnapshot() async -> [String: Int]
}
extension VisionGenerating {
    /// Waits for active vision requests to finish; default implementations return immediately.
    public func waitUntilIdle() async {}
    /// Reports backend admission state; the default reports idle.
    public func isBusy() async -> Bool { false }
    /// Returns backend memory counters; the default has no counters.
    public func memorySnapshot() async -> [String: Int] { [:] }
}

/// A separately launched loopback listener; no dependency points from the text executable to this module.
public final class VisionHTTPServer: Sendable {
    let backend: any VisionGenerating
    let model: String
    let controlToken: String?
    /// Creates a loopback vision server with an optional control token.
    public init(backend: any VisionGenerating, model: String, controlToken: String? = nil) {
        self.backend = backend
        self.model = model
        self.controlToken = controlToken
    }
    /// Starts the loopback HTTP listener and optionally writes an atomic readiness file.
    public func run(port: Int, readyFile: URL? = nil) async throws {
        let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        do {
            let channel = try await ServerBootstrap(group: group)
                .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
                .childChannelInitializer { channel in
                    channel.pipeline.configureHTTPServerPipeline(withPipeliningAssistance: false)
                        .flatMap { channel.pipeline.addHandler(VisionHandler(server: self)) }
                }
                .bind(host: "127.0.0.1", port: port).get()
            let actualPort = channel.localAddress!.port!
            if let readyFile {
                let ready =
                    [
                        "port": actualPort, "pid": Int(ProcessInfo.processInfo.processIdentifier),
                        "token": controlToken ?? "", "model": model,
                    ] as [String: Any]
                try JSONSerialization.data(withJSONObject: ready).write(to: readyFile, options: .atomic)
            }
            FileHandle.standardError.write(Data("Vision ready: http://127.0.0.1:\(actualPort)/v1 (\(model))\n".utf8))
            try await channel.closeFuture.get()
            try await group.shutdownGracefully()
        } catch {
            try? await group.shutdownGracefully()
            throw error
        }
    }
    /// Handles one HTTP request and returns its status and JSON body.
    public func response(method: HTTPMethod, uri: String, data: Data, authorization: String? = nil) async -> (
        HTTPResponseStatus, Data
    ) {
        if let controlToken, authorization != "Bearer " + controlToken {
            return Self.error(
                .unauthorized,
                VisionError("unauthorized", "This managed worker accepts requests from its owning Midnight listener."))
        }
        do {
            switch (method, uri) {
            case (.GET, "/v1/models"):
                return (
                    .ok,
                    try JSONSerialization.data(withJSONObject: [
                        "object": "list",
                        "data": [["id": model, "object": "model", "created": 0, "owned_by": "midnight-vision"]],
                    ])
                )
            case (.GET, "/v1/vision/status"):
                return (
                    .ok,
                    try JSONSerialization.data(withJSONObject: [
                        "phase": "ready", "model": model, "runtime": "native-mlx-vision", "persistent_model": true,
                        "maximum_images": 1, "busy": await backend.isBusy(), "memory_scope": "vision_worker",
                        "memory": await backend.memorySnapshot(),
                    ])
                )
            case (.POST, "/v1/vision/drain"):
                await backend.waitUntilIdle()
                return (.ok, Data("{\"idle\":true}".utf8))
            case (.POST, "/v1/chat/completions"):
                let request = try VisionRequest.parse(data, expectedModel: model)
                let answer = try await backend.answer(request)
                let result: [String: Any] = [
                    "id": "chatcmpl-vision-" + UUID().uuidString, "object": "chat.completion",
                    "created": Int(Date().timeIntervalSince1970), "model": model,
                    "choices": [
                        [
                            "index": 0, "message": ["role": "assistant", "content": answer.text],
                            "finish_reason": answer.finishReason,
                        ]
                    ],
                    "usage": [
                        "prompt_tokens": answer.promptTokens, "completion_tokens": answer.completionTokens,
                        "total_tokens": answer.promptTokens + answer.completionTokens,
                    ],
                ]
                return (.ok, try JSONSerialization.data(withJSONObject: result))
            default:
                return Self.error(
                    .notFound,
                    VisionError(
                        "unsupported_route",
                        "This optional vision listener supports /v1/models, /v1/vision/status, and nonstreaming /v1/chat/completions."
                    ))
            }
        } catch let error as VisionError {
            return Self.error(error.code == "vision_busy" ? .tooManyRequests : .badRequest, error)
        } catch is CancellationError {
            return Self.error(.requestTimeout, VisionError("cancelled", "Vision request cancelled."))
        } catch {
            return Self.error(.internalServerError, VisionError("vision_failed", String(describing: error)))
        }
    }
    static func error(_ status: HTTPResponseStatus, _ error: VisionError) -> (HTTPResponseStatus, Data) {
        struct Envelope: Encodable { let error: VisionError }
        return (status, (try? JSONEncoder().encode(Envelope(error: error))) ?? Data())
    }
}

private final class VisionHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart
    let server: VisionHTTPServer
    var head: HTTPRequestHead?
    var body = Data()
    var inFlight = false
    var work: Task<Void, Never>?
    init(server: VisionHTTPServer) { self.server = server }
    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        guard !inFlight else {
            return
        }
        switch unwrapInboundIn(data) {
        case .head(let head): self.head = head
        case .body(var buffer):
            guard body.count + buffer.readableBytes <= VisionRequest.maximumBodyBytes else {
                inFlight = true
                let result = VisionHTTPServer.error(
                    .payloadTooLarge, VisionError("request_too_large", "Vision request exceeds 12 MiB."))
                write(result, channel: context.channel)
                return
            }
            body.append(contentsOf: buffer.readBytes(length: buffer.readableBytes) ?? [])
        case .end:
            guard let head else {
                context.close(promise: nil)
                return
            }
            inFlight = true
            let channel = context.channel
            let payload = body
            let server = server
            body = Data()
            work = Task {
                let result = await server.response(
                    method: head.method, uri: head.uri, data: payload,
                    authorization: head.headers.first(name: "authorization"))
                guard !Task.isCancelled else {
                    return
                }
                self.write(result, channel: channel)
            }
        }
    }
    private func write(_ result: (HTTPResponseStatus, Data), channel: Channel) {
        channel.eventLoop.execute {
            guard channel.isActive else {
                return
            }
            let headers: HTTPHeaders = [
                "content-type": "application/json", "content-length": String(result.1.count), "connection": "close",
            ]
            channel.write(
                HTTPServerResponsePart.head(HTTPResponseHead(version: .http1_1, status: result.0, headers: headers)),
                promise: nil)
            var buffer = channel.allocator.buffer(capacity: result.1.count)
            buffer.writeBytes(result.1)
            channel.write(HTTPServerResponsePart.body(.byteBuffer(buffer)), promise: nil)
            channel.writeAndFlush(HTTPServerResponsePart.end(nil)).whenComplete { _ in channel.close(promise: nil) }
        }
    }
    func channelInactive(context: ChannelHandlerContext) {
        work?.cancel()
        context.fireChannelInactive()
    }
    func errorCaught(context: ChannelHandlerContext, error: Error) {
        work?.cancel()
        context.close(promise: nil)
    }
}
