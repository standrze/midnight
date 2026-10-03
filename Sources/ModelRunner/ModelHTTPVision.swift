import Foundation
import ModelRunnerCore
import ModelRunnerProtocol
import NIOCore
import NIOHTTP1

// Shares the listener and its single model lifecycle; this is not a separate server.
extension ModelHTTPServer {
    func handleVision(
        _ vision: any VisionModelServing, head: HTTPRequestHead,
        body: Data, channel: Channel
    ) async throws {
        let response: VisionHTTPResponse
        switch (head.method, head.uri) {
        case (.POST, "/v1/chat/completions"):
            guard body.count <= 12 * 1024 * 1024 else {
                throw ModelHTTPError(
                    status: .payloadTooLarge,
                    message: "Vision requests may not exceed 12 MiB.", code: "request_too_large")
            }
            response = try await vision.complete(body)
        case (.GET, "/v1/vision/status"):
            response = try await vision.status()
        default:
            throw ModelHTTPError(
                status: .notImplemented,
                message:
                    "This vision model supports non-streaming Chat Completions with an image. Load a text or voice model for other operations.",
                code: "unsupported_model_feature")
        }
        try await send(
            status: HTTPResponseStatus(statusCode: response.status),
            contentType: "application/json; charset=utf-8", data: response.body, on: channel)
    }
}
