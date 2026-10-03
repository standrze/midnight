import Foundation
import ModelRunnerCore
import ModelRunnerProtocol
import NIOCore
import NIOHTTP1

// Shares the listener and its single model lifecycle; this is not a separate server.
extension ModelHTTPServer {
    func handleInspector(body: Data?, channel: Channel) async throws {
        guard let runner else {
            throw ModelHTTPError(
                status: .notImplemented,
                message: "The loaded model does not provide layer inspection.",
                type: "invalid_request_error",
                code: "unsupported_model_feature"
            )
        }
        do {
            if let body {
                guard body.count <= 32 * 1024 else {
                    throw ModelHTTPError(
                        status: .payloadTooLarge,
                        message: "Inspector requests may not exceed 32 KiB.",
                        code: "request_too_large"
                    )
                }
                let request: InspectorTraceRequest
                do {
                    request = try JSONDecoder().decode(InspectorTraceRequest.self, from: body)
                } catch {
                    throw ModelHTTPError(
                        status: .badRequest,
                        message: "Invalid Inspector request: \(error.localizedDescription)",
                        code: "invalid_json"
                    )
                }
                try await sendJSON(try await runner.inspectorTrace(request: request), on: channel)
            } else {
                var descriptor = try await runner.inspectorModel()
                descriptor.runtimeGeneration = inspectorRuntimeGeneration
                descriptor.modelCard = modelCard
                try await sendJSON(descriptor, on: channel)
            }
        } catch LocalModelRunnerError.busy {
            throw ModelHTTPError(
                status: .conflict,
                message: LocalModelRunnerError.busy.localizedDescription,
                type: "server_error",
                code: "model_busy"
            )
        } catch let error as ModelInspectionError {
            switch error {
            case .invalidRequest(let message):
                throw ModelHTTPError(status: .badRequest, message: message, code: "invalid_inspector_request")
            case .unsupported(let message):
                throw ModelHTTPError(status: .unprocessableEntity, message: message, code: "unsupported_model_feature")
            case .invalidGraph(let message):
                throw ModelHTTPError(
                    status: .internalServerError, message: message, type: "server_error", code: "invalid_model_graph")
            }
        }
    }
}
