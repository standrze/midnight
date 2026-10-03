import Foundation
import ModelRunnerCore
import ModelRunnerProtocol
import NIOCore
import NIOHTTP1

enum InspectorRecordingRoute: Equatable {
    case create
    case session(String)

    static func parse(uri: String) -> Self? {
        let base = "/v1/inspector/recordings"
        if uri == base { return .create }
        guard uri.hasPrefix(base + "/"),
            let id = UUID(uuidString: String(uri.dropFirst(base.count + 1)))
        else { return nil }
        return .session(id.uuidString)
    }

    func validate(method: HTTPMethod) throws {
        switch (self, method) {
        case (.create, .POST), (.session, .GET), (.session, .DELETE): return
        default:
            throw ModelHTTPError(
                status: .methodNotAllowed, message: "Method is not allowed for this recording route.",
                code: "method_not_allowed")
        }
    }
}

extension ModelHTTPServer {
    func handleRecording(
        route: InspectorRecordingRoute, head: HTTPRequestHead, body: Data, channel: Channel
    ) async throws {
        try route.validate(method: head.method)
        if case .session(let id) = route {
            let session = head.method == .DELETE ? recordingStore.cancel(id: id) : recordingStore.status(id: id)
            guard let session else {
                throw ModelHTTPError(
                    status: .notFound, message: "Recording not found or no longer retained.",
                    param: "recording_id", code: "recording_not_found")
            }
            try await sendJSON(session, on: channel)
            return
        }
        guard body.count <= 32 * 1024 else {
            throw ModelHTTPError(
                status: .payloadTooLarge, message: "Inspector requests may not exceed 32 KiB.",
                code: "request_too_large")
        }
        var request: InspectorRecordingRequest
        do {
            request = try JSONDecoder().decode(InspectorRecordingRequest.self, from: body)
        } catch {
            throw ModelHTTPError(
                status: .badRequest, message: "Invalid recording request: \(error.localizedDescription)",
                code: "invalid_json")
        }
        guard let runner else {
            throw ModelHTTPError(
                status: .notImplemented, message: "The loaded model does not provide activation recording.",
                code: "unsupported_model_feature")
        }
        // Managed admission has already compared the requested generation under
        // the model lease; store the actual generation in the saved descriptor.
        request.runtimeGeneration = inspectorRuntimeGeneration
        do {
            let session = try await runner.armInspectorRecording(request: request, store: recordingStore)
            try await sendJSON(session, status: .accepted, on: channel)
        } catch LocalModelRunnerError.busy {
            throw ModelHTTPError(
                status: .conflict, message: LocalModelRunnerError.busy.localizedDescription,
                type: "server_error", code: "model_busy")
        } catch let error as InspectorRecordingError {
            throw ModelHTTPError(
                status: .conflict, message: error.localizedDescription, code: "recording_busy")
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
