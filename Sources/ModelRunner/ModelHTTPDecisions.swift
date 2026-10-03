import Foundation
import ModelRunnerCore
import ModelRunnerProtocol
import NIOCore
import NIOHTTP1

extension ModelHTTPServer {
    func handleDecisions(body: Data, channel: Channel) async throws {
        guard body.count <= 64 * 1024 else {
            throw ModelHTTPError(
                status: .payloadTooLarge, message: "Decision requests may not exceed 64 KiB.", code: "request_too_large"
            )
        }
        guard let runner else {
            throw ModelHTTPError(
                status: .unprocessableEntity, message: "This model does not support decisions.",
                code: "unsupported_model_feature")
        }
        let request: DecisionRequest
        do {
            let object = try JSONSerialization.jsonObject(with: body) as? [String: Any]
            guard let object,
                Set(object.keys).isSubset(of: ["model", "context", "schema", "field_order", "score_fields"])
            else {
                throw DecisionError.invalidRequest("Unknown decision request fields.")
            }
            request = try DecisionRequest.decode(body)
            guard !request.model.isEmpty else { throw DecisionError.invalidRequest("model is required.") }
        } catch {
            throw ModelHTTPError(
                status: .badRequest, message: error.localizedDescription, code: "invalid_decision_request")
        }
        do {
            guard request.model == servedModelName else {
                throw ModelHTTPError(
                    status: .notFound, message: "The requested model is not loaded.", param: "model",
                    code: "model_not_found")
            }
            try await sendJSON(try await runner.scoreDecisions(request: request), on: channel)
        } catch let error as DecisionError {
            switch error {
            case .invalidRequest(let message):
                throw ModelHTTPError(status: .badRequest, message: message, code: "invalid_decision_request")
            case .unsupported(let message):
                throw ModelHTTPError(status: .unprocessableEntity, message: message, code: "unsupported_model_feature")
            case .invalidContract(let message), .numericalFailure(let message):
                throw ModelHTTPError(
                    status: .internalServerError, message: message, type: "server_error", code: "invalid_decision_model"
                )
            }
        } catch LocalModelRunnerError.busy {
            throw ModelHTTPError(status: .conflict, message: "Model execution queue is full.", code: "model_busy")
        } catch let error as RequestAdmissionError {
            throw ModelHTTPError(
                status: .badRequest, message: error.localizedDescription, code: "request_exceeds_limits")
        } catch let error as DecodingError {
            throw ModelHTTPError(status: .badRequest, message: error.localizedDescription, code: "invalid_json")
        }
    }
}
