import Foundation
import ModelRunnerCore
import ModelRunnerProtocol
import NIOCore
import NIOHTTP1

// Shares the listener and its single model lifecycle; this is not a separate server.
extension ModelHTTPServer {
    func handleManaged(
        head: HTTPRequestHead, body: Data, channel: Channel,
        manager: ModelLifecycleManager
    ) async {
        do {
            if let route = InspectorRecordingRoute.parse(uri: head.uri) {
                try route.validate(method: head.method)
                if case .session = route {
                    try await handleRecording(route: route, head: head, body: body, channel: channel)
                    return
                }
            }
            // Stored responses belong to the listener, so retrieval and deletion
            // continue to work while a model is unloaded or being replaced.
            if let route = ResponsesAPIRoute.parse(uri: head.uri) {
                guard route.allows(method: head.method.rawValue) else {
                    throw ModelHTTPError(
                        status: .methodNotAllowed,
                        message: "Method is not allowed for this Responses route.", code: "method_not_allowed")
                }
                if route != .create {
                    try await handleResponses(
                        route: route, head: head, body: body,
                        channel: channel, requestID: "stored-response")
                    return
                }
            }
            switch (head.method, head.uri) {
            case (.POST, "/v1/runtime/reset-cache"):
                try Self.validateControlRequest(head: head, remoteAddress: channel.remoteAddress)
                guard body.count <= 32 * 1024 else {
                    throw ModelHTTPError(
                        status: .payloadTooLarge,
                        message: "Cache reset requests may not exceed 32 KiB.", code: "request_too_large")
                }
                guard let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
                    object.isEmpty
                else {
                    throw ModelHTTPError(
                        status: .badRequest,
                        message: "Cache reset expects an empty JSON object.", code: "invalid_parameter")
                }
                try await manager.withModel { model in
                    guard let runner = model.runner else {
                        throw ModelHTTPError(
                            status: .badRequest,
                            message: "KV-cache reset is supported only for a loaded text model.",
                            code: "unsupported_model")
                    }
                    do { try await runner.resetKVCache() } catch LocalModelRunnerError.busy {
                        throw ModelHTTPError(
                            status: .conflict,
                            message: "The runner is busy. Retry after generation finishes.", code: "model_busy")
                    }
                }
                struct CacheReset: Encodable { let reset = true }
                try await sendJSON(CacheReset(), on: channel)
            case (.GET, "/v1/runtime/downloads"):
                try Self.validateControlRequest(head: head, remoteAddress: channel.remoteAddress)
                try await sendJSON(try await manager.downloadedModels(), on: channel)
            case (.POST, "/v1/runtime/remove"):
                try Self.validateControlRequest(head: head, remoteAddress: channel.remoteAddress)
                guard body.count <= 32 * 1024 else {
                    throw ModelHTTPError(
                        status: .payloadTooLarge,
                        message: "Model control requests may not exceed 32 KiB.", code: "request_too_large")
                }
                guard let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
                    object.count == 1, let name = object["model"] as? String
                else {
                    throw ModelHTTPError(
                        status: .badRequest,
                        message: "Remove expects a JSON object containing only a model folder name.",
                        code: "invalid_model_name")
                }
                try await sendJSON(try await manager.removeDownload(named: name), on: channel)
            case (.GET, "/v1/runtime"), (.GET, "/v1/inspector/runtime"):
                try await sendJSON(await manager.state(), on: channel)
            case (.POST, "/v1/runtime/load"), (.POST, "/v1/runtime/unload"):
                try Self.validateControlRequest(head: head, remoteAddress: channel.remoteAddress)
                guard body.count <= 32 * 1024 else {
                    throw ModelHTTPError(
                        status: .payloadTooLarge,
                        message: "Model control requests may not exceed 32 KiB.", code: "request_too_large")
                }
                let state: ModelLifecycleState
                do {
                    if head.uri == "/v1/runtime/load" {
                        let request = try JSONDecoder().decode(ModelLoadRequest.self, from: body)
                        state = try await manager.load(request)
                    } else {
                        guard let object = try JSONSerialization.jsonObject(with: body) as? [String: Any],
                            Set(object.keys).isSubset(of: ["expectedGeneration", "expectedInstanceID"])
                        else {
                            throw ModelHTTPError(
                                status: .badRequest,
                                message:
                                    "Unload accepts only optional expectedGeneration and expectedInstanceID guards.",
                                code: "invalid_parameter")
                        }
                        let request = try JSONDecoder().decode(ModelUnloadRequest.self, from: body)
                        state = try await manager.unload(
                            expectedGeneration: request.expectedGeneration,
                            expectedInstanceID: request.expectedInstanceID)
                    }
                } catch let error as ModelLifecycleError {
                    throw error
                } catch let error as ModelHTTPError {
                    throw error
                } catch {
                    throw ModelHTTPError(
                        status: .badRequest, message: error.localizedDescription,
                        code: "invalid_model_selection")
                }
                try await sendJSON(state, status: .accepted, on: channel)
            case (.GET, "/v1/models"):
                struct Models: Encodable {
                    let object = "list"
                    let data: [ModelLifecycleDescriptor]
                }
                try await sendJSON(Models(data: await manager.availableModels()), on: channel)
            case (.GET, let uri) where uri.hasPrefix("/v1/models/"):
                let name = String(uri.dropFirst("/v1/models/".count)).removingPercentEncoding
                guard let descriptor = await manager.availableModels().first(where: { $0.id == name }) else {
                    throw ModelHTTPError(
                        status: .notFound, message: "The requested model is not available.",
                        param: "model", code: "model_not_found")
                }
                try await sendJSON(descriptor, on: channel)
            case (_, let uri) where uri == "/v1/runtime" || uri.hasPrefix("/v1/runtime/"):
                throw ModelHTTPError(status: .notFound, message: "Route not found")
            default:
                try await serveManagedModelRoute(
                    head: head, body: body, channel: channel, manager: manager
                )
            }
        } catch let error as ModelRemovalError {
            let status: HTTPResponseStatus
            switch error {
            case .notFound: status = .notFound
            case .inUse: status = .conflict
            case .cleanupFailed: status = .internalServerError
            default: status = .badRequest
            }
            try? await sendJSON(
                OpenAIErrorEnvelope(
                    message: error.localizedDescription,
                    type: "invalid_request_error", param: "model", code: error.code),
                status: status, on: channel)
        } catch let error as ModelLifecycleError {
            let status: HTTPResponseStatus
            let code: String
            switch error {
            case .busy:
                status = .conflict
                code = "model_transition_in_progress"
            case .staleSelection:
                status = .conflict
                code = "stale_model_selection"
            case .unavailable:
                status = .serviceUnavailable
                code = "model_unavailable"
            case .disabledModel:
                status = .serviceUnavailable
                code = "model_unavailable"
            case .unknownModel:
                status = .notFound
                code = "model_not_found"
            case .loadFailed:
                status = .serviceUnavailable
                code = "model_load_failed"
            }
            try? await sendJSON(
                OpenAIErrorEnvelope(
                    message: error.localizedDescription,
                    type: "server_error", code: code), status: status, on: channel)
        } catch let error as ModelHTTPError {
            try? await sendJSON(
                OpenAIErrorEnvelope(
                    message: error.message, type: error.type,
                    param: error.param, code: error.code), status: error.status, on: channel)
        } catch {
            try? await sendJSON(
                OpenAIErrorEnvelope(
                    message: error.localizedDescription,
                    type: "server_error", code: "internal_error"), status: .internalServerError, on: channel)
        }
    }

    private func serveManagedModelRoute(
        head: HTTPRequestHead, body: Data, channel: Channel, manager: ModelLifecycleManager
    ) async throws {
        let isModelRoute =
            AudioAPIRoute.parse(uri: head.uri) != nil
            || ResponsesAPIRoute.parse(uri: head.uri) == .create
            || (head.method == .GET && head.uri == "/v1/inspector/model")
            || (head.method == .GET && head.uri == "/v1/vision/status")
            || (head.method == .POST
                && ["/v1/inspector/trace", "/v1/inspector/recordings", "/v1/chat/completions", "/v1/decisions"]
                    .contains(head.uri))
        guard isModelRoute else {
            throw ModelHTTPError(status: .notFound, message: "Route not found")
        }
        // The lease spans decoding/preflight, generation, and response
        // completion. Model metadata never changes inside a response.
        let verbose = self.verbose
        var expectedGeneration: UInt64?
        if head.method == .POST && ["/v1/inspector/trace", "/v1/inspector/recordings"].contains(head.uri) {
            guard body.count <= 32 * 1024 else {
                throw ModelHTTPError(
                    status: .payloadTooLarge,
                    message: "Inspector requests may not exceed 32 KiB.", code: "request_too_large")
            }
            // Preserve the empty-runtime response and let the dedicated
            // handler report malformed inspection parameters.
            if head.uri == "/v1/inspector/recordings" {
                expectedGeneration =
                    (try? JSONDecoder().decode(InspectorRecordingRequest.self, from: body))?.runtimeGeneration
            } else {
                expectedGeneration =
                    (try? JSONDecoder().decode(InspectorTraceRequest.self, from: body))?.runtimeGeneration
            }
        }
        let serve: @Sendable (LoadedModel) async throws -> Void = { model in
            let inspectionGeneration =
                ["/v1/inspector/model", "/v1/inspector/recordings"].contains(head.uri)
                ? await manager.state().modelGeneration : nil
            let handler = ModelHTTPServer(
                runner: model.runner,
                servedModelName: model.servedModelName, tokenLimit: model.tokenLimit,
                verbose: verbose, speechSynthesizer: model.speechSynthesizer,
                vision: model.vision,
                modelCreated: model.created, modelCard: model.modelCard,
                inspectorRuntimeGeneration: inspectionGeneration,
                responsesStore: self.responsesStore, recordingStore: self.recordingStore)
            await handler.handle(head: head, body: body, channel: channel)
        }
        if head.method == .GET, AudioAPIRoute.parse(uri: head.uri) == .voices,
            let selected = URLComponents(string: head.uri)?.queryItems?.first(where: { $0.name == "model" })?
                .value,
            !selected.isEmpty
        {
            try await manager.withRequestedModel(selected, serve)
        } else if head.method == .POST,
            ["/v1/chat/completions", "/v1/responses", "/v1/audio/speech", "/v1/decisions"].contains(head.uri),
            let selected = try Self.requestedModel(path: head.uri, body: body)
        {
            try await manager.withRequestedModel(selected, serve)
        } else {
            try await manager.withModel(expectedGeneration: expectedGeneration, serve)
        }
    }

    static func requestedModel(path: String, body: Data) throws -> String? {
        if path == "/v1/decisions", body.count > 64 * 1024 {
            throw ModelHTTPError(
                status: .payloadTooLarge, message: "Decision requests may not exceed 64 KiB.", code: "request_too_large"
            )
        }
        do {
            // Requests without an explicit model preserve current-model behavior.
            if let object = try JSONSerialization.jsonObject(with: body) as? [String: Any],
                object["model"] == nil
            {
                return nil
            }
            let model: String?
            switch path {
            case "/v1/chat/completions":
                // Vision bodies are forwarded intact and are not decodable as
                // the text-only ChatCompletionRequest. Inspect only routing here.
                guard let object = try JSONSerialization.jsonObject(with: body) as? [String: Any],
                    let selected = object["model"] as? String
                else {
                    throw ModelHTTPError(
                        status: .badRequest, message: "model must be a string", code: "invalid_parameter")
                }
                model = selected
            case "/v1/decisions": model = try DecisionRequest.decode(body).model
            case "/v1/responses": model = try decodeResponsesRequest(body).model
            case "/v1/audio/speech":
                switch try AudioSpeechRequest.decode(from: body) {
                case .openAI(let request): model = request.model
                case .mistral(let request): model = request.model
                }
            default: return nil
            }
            if let model, model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                throw ModelHTTPError(status: .badRequest, message: "model must not be empty", code: "invalid_parameter")
            }
            return model
        } catch let error as ModelHTTPError {
            throw error
        } catch {
            throw ModelHTTPError(status: .badRequest, message: error.localizedDescription, code: "invalid_request")
        }
    }

    static func validateControlRequest(head: HTTPRequestHead, remoteAddress: SocketAddress?) throws {
        let address = remoteAddress?.ipAddress ?? ""
        guard address == "127.0.0.1" || address == "::1" || address == "::ffff:127.0.0.1",
            head.headers["origin"].isEmpty,
            !head.headers["sec-fetch-site"].contains(where: { $0.lowercased() == "cross-site" })
        else {
            throw ModelHTTPError(
                status: .forbidden,
                message: "Model control is available to local native clients only.", code: "control_forbidden")
        }
        guard head.headers["content-type"].count == 1,
            head.headers.first(name: "content-type")?.split(separator: ";").first?
                .trimmingCharacters(in: .whitespaces).lowercased() == "application/json"
        else {
            throw ModelHTTPError(
                status: .unsupportedMediaType,
                message: "Model control requires Content-Type: application/json.", code: "invalid_content_type")
        }
    }
}
