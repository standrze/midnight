import Foundation
import ModelRunnerCore
import ModelRunnerProtocol
import NIOCore
import NIOHTTP1

extension ModelHTTPServer {
    func handleResponses(
        route: ResponsesAPIRoute, head: HTTPRequestHead, body: Data,
        channel: Channel, requestID: String
    ) async throws {
        guard route.allows(method: head.method.rawValue) else {
            throw ModelHTTPError(
                status: .methodNotAllowed,
                message: "Method is not allowed for this Responses route.", code: "method_not_allowed")
        }
        let query = URLComponents(string: head.uri)?.queryItems ?? []
        switch route {
        case .create:
            _ = try Self.responsesQuery(query, allowed: [])
            try await createResponse(body: body, channel: channel, requestID: requestID)
        case .response(let id):
            let options = try Self.responsesQuery(query, allowed: head.method == .GET ? ["stream"] : [])
            if let stream = options["stream"], stream != "false" {
                throw Self.responsesParameterError("Stored response streaming is not supported.", "stream")
            }
            if head.method == .DELETE {
                guard await responsesStore.remove(id: id) else {
                    throw Self.responseNotFound(id)
                }
                try await sendJSON(
                    OpenAIJSONValue.object([
                        "id": .string(id), "object": .string("response.deleted"), "deleted": .bool(true),
                    ]), on: channel)
            } else {
                guard let entry = await responsesStore.get(id: id) else {
                    throw Self.responseNotFound(id)
                }
                try await sendJSON(entry.response, on: channel)
            }
        case .inputItems(let id):
            guard let entry = await responsesStore.get(id: id) else {
                throw Self.responseNotFound(id)
            }
            try await sendJSON(Self.responsesInputPage(entry.inputItems, query: query), on: channel)
        }
    }

    private func createResponse(body: Data, channel: Channel, requestID: String) async throws {
        let request = try Self.decodeResponsesRequest(body)
        guard request.model == servedModelName else {
            throw ModelHTTPError(
                status: .notFound,
                message: "The model '\(request.model)' does not exist or is not loaded.",
                param: "model", code: "model_not_found")
        }
        let history = try await responsesHistory(for: request)
        var inputIDs = Set<String>()
        for item in history.inputItems {
            if case .object(let object) = item, case .string(let id) = object["id"],
                !inputIDs.insert(id).inserted
            {
                throw Self.responsesParameterError(
                    "Input item IDs must be unique across the continued conversation.", "input")
            }
        }
        var messages = history.messages
        if let instructions = request.instructions {
            messages.insert(.init(role: "system", content: instructions), at: 0)
        }
        guard !messages.isEmpty else {
            throw Self.responsesParameterError("Provide input, instructions, or a previous_response_id.", "input")
        }
        try Self.validateResponsesToolHistory(messages)
        var maximumTokens: Int
        do { maximumTokens = try tokenLimit.resolve(requested: request.maxOutputTokens) } catch {
            throw Self.responsesParameterError(error.localizedDescription, "max_output_tokens")
        }
        let plan: ToolChoicePlan
        do { plan = try ToolChoicePlan.resolve(choice: request.toolChoice, tools: request.tools) } catch {
            throw Self.responsesParameterError(error.localizedDescription, "tool_choice")
        }
        do { try StructuredOutputRequest.validate(format: request.responseFormat, tools: plan.tools, stop: []) } catch {
            throw Self.responsesParameterError(error.localizedDescription, "text.format")
        }
        guard let runner else {
            throw ModelHTTPError(
                status: .notImplemented,
                message: "The loaded model does not provide text responses.",
                param: "model", code: "unsupported_model_feature")
        }
        let prepared: PreparedModelPrompt
        do {
            prepared = try await runner.preparePrompt(
                messages: messages, maximumTokens: maximumTokens,
                tools: plan.tools, toolChoice: plan.constraint, reasoningEffort: request.reasoningEffort,
                responseFormat: request.responseFormat, thinkingEnabled: request.includeReasoning)
        } catch let error as StructuredOutputRequestError {
            throw ModelHTTPError(
                status: .unprocessableEntity, message: error.localizedDescription,
                param: "text.format", code: "unsupported_model_feature")
        } catch let error as RequestAdmissionError {
            throw ModelHTTPError(
                status: .badRequest, message: error.localizedDescription,
                param: "input", code: "request_exceeds_limits")
        } catch let error as LocalModelRunnerError {
            switch error {
            case .busy:
                throw ModelHTTPError(status: .conflict, message: error.localizedDescription, code: "model_busy")
            case .unsupportedForcedToolChoiceFormat, .unsupportedForcedToolChoiceModel,
                .invalidForcedToolChoiceTokens, .emptyForcedToolChoicePrefix:
                throw ModelHTTPError(
                    status: .unprocessableEntity, message: error.localizedDescription,
                    param: "tool_choice", code: "unsupported_model_feature")
            default:
                throw Self.responsesParameterError(error.localizedDescription, "input")
            }
        } catch {
            throw Self.responsesParameterError(error.localizedDescription, "input")
        }

        maximumTokens = try tokenLimit.resolve(
            requested: maximumTokens,
            promptTokens: prepared.promptTokenCount, contextLength: runner.contextLength)
        var fields = request.responseFields()
        fields["temperature"] = .number(request.temperature ?? 1)
        fields["top_p"] = .number(request.topP ?? 0.95)
        fields["max_output_tokens"] = .integer(maximumTokens)
        let id = "resp_" + UUID().uuidString.replacingOccurrences(of: "-", with: "")
        var output = ResponsesOutput(
            id: id, model: servedModelName,
            createdAt: Int(Date().timeIntervalSince1970), fields: fields)
        if request.stream {
            try await beginEventStream(on: channel)
        }
        do {
            if request.stream {
                try await writeResponsesEvents(output.started(), on: channel)
            }
            let events = await runner.stream(
                messages: messages, maximumTokens: maximumTokens,
                temperature: request.temperature, topP: request.topP, tools: plan.tools,
                toolChoice: plan.constraint, reasoningEffort: request.reasoningEffort,
                thinkingEnabled: request.includeReasoning, preparedPrompt: prepared,
                responseFormat: request.responseFormat)
            var metrics: LocalModelRunnerMetrics?
            var calls = Set<String>()
            let allowedTools = Set(plan.tools?.map { $0.function.name } ?? [])
            for try await event in events {
                try Task.checkCancellation()
                let wireEvents: [OpenAIJSONValue]
                switch event {
                case .reasoning(let text):
                    wireEvents = request.includeReasoning ? output.appendReasoning(text) : []
                case .content(let text): wireEvents = output.appendText(text)
                case .toolCall(let call):
                    guard allowedTools.contains(call.function.name),
                        calls.insert(call.id).inserted,
                        request.parallelToolCalls || calls.count == 1
                    else {
                        throw ModelHTTPError(
                            status: .internalServerError,
                            message:
                                "The model generated a tool call that violates this request's tools or parallel_tool_calls setting.",
                            type: "server_error", code: "invalid_tool_call")
                    }
                    wireEvents = output.appendToolCall(call)
                case .metrics(let value):
                    metrics = value
                    logGeneration(value, requestID: requestID)
                    wireEvents = []
                }
                if request.stream {
                    try await writeResponsesEvents(wireEvents, on: channel)
                }
            }
            try Task.checkCancellation()
            guard let metrics else {
                throw ModelHTTPError(
                    status: .internalServerError,
                    message: "Generation completed without usage information.",
                    type: "server_error", code: "missing_usage")
            }
            // Stage completion until storage succeeds: never emit completed then failed.
            var completed = output
            let finalEvents = completed.finish(
                inputTokens: metrics.promptTokenCount,
                cachedInputTokens: metrics.cachedPromptTokenCount,
                outputTokens: metrics.generationTokenCount, stopReason: metrics.stopReason)
            if request.store {
                let entry = ResponsesStore.Entry(
                    response: completed.response,
                    messages: history.messages + completed.assistantMessages,
                    inputItems: history.inputItems, model: servedModelName)
                guard await responsesStore.store(id: id, entry: entry) else {
                    throw ModelHTTPError(
                        status: .insufficientStorage,
                        message:
                            "This response exceeds the local history store's capacity. Retry with store:false or a shorter input.",
                        type: "server_error", code: "response_storage_limit")
                }
            }
            if request.stream {
                try await writeResponsesEvents(finalEvents, on: channel)
                try? await finishStream(on: channel)
            } else {
                try await sendJSON(completed.response, on: channel)
            }
        } catch {
            if request.stream {
                if !Task.isCancelled {
                    let message = (error as? ModelHTTPError)?.message ?? error.localizedDescription
                    // Response.error.code is an OpenAI enum, unlike HTTP error codes.
                    try? await writeResponsesEvents(output.fail(message: message, code: "server_error"), on: channel)
                }
                try? await finishStream(on: channel)
            } else {
                if let error = error as? ModelHTTPError {
                    throw error
                }
                if let error = error as? LocalModelRunnerError, case .busy = error {
                    throw ModelHTTPError(status: .conflict, message: error.localizedDescription, code: "model_busy")
                }
                throw ModelHTTPError(
                    status: .internalServerError,
                    message: error.localizedDescription, type: "server_error", code: "generation_failed")
            }
        }
    }

    private func responsesHistory(for request: ResponsesRequest) async throws
        -> (messages: [OpenAIMessage], inputItems: [OpenAIJSONValue])
    {
        guard let previous = request.previousResponseID else {
            return (request.inputMessages, request.inputItems)
        }
        guard let entry = await responsesStore.get(id: previous) else {
            throw Self.responseNotFound(previous, param: "previous_response_id")
        }
        guard entry.model == request.model else {
            throw Self.responsesParameterError(
                "The previous response belongs to a different model.", "previous_response_id")
        }
        let previousOutput: [OpenAIJSONValue]
        if case .object(let object) = entry.response, case .array(let items) = object["output"] {
            previousOutput = items
        } else {
            previousOutput = []
        }
        return (
            entry.messages + request.inputMessages,
            entry.inputItems + previousOutput + request.inputItems
        )
    }

    private func writeResponsesEvents(_ events: [OpenAIJSONValue], on channel: Channel) async throws {
        for event in events {
            guard case .object(let object) = event, case .string(let name) = object["type"] else {
                continue
            }
            try await writeNamedEvent(name, value: event, on: channel)
        }
    }

    static func decodeResponsesRequest(_ body: Data) throws -> ResponsesRequest {
        do { return try JSONDecoder().decode(ResponsesRequest.self, from: body) } catch let error
            as ResponsesRequestError
        {
            throw responsesParameterError(error.message, error.param)
        } catch {
            let issue = chatDecodingIssue(error)
            throw ModelHTTPError(
                status: .badRequest, message: "Invalid JSON request: \(issue.message)",
                param: issue.param, code: "invalid_json")
        }
    }

    static func validateResponsesToolHistory(_ messages: [OpenAIMessage]) throws {
        var pending = Set<String>()
        var seen = Set<String>()
        for message in messages {
            if message.role == "tool" {
                guard let id = message.toolCallID, pending.remove(id) != nil else {
                    throw responsesParameterError(
                        "Each function_call_output must answer one preceding function_call exactly once.", "input")
                }
            } else {
                guard pending.isEmpty || message.role == "assistant" else {
                    throw responsesParameterError(
                        "Provide function_call_output for every pending function_call before continuing the conversation.",
                        "input")
                }
                for call in message.toolCalls ?? [] {
                    guard !call.id.isEmpty, seen.insert(call.id).inserted else {
                        throw responsesParameterError(
                            "Function call IDs must be nonempty and unique in the conversation.", "input")
                    }
                    pending.insert(call.id)
                }
            }
        }
        guard pending.isEmpty else {
            throw responsesParameterError("A function_call is missing its function_call_output.", "input")
        }
    }

    static func responsesInputPage(_ items: [OpenAIJSONValue], query: [URLQueryItem]) throws -> OpenAIJSONValue {
        let options = try responsesQuery(query, allowed: ["limit", "order", "after"])
        let limit: Int
        if let raw = options["limit"] {
            guard let value = Int(raw), (1...100).contains(value) else {
                throw responsesParameterError("limit must be an integer between 1 and 100.", "limit")
            }
            limit = value
        } else {
            limit = 20
        }
        let order = options["order"] ?? "desc"
        guard order == "asc" || order == "desc" else {
            throw responsesParameterError("order must be asc or desc.", "order")
        }
        let ordered = order == "asc" ? items : Array(items.reversed())
        func itemID(_ item: OpenAIJSONValue) -> OpenAIJSONValue {
            if case .object(let object) = item {
                return object["id"] ?? .null
            }
            return .null
        }
        var start = 0
        if let after = options["after"] {
            guard let index = ordered.firstIndex(where: { itemID($0) == .string(after) }) else {
                throw responsesParameterError("The after item does not exist in this response's inputs.", "after")
            }
            start = index + 1
        }
        let end = min(ordered.count, start + limit)
        let page = Array(ordered[start..<end])
        return .object([
            "object": .string("list"), "data": .array(page),
            "first_id": page.first.map(itemID) ?? .null, "last_id": page.last.map(itemID) ?? .null,
            "has_more": .bool(end < ordered.count),
        ])
    }

    private static func responsesQuery(_ items: [URLQueryItem], allowed: Set<String>) throws -> [String: String] {
        var values: [String: String] = [:]
        for item in items {
            guard allowed.contains(item.name), values[item.name] == nil,
                let value = item.value, !value.isEmpty
            else {
                throw responsesParameterError("Unknown, repeated, or empty query parameter '\(item.name)'.", item.name)
            }
            values[item.name] = value
        }
        return values
    }

    private static func responsesParameterError(_ message: String, _ param: String) -> ModelHTTPError {
        ModelHTTPError(status: .badRequest, message: message, param: param, code: "invalid_parameter")
    }

    private static func responseNotFound(_ id: String, param: String = "response_id") -> ModelHTTPError {
        ModelHTTPError(
            status: .notFound,
            message: "Response '\(id)' is not stored, has expired, or was deleted.", param: param,
            code: "response_not_found")
    }
}
