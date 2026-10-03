import Foundation
import ModelRunnerCore
import ModelRunnerProtocol
import NIOCore
import NIOHTTP1
import NIOPosix

final class ModelHTTPServer: @unchecked Sendable {
    private let manager: ModelLifecycleManager?
    let authentication: APIKeyAuthentication?
    let runner: LocalModelRunner?
    let tokenLimit: GenerationTokenLimit
    let servedModelName: String
    let responsesStore: ResponsesStore
    let recordingStore: InspectorRecordingStore
    let modelCard: ModelCard?
    private let modelCreated: Int
    let verbose: Bool
    let speechSynthesizer: (any LocalSpeechSynthesizing)?
    private let vision: (any VisionModelServing)?
    let voiceCatalog: VoxtralVoiceCatalog?
    let inspectorRuntimeGeneration: UInt64?

    init(
        runner: LocalModelRunner? = nil,
        servedModelName: String,
        tokenLimit: GenerationTokenLimit,
        verbose: Bool = false,
        speechSynthesizer: (any LocalSpeechSynthesizing)? = nil,
        vision: (any VisionModelServing)? = nil,
        voiceCatalog: VoxtralVoiceCatalog? = nil,
        modelCreated: Int = Int(Date().timeIntervalSince1970),
        modelCard: ModelCard? = nil,
        inspectorRuntimeGeneration: UInt64? = nil,
        manager: ModelLifecycleManager? = nil,
        authentication: APIKeyAuthentication? = nil,
        responsesStore: ResponsesStore = ResponsesStore(),
        recordingStore: InspectorRecordingStore = InspectorRecordingStore()
    ) {
        self.manager = manager
        self.authentication = authentication
        self.responsesStore = responsesStore
        self.recordingStore = recordingStore
        self.runner = runner
        self.tokenLimit = tokenLimit
        self.servedModelName = servedModelName
        self.modelCreated = modelCreated
        self.modelCard = (modelCard ?? ModelCard(name: servedModelName)).withCapabilities(
            .init(
                vision: vision != nil, audioInput: speechSynthesizer?.supportsReferenceAudio ?? false,
                audioOutput: speechSynthesizer != nil, decisions: runner?.supportsDecisions == true ? true : nil)
        ).withVoices(speechSynthesizer?.voiceCatalog.modelCardVoices)
        self.inspectorRuntimeGeneration = inspectorRuntimeGeneration
        self.verbose = verbose
        self.speechSynthesizer = speechSynthesizer
        self.vision = vision
        self.voiceCatalog = speechSynthesizer?.voiceCatalog ?? voiceCatalog
    }

    convenience init(
        manager: ModelLifecycleManager, verbose: Bool = false,
        authentication: APIKeyAuthentication? = nil,
        responsesStore: ResponsesStore = ResponsesStore(),
        recordingStore: InspectorRecordingStore = InspectorRecordingStore()
    ) {
        // Model handlers run on a request-scoped server with the selected
        // model's real limits. This listener only handles admission and control.
        self.init(
            servedModelName: "", tokenLimit: try! GenerationTokenLimit(configuredMaximum: 512),
            verbose: verbose, manager: manager, authentication: authentication, responsesStore: responsesStore,
            recordingStore: recordingStore)
    }

    func run(
        host: String, port: Int,
        onListening: (@Sendable () async -> Void)? = nil
    ) async throws {
        let group = MultiThreadedEventLoopGroup(numberOfThreads: System.coreCount)
        do {
            let channel = try await ServerBootstrap(group: group)
                .serverChannelOption(ChannelOptions.backlog, value: 256)
                .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
                .childChannelInitializer { channel in
                    Self.configureHTTPPipeline(on: channel).flatMap {
                        channel.pipeline.addHandler(ModelHTTPRequestHandler(server: self))
                    }
                }
                .childChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
                .bind(host: host, port: port)
                .get()

            try await withTaskCancellationHandler {
                await onListening?()
                try Task.checkCancellation()
                try await channel.closeFuture.get()
            } onCancel: {
                channel.close(promise: nil)
            }
            try await group.shutdownGracefully()
        } catch {
            try? await group.shutdownGracefully()
            throw error
        }
    }

    static func configureHTTPPipeline(on channel: Channel) -> EventLoopFuture<Void> {
        // Responses close the connection, and responseInFlight ignores extra requests.
        // Keep socket reads active while streaming so a disconnect cancels work even
        // during prefill or hidden reasoning, when no response chunks are written.
        channel.pipeline.configureHTTPServerPipeline(withPipeliningAssistance: false)
    }

    func handle(head: HTTPRequestHead, body: Data, channel: Channel) async {
        if let manager {
            await handleManaged(head: head, body: body, channel: channel, manager: manager)
            return
        }
        let requestID = String(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12))
        let started = ContinuousClock.now
        log(
            requestID,
            "incoming method=\(head.method.rawValue) uri=\(head.uri) "
                + "client=\(channel.remoteAddress?.description ?? "unknown") bytes=\(body.count)"
        )
        do {
            // Recording status belongs to the listener and must stay readable
            // during generation and when the selected runtime changes.
            if let route = InspectorRecordingRoute.parse(uri: head.uri) {
                try await handleRecording(route: route, head: head, body: body, channel: channel)
                return
            }
            // Discovery belongs to this listener, including for vision cards.
            // The optional worker owns image decoding and request validation;
            // dispatch before the text-only message decoder sees image parts.
            let isModelDiscovery =
                head.method == .GET
                && (head.uri == "/v1/models" || head.uri.hasPrefix("/v1/models/"))
            if let vision, !isModelDiscovery {
                try await handleVision(vision, head: head, body: body, channel: channel)
                return
            }
            if let route = ResponsesAPIRoute.parse(uri: head.uri) {
                try await handleResponses(
                    route: route, head: head, body: body,
                    channel: channel, requestID: requestID)
                return
            }
            if let audioRoute = AudioAPIRoute.parse(uri: head.uri) {
                guard audioRoute.allows(method: head.method.rawValue) else {
                    throw AudioHTTPError(
                        status: .methodNotAllowed,
                        message: "Method \(head.method.rawValue) is not allowed for this audio route.",
                        code: "method_not_allowed"
                    )
                }
                try await handleAudio(
                    route: audioRoute,
                    head: head,
                    body: body,
                    channel: channel,
                    requestID: requestID
                )
                log(
                    requestID,
                    "request-finished elapsed=\(formatDuration(started.duration(to: .now)))"
                )
                return
            }
            switch (head.method, head.uri) {
            case (.GET, "/v1/inspector/runtime"):
                try await sendJSON(
                    InspectorRuntimeIdentity(
                        instanceID: ProcessInfo.processInfo.environment["MIDNIGHT_CONTROL_INSTANCE"],
                        processID: ProcessInfo.processInfo.processIdentifier
                    ), on: channel
                )
            case (.GET, "/v1/inspector/model"):
                try await handleInspector(body: nil, channel: channel)
            case (.POST, "/v1/inspector/trace"):
                try await handleInspector(body: body, channel: channel)
            case (.GET, "/v1/models"):
                try await sendJSON(
                    ModelListResponse(models: [modelDescriptor()]),
                    on: channel
                )
            case (.GET, let uri) where uri.hasPrefix("/v1/models/"):
                let encodedName = String(uri.dropFirst("/v1/models/".count))
                guard !encodedName.isEmpty,
                    let modelName = encodedName.removingPercentEncoding,
                    modelName == servedModelName
                else {
                    throw ModelHTTPError(
                        status: .notFound,
                        message: "The model '\(encodedName)' does not exist.",
                        type: "invalid_request_error",
                        param: "model",
                        code: "model_not_found"
                    )
                }
                try await sendJSON(modelDescriptor(), on: channel)
            case (.POST, "/v1/decisions"):
                try await handleDecisions(body: body, channel: channel)
            case (.POST, "/v1/chat/completions"):
                try await handleChat(body: body, channel: channel, requestID: requestID)
            default:
                throw ModelHTTPError(status: .notFound, message: "Route not found")
            }
        } catch let error as AudioHTTPError {
            log(requestID, "rejected status=\(error.status.code) error=\(error.message)")
            try? await sendAudioError(error, on: channel)
        } catch let error as ModelHTTPError {
            log(requestID, "rejected status=\(error.status.code) error=\(error.message)")
            try? await sendJSON(
                OpenAIErrorEnvelope(
                    message: error.message,
                    type: error.type,
                    param: error.param,
                    code: error.code
                ),
                status: error.status,
                on: channel
            )
        } catch {
            log(requestID, "failed status=500 error=\(error.localizedDescription)")
            try? await sendJSON(
                OpenAIErrorEnvelope(
                    message: error.localizedDescription,
                    type: "server_error",
                    code: "internal_error"
                ),
                status: .internalServerError,
                on: channel
            )
        }
        log(requestID, "request-finished elapsed=\(formatDuration(started.duration(to: .now)))")
    }

    fileprivate func rejectPayloadTooLarge(head: HTTPRequestHead, channel: Channel) async {
        let message = "Request body exceeds the 32 MiB local server limit."
        if let route = AudioAPIRoute.parse(uri: head.uri), route != .speech {
            try? await sendJSON(
                MistralError(
                    message: message,
                    param: "body",
                    code: "request_too_large"
                ),
                status: .payloadTooLarge,
                on: channel
            )
        } else {
            try? await sendJSON(
                OpenAIErrorEnvelope(
                    message: message,
                    param: "body",
                    code: "request_too_large"
                ),
                status: .payloadTooLarge,
                on: channel
            )
        }
    }

    private func modelDescriptor() -> ModelResponse {
        ModelResponse(
            id: servedModelName, created: modelCreated,
            contextLength: runner?.contextLength, prefillStepSize: runner?.prefillStepSize,
            kvCompression: runner?.kvCompression, memoryLimitBytes: runner?.memoryLimitBytes,
            maximumOutputTokens: runner == nil ? nil : tokenLimit.configuredMaximum,
            defaultOutputTokens: runner == nil ? nil : tokenLimit.defaultTokens, modelCard: modelCard,
            nativeProtocol: nativeModelProtocol(at: runner?.modelPath))
    }

    static func chatDecodingIssue(_ error: Error) -> (message: String, param: String?) {
        let context: DecodingError.Context
        let codingPath: [any CodingKey]
        switch error {
        case DecodingError.keyNotFound(let key, let errorContext):
            context = errorContext
            codingPath = context.codingPath + [key]
        case DecodingError.valueNotFound(_, let errorContext),
            DecodingError.typeMismatch(_, let errorContext),
            DecodingError.dataCorrupted(let errorContext):
            context = errorContext
            codingPath = context.codingPath
        default:
            return (error.localizedDescription, nil)
        }

        let path = codingPath.reduce(into: "") { result, key in
            if let index = key.intValue {
                result += "[\(index)]"
            } else {
                result += (result.isEmpty ? "" : ".") + key.stringValue
            }
        }
        guard !path.isEmpty else {
            return (context.debugDescription, nil)
        }
        return ("\(path): \(context.debugDescription)", path)
    }

    struct ValidatedChatRequest {
        let completion: ChatCompletionRequest
        let requestedMaximumTokens: Int?
        let stop: [String]
        let toolChoicePlan: ToolChoicePlan
    }

    // Keep request decoding and validation independent of the NIO channel.
    func decodeAndValidateChatRequest(_ body: Data) throws -> ValidatedChatRequest {
        let completion: ChatCompletionRequest
        do {
            completion = try JSONDecoder().decode(ChatCompletionRequest.self, from: body)
        } catch {
            let issue = Self.chatDecodingIssue(error)
            throw ModelHTTPError(
                status: .badRequest,
                message: "Invalid JSON request: \(issue.message)",
                param: issue.param,
                code: "invalid_json"
            )
        }
        guard completion.model == servedModelName else {
            throw ModelHTTPError(
                status: .notFound,
                message: "The model '\(completion.model)' does not exist or is not loaded.",
                param: "model",
                code: "model_not_found"
            )
        }
        guard completion.maxTokens == nil || completion.maxCompletionTokens == nil else {
            throw ModelHTTPError(
                status: .badRequest,
                message: "Specify either max_tokens or max_completion_tokens, not both.",
                param: "max_completion_tokens",
                code: "invalid_parameter"
            )
        }
        let requestedMaximumTokens = completion.maxCompletionTokens ?? completion.maxTokens
        do {
            _ = try tokenLimit.resolve(requested: requestedMaximumTokens)
        } catch {
            throw ModelHTTPError(
                status: .badRequest,
                message: error.localizedDescription,
                param: completion.maxCompletionTokens == nil ? "max_tokens" : "max_completion_tokens",
                code: "invalid_parameter"
            )
        }
        if let temperature = completion.temperature,
            !temperature.isFinite || temperature < 0 || temperature > 2
        {
            throw ModelHTTPError(
                status: .badRequest,
                message: "temperature must be between zero and two.",
                param: "temperature",
                code: "invalid_parameter"
            )
        }
        if let topP = completion.topP,
            !topP.isFinite || topP <= 0 || topP > 1
        {
            throw ModelHTTPError(
                status: .badRequest,
                message: "top_p must be greater than zero and at most one.",
                param: "top_p",
                code: "invalid_parameter"
            )
        }
        let stop = completion.stop?.values ?? []
        guard stop.count <= 4, stop.allSatisfy({ !$0.isEmpty }) else {
            throw ModelHTTPError(
                status: .badRequest,
                message: "stop must contain between one and four non-empty strings.",
                param: "stop",
                code: "invalid_parameter"
            )
        }
        if completion.stream != true, completion.streamOptions != nil {
            throw ModelHTTPError(
                status: .badRequest,
                message: "stream_options may only be used when stream is true.",
                param: "stream_options",
                code: "invalid_parameter"
            )
        }
        let toolChoicePlan: ToolChoicePlan
        do {
            toolChoicePlan = try ToolChoicePlan.resolve(
                choice: completion.toolChoice,
                tools: completion.tools
            )
        } catch let error as ToolChoiceValidationError {
            throw ModelHTTPError(
                status: .badRequest,
                message: error.localizedDescription,
                param: "tool_choice",
                code: "invalid_parameter"
            )
        }
        do {
            try StructuredOutputRequest.validate(
                format: completion.responseFormat,
                tools: toolChoicePlan.tools, stop: stop)
        } catch {
            throw ModelHTTPError(
                status: .badRequest, message: error.localizedDescription,
                param: "response_format", code: "invalid_parameter")
        }
        if completion.nativeProtocol != nil && !stop.isEmpty {
            throw ModelHTTPError(
                status: .badRequest,
                message: "Native protocol output requires complete framing; custom stop strings are unsupported.",
                param: "stop", code: "invalid_parameter")
        }
        return ValidatedChatRequest(
            completion: completion,
            requestedMaximumTokens: requestedMaximumTokens,
            stop: stop,
            toolChoicePlan: toolChoicePlan
        )
    }

    private func handleChat(body: Data, channel: Channel, requestID: String) async throws {
        guard let runner else {
            throw ModelHTTPError(
                status: .notImplemented,
                message: "The loaded model does not provide chat completions.",
                type: "invalid_request_error",
                param: "model",
                code: "unsupported_model_feature"
            )
        }
        let request = try decodeAndValidateChatRequest(body)
        let completion = request.completion
        let requestedMaximumTokens = request.requestedMaximumTokens
        let stop = request.stop
        let toolChoicePlan = request.toolChoicePlan
        let preparedPrompt: PreparedModelPrompt
        do {
            preparedPrompt = try await runner.preparePrompt(
                messages: completion.messages,
                maximumTokens: requestedMaximumTokens, tools: toolChoicePlan.tools,
                toolChoice: toolChoicePlan.constraint,
                reasoningEffort: completion.reasoningEffort,
                responseFormat: completion.responseFormat, thinkingEnabled: completion.includeReasoning,
                nativeProtocol: completion.nativeProtocol)
        } catch let error as StructuredOutputRequestError {
            throw ModelHTTPError(
                status: .unprocessableEntity, message: error.localizedDescription,
                param: "response_format", code: "unsupported_model_feature")
        } catch let error as RequestAdmissionError {
            throw ModelHTTPError(
                status: .badRequest, message: error.localizedDescription,
                code: "request_exceeds_limits")
        } catch LocalModelRunnerError.unsupportedForcedToolChoiceFormat(let format) {
            throw ModelHTTPError(
                status: .unprocessableEntity,
                message: "The loaded model's '\(format)' tool-call format cannot enforce tool_choice.",
                param: "tool_choice",
                code: "unsupported_model_feature"
            )
        } catch LocalModelRunnerError.unsupportedForcedToolChoiceModel(let model) {
            throw ModelHTTPError(
                status: .unprocessableEntity,
                message: "The loaded model type '\(model)' cannot enforce tool_choice.",
                param: "tool_choice",
                code: "unsupported_model_feature"
            )
        } catch LocalModelRunnerError.invalidForcedToolChoiceTokens {
            throw ModelHTTPError(
                status: .unprocessableEntity,
                message: LocalModelRunnerError.invalidForcedToolChoiceTokens.localizedDescription,
                param: "tool_choice",
                code: "unsupported_model_feature"
            )
        } catch LocalModelRunnerError.emptyForcedToolChoicePrefix {
            throw ModelHTTPError(
                status: .unprocessableEntity,
                message: LocalModelRunnerError.emptyForcedToolChoicePrefix.localizedDescription,
                param: "tool_choice",
                code: "unsupported_model_feature"
            )
        } catch LocalModelRunnerError.busy {
            throw ModelHTTPError(
                status: .conflict, message: LocalModelRunnerError.busy.localizedDescription,
                code: "model_busy")
        } catch {
            throw ModelHTTPError(
                status: .badRequest, message: error.localizedDescription,
                code: "invalid_prompt")
        }
        let maximumTokensDescription = requestedMaximumTokens.map { String($0) } ?? "default"
        let temperatureDescription = completion.temperature.map { String($0) } ?? "default"
        let topPDescription = completion.topP.map { String($0) } ?? "default"
        let toolCount = completion.tools?.count ?? 0
        log(
            requestID,
            "chat model=\(completion.model) messages=\(completion.messages.count) "
                + "tools=\(toolCount) max_tokens=\(maximumTokensDescription) "
                + "temperature=\(temperatureDescription) top_p=\(topPDescription) "
                + "stop_sequences=\(stop.count) stream=\(completion.stream == true)"
        )

        if completion.stream == true {
            try await handleStreamingChat(
                completion,
                toolChoicePlan: toolChoicePlan,
                preparedPrompt: preparedPrompt,
                runner: runner,
                maximumTokens: requestedMaximumTokens,
                stop: stop,
                channel: channel,
                requestID: requestID
            )
        } else {
            try await handleNonStreamingChat(
                completion,
                toolChoicePlan: toolChoicePlan,
                preparedPrompt: preparedPrompt,
                runner: runner,
                maximumTokens: requestedMaximumTokens,
                stop: stop,
                channel: channel,
                requestID: requestID
            )
        }
    }

    private func handleStreamingChat(
        _ completion: ChatCompletionRequest,
        toolChoicePlan: ToolChoicePlan,
        preparedPrompt: PreparedModelPrompt,
        runner: LocalModelRunner,
        maximumTokens: Int?,
        stop: [String],
        channel: Channel,
        requestID: String
    ) async throws {
        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: "text/event-stream; charset=utf-8")
        headers.add(name: "cache-control", value: "no-cache")
        headers.add(name: "transfer-encoding", value: "chunked")
        headers.add(name: "connection", value: "close")
        try await channel.writeAndFlush(
            HTTPServerResponsePart.head(.init(version: .http1_1, status: .ok, headers: headers))
        ).get()

        let completionID = "chatcmpl-\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))"
        var contentChunks = 0
        var contentCharacters = 0
        do {
            try await writeEvent(
                ChatCompletionChunk(
                    id: completionID,
                    model: servedModelName,
                    choices: [.init(delta: .init(role: "assistant"))]
                ),
                on: channel
            )
            let chunks = await runner.stream(
                messages: completion.messages,
                maximumTokens: maximumTokens,
                temperature: completion.temperature,
                topP: completion.topP,
                stop: stop,
                tools: toolChoicePlan.tools,
                toolChoice: toolChoicePlan.constraint,
                reasoningEffort: completion.reasoningEffort,
                thinkingEnabled: completion.includeReasoning,
                preparedPrompt: preparedPrompt,
                responseFormat: completion.responseFormat
            )
            var finishReason = "stop"
            var toolCallIndex = 0
            var generationMetrics: LocalModelRunnerMetrics?
            for try await event in chunks {
                switch event {
                case .reasoning(let text):
                    if completion.includeReasoning == true {
                        try await writeEvent(
                            ChatCompletionChunk(
                                id: completionID,
                                model: servedModelName,
                                choices: [.init(delta: .init(reasoningContent: text))]), on: channel)
                    }
                case .content(let text):
                    contentChunks += 1
                    contentCharacters += text.count
                    try await writeEvent(
                        ChatCompletionChunk(
                            id: completionID,
                            model: servedModelName,
                            choices: [.init(delta: .init(content: text))]
                        ),
                        on: channel
                    )
                case .toolCall(let call):
                    finishReason = "tool_calls"
                    let delta = OpenAIToolCallDelta(
                        index: toolCallIndex,
                        id: call.id,
                        type: call.type,
                        function: .init(
                            name: call.function.name,
                            arguments: call.function.arguments
                        )
                    )
                    toolCallIndex += 1
                    try await writeEvent(
                        ChatCompletionChunk(
                            id: completionID,
                            model: servedModelName,
                            choices: [.init(delta: .init(toolCalls: [delta]))]
                        ),
                        on: channel
                    )
                case .metrics(let metrics):
                    generationMetrics = metrics
                    if finishReason != "tool_calls" {
                        finishReason = self.finishReason(for: metrics)
                    }
                    log(
                        requestID,
                        "generation prompt_tokens=\(metrics.promptTokenCount) "
                            + promptCacheLogSuffix(metrics)
                            + "prompt_tokens_per_second=\(formatRate(metrics.promptTokensPerSecond)) "
                            + "generated_tokens=\(metrics.generationTokenCount) "
                            + "tokens_per_second=\(formatRate(metrics.tokensPerSecond)) "
                            + "stop=\(metrics.stopReason)"
                            + speculativeLogSuffix(metrics)
                    )
                }
            }
            try await writeEvent(
                ChatCompletionChunk(
                    id: completionID,
                    model: servedModelName,
                    choices: [.init(delta: .init(), finishReason: finishReason)]
                ),
                on: channel
            )
            if let generationMetrics {
                try await writeEvent(
                    ChatCompletionChunk(
                        id: completionID,
                        model: servedModelName,
                        choices: [],
                        usage: completion.streamOptions?.includeUsage == true
                            ? ChatCompletionUsage(
                                promptTokens: generationMetrics.promptTokenCount,
                                completionTokens: generationMetrics.generationTokenCount,
                                cachedTokens: generationMetrics.cachedPromptTokenCount
                            )
                            : nil
                    ),
                    on: channel
                )
            }
            log(
                requestID,
                "chat-complete finish_reason=\(finishReason) chunks=\(contentChunks) "
                    + "characters=\(contentCharacters) tool_calls=\(toolCallIndex)"
            )
        } catch {
            log(requestID, "generation-failed error=\(error.localizedDescription)")
            try? await writeEvent(
                OpenAIErrorEnvelope(
                    message: error.localizedDescription,
                    type: "server_error",
                    code: "generation_failed"
                ),
                on: channel
            )
        }
        try await writeBody(Data("data: [DONE]\n\n".utf8), on: channel)
        try await channel.writeAndFlush(HTTPServerResponsePart.end(nil)).get()
        try? await channel.close().get()
    }

    private func handleNonStreamingChat(
        _ completion: ChatCompletionRequest,
        toolChoicePlan: ToolChoicePlan,
        preparedPrompt: PreparedModelPrompt,
        runner: LocalModelRunner,
        maximumTokens: Int?,
        stop: [String],
        channel: Channel,
        requestID: String
    ) async throws {
        let completionID = "chatcmpl-\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))"
        var content = ""
        var reasoning = ""
        var toolCalls: [OpenAIToolCall] = []
        var generationMetrics: LocalModelRunnerMetrics?
        let events = await runner.stream(
            messages: completion.messages,
            maximumTokens: maximumTokens,
            temperature: completion.temperature,
            topP: completion.topP,
            stop: stop,
            tools: toolChoicePlan.tools,
            toolChoice: toolChoicePlan.constraint,
            reasoningEffort: completion.reasoningEffort,
            thinkingEnabled: completion.includeReasoning,
            preparedPrompt: preparedPrompt,
            responseFormat: completion.responseFormat
        )
        do {
            for try await event in events {
                switch event {
                case .reasoning(let text):
                    if completion.includeReasoning == true {
                        reasoning += text
                    }
                case .content(let text): content += text
                case .toolCall(let call): toolCalls.append(call)
                case .metrics(let metrics):
                    generationMetrics = metrics
                    logGeneration(metrics, requestID: requestID)
                }
            }
        } catch let error as RequestAdmissionError {
            throw ModelHTTPError(
                status: .badRequest, message: error.localizedDescription,
                code: "request_exceeds_limits")
        } catch LocalModelRunnerError.busy {
            throw ModelHTTPError(
                status: .conflict,
                message: LocalModelRunnerError.busy.localizedDescription,
                code: "model_busy"
            )
        } catch {
            throw ModelHTTPError(
                status: .internalServerError,
                message: error.localizedDescription,
                type: "server_error",
                code: "generation_failed"
            )
        }
        guard let generationMetrics else {
            throw ModelHTTPError(
                status: .internalServerError,
                message: "Generation completed without usage information.",
                type: "server_error",
                code: "missing_usage"
            )
        }
        let finishReason = toolCalls.isEmpty ? finishReason(for: generationMetrics) : "tool_calls"
        let message = OpenAIMessage(
            role: "assistant",
            content: content.isEmpty ? nil : content,
            toolCalls: toolCalls.isEmpty ? nil : toolCalls,
            reasoningContent: reasoning.isEmpty ? nil : reasoning
        )
        try await sendJSON(
            ChatCompletionResponse(
                id: completionID,
                model: servedModelName,
                choices: [.init(message: message, finishReason: finishReason)],
                usage: ChatCompletionUsage(
                    promptTokens: generationMetrics.promptTokenCount,
                    completionTokens: generationMetrics.generationTokenCount,
                    cachedTokens: generationMetrics.cachedPromptTokenCount
                )
            ),
            on: channel
        )
        log(
            requestID,
            "chat-complete finish_reason=\(finishReason) characters=\(content.count) "
                + "tool_calls=\(toolCalls.count)"
        )
    }

    private func finishReason(for metrics: LocalModelRunnerMetrics) -> String {
        metrics.stopReason == "length" ? "length" : "stop"
    }

    func logGeneration(_ metrics: LocalModelRunnerMetrics, requestID: String) {
        log(
            requestID,
            "generation prompt_tokens=\(metrics.promptTokenCount) "
                + promptCacheLogSuffix(metrics)
                + "prompt_tokens_per_second=\(formatRate(metrics.promptTokensPerSecond)) "
                + "generated_tokens=\(metrics.generationTokenCount) "
                + "tokens_per_second=\(formatRate(metrics.tokensPerSecond)) "
                + "stop=\(metrics.stopReason)"
                + speculativeLogSuffix(metrics)
        )
    }

    private func speculativeLogSuffix(_ metrics: LocalModelRunnerMetrics) -> String {
        var suffix = ""
        if let proposed = metrics.proposedDraftTokens,
            let accepted = metrics.acceptedDraftTokens
        {
            suffix += " dflash_accepted=\(accepted)/\(proposed)"
        }
        if let reason = metrics.speculativePassthroughReason {
            suffix += " dflash_passthrough=\(reason)"
        }
        return suffix
    }

    private func promptCacheLogSuffix(_ metrics: LocalModelRunnerMetrics) -> String {
        guard metrics.cachedPromptTokenCount > 0 else {
            return ""
        }
        return "prompt_cached=\(metrics.cachedPromptTokenCount) "
            + "prompt_prefilled=\(metrics.prefilledPromptTokenCount) "
    }

    func sendJSON<Value: Encodable>(
        _ value: Value,
        status: HTTPResponseStatus = .ok,
        additionalHeaders: HTTPHeaders = HTTPHeaders(),
        on channel: Channel
    ) async throws {
        try await send(
            status: status,
            contentType: "application/json; charset=utf-8",
            data: try JSONEncoder().encode(value),
            additionalHeaders: additionalHeaders,
            on: channel
        )
    }

    func send(
        status: HTTPResponseStatus,
        contentType: String,
        data: Data,
        additionalHeaders: HTTPHeaders = HTTPHeaders(),
        on channel: Channel
    ) async throws {
        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: contentType)
        headers.add(name: "content-length", value: String(data.count))
        headers.add(name: "connection", value: "close")
        for (name, value) in additionalHeaders {
            headers.add(name: name, value: value)
        }
        try await channel.writeAndFlush(
            HTTPServerResponsePart.head(
                .init(version: .http1_1, status: status, headers: headers)
            )
        ).get()
        if !data.isEmpty {
            try await writeBody(data, on: channel)
        }
        try await channel.writeAndFlush(HTTPServerResponsePart.end(nil)).get()
        try? await channel.close().get()
    }

    func writeEvent<Value: Encodable>(_ value: Value, on channel: Channel) async throws {
        let data = try JSONEncoder().encode(value)
        var event = Data("data: ".utf8)
        event.append(data)
        event.append(Data("\n\n".utf8))
        try await writeBody(event, on: channel)
    }

    func beginEventStream(on channel: Channel) async throws {
        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: "text/event-stream; charset=utf-8")
        headers.add(name: "cache-control", value: "no-cache")
        headers.add(name: "transfer-encoding", value: "chunked")
        headers.add(name: "connection", value: "close")
        try await channel.writeAndFlush(
            HTTPServerResponsePart.head(.init(version: .http1_1, status: .ok, headers: headers))
        ).get()
    }

    func writeNamedEvent<Value: Encodable>(
        _ name: String,
        value: Value,
        on channel: Channel
    ) async throws {
        let data = try JSONEncoder().encode(value)
        var event = Data("event: \(name)\ndata: ".utf8)
        event.append(data)
        event.append(Data("\n\n".utf8))
        try await writeBody(event, on: channel)
    }

    func finishStream(on channel: Channel) async throws {
        try await channel.writeAndFlush(HTTPServerResponsePart.end(nil)).get()
        try? await channel.close().get()
    }

    func writeBody(_ data: Data, on channel: Channel) async throws {
        var buffer = channel.allocator.buffer(capacity: data.count)
        buffer.writeBytes(data)
        try await channel.writeAndFlush(
            HTTPServerResponsePart.body(.byteBuffer(buffer))
        ).get()
    }

    func log(_ requestID: String, _ message: @autoclosure () -> String) {
        guard verbose else {
            return
        }
        print("[verbose] request=\(requestID) \(message())")
    }

    func formatDuration(_ duration: Duration) -> String {
        let components = duration.components
        let milliseconds =
            Double(components.seconds) * 1_000
            + Double(components.attoseconds) / 1_000_000_000_000_000
        return String(format: "%.1fms", milliseconds)
    }

    func formatRate(_ rate: Double) -> String {
        guard rate.isFinite else {
            return "n/a"
        }
        return String(format: "%.2f", rate)
    }
}

final class ModelHTTPRequestHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = HTTPServerRequestPart
    typealias ResponseAction =
        @Sendable (
            HTTPRequestHead, Data, Bool, Channel
        ) async -> Void

    private let responseAction: ResponseAction
    private let authentication: APIKeyAuthentication?
    private let maximumRequestBodyBytes = 32 * 1_024 * 1_024
    private var requestHead: HTTPRequestHead?
    private var body = Data()
    private var bodyExceededLimit = false
    private var authenticationRejected = false
    private var responseInFlight = false
    private var responseTask: Task<Void, Never>?

    init(server: ModelHTTPServer) {
        authentication = server.authentication
        responseAction = { head, body, exceededLimit, channel in
            if let authentication = server.authentication, !authentication.accepts(head.headers) {
                var headers = HTTPHeaders()
                headers.add(name: "www-authenticate", value: "Bearer realm=\"Midnight\"")
                try? await server.sendJSON(
                    OpenAIErrorEnvelope(
                        message: "A valid API key is required.",
                        type: "authentication_error",
                        code: "invalid_api_key"
                    ),
                    status: .unauthorized,
                    additionalHeaders: headers,
                    on: channel
                )
            } else if exceededLimit {
                await server.rejectPayloadTooLarge(head: head, channel: channel)
            } else {
                await server.handle(head: head, body: body, channel: channel)
            }
        }
    }

    init(responseAction: @escaping ResponseAction) {
        authentication = nil
        self.responseAction = responseAction
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        // Every response advertises `Connection: close`. Ignore a request that
        // NIO's pipelining helper releases while the first response is being
        // flushed, so it cannot start a second response task on this channel.
        guard !responseInFlight else {
            return
        }
        switch unwrapInboundIn(data) {
        case .head(let head):
            requestHead = head
            body.removeAll(keepingCapacity: true)
            authenticationRejected = authentication.map { !$0.accepts(head.headers) } ?? false
            bodyExceededLimit =
                head.headers.first(name: "content-length")
                .flatMap(Int.init)
                .map { $0 > maximumRequestBodyBytes } ?? false
        case .body(var buffer):
            let incomingBytes = buffer.readableBytes
            if authenticationRejected {
                buffer.moveReaderIndex(forwardBy: incomingBytes)
                return
            }
            guard !bodyExceededLimit, body.count <= maximumRequestBodyBytes - incomingBytes else {
                bodyExceededLimit = true
                body.removeAll(keepingCapacity: false)
                buffer.moveReaderIndex(forwardBy: incomingBytes)
                return
            }
            if let bytes = buffer.readBytes(length: incomingBytes) {
                body.append(contentsOf: bytes)
            }
        case .end:
            guard let head = requestHead else {
                return
            }
            responseInFlight = true
            let requestBody = body
            let exceededLimit = bodyExceededLimit
            requestHead = nil
            body.removeAll(keepingCapacity: false)
            bodyExceededLimit = false
            authenticationRejected = false
            let channel = context.channel
            let responseAction = responseAction
            responseTask = Task {
                guard !Task.isCancelled else {
                    return
                }
                await responseAction(head, requestBody, exceededLimit, channel)
            }
        }
    }

    func channelInactive(context: ChannelHandlerContext) {
        cancelResponse()
        context.fireChannelInactive()
    }

    func errorCaught(context: ChannelHandlerContext, error: Error) {
        cancelResponse()
        context.close(promise: nil)
    }

    private func cancelResponse() {
        responseTask?.cancel()
        responseTask = nil
    }
}

private struct ModelListResponse: Encodable {
    let object = "list"
    let data: [ModelResponse]

    init(models: [ModelResponse]) {
        data = models
    }
}

private struct InspectorRuntimeIdentity: Encodable {
    let instanceID: String?
    let processID: Int32
}

private struct ModelResponse: Encodable {
    let id: String
    let created: Int
    let object = "model"
    let ownedBy = "midnight"
    let contextLength: Int?
    let prefillStepSize: Int?
    let kvCompression: String?
    let memoryLimitBytes: Int?
    let maximumOutputTokens: Int?
    let defaultOutputTokens: Int?

    var modelCard: ModelCard? = nil
    var nativeProtocol: String? = nil

    enum CodingKeys: String, CodingKey {
        case id, created, object
        case modelCard = "model_card"
        case nativeProtocol = "native_protocol"
        case ownedBy = "owned_by"
        case contextLength = "context_length"
        case prefillStepSize = "prefill_step_size"
        case kvCompression = "kv_compression"
        case memoryLimitBytes = "memory_limit_bytes"
        case maximumOutputTokens = "max_output_tokens"
        case defaultOutputTokens = "default_output_tokens"
    }
}

struct ModelHTTPError: Error {
    let status: HTTPResponseStatus
    let message: String
    let type: String
    let param: String?
    let code: String?

    init(
        status: HTTPResponseStatus,
        message: String,
        type: String = "invalid_request_error",
        param: String? = nil,
        code: String? = nil
    ) {
        self.status = status
        self.message = message
        self.type = type
        self.param = param
        self.code = code
    }
}
