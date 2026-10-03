import Foundation
import ModelRunnerCore
import ModelRunnerProtocol
import NIOCore
import NIOHTTP1

// Shares the listener and its single model lifecycle; this is not a separate server.
extension ModelHTTPServer {
    func handleAudio(
        route: AudioAPIRoute,
        head: HTTPRequestHead,
        body: Data,
        channel: Channel,
        requestID: String
    ) async throws {
        switch route {
        case .speech:
            try await handleSpeech(body: body, channel: channel, requestID: requestID)
        case .voices where head.method == .GET:
            try await handleVoiceList(uri: head.uri, channel: channel)
        case .voices where head.method == .POST:
            if requestUsesMultipartFormData(head) {
                throw AudioHTTPError(
                    status: .notImplemented,
                    message:
                        "OpenAI custom voice creation is recognized, but this local checkpoint does not include the encoder needed to create a voice from audio.",
                    type: "invalid_request_error",
                    param: "audio_sample",
                    code: "unsupported_model_feature",
                    style: .openAI
                )
            }
            try await handleCreateVoice(body: body)
        case .voice(let voiceID) where head.method == .GET:
            try await sendJSON(try voiceResponse(id: voiceID), on: channel)
        case .voice(let voiceID) where head.method == .PATCH:
            try handleUpdateVoice(id: voiceID, body: body)
        case .voice(let voiceID) where head.method == .DELETE:
            try handleDeleteVoice(id: voiceID)
        case .voiceSample(let voiceID):
            try handleVoiceSample(id: voiceID)
        default:
            throw AudioHTTPError(
                status: .methodNotAllowed,
                message: "Method \(head.method.rawValue) is not allowed for this audio route.",
                code: "method_not_allowed"
            )
        }
    }

    private func handleSpeech(
        body: Data,
        channel: Channel,
        requestID: String
    ) async throws {
        let request: AudioSpeechRequest
        do {
            request = try AudioSpeechRequest.decode(from: body)
        } catch {
            let style: AudioHTTPError.Style = speechBodyLooksOpenAI(body) ? .openAI : .mistralValidation
            let issue = decodingIssue(error)
            throw AudioHTTPError(
                status: style == .openAI ? .badRequest : .unprocessableEntity,
                message: "Invalid speech request: \(issue.message)",
                param: issue.param,
                code: issue.code,
                style: style
            )
        }

        switch request {
        case .openAI(let openAI):
            try await handleOpenAISpeech(openAI, channel: channel, requestID: requestID)
        case .mistral(let mistral):
            try await handleMistralSpeech(mistral, channel: channel, requestID: requestID)
        }
    }

    private func handleOpenAISpeech(
        _ request: OpenAISpeechRequest,
        channel: Channel,
        requestID: String
    ) async throws {
        guard !request.input.isEmpty, request.input.count <= 4_096 else {
            throw AudioHTTPError(
                status: .badRequest,
                message: "input must contain between 1 and 4096 characters.",
                param: "input",
                code: "invalid_parameter",
                style: .openAI
            )
        }
        guard request.speed.isFinite, (0.25...4).contains(request.speed) else {
            throw AudioHTTPError(
                status: .badRequest,
                message: "speed must be between 0.25 and 4.0.",
                param: "speed",
                code: "invalid_parameter",
                style: .openAI
            )
        }
        if let instructions = request.instructions, instructions.count > 4_096 {
            throw AudioHTTPError(
                status: .badRequest,
                message: "instructions may not exceed 4096 characters.",
                param: "instructions",
                code: "invalid_parameter",
                style: .openAI
            )
        }
        let synthesizer = try requireSpeechSynthesizer(style: .openAI)
        guard request.model == synthesizer.servedModelName else {
            throw AudioHTTPError(
                status: .notFound,
                message: "The speech model '\(request.model)' is not loaded.",
                param: "model",
                code: "model_not_found",
                style: .openAI
            )
        }
        let requestedVoiceID = request.voice.value
        guard let voice = synthesizer.voiceCatalog.voice(id: requestedVoiceID) else {
            throw AudioHTTPError(
                status: .badRequest,
                message: "The voice '\(requestedVoiceID)' is not available.",
                param: "voice",
                code: "voice_not_found",
                style: .openAI
            )
        }
        guard let format = LocalSpeechAudioFormat(rawValue: request.responseFormat.rawValue) else {
            throw AudioHTTPError(
                status: .badRequest,
                message: "The requested response format is not supported.",
                param: "response_format",
                code: "unsupported_format",
                style: .openAI
            )
        }
        guard synthesizer.supportedAudioFormats.contains(format) else {
            throw AudioHTTPError(
                status: .badRequest,
                message: "The loaded speech model does not support \(format.rawValue) output.",
                param: "response_format",
                code: "unsupported_format",
                style: .openAI
            )
        }
        guard synthesizer.supportedSpeedRange.contains(request.speed) else {
            throw AudioHTTPError(
                status: .badRequest,
                message: "The loaded speech model currently supports speed 1.0 only.",
                param: "speed",
                code: "unsupported_model_feature",
                style: .openAI
            )
        }
        guard request.instructions == nil || synthesizer.supportsInstructions else {
            throw AudioHTTPError(
                status: .badRequest,
                message: "The loaded speech model does not support instructions.",
                param: "instructions",
                code: "unsupported_model_feature",
                style: .openAI
            )
        }
        log(
            requestID,
            "speech protocol=openai model=\(request.model) voice=\(voice.id) "
                + "format=\(format.rawValue) characters=\(request.input.count)"
        )
        let synthesis = LocalSpeechSynthesisRequest(
            input: request.input,
            voiceID: voice.id,
            format: format,
            pcmEncoding: .signedInt16LittleEndian,
            instructions: request.instructions,
            speed: request.speed
        )
        switch request.streamFormat {
        case .audio:
            try await streamOpenAIAudio(
                synthesizer: synthesizer,
                request: synthesis,
                channel: channel,
                requestID: requestID
            )
        case .sse:
            try await streamOpenAISSE(
                synthesizer: synthesizer,
                request: synthesis,
                channel: channel,
                requestID: requestID
            )
        }
    }

    private func handleMistralSpeech(
        _ request: MistralSpeechRequest,
        channel: Channel,
        requestID: String
    ) async throws {
        guard !request.input.isEmpty, request.input.count <= 4_096 else {
            throw mistralValidation(
                "input must contain between 1 and 4096 characters.",
                param: "input")
        }
        let synthesizer = try requireSpeechSynthesizer(style: .mistral)
        if request.refAudio != nil, !synthesizer.supportsReferenceAudio {
            throw mistralValidation(
                "The loaded model does not support reference audio.", param: "ref_audio",
                code: "unsupported_model_feature")
        }
        let referenceAudio = try request.refAudio.map { try decodeReferenceAudio($0) }
        if referenceAudio != nil, synthesizer.requiresReferenceText,
            request.refText?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false
        {
            throw mistralValidation("ref_text is required with ref_audio.", param: "ref_text")
        }
        if referenceAudio == nil, request.refText != nil {
            throw mistralValidation("ref_text requires ref_audio.", param: "ref_text")
        }
        if let model = request.model, model != synthesizer.servedModelName {
            throw AudioHTTPError(
                status: .notFound,
                message: "The speech model '\(model)' is not loaded.",
                param: "model",
                code: "model_not_found"
            )
        }
        guard let requestedVoiceID = request.voiceID ?? synthesizer.voiceCatalog.voices.first?.id else {
            throw mistralValidation("No preset voice is available.", param: "voice_id")
        }
        guard let voice = synthesizer.voiceCatalog.voice(id: requestedVoiceID) else {
            throw mistralValidation(
                "The voice '\(requestedVoiceID)' is not available.",
                param: "voice_id",
                code: "voice_not_found"
            )
        }
        let requestedFormat = request.responseFormat ?? .mp3
        guard let format = LocalSpeechAudioFormat(rawValue: requestedFormat.rawValue) else {
            throw mistralValidation(
                "The response format '\(requestedFormat.rawValue)' is not supported.",
                param: "response_format",
                code: "unsupported_format"
            )
        }
        guard synthesizer.supportedAudioFormats.contains(format) else {
            throw mistralValidation(
                "The loaded speech model does not support \(format.rawValue) output.",
                param: "response_format",
                code: "unsupported_format"
            )
        }
        log(
            requestID,
            "speech protocol=mistral model=\(synthesizer.servedModelName) voice=\(voice.id) "
                + "format=\(format.rawValue) characters=\(request.input.count) stream=\(request.stream)"
        )
        let synthesis = LocalSpeechSynthesisRequest(
            input: request.input,
            voiceID: voice.id,
            format: format,
            pcmEncoding: .float32LittleEndian,
            referenceAudio: referenceAudio,
            referenceText: request.refText
        )
        if request.stream {
            try await streamMistralAudio(
                synthesizer: synthesizer,
                request: synthesis,
                channel: channel,
                requestID: requestID
            )
        } else {
            let audio: Data
            let usage: LocalSpeechUsage
            do {
                (audio, usage) = try await collectSpeech(
                    synthesizer: synthesizer,
                    request: synthesis
                )
            } catch {
                throw speechGenerationError(error, style: .mistral)
            }
            try await sendJSON(
                MistralSpeechResponse(audioData: audio.base64EncodedString()),
                on: channel
            )
            log(
                requestID,
                "speech-complete bytes=\(audio.count) prompt_tokens=\(usage.promptTokens) "
                    + "completion_tokens=\(usage.completionTokens)"
            )
        }
    }

    private func streamMistralAudio(
        synthesizer: any LocalSpeechSynthesizing,
        request: LocalSpeechSynthesisRequest,
        channel: Channel,
        requestID: String
    ) async throws {
        let events = await synthesizer.stream(request: request)
        var iterator = events.makeAsyncIterator()
        let firstEvent: LocalSpeechSynthesisEvent
        do {
            guard let event = try await iterator.next() else {
                throw AudioHTTPError(
                    status: .internalServerError,
                    message: "Speech generation ended before producing audio or usage information.",
                    type: "server_error",
                    code: "missing_usage"
                )
            }
            firstEvent = event
        } catch {
            throw speechGenerationError(error, style: .mistral)
        }

        try await beginEventStream(on: channel)
        var usage: LocalSpeechUsage?
        var bytes = 0
        var nextEvent: LocalSpeechSynthesisEvent? = firstEvent
        do {
            while let event = nextEvent {
                switch event {
                case .audio(let data):
                    bytes += data.count
                    try await writeNamedEvent(
                        "speech.audio.delta",
                        value: MistralSpeechAudioDeltaEvent(
                            audioData: data.base64EncodedString()
                        ),
                        on: channel
                    )
                case .completed(let completedUsage):
                    usage = completedUsage
                }
                nextEvent = try await iterator.next()
            }
        } catch {
            log(requestID, "speech-generation-failed error=\(error.localizedDescription)")
            // The official stream has no error-event variant. Closing before
            // the terminating chunk makes a partial stream visibly fail.
            try? await channel.close().get()
            return
        }
        guard let usage else {
            log(requestID, "speech-generation-failed error=missing_usage")
            try? await channel.close().get()
            return
        }
        do {
            try await writeNamedEvent(
                "speech.audio.done",
                value: MistralSpeechAudioDoneEvent(
                    usage: MistralUsageInfo(
                        promptTokens: usage.promptTokens,
                        totalTokens: usage.totalTokens,
                        completionTokens: usage.completionTokens
                    )
                ),
                on: channel
            )
            log(
                requestID,
                "speech-complete bytes=\(bytes) prompt_tokens=\(usage.promptTokens) "
                    + "completion_tokens=\(usage.completionTokens)"
            )
        } catch {
            try? await channel.close().get()
            return
        }
        try await finishStream(on: channel)
    }

    private func streamOpenAIAudio(
        synthesizer: any LocalSpeechSynthesizing,
        request: LocalSpeechSynthesisRequest,
        channel: Channel,
        requestID: String
    ) async throws {
        let events = await synthesizer.stream(request: request)
        var iterator = events.makeAsyncIterator()
        let firstEvent: LocalSpeechSynthesisEvent
        do {
            guard let event = try await iterator.next() else {
                throw AudioHTTPError(
                    status: .internalServerError,
                    message: "Speech generation ended before producing audio or usage information.",
                    type: "server_error",
                    code: "missing_usage",
                    style: .openAI
                )
            }
            firstEvent = event
        } catch {
            throw speechGenerationError(error, style: .openAI)
        }

        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: contentType(for: request.format))
        headers.add(name: "transfer-encoding", value: "chunked")
        headers.add(name: "connection", value: "close")
        try await channel.writeAndFlush(
            HTTPServerResponsePart.head(.init(version: .http1_1, status: .ok, headers: headers))
        ).get()
        var bytes = 0
        var usage: LocalSpeechUsage?
        var nextEvent: LocalSpeechSynthesisEvent? = firstEvent
        do {
            while let event = nextEvent {
                switch event {
                case .audio(let data):
                    bytes += data.count
                    try await writeBody(data, on: channel)
                case .completed(let completedUsage):
                    usage = completedUsage
                    log(
                        requestID,
                        "speech-usage prompt_tokens=\(completedUsage.promptTokens) "
                            + "completion_tokens=\(completedUsage.completionTokens)"
                    )
                }
                nextEvent = try await iterator.next()
            }
        } catch {
            log(requestID, "speech-generation-failed error=\(error.localizedDescription)")
            try? await channel.close().get()
            return
        }
        guard usage != nil else {
            log(requestID, "speech-generation-failed error=missing_usage")
            try? await channel.close().get()
            return
        }
        log(requestID, "speech-complete bytes=\(bytes)")
        try await finishStream(on: channel)
    }

    private func streamOpenAISSE(
        synthesizer: any LocalSpeechSynthesizing,
        request: LocalSpeechSynthesisRequest,
        channel: Channel,
        requestID: String
    ) async throws {
        let events = await synthesizer.stream(request: request)
        var iterator = events.makeAsyncIterator()
        let firstEvent: LocalSpeechSynthesisEvent
        do {
            guard let event = try await iterator.next() else {
                throw AudioHTTPError(
                    status: .internalServerError,
                    message: "Speech generation ended before producing audio or usage information.",
                    type: "server_error",
                    code: "missing_usage",
                    style: .openAI
                )
            }
            firstEvent = event
        } catch {
            throw speechGenerationError(error, style: .openAI)
        }

        try await beginEventStream(on: channel)
        var bytes = 0
        var usage: LocalSpeechUsage?
        var nextEvent: LocalSpeechSynthesisEvent? = firstEvent
        do {
            while let event = nextEvent {
                switch event {
                case .audio(let data):
                    bytes += data.count
                    try await writeEvent(
                        OpenAISpeechAudioDeltaEvent(audio: data.base64EncodedString()),
                        on: channel
                    )
                case .completed(let completedUsage):
                    usage = completedUsage
                }
                nextEvent = try await iterator.next()
            }
        } catch {
            log(requestID, "speech-generation-failed error=\(error.localizedDescription)")
            try? await channel.close().get()
            return
        }
        guard let usage else {
            log(requestID, "speech-generation-failed error=missing_usage")
            try? await channel.close().get()
            return
        }
        try await writeEvent(
            OpenAISpeechAudioDoneEvent(
                usage: OpenAISpeechUsage(
                    inputTokens: usage.promptTokens,
                    outputTokens: usage.completionTokens,
                    totalTokens: usage.totalTokens
                )
            ),
            on: channel
        )
        log(
            requestID,
            "speech-complete bytes=\(bytes) prompt_tokens=\(usage.promptTokens) "
                + "completion_tokens=\(usage.completionTokens)"
        )
        try await finishStream(on: channel)
    }

    private func collectSpeech(
        synthesizer: any LocalSpeechSynthesizing,
        request: LocalSpeechSynthesisRequest
    ) async throws -> (Data, LocalSpeechUsage) {
        var audio = Data()
        var usage: LocalSpeechUsage?
        let events = await synthesizer.stream(request: request)
        for try await event in events {
            switch event {
            case .audio(let data): audio.append(data)
            case .completed(let completedUsage): usage = completedUsage
            }
        }
        guard let usage else {
            throw AudioHTTPError(
                status: .internalServerError,
                message: "Speech generation completed without usage information.",
                code: "missing_usage"
            )
        }
        return (audio, usage)
    }

    private func handleVoiceList(uri: String, channel: Channel) async throws {
        let query = try VoiceListQuery(items: AudioAPIRoute.queryItems(uri: uri))
        let allVoices = query.type == .custom ? [] : (voiceCatalog?.voices ?? [])
        let selected = Array(allVoices.dropFirst(query.offset).prefix(query.limit))
        let total = allVoices.count
        let totalPages = total == 0 ? 0 : (total + query.limit - 1) / query.limit
        let page = query.offset / query.limit + 1
        try await sendJSON(
            VoiceListResponse(
                items: selected.map(voiceResponse),
                total: total,
                page: page,
                pageSize: query.limit,
                totalPages: totalPages
            ),
            on: channel
        )
    }

    private func handleCreateVoice(body: Data) async throws {
        let request: VoiceCreateRequest = try decodeMistralBody(body)
        guard !request.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw mistralValidation("name must not be empty.", param: "name")
        }
        guard let sampleAudio = Data(base64Encoded: request.sampleAudio), !sampleAudio.isEmpty else {
            throw mistralValidation(
                "sample_audio must contain non-empty valid base64.",
                param: "sample_audio"
            )
        }
        throw mistralValidation(
            "Custom voice creation is unavailable because the open Voxtral checkpoint does not include audio encoder weights.",
            param: "sample_audio",
            code: "unsupported_model_feature"
        )
    }

    private func handleUpdateVoice(id: String, body: Data) throws {
        let request: VoiceUpdateRequest = try decodeMistralBody(body)
        if case .value(let description) = request.description, description.count > 500 {
            throw mistralValidation(
                "description may not exceed 500 characters.",
                param: "description"
            )
        }
        guard voiceCatalog?.voice(id: id) != nil else {
            throw voiceNotFound(id)
        }
        throw mistralValidation(
            "Bundled preset voices cannot be updated.",
            param: "voice_id",
            code: "preset_voice_read_only"
        )
    }

    private func handleDeleteVoice(id: String) throws {
        guard voiceCatalog?.voice(id: id) != nil else {
            throw voiceNotFound(id)
        }
        throw mistralValidation(
            "Bundled preset voices cannot be deleted.",
            param: "voice_id",
            code: "preset_voice_read_only"
        )
    }

    private func handleVoiceSample(id: String) throws {
        guard voiceCatalog?.voice(id: id) != nil else {
            throw voiceNotFound(id)
        }
        throw AudioHTTPError(
            status: .notFound,
            message: "The open checkpoint contains an embedding for voice '\(id)', but no original WAV sample.",
            param: "voice_id",
            code: "voice_sample_not_found"
        )
    }

    private func voiceResponse(id: String) throws -> VoiceResponse {
        guard let voice = voiceCatalog?.voice(id: id) else {
            throw voiceNotFound(id)
        }
        return voiceResponse(voice)
    }

    private func voiceResponse(_ voice: VoxtralPresetVoice) -> VoiceResponse {
        VoiceResponse(
            name: voice.name,
            id: voice.apiID,
            createdAt: voiceCatalog?.createdAt ?? Date(timeIntervalSince1970: 0),
            userID: nil,
            slug: voice.id,
            languages: voice.languages,
            gender: voice.gender,
            tags: ["preset"],
            description: "Voice selection for the loaded speech model"
        )
    }

    private func requireSpeechSynthesizer(
        style: AudioHTTPError.Style
    ) throws -> any LocalSpeechSynthesizing {
        guard let speechSynthesizer else {
            throw AudioHTTPError(
                status: .notImplemented,
                message: "The speech API is installed, but the native Voxtral MLX generator is not connected yet.",
                code: "speech_engine_unavailable",
                style: style
            )
        }
        return speechSynthesizer
    }

    private func speechGenerationError(
        _ error: Error,
        style: AudioHTTPError.Style
    ) -> AudioHTTPError {
        if let error = error as? AudioHTTPError {
            return error
        }
        return AudioHTTPError(
            status: .internalServerError,
            message: error.localizedDescription,
            type: "server_error",
            code: "generation_failed",
            style: style
        )
    }

    private func decodeMistralBody<Value: Decodable>(_ body: Data) throws -> Value {
        do { return try JSONDecoder().decode(Value.self, from: body) } catch {
            let issue = decodingIssue(error)
            throw mistralValidation(
                "Invalid JSON request: \(issue.message)",
                param: issue.param,
                code: issue.code
            )
        }
    }

    private func decodingIssue(_ error: Error) -> (message: String, param: String?, code: String) {
        switch error {
        case DecodingError.keyNotFound(let key, _):
            return ("Field '\(key.stringValue)' is required.", key.stringValue, "missing")
        case DecodingError.valueNotFound(_, let context):
            return (
                context.debugDescription,
                context.codingPath.last?.stringValue,
                "value_error"
            )
        case DecodingError.typeMismatch(_, let context):
            return (
                context.debugDescription,
                context.codingPath.last?.stringValue,
                "type_error"
            )
        case DecodingError.dataCorrupted(let context):
            return (
                context.debugDescription,
                context.codingPath.last?.stringValue,
                "value_error"
            )
        case AudioSpeechRequestError.mixedProtocols:
            return (error.localizedDescription, "voice", "invalid_request")
        case AudioSpeechRequestError.expectedObject:
            return (error.localizedDescription, nil, "invalid_request")
        default:
            return (error.localizedDescription, nil, "invalid_json")
        }
    }

    private func mistralValidation(
        _ message: String,
        param: String?,
        code: String = "invalid_parameter"
    ) -> AudioHTTPError {
        AudioHTTPError(
            status: .unprocessableEntity,
            message: message,
            param: param,
            code: code,
            style: .mistralValidation
        )
    }

    private func voiceNotFound(_ id: String) -> AudioHTTPError {
        AudioHTTPError(
            status: .notFound,
            message: "The voice '\(id)' does not exist.",
            param: "voice_id",
            code: "voice_not_found"
        )
    }

    func speechBodyLooksOpenAI(_ body: Data) -> Bool {
        guard let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any]
        else {
            return false
        }
        return object["voice"] != nil
    }

    func requestUsesMultipartFormData(_ head: HTTPRequestHead) -> Bool {
        guard let contentType = head.headers.first(name: "content-type") else {
            return false
        }
        return contentType.split(separator: ";", maxSplits: 1).first?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() == "multipart/form-data"
    }

    private func contentType(for format: LocalSpeechAudioFormat) -> String {
        switch format {
        case .mp3: "audio/mpeg"
        case .opus: "audio/ogg"
        case .aac: "audio/aac"
        case .flac: "audio/flac"
        case .wav: "audio/wav"
        case .pcm: "application/octet-stream"
        }
    }

    func sendAudioError(_ error: AudioHTTPError, on channel: Channel) async throws {
        switch error.style {
        case .openAI:
            try await sendJSON(
                OpenAIErrorEnvelope(
                    message: error.message,
                    type: error.type,
                    param: error.param,
                    code: error.code
                ),
                status: error.status,
                on: channel
            )
        case .mistralValidation:
            let location: [ValidationLocation] =
                error.param.map {
                    [.string(error.location), .string($0)]
                } ?? [.string(error.location)]
            try await sendJSON(
                HTTPValidationError(
                    detail: [
                        ValidationErrorDetail(
                            loc: location,
                            msg: error.message,
                            type: error.code ?? "value_error"
                        )
                    ]
                ),
                status: error.status,
                on: channel
            )
        case .mistral:
            try await sendJSON(
                MistralError(
                    message: error.message,
                    type: error.type,
                    param: error.param,
                    code: error.code
                ),
                status: error.status,
                on: channel
            )
        }
    }
}
struct AudioHTTPError: Error {
    enum Style: Equatable {
        case openAI
        case mistralValidation
        case mistral
    }

    let status: HTTPResponseStatus
    let message: String
    let type: String
    let param: String?
    let code: String?
    let style: Style
    let location: String

    init(
        status: HTTPResponseStatus,
        message: String,
        type: String = "invalid_request_error",
        param: String? = nil,
        code: String? = nil,
        style: Style = .mistral,
        location: String = "body"
    ) {
        self.status = status
        self.message = message
        self.type = type
        self.param = param
        self.code = code
        self.style = style
        self.location = location
    }
}

private func decodeReferenceAudio(_ encoded: String) throws -> Data {
    guard encoded.utf8.count <= 12 * 1024 * 1024 else {
        throw AudioHTTPError(
            status: .unprocessableEntity, message: "Reference audio is too large.", param: "ref_audio",
            style: .mistralValidation)
    }
    let payload =
        encoded.hasPrefix("data:")
        ? String(encoded.split(separator: ",", maxSplits: 1).last ?? "")
        : encoded
    guard let audio = Data(base64Encoded: payload), !audio.isEmpty else {
        throw AudioHTTPError(
            status: .unprocessableEntity,
            message: "ref_audio must be nonempty base64 audio data (a data URL is also accepted).",
            param: "ref_audio",
            code: "invalid_reference_audio",
            style: .mistralValidation
        )
    }
    return audio
}

private struct VoiceListQuery {
    enum VoiceType: String {
        case all, custom, preset
    }

    let limit: Int
    let offset: Int
    let type: VoiceType

    init(items: [URLQueryItem]) throws {
        func value(named name: String) throws -> String? {
            let values = items.filter { $0.name == name }
            guard values.count <= 1 else {
                throw AudioHTTPError(
                    status: .unprocessableEntity,
                    message: "Query parameter '\(name)' may only be provided once.",
                    param: name,
                    code: "invalid_query_parameter",
                    style: .mistralValidation,
                    location: "query"
                )
            }
            guard let item = values.first else {
                return nil
            }
            guard let value = item.value, !value.isEmpty else {
                throw AudioHTTPError(
                    status: .unprocessableEntity,
                    message: "Query parameter '\(name)' requires a value.",
                    param: name,
                    code: "invalid_query_parameter",
                    style: .mistralValidation,
                    location: "query"
                )
            }
            return value
        }

        let rawLimit = try value(named: "limit")
        let limit: Int
        if let rawLimit, let parsed = Int(rawLimit) {
            limit = parsed
        } else if rawLimit == nil {
            limit = 10
        } else {
            throw AudioHTTPError(
                status: .unprocessableEntity,
                message: "limit must be an integer between 1 and 100.",
                param: "limit",
                code: "invalid_query_parameter",
                style: .mistralValidation,
                location: "query"
            )
        }
        guard (1...100).contains(limit) else {
            throw AudioHTTPError(
                status: .unprocessableEntity,
                message: "limit must be an integer between 1 and 100.",
                param: "limit",
                code: "invalid_query_parameter",
                style: .mistralValidation,
                location: "query"
            )
        }
        let rawOffset = try value(named: "offset")
        let offset: Int
        if let rawOffset, let parsed = Int(rawOffset) {
            offset = parsed
        } else if rawOffset == nil {
            offset = 0
        } else {
            throw AudioHTTPError(
                status: .unprocessableEntity,
                message: "offset must be a non-negative integer.",
                param: "offset",
                code: "invalid_query_parameter",
                style: .mistralValidation,
                location: "query"
            )
        }
        guard offset >= 0 else {
            throw AudioHTTPError(
                status: .unprocessableEntity,
                message: "offset must be a non-negative integer.",
                param: "offset",
                code: "invalid_query_parameter",
                style: .mistralValidation,
                location: "query"
            )
        }
        guard offset / limit < Int.max else {
            throw AudioHTTPError(
                status: .unprocessableEntity,
                message: "offset is too large for the requested page size.",
                param: "offset",
                code: "invalid_query_parameter",
                style: .mistralValidation,
                location: "query"
            )
        }
        let typeValue = try value(named: "type") ?? "all"
        guard let type = VoiceType(rawValue: typeValue) else {
            throw AudioHTTPError(
                status: .unprocessableEntity,
                message: "type must be one of: all, custom, preset.",
                param: "type",
                code: "invalid_query_parameter",
                style: .mistralValidation,
                location: "query"
            )
        }
        self.limit = limit
        self.offset = offset
        self.type = type
    }
}
