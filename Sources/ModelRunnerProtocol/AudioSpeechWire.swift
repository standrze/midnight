import Foundation

public enum OpenAISpeechVoice: Codable, Equatable, Sendable {
    case name(String)
    case id(String)

    public var value: String {
        switch self {
        case .name(let value), .id(let value): value
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let name = try? container.decode(String.self) {
            self = .name(name)
            return
        }
        self = .id(try container.decode(VoiceID.self).id)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .name(let name): try container.encode(name)
        case .id(let id): try container.encode(VoiceID(id: id))
        }
    }

    private struct VoiceID: Codable {
        let id: String
    }
}

public enum OpenAISpeechResponseFormat: String, Codable, CaseIterable, Sendable {
    case mp3, opus, aac, flac, wav, pcm
}

public enum OpenAISpeechStreamFormat: String, Codable, CaseIterable, Sendable {
    case sse, audio
}

public struct OpenAISpeechRequest: Codable, Equatable, Sendable {
    public let model: String
    public let input: String
    public let voice: OpenAISpeechVoice
    public let instructions: String?
    public let responseFormat: OpenAISpeechResponseFormat
    public let speed: Double
    public let streamFormat: OpenAISpeechStreamFormat

    public init(
        model: String,
        input: String,
        voice: OpenAISpeechVoice,
        instructions: String? = nil,
        responseFormat: OpenAISpeechResponseFormat = .mp3,
        speed: Double = 1,
        streamFormat: OpenAISpeechStreamFormat = .audio
    ) {
        self.model = model
        self.input = input
        self.voice = voice
        self.instructions = instructions
        self.responseFormat = responseFormat
        self.speed = speed
        self.streamFormat = streamFormat
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decode(String.self, forKey: .model)
        input = try container.decode(String.self, forKey: .input)
        voice = try container.decode(OpenAISpeechVoice.self, forKey: .voice)
        instructions = try container.decodeNonNullIfPresent(String.self, forKey: .instructions)
        responseFormat = try container.decodeNonNullIfPresent(
            OpenAISpeechResponseFormat.self,
            forKey: .responseFormat
        ) ?? .mp3
        speed = try container.decodeNonNullIfPresent(Double.self, forKey: .speed) ?? 1
        streamFormat = try container.decodeNonNullIfPresent(
            OpenAISpeechStreamFormat.self,
            forKey: .streamFormat
        ) ?? .audio
    }

    enum CodingKeys: String, CodingKey {
        case model, input, voice, instructions, speed
        case responseFormat = "response_format"
        case streamFormat = "stream_format"
    }
}

public struct OpenAISpeechUsage: Codable, Equatable, Sendable {
    public let inputTokens: Int
    public let outputTokens: Int
    public let totalTokens: Int

    public init(inputTokens: Int, outputTokens: Int, totalTokens: Int) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.totalTokens = totalTokens
    }

    enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case totalTokens = "total_tokens"
    }
}

public struct OpenAISpeechAudioDeltaEvent: Codable, Equatable, Sendable {
    public let type: String
    public let audio: String

    public init(audio: String) {
        self.type = "speech.audio.delta"
        self.audio = audio
    }
}

public struct OpenAISpeechAudioDoneEvent: Codable, Equatable, Sendable {
    public let type: String
    public let usage: OpenAISpeechUsage

    public init(usage: OpenAISpeechUsage) {
        self.type = "speech.audio.done"
        self.usage = usage
    }
}

public enum MistralSpeechOutputFormat: String, Codable, CaseIterable, Sendable {
    case pcm, wav, mp3, flac, opus
}

public struct MistralSpeechRequest: Codable, Equatable, Sendable {
    public let input: String
    public let model: String?
    public let metadata: [String: OpenAIJSONValue]?
    public let stream: Bool
    public let promptCacheKey: String?
    public let voiceID: String?
    public let refAudio: String?
    public let responseFormat: MistralSpeechOutputFormat?

    public init(
        input: String,
        model: String? = nil,
        metadata: [String: OpenAIJSONValue]? = nil,
        stream: Bool = false,
        promptCacheKey: String? = nil,
        voiceID: String? = nil,
        refAudio: String? = nil,
        responseFormat: MistralSpeechOutputFormat? = nil
    ) {
        self.input = input
        self.model = model
        self.metadata = metadata
        self.stream = stream
        self.promptCacheKey = promptCacheKey
        self.voiceID = voiceID
        self.refAudio = refAudio
        self.responseFormat = responseFormat
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        input = try container.decode(String.self, forKey: .input)
        model = try container.decodeIfPresent(String.self, forKey: .model)
        metadata = try container.decodeIfPresent(
            [String: OpenAIJSONValue].self,
            forKey: .metadata
        )
        stream = try container.decodeNonNullIfPresent(Bool.self, forKey: .stream) ?? false
        promptCacheKey = try container.decodeIfPresent(String.self, forKey: .promptCacheKey)
        voiceID = try container.decodeIfPresent(String.self, forKey: .voiceID)
        refAudio = try container.decodeIfPresent(String.self, forKey: .refAudio)
        responseFormat = try container.decodeNonNullIfPresent(
            MistralSpeechOutputFormat.self,
            forKey: .responseFormat
        )
    }

    enum CodingKeys: String, CodingKey {
        case input, model, metadata, stream
        case promptCacheKey = "prompt_cache_key"
        case voiceID = "voice_id"
        case refAudio = "ref_audio"
        case responseFormat = "response_format"
    }
}

public enum AudioSpeechRequest: Equatable, Sendable {
    case openAI(OpenAISpeechRequest)
    case mistral(MistralSpeechRequest)

    public static func decode(from data: Data, decoder: JSONDecoder = JSONDecoder()) throws -> Self {
        let object = try JSONSerialization.jsonObject(with: data)
        guard let dictionary = object as? [String: Any] else {
            throw AudioSpeechRequestError.expectedObject
        }
        let keys = Set(dictionary.keys)
        // `voice` is required by OpenAI and absent from Mistral's schema. The
        // remaining OpenAI-only names are not safe discriminators because
        // Mistral explicitly allows additional request properties.
        let openAIKeys: Set<String> = ["voice"]
        let mistralKeys: Set<String> = [
            "voice_id", "ref_audio", "metadata", "prompt_cache_key", "stream",
        ]
        let hasOpenAIKeys = !keys.isDisjoint(with: openAIKeys)
        let hasMistralKeys = !keys.isDisjoint(with: mistralKeys)
        guard !(hasOpenAIKeys && hasMistralKeys) else {
            throw AudioSpeechRequestError.mixedProtocols
        }
        if hasOpenAIKeys {
            return .openAI(try decoder.decode(OpenAISpeechRequest.self, from: data))
        }
        return .mistral(try decoder.decode(MistralSpeechRequest.self, from: data))
    }
}

public enum AudioSpeechRequestError: LocalizedError, Equatable {
    case expectedObject
    case mixedProtocols

    public var errorDescription: String? {
        switch self {
        case .expectedObject: "The speech request must be a JSON object."
        case .mixedProtocols:
            "Do not mix OpenAI voice fields with Mistral voice_id, ref_audio, metadata, prompt_cache_key, or stream fields."
        }
    }
}

public struct MistralSpeechResponse: Codable, Equatable, Sendable {
    public let audioData: String

    public init(audioData: String) { self.audioData = audioData }

    enum CodingKeys: String, CodingKey {
        case audioData = "audio_data"
    }
}

public struct MistralUsageInfo: Codable, Equatable, Sendable {
    public let promptAudioSeconds: Int?
    public let promptTokens: Int
    public let totalTokens: Int
    public let completionTokens: Int?
    public let requestCount: Int?
    public let promptTokensDetails: OpenAIJSONValue?
    public let completionTokensDetails: OpenAIJSONValue?
    public let serviceTier: String?
    public let promptTokenDetails: OpenAIJSONValue?
    public let numCachedTokens: Int?

    public init(
        promptAudioSeconds: Int? = nil,
        promptTokens: Int = 0,
        totalTokens: Int = 0,
        completionTokens: Int? = nil,
        requestCount: Int? = nil,
        promptTokensDetails: OpenAIJSONValue? = nil,
        completionTokensDetails: OpenAIJSONValue? = nil,
        serviceTier: String? = nil,
        promptTokenDetails: OpenAIJSONValue? = nil,
        numCachedTokens: Int? = nil
    ) {
        self.promptAudioSeconds = promptAudioSeconds
        self.promptTokens = promptTokens
        self.totalTokens = totalTokens
        self.completionTokens = completionTokens
        self.requestCount = requestCount
        self.promptTokensDetails = promptTokensDetails
        self.completionTokensDetails = completionTokensDetails
        self.serviceTier = serviceTier
        self.promptTokenDetails = promptTokenDetails
        self.numCachedTokens = numCachedTokens
    }

    enum CodingKeys: String, CodingKey {
        case promptAudioSeconds = "prompt_audio_seconds"
        case promptTokens = "prompt_tokens"
        case totalTokens = "total_tokens"
        case completionTokens = "completion_tokens"
        case requestCount = "request_count"
        case promptTokensDetails = "prompt_tokens_details"
        case completionTokensDetails = "completion_tokens_details"
        case serviceTier = "service_tier"
        case promptTokenDetails = "prompt_token_details"
        case numCachedTokens = "num_cached_tokens"
    }
}

public struct MistralSpeechAudioDeltaEvent: Codable, Equatable, Sendable {
    public let type: String
    public let audioData: String

    public init(audioData: String) {
        self.type = "speech.audio.delta"
        self.audioData = audioData
    }

    enum CodingKeys: String, CodingKey {
        case type
        case audioData = "audio_data"
    }
}

public struct MistralSpeechAudioDoneEvent: Codable, Equatable, Sendable {
    public let type: String
    public let usage: MistralUsageInfo

    public init(usage: MistralUsageInfo) {
        self.type = "speech.audio.done"
        self.usage = usage
    }
}

public enum PatchField<Value: Codable & Equatable & Sendable>: Equatable, Sendable {
    case missing
    case null
    case value(Value)
}

public struct VoiceCreateRequest: Codable, Equatable, Sendable {
    public let name: String
    public let sampleAudio: String
    public let slug: String?
    public let languages: [String]
    public let gender: String?
    public let age: Int?
    public let tags: [String]?
    public let color: String?
    public let description: String?
    public let retentionNotice: Int
    public let sampleFilename: String?

    public init(
        name: String,
        sampleAudio: String,
        slug: String? = nil,
        languages: [String] = [],
        gender: String? = nil,
        age: Int? = nil,
        tags: [String]? = nil,
        color: String? = nil,
        description: String? = nil,
        retentionNotice: Int = 30,
        sampleFilename: String? = nil
    ) {
        self.name = name
        self.sampleAudio = sampleAudio
        self.slug = slug
        self.languages = languages
        self.gender = gender
        self.age = age
        self.tags = tags
        self.color = color
        self.description = description
        self.retentionNotice = retentionNotice
        self.sampleFilename = sampleFilename
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        sampleAudio = try container.decode(String.self, forKey: .sampleAudio)
        slug = try container.decodeIfPresent(String.self, forKey: .slug)
        languages = try container.decodeNonNullIfPresent([String].self, forKey: .languages) ?? []
        gender = try container.decodeIfPresent(String.self, forKey: .gender)
        age = try container.decodeIfPresent(Int.self, forKey: .age)
        tags = try container.decodeIfPresent([String].self, forKey: .tags)
        color = try container.decodeIfPresent(String.self, forKey: .color)
        description = try container.decodeIfPresent(String.self, forKey: .description)
        retentionNotice = try container.decodeNonNullIfPresent(
            Int.self,
            forKey: .retentionNotice
        ) ?? 30
        sampleFilename = try container.decodeIfPresent(String.self, forKey: .sampleFilename)
    }

    enum CodingKeys: String, CodingKey {
        case name, slug, languages, gender, age, tags, color, description
        case sampleAudio = "sample_audio"
        case retentionNotice = "retention_notice"
        case sampleFilename = "sample_filename"
    }
}

public struct VoiceUpdateRequest: Codable, Equatable, Sendable {
    public let name: PatchField<String>
    public let languages: PatchField<[String]>
    public let gender: PatchField<String>
    public let age: PatchField<Int>
    public let tags: PatchField<[String]>
    public let description: PatchField<String>

    public init(
        name: PatchField<String> = .missing,
        languages: PatchField<[String]> = .missing,
        gender: PatchField<String> = .missing,
        age: PatchField<Int> = .missing,
        tags: PatchField<[String]> = .missing,
        description: PatchField<String> = .missing
    ) {
        self.name = name
        self.languages = languages
        self.gender = gender
        self.age = age
        self.tags = tags
        self.description = description
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try Self.decodePatch(String.self, key: .name, from: container)
        languages = try Self.decodePatch([String].self, key: .languages, from: container)
        gender = try Self.decodePatch(String.self, key: .gender, from: container)
        age = try Self.decodePatch(Int.self, key: .age, from: container)
        tags = try Self.decodePatch([String].self, key: .tags, from: container)
        description = try Self.decodePatch(String.self, key: .description, from: container)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try Self.encodePatch(name, key: .name, to: &container)
        try Self.encodePatch(languages, key: .languages, to: &container)
        try Self.encodePatch(gender, key: .gender, to: &container)
        try Self.encodePatch(age, key: .age, to: &container)
        try Self.encodePatch(tags, key: .tags, to: &container)
        try Self.encodePatch(description, key: .description, to: &container)
    }

    private static func decodePatch<Value: Codable & Equatable & Sendable>(
        _ type: Value.Type,
        key: CodingKeys,
        from container: KeyedDecodingContainer<CodingKeys>
    ) throws -> PatchField<Value> {
        guard container.contains(key) else { return .missing }
        if try container.decodeNil(forKey: key) { return .null }
        return .value(try container.decode(Value.self, forKey: key))
    }

    private static func encodePatch<Value: Codable & Equatable & Sendable>(
        _ field: PatchField<Value>,
        key: CodingKeys,
        to container: inout KeyedEncodingContainer<CodingKeys>
    ) throws {
        switch field {
        case .missing: break
        case .null: try container.encodeNil(forKey: key)
        case .value(let value): try container.encode(value, forKey: key)
        }
    }

    enum CodingKeys: String, CodingKey {
        case name, languages, gender, age, tags, description
    }
}

public struct VoiceResponse: Codable, Equatable, Sendable {
    public let name: String
    public let id: String
    public let createdAt: Date
    public let userID: String?
    public let slug: String?
    public let languages: [String]
    public let gender: String?
    public let age: Int?
    public let tags: [String]?
    public let color: String?
    public let description: String?
    public let retentionNotice: Int
    public let trimmedSeconds: Double?

    public init(
        name: String,
        id: String,
        createdAt: Date,
        userID: String?,
        slug: String? = nil,
        languages: [String] = [],
        gender: String? = nil,
        age: Int? = nil,
        tags: [String]? = nil,
        color: String? = nil,
        description: String? = nil,
        retentionNotice: Int = 30,
        trimmedSeconds: Double? = nil
    ) {
        self.name = name
        self.id = id
        self.createdAt = createdAt
        self.userID = userID
        self.slug = slug
        self.languages = languages
        self.gender = gender
        self.age = age
        self.tags = tags
        self.color = color
        self.description = description
        self.retentionNotice = retentionNotice
        self.trimmedSeconds = trimmedSeconds
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        id = try container.decode(String.self, forKey: .id)
        createdAt = try Self.decodeDate(try container.decode(String.self, forKey: .createdAt))
        userID = try container.decodeIfPresent(String.self, forKey: .userID)
        slug = try container.decodeIfPresent(String.self, forKey: .slug)
        languages = try container.decodeIfPresent([String].self, forKey: .languages) ?? []
        gender = try container.decodeIfPresent(String.self, forKey: .gender)
        age = try container.decodeIfPresent(Int.self, forKey: .age)
        tags = try container.decodeIfPresent([String].self, forKey: .tags)
        color = try container.decodeIfPresent(String.self, forKey: .color)
        description = try container.decodeIfPresent(String.self, forKey: .description)
        retentionNotice = try container.decodeIfPresent(Int.self, forKey: .retentionNotice) ?? 30
        trimmedSeconds = try container.decodeIfPresent(Double.self, forKey: .trimmedSeconds)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(id, forKey: .id)
        try container.encode(Self.encodeDate(createdAt), forKey: .createdAt)
        try container.encode(userID, forKey: .userID)
        try container.encodeIfPresent(slug, forKey: .slug)
        try container.encode(languages, forKey: .languages)
        try container.encodeIfPresent(gender, forKey: .gender)
        try container.encodeIfPresent(age, forKey: .age)
        try container.encodeIfPresent(tags, forKey: .tags)
        try container.encodeIfPresent(color, forKey: .color)
        try container.encodeIfPresent(description, forKey: .description)
        try container.encode(retentionNotice, forKey: .retentionNotice)
        try container.encodeIfPresent(trimmedSeconds, forKey: .trimmedSeconds)
    }

    private static func encodeDate(_ date: Date) -> String {
        ISO8601DateFormatter().string(from: date)
    }

    private static func decodeDate(_ value: String) throws -> Date {
        let regular = ISO8601DateFormatter()
        if let date = regular.date(from: value) { return date }
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions.insert(.withFractionalSeconds)
        if let date = fractional.date(from: value) { return date }
        throw DecodingError.dataCorrupted(
            .init(codingPath: [], debugDescription: "Invalid ISO-8601 date: \(value)")
        )
    }

    enum CodingKeys: String, CodingKey {
        case name, id, slug, languages, gender, age, tags, color, description
        case createdAt = "created_at"
        case userID = "user_id"
        case retentionNotice = "retention_notice"
        case trimmedSeconds = "trimmed_seconds"
    }
}

public struct VoiceListResponse: Codable, Equatable, Sendable {
    public let items: [VoiceResponse]
    public let total: Int
    public let page: Int
    public let pageSize: Int
    public let totalPages: Int

    public init(items: [VoiceResponse], total: Int, page: Int, pageSize: Int, totalPages: Int) {
        self.items = items
        self.total = total
        self.page = page
        self.pageSize = pageSize
        self.totalPages = totalPages
    }

    enum CodingKeys: String, CodingKey {
        case items, total, page
        case pageSize = "page_size"
        case totalPages = "total_pages"
    }
}

public enum ValidationLocation: Codable, Equatable, Sendable {
    case string(String)
    case integer(Int)

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) { self = .string(value) }
        else { self = .integer(try container.decode(Int.self)) }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .integer(let value): try container.encode(value)
        }
    }
}

public struct ValidationErrorDetail: Codable, Equatable, Sendable {
    public let loc: [ValidationLocation]
    public let msg: String
    public let type: String
    public let input: OpenAIJSONValue?
    public let ctx: [String: OpenAIJSONValue]?

    public init(
        loc: [ValidationLocation],
        msg: String,
        type: String,
        input: OpenAIJSONValue? = nil,
        ctx: [String: OpenAIJSONValue]? = nil
    ) {
        self.loc = loc
        self.msg = msg
        self.type = type
        self.input = input
        self.ctx = ctx
    }
}

public struct HTTPValidationError: Codable, Equatable, Sendable {
    public let detail: [ValidationErrorDetail]?

    public init(detail: [ValidationErrorDetail]? = nil) { self.detail = detail }
}

/// Local runner error envelope for non-validation Mistral routes. Mistral's
/// public schema specifies the 422 envelope but leaves other error bodies open.
public struct MistralError: Codable, Equatable, Sendable {
    public let object: String
    public let message: String
    public let type: String
    public let param: String?
    public let code: String?

    public init(
        message: String,
        type: String = "invalid_request_error",
        param: String? = nil,
        code: String? = nil
    ) {
        self.object = "error"
        self.message = message
        self.type = type
        self.param = param
        self.code = code
    }
}

private extension KeyedDecodingContainer {
    /// Decode an optional schema property while still rejecting an explicit
    /// JSON `null`. `decodeIfPresent` intentionally treats missing and null as
    /// the same value, which is too permissive for defaulted non-null fields.
    func decodeNonNullIfPresent<Value: Decodable>(
        _ type: Value.Type,
        forKey key: Key
    ) throws -> Value? {
        guard contains(key) else { return nil }
        guard try !decodeNil(forKey: key) else {
            throw DecodingError.valueNotFound(
                Value.self,
                .init(
                    codingPath: codingPath + [key],
                    debugDescription: "Expected \(Value.self), but found null instead."
                )
            )
        }
        return try decode(Value.self, forKey: key)
    }
}
