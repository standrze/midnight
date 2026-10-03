import Foundation

/// Container or codec requested for synthesized local speech.
public enum LocalSpeechAudioFormat: String, CaseIterable, Hashable, Sendable {
    case pcm
    case wav
    case mp3
    case flac
    case opus
    case aac
}

/// Byte representation used for raw PCM audio samples.
public enum LocalSpeechPCMEncoding: String, Sendable {
    case float32LittleEndian
    case signedInt16LittleEndian
}

/// Text, voice, format, and optional references for one synthesis request.
public struct LocalSpeechSynthesisRequest: Equatable, Sendable {
    public let input: String
    public let voiceID: String
    public let format: LocalSpeechAudioFormat
    public let pcmEncoding: LocalSpeechPCMEncoding
    public let instructions: String?
    public let referenceAudio: Data?
    public let referenceText: String?
    /// Requested playback rate relative to normal speed (1.0).
    public let speed: Double

    /// Creates a speech request with output format, voice, and optional references.
    public init(
        input: String,
        voiceID: String,
        format: LocalSpeechAudioFormat,
        pcmEncoding: LocalSpeechPCMEncoding = .float32LittleEndian,
        instructions: String? = nil,
        referenceAudio: Data? = nil,
        referenceText: String? = nil,
        speed: Double = 1
    ) {
        self.input = input
        self.voiceID = voiceID
        self.format = format
        self.pcmEncoding = pcmEncoding
        self.instructions = instructions
        self.referenceAudio = referenceAudio
        self.referenceText = referenceText
        self.speed = speed
    }
}

/// Prompt and generated token counts from speech synthesis.
public struct LocalSpeechUsage: Equatable, Sendable {
    public let promptTokens: Int
    public let completionTokens: Int

    /// Records prompt and completion token counts for speech usage.
    public init(promptTokens: Int, completionTokens: Int) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
    }

    /// Sum of prompt and completion token counts.
    public var totalTokens: Int { promptTokens + completionTokens }
}

/// An audio chunk or terminal usage report from a speech stream.
public enum LocalSpeechSynthesisEvent: Equatable, Sendable {
    case audio(Data)
    case completed(LocalSpeechUsage)
}

/// A local speech backend with declared voice and format capabilities.
public protocol LocalSpeechSynthesizing: Sendable {
    var servedModelName: String { get }
    var voiceCatalog: VoxtralVoiceCatalog { get }
    var supportedAudioFormats: Set<LocalSpeechAudioFormat> { get }
    var supportsInstructions: Bool { get }
    var supportsReferenceAudio: Bool { get }
    var requiresReferenceText: Bool { get }
    var supportedSpeedRange: ClosedRange<Double> { get }

    /// Starts synthesis and yields audio chunks followed by terminal usage on success.
    /// Validation and synthesis failures are thrown while consuming the stream.
    /// Cancelling consumption cancels the producer; use `waitUntilIdle()` to await its cleanup.
    func stream(
        request: LocalSpeechSynthesisRequest
    ) async -> AsyncThrowingStream<LocalSpeechSynthesisEvent, Error>

    /// After admission is closed, wait for every producer to finish cleanup,
    /// including producers whose consumers have already cancelled their streams.
    func waitUntilIdle() async
}

extension LocalSpeechSynthesizing {
    /// Default backends do not accept reference audio.
    public var supportsReferenceAudio: Bool { false }
    /// Default backends require transcript text with reference audio.
    public var requiresReferenceText: Bool { true }
    /// Default backend support includes every declared output format.
    public var supportedAudioFormats: Set<LocalSpeechAudioFormat> {
        Set(LocalSpeechAudioFormat.allCases)
    }

    /// Default backends accept synthesis instructions.
    public var supportsInstructions: Bool { true }

    /// Default supported playback speed multipliers.
    public var supportedSpeedRange: ClosedRange<Double> { 0.25...4 }
}
