import Foundation

public enum LocalSpeechAudioFormat: String, CaseIterable, Hashable, Sendable {
    case pcm
    case wav
    case mp3
    case flac
    case opus
    case aac
}

public enum LocalSpeechPCMEncoding: String, Sendable {
    case float32LittleEndian
    case signedInt16LittleEndian
}

public struct LocalSpeechSynthesisRequest: Equatable, Sendable {
    public let input: String
    public let voiceID: String
    public let format: LocalSpeechAudioFormat
    public let pcmEncoding: LocalSpeechPCMEncoding
    public let instructions: String?
    public let speed: Double

    public init(
        input: String,
        voiceID: String,
        format: LocalSpeechAudioFormat,
        pcmEncoding: LocalSpeechPCMEncoding = .float32LittleEndian,
        instructions: String? = nil,
        speed: Double = 1
    ) {
        self.input = input
        self.voiceID = voiceID
        self.format = format
        self.pcmEncoding = pcmEncoding
        self.instructions = instructions
        self.speed = speed
    }
}

public struct LocalSpeechUsage: Equatable, Sendable {
    public let promptTokens: Int
    public let completionTokens: Int

    public init(promptTokens: Int, completionTokens: Int) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
    }

    public var totalTokens: Int { promptTokens + completionTokens }
}

public enum LocalSpeechSynthesisEvent: Equatable, Sendable {
    case audio(Data)
    case completed(LocalSpeechUsage)
}

public protocol LocalSpeechSynthesizing: Sendable {
    var servedModelName: String { get }
    var voiceCatalog: VoxtralVoiceCatalog { get }
    var supportedAudioFormats: Set<LocalSpeechAudioFormat> { get }
    var supportsInstructions: Bool { get }
    var supportedSpeedRange: ClosedRange<Double> { get }

    func stream(
        request: LocalSpeechSynthesisRequest
    ) async -> AsyncThrowingStream<LocalSpeechSynthesisEvent, Error>
}

public extension LocalSpeechSynthesizing {
    var supportedAudioFormats: Set<LocalSpeechAudioFormat> {
        Set(LocalSpeechAudioFormat.allCases)
    }

    var supportsInstructions: Bool { true }

    var supportedSpeedRange: ClosedRange<Double> { 0.25 ... 4 }
}
