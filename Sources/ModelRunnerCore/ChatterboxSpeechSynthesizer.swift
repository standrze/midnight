#if os(macOS)
import Foundation
import MLX
import MLXAudioCore
import MLXAudioTTS
import MLXLMCommon
import ModelRunnerProtocol

/// A model-local adapter: HTTP and Voxtral generation remain unchanged.
public actor ChatterboxSpeechSynthesizer: LocalSpeechSynthesizing {
    public nonisolated let servedModelName: String
    public nonisolated let voiceCatalog: VoxtralVoiceCatalog
    public nonisolated let supportedAudioFormats: Set<LocalSpeechAudioFormat>
    public nonisolated let supportsInstructions = false
    public nonisolated let supportedSpeedRange: ClosedRange<Double>
    private let ffmpeg: URL?
    private let model: ChatterboxModel
    private let settings: ChatterboxSettings
    private let directory: URL
    private var busy = false

    public init(modelPath: String, servedModelName: String, settings: ChatterboxSettings) async throws {
        self.servedModelName = servedModelName
        self.settings = settings
        ffmpeg = try ChatterboxAudioEncoding.executable(configuredPath: settings.ffmpegPath)
        supportedAudioFormats = ffmpeg == nil ? [.wav, .pcm] : Set(LocalSpeechAudioFormat.allCases)
        supportedSpeedRange = ffmpeg == nil ? 1...1 : 0.25...4
        directory = URL(fileURLWithPath: modelPath, isDirectory: true)
        let modelDirectory = directory
        let speechTokenizer = settings.speechTokenizer.map { Self.voiceURL($0, directory: modelDirectory) }
        model = try await MLXPinnedRuntime.shared.run {
            let limits = try MLXResourceLimits.resolve(for: .metal,
                physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
                recommendedWorkingSetBytes: GPU.maxRecommendedWorkingSetBytes())
            try MLXResourceGuard.apply(limits)
            return try await ChatterboxModel.fromModelDirectory(modelDirectory, hfToken: nil,
                localSpeechTokenizer: speechTokenizer, allowTokenizerDownload: false)
        }
        guard model.tokenizer != nil else { throw Failure("Chatterbox tokenizer could not be loaded") }
        guard (settings.variant == .turbo) == model.config.meanflow else {
            throw Failure("chatterbox.json variant does not match the checkpoint")
        }
        if settings.variant == .multilingual, !model.config.t3Config.isMultilingual {
            throw Failure("Multilingual requires a multilingual checkpoint, not the original English Chatterbox model")
        }
        guard settings.voices.isEmpty || model.s3Tokenizer != nil else {
            throw Failure("Configured reference voices require S3TokenizerV2; default-token fallback is not supported")
        }
        for path in settings.voices.values {
            let url = Self.voiceURL(path, directory: directory)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw Failure("Reference voice audio does not exist: \(url.path)")
            }
        }
        var names = settings.voices.keys.sorted()
        if model.defaultConditioning != nil { names.insert("default", at: 0) }
        guard !names.isEmpty else { throw Failure("Provide a reference voice or checkpoint conds.safetensors") }
        voiceCatalog = VoxtralVoiceCatalog(chatterboxVoices: names, language: settings.language)
    }

    public func stream(request: LocalSpeechSynthesisRequest) async -> AsyncThrowingStream<LocalSpeechSynthesisEvent, Error> {
        let (stream, continuation) = AsyncThrowingStream<LocalSpeechSynthesisEvent, Error>.makeStream()
        let task = Task {
            do {
                guard !busy else { throw Failure("Chatterbox is already generating speech") }
                busy = true
                defer { busy = false }
                guard voiceCatalog.voice(id: request.voiceID) != nil,
                      supportedAudioFormats.contains(request.format), supportedSpeedRange.contains(request.speed),
                      request.instructions == nil else { throw Failure("Unsupported speech request") }
                let model = self.model
                let settings = self.settings
                let ffmpeg = self.ffmpeg
                let reference = settings.voices[request.voiceID].map { Self.voiceURL($0, directory: directory) }
                let data = try await MLXPinnedRuntime.shared.run {
                    try Task.checkCancellation()
                    model.cfgWeightOverride = settings.cfgWeight
                    model.emotionAdvOverride = settings.exaggeration
                    let refAudio = try reference.map { try loadAudioArray(from: $0, sampleRate: 24000).1 }
                    let audio = try await model.generate(
                        text: settings.preparedText(request.input), voice: nil,
                        refAudio: refAudio, refText: nil, language: nil,
                        generationParameters: GenerateParameters(maxTokens: settings.maxTokens,
                            temperature: settings.temperature, topP: settings.topP, topK: settings.topK,
                            minP: settings.minP, repetitionPenalty: settings.repetitionPenalty)
                    )
                    try Task.checkCancellation()
                    eval(audio)
                    return try ChatterboxAudioEncoding.encode(
                        samples: audio.reshaped([-1]).asArray(Float.self), sampleRate: model.sampleRate,
                        request: request, ffmpeg: ffmpeg)
                }
                continuation.yield(.audio(data))
                // The backend does not expose speech token counts; do not count audio samples as tokens.
                continuation.yield(.completed(LocalSpeechUsage(promptTokens: 0, completionTokens: 0)))
                continuation.finish()
            } catch { continuation.finish(throwing: error) }
        }
        continuation.onTermination = { _ in task.cancel() }
        return stream
    }

    private static func voiceURL(_ path: String, directory: URL) -> URL {
        let expanded = NSString(string: path).expandingTildeInPath
        return NSString(string: expanded).isAbsolutePath
            ? URL(fileURLWithPath: expanded) : directory.appendingPathComponent(expanded)
    }

    private struct Failure: LocalizedError {
        let message: String
        init(_ message: String) { self.message = message }
        var errorDescription: String? { message }
    }
}

// Reuse the existing voice descriptor shape without changing Voxtral's catalog loader.
extension VoxtralVoiceCatalog {
    init(chatterboxVoices: [String], language: String) {
        voices = chatterboxVoices.enumerated().map { index, name in
            VoxtralPresetVoice(id: name, apiID: name, name: name, position: index, languages: [language], gender: nil)
        }
        createdAt = Date(timeIntervalSince1970: 0)
    }
}
#endif
