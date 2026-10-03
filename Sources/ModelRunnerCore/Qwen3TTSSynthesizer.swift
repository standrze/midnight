import Foundation
@preconcurrency import MLX
import MLXAudioTTS
import ModelRunnerProtocol

/// Native Qwen3-TTS inference backed by Midnight's Qwen-only MLX Audio slice.
public actor Qwen3TTSSynthesizer: LocalSpeechSynthesizing {
    public nonisolated let servedModelName: String
    public nonisolated let voiceCatalog: VoxtralVoiceCatalog
    public nonisolated let supportedAudioFormats: Set<LocalSpeechAudioFormat> = [.wav, .pcm]
    public nonisolated let supportsInstructions = true
    /// Qwen3-TTS Base accepts reference audio for voice cloning.
    public nonisolated var supportsReferenceAudio: Bool { modelType == "base" }
    public nonisolated let supportedSpeedRange: ClosedRange<Double> = 1...1

    private let model: Qwen3TTSModel
    private let modelType: String
    private let producerLifetime = StreamProducerLifetime()

    /// Loads a Qwen3-TTS checkpoint and selects its runtime engine.
    public init(modelPath: String, servedModelName: String? = nil, engine: ModelEngine = .auto) async throws {
        guard try engine.resolve() == .metal else {
            throw Qwen3TTSError.invalidConfiguration("Qwen3-TTS currently requires the Metal engine")
        }
        let directory = URL(fileURLWithPath: NSString(string: modelPath).expandingTildeInPath, isDirectory: true)
            .standardizedFileURL.resolvingSymlinksInPath()
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            root["model_type"] as? String == "qwen3_tts"
        else {
            throw Qwen3TTSError.invalidConfiguration("model_type must be qwen3_tts")
        }
        let type = (root["tts_model_type"] as? String ?? "base").lowercased()
        let speakers = Self.speakers(in: root)
        guard type != "custom_voice" || !speakers.isEmpty else {
            throw Qwen3TTSError.invalidConfiguration("Qwen3-TTS CustomVoice checkpoint has no preset speakers")
        }
        self.model = try await Qwen3TTSModel.fromModelDirectory(directory)
        self.modelType = type
        let requestedName = servedModelName?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        self.servedModelName = requestedName.isEmpty ? directory.lastPathComponent : requestedName
        self.voiceCatalog = try Self.availableVoices(modelPath: directory.path)
    }

    /// Reads the preset voices available from a Qwen3-TTS checkpoint.
    public static func availableVoices(modelPath: String) throws -> VoxtralVoiceCatalog {
        let data = try Data(contentsOf: URL(fileURLWithPath: modelPath).appendingPathComponent("config.json"))
        let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let type = (root["tts_model_type"] as? String ?? "base").lowercased()
        let availableVoices = type == "base" ? ["clone"] : speakers(in: root)
        return VoxtralVoiceCatalog(
            voices: availableVoices.enumerated().map { index, speaker in
                VoxtralPresetVoice(
                    id: speaker, apiID: speaker, name: speaker.capitalized,
                    position: index, languages: ["en"], gender: nil)
            })
    }

    /// Returns once this synthesizer has no outstanding producer work.
    public func waitUntilIdle() async {
        await producerLifetime.waitUntilIdle()
    }

    /// Emits generated audio chunks and final usage for a speech request.
    public func stream(request: LocalSpeechSynthesisRequest) async -> AsyncThrowingStream<
        LocalSpeechSynthesisEvent, Error
    > {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    guard request.speed == 1 else {
                        throw Qwen3TTSError.unsupportedFeature("speed")
                    }
                    guard supportedAudioFormats.contains(request.format) else {
                        throw Qwen3TTSError.unsupportedFormat(request.format.rawValue)
                    }
                    guard !request.input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                        throw Qwen3TTSError.invalidConfiguration("speech input must not be empty")
                    }
                    guard voiceCatalog.voice(id: request.voiceID) != nil else {
                        throw Qwen3TTSError.unknownVoice(request.voiceID)
                    }
                    let voice: String?
                    let referenceAudio: MLXArray?
                    let referenceText: String?
                    if modelType == "base" {
                        guard let data = request.referenceAudio,
                            let text = request.referenceText?.trimmingCharacters(in: .whitespacesAndNewlines),
                            !text.isEmpty
                        else {
                            throw Qwen3TTSError.unsupportedFeature("Base cloning requires ref_audio and ref_text")
                        }
                        voice = nil
                        referenceAudio = try Self.decodeReferenceWAV(data, sampleRate: model.sampleRate)
                        referenceText = text
                    } else {
                        voice = request.instructions.map { "\(request.voiceID), \($0)" } ?? request.voiceID
                        referenceAudio = nil
                        referenceText = nil
                    }
                    let waveform = try await model.generate(
                        text: request.input, voice: voice,
                        refAudio: referenceAudio, refText: referenceText, language: "en")
                    eval(waveform)
                    let samples = waveform.asType(.float32).asArray(Float.self)
                    let audio = try VoxtralAudioEncoding.encode(
                        samples: samples, sampleRate: model.sampleRate,
                        format: request.format, pcmEncoding: request.pcmEncoding)
                    continuation.yield(.audio(audio))
                    continuation.yield(
                        .completed(.init(promptTokens: request.input.count, completionTokens: samples.count)))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            producerLifetime.track(task)
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private static func speakers(in root: [String: Any]) -> [String] {
        let talker = root["talker_config"] as? [String: Any]
        let ids = talker?["spk_id"] as? [String: Any] ?? [:]
        return ids.keys.sorted()
    }

    /// Qwen's speaker encoder expects a mono 24-kHz waveform. Keeping this
    /// narrow avoids silently cloning from a differently interpreted format.
    private static func decodeReferenceWAV(_ data: Data, sampleRate: Int) throws -> MLXArray {
        guard data.count >= 44, data.prefix(4) == Data("RIFF".utf8), data[8..<12] == Data("WAVE".utf8) else {
            throw Qwen3TTSError.invalidConfiguration("ref_audio must be a WAV file")
        }
        var offset = 12
        var format: UInt16?
        var channels: UInt16?
        var rate: UInt32?
        var bits: UInt16?
        var payload: Data?
        while offset + 8 <= data.count {
            let id = data[offset..<offset + 4]
            let length =
                Int(data[offset + 4]) | Int(data[offset + 5]) << 8 | Int(data[offset + 6]) << 16 | Int(data[offset + 7])
                << 24
            let start = offset + 8
            guard start + length <= data.count else {
                break
            }
            if id == Data("fmt ".utf8), length >= 16 {
                format = UInt16(data[start]) | UInt16(data[start + 1]) << 8
                channels = UInt16(data[start + 2]) | UInt16(data[start + 3]) << 8
                rate =
                    UInt32(data[start + 4]) | UInt32(data[start + 5]) << 8 | UInt32(data[start + 6]) << 16 | UInt32(
                        data[start + 7]) << 24
                bits = UInt16(data[start + 14]) | UInt16(data[start + 15]) << 8
            } else if id == Data("data".utf8) {
                payload = Data(data[start..<start + length])
            }
            offset = start + length + length % 2
        }
        guard channels == 1, rate == UInt32(sampleRate), let format, let bits, let payload else {
            throw Qwen3TTSError.invalidConfiguration("ref_audio must be mono \(sampleRate)-Hz WAV audio")
        }
        let samples: [Float]
        switch (format, bits) {
        case (1, 16):
            samples = stride(from: 0, to: payload.count - 1, by: 2).map {
                let value = Int16(bitPattern: UInt16(payload[$0]) | UInt16(payload[$0 + 1]) << 8)
                return Float(value) / Float(Int16.max)
            }
        case (3, 32):
            samples = stride(from: 0, to: payload.count - 3, by: 4).map {
                Float(
                    bitPattern: UInt32(payload[$0]) | UInt32(payload[$0 + 1]) << 8 | UInt32(payload[$0 + 2]) << 16
                        | UInt32(payload[$0 + 3]) << 24)
            }
        default:
            throw Qwen3TTSError.invalidConfiguration("ref_audio must use 16-bit PCM or Float32 WAV encoding")
        }
        guard !samples.isEmpty else {
            throw Qwen3TTSError.invalidConfiguration("ref_audio contains no samples")
        }
        return MLXArray(samples)
    }
}

/// An invalid Qwen3-TTS configuration, voice, feature, or audio format.
public enum Qwen3TTSError: LocalizedError {
    case invalidConfiguration(String)
    case unsupportedFeature(String)
    case unsupportedFormat(String)
    case unknownVoice(String)

    /// User-facing explanation for this error.
    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message): message
        case .unsupportedFeature(let feature): "Qwen3-TTS does not support \(feature)."
        case .unsupportedFormat(let format): "Qwen3-TTS supports WAV and PCM output, not \(format)."
        case .unknownVoice(let voice): "Qwen3-TTS voice '\(voice)' is not available."
        }
    }
}
