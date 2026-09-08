import Foundation

/// Optional, model-local settings. These never change the HTTP request schema.
public struct ChatterboxSettings: Decodable, Sendable {
    public enum Variant: String, Decodable, Sendable { case turbo, multilingual }
    public let variant: Variant
    public let language: String
    public let temperature: Float
    public let topP: Float
    public let exaggeration: Float?
    public let cfgWeight: Float?
    public let voices: [String: String]
    public let speechTokenizer: String?
    public let ffmpegPath: String?
    public let maxTokens: Int
    public let topK: Int
    public let minP: Float
    public let repetitionPenalty: Float

    public static func load(modelDirectory: String) throws -> Self? {
        let url = URL(fileURLWithPath: modelDirectory).appendingPathComponent("chatterbox.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        do { return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url)) }
        catch { throw ModelStackSettingsError.invalidFile(url.path, error.localizedDescription) }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Key.self)
        let known: Set<String> = ["variant", "language", "temperature", "top_p", "exaggeration", "cfg_weight", "voices", "speech_tokenizer", "ffmpeg_path", "max_tokens", "top_k", "min_p", "repetition_penalty"]
        func invalid(_ message: String) -> DecodingError {
            .dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: message))
        }
        guard c.allKeys.allSatisfy({ known.contains($0.stringValue) }) else {
            throw invalid("Unknown Chatterbox setting; unsupported controls are not silently ignored")
        }
        variant = try c.decode(Variant.self, forKey: Key("variant"))
        language = try c.decodeIfPresent(String.self, forKey: Key("language")) ?? "en"
        temperature = try c.decodeIfPresent(Float.self, forKey: Key("temperature")) ?? 0.8
        topP = try c.decodeIfPresent(Float.self, forKey: Key("top_p")) ?? (variant == .turbo ? 0.95 : 1)
        exaggeration = try c.decodeIfPresent(Float.self, forKey: Key("exaggeration"))
        cfgWeight = try c.decodeIfPresent(Float.self, forKey: Key("cfg_weight"))
        voices = try c.decodeIfPresent([String: String].self, forKey: Key("voices")) ?? [:]
        speechTokenizer = try c.decodeIfPresent(String.self, forKey: Key("speech_tokenizer"))
        ffmpegPath = try c.decodeIfPresent(String.self, forKey: Key("ffmpeg_path"))
        maxTokens = try c.decodeIfPresent(Int.self, forKey: Key("max_tokens")) ?? 1000
        topK = try c.decodeIfPresent(Int.self, forKey: Key("top_k")) ?? (variant == .turbo ? 1000 : 0)
        minP = try c.decodeIfPresent(Float.self, forKey: Key("min_p")) ?? (variant == .turbo ? 0 : 0.05)
        repetitionPenalty = try c.decodeIfPresent(Float.self, forKey: Key("repetition_penalty")) ?? 1.2
        guard (1...4096).contains(maxTokens), (0...6563).contains(topK), minP.isFinite,
              (0...1).contains(minP), repetitionPenalty.isFinite, repetitionPenalty > 0 else {
            throw invalid("Invalid max_tokens, top_k, min_p, or repetition_penalty")
        }
        guard voices.isEmpty || speechTokenizer?.isEmpty == false else {
            throw invalid("Reference voices require a local speech_tokenizer directory")
        }
        guard temperature.isFinite, temperature > 0, temperature <= 2,
              topP.isFinite, topP > 0, topP <= 1 else {
            throw invalid("temperature must be in (0, 2] and top_p in (0, 1]")
        }
        for value in [exaggeration, cfgWeight].compactMap({ $0 }) {
            guard value.isFinite, (0...1).contains(value) else {
                throw invalid("exaggeration and cfg_weight must be in [0, 1]")
            }
        }
        if variant == .turbo {
            guard language == "en", exaggeration == nil, cfgWeight == nil, minP == 0 else {
                throw invalid("Turbo supports English only and has no exaggeration or cfg_weight control")
            }
        } else {
            guard topK == 0 else { throw invalid("The native multilingual backend does not support top_k") }
            guard Self.languages.contains(language) else {
                throw invalid("Unsupported language '\(language)'; native Japanese and Chinese text preparation is not available")
            }
        }
        guard voices.allSatisfy({ !$0.key.isEmpty && $0.key != "default" && !$0.value.isEmpty }) else {
            throw invalid("Voice names and audio paths must be nonempty; 'default' is reserved")
        }
    }

    public static let languages = ["ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr"]

    /// Matches the reference multilingual language prefix, NFKD and space tokens.
    public func preparedText(_ text: String) -> String {
        guard variant == .multilingual else { return text }
        let normalized = text.lowercased().decomposedStringWithCompatibilityMapping
        return "[\(language)]" + normalized.replacingOccurrences(of: " ", with: "[SPACE]")
    }

    private struct Key: CodingKey {
        var stringValue: String
        var intValue: Int? { nil }
        init(_ value: String) { stringValue = value }
        init?(stringValue: String) { self.init(stringValue) }
        init?(intValue: Int) { return nil }
    }
}
