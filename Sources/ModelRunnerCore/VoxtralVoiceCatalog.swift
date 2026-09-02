import Foundation

public struct VoxtralPresetVoice: Equatable, Sendable {
    public let id: String
    public let apiID: String
    public let name: String
    public let position: Int
    public let languages: [String]
    public let gender: String?

    public init(
        id: String,
        apiID: String,
        name: String,
        position: Int,
        languages: [String],
        gender: String?
    ) {
        self.id = id
        self.apiID = apiID
        self.name = name
        self.position = position
        self.languages = languages
        self.gender = gender
    }
}

public struct VoxtralVoiceCatalog: Equatable, Sendable {
    public let voices: [VoxtralPresetVoice]
    public let createdAt: Date

    public init?(modelDirectory: String, fileManager: FileManager = .default) throws {
        let directory = URL(
            fileURLWithPath: NSString(string: modelDirectory).expandingTildeInPath,
            isDirectory: true
        ).standardizedFileURL
        let configurationURL = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configurationURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            root["model_type"] as? String == "voxtral_tts",
            let multimodal = root["multimodal"] as? [String: Any],
            let tokenizer = multimodal["audio_tokenizer_args"] as? [String: Any],
            let voiceMap = tokenizer["voice"] as? [String: Any],
            !voiceMap.isEmpty
        else { return nil }

        let voices = voiceMap.compactMap { id, rawPosition -> VoxtralPresetVoice? in
            guard let position = (rawPosition as? NSNumber)?.intValue else { return nil }
            return VoxtralPresetVoice(
                id: id,
                apiID: Self.stableAPIID(for: id),
                name: Self.displayName(for: id),
                position: position,
                languages: [Self.language(for: id)],
                gender: Self.gender(for: id)
            )
        }.sorted { lhs, rhs in
            lhs.position == rhs.position ? lhs.id < rhs.id : lhs.position < rhs.position
        }
        guard voices.count == voiceMap.count else { return nil }

        let values = try? configurationURL.resourceValues(
            forKeys: [.creationDateKey, .contentModificationDateKey]
        )
        self.voices = voices
        self.createdAt = values?.creationDate ?? values?.contentModificationDate ?? .distantPast
    }

    public func voice(id: String) -> VoxtralPresetVoice? {
        voices.first { $0.id == id || $0.apiID.caseInsensitiveCompare(id) == .orderedSame }
    }

    /// Mistral declares voice resource IDs as UUIDs, while Voxtral stores
    /// human-readable preset slugs. Expose a deterministic UUID and retain the
    /// checkpoint value separately as `slug`/`id` for model generation.
    private static func stableAPIID(for id: String) -> String {
        var hash: UInt64 = 14_695_981_039_346_656_037
        for byte in id.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1_099_511_628_211
        }
        return String(
            format: "564f5854-5241-4c54-8000-%012llx",
            hash & 0x0000_FFFF_FFFF_FFFF
        )
    }

    private static func displayName(for id: String) -> String {
        id.split(separator: "_")
            .map { component in
                let value = String(component)
                if let language = languageNames[value] { return language }
                return value.prefix(1).uppercased() + value.dropFirst()
            }
            .joined(separator: " ")
    }

    private static func language(for id: String) -> String {
        let prefix = id.split(separator: "_").first.map(String.init) ?? "en"
        return languageNames[prefix] == nil ? "en" : prefix
    }

    private static func gender(for id: String) -> String? {
        let suffix = id.split(separator: "_").last.map(String.init)
        return suffix == "male" || suffix == "female" ? suffix : nil
    }

    private static let languageNames: [String: String] = [
        "ar": "Arabic",
        "de": "German",
        "es": "Spanish",
        "fr": "French",
        "hi": "Hindi",
        "it": "Italian",
        "nl": "Dutch",
        "pt": "Portuguese",
    ]
}
