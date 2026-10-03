import Foundation

/// Editable metadata plus capabilities supplied by the runtime.
/// Assistant references are discovery hints and must pass runtime compatibility checks.
public struct ModelCard: Codable, Equatable, Sendable {
    public static let filename = "model-card.json"
    public var name: String
    public var description: String?
    /// Absent on legacy cards; resolved from the backend before serving.
    public var capabilities: Capabilities?
    /// Runtime-derived voice choices; nil means unavailable/not applicable.
    public var voices: [Voice]?
    /// Local assistant path or Hugging Face repository ID; does not change model identity.
    public var assistantModel: String?

    /// Creates editable model metadata with optional runtime capabilities.
    public init(
        name: String, description: String? = nil, capabilities: Capabilities? = nil, voices: [Voice]? = nil,
        assistantModel: String? = nil
    ) {
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.voices = voices
        self.assistantModel = assistantModel
    }

    enum CodingKeys: String, CodingKey {
        case name, description, capabilities, voices
        case assistantModel = "assistant_model"
    }

    /// A selectable voice advertised in a model's capability card.
    public struct Voice: Codable, Equatable, Sendable {
        public let id: String
        public let slug: String
        public let name: String
        public let languages: [String]
        public let gender: String?
        public let requiresReferenceAudio: Bool

        /// Creates an advertised voice with its API ID and checkpoint slug.
        public init(
            id: String, slug: String, name: String, languages: [String], gender: String? = nil,
            requiresReferenceAudio: Bool = false
        ) {
            self.id = id
            self.slug = slug
            self.name = name
            self.languages = languages
            self.gender = gender
            self.requiresReferenceAudio = requiresReferenceAudio
        }
        enum CodingKeys: String, CodingKey {
            case id, slug, name, languages, gender
            case requiresReferenceAudio = "requires_reference_audio"
        }
    }

    /// Returns a copy with the advertised voice list replaced.
    public func withVoices(_ voices: [Voice]?) -> Self {
        var card = self
        card.voices = voices
        return card
    }

    /// Support available through Midnight, not claims about the upstream model.
    public struct Capabilities: Codable, Equatable, Sendable {
        public var decisions: Bool?
        public var vision: Bool
        public var audioInput: Bool
        public var audioOutput: Bool

        /// Creates the capability flags actually supported by the local backend.
        public init(vision: Bool = false, audioInput: Bool = false, audioOutput: Bool = false, decisions: Bool? = nil) {
            self.decisions = decisions
            self.vision = vision
            self.audioInput = audioInput
            self.audioOutput = audioOutput
        }

        enum CodingKeys: String, CodingKey {
            case vision, decisions
            case audioInput = "audio_input"
            case audioOutput = "audio_output"
        }
    }

    /// File metadata must never override what the selected backend can serve.
    public func withCapabilities(_ capabilities: Capabilities) -> Self {
        var card = self
        card.capabilities = capabilities
        return card
    }

    /// Reads a card of at most 64 KiB; returns nil when no card file exists.
    public static func load(directory: String) throws -> Self? {
        let url = URL(fileURLWithPath: directory, isDirectory: true).appendingPathComponent(filename)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        do {
            let file = try FileHandle(forReadingFrom: url)
            defer { try? file.close() }
            let data = try file.read(upToCount: 65_537) ?? Data()
            guard data.count <= 65_536 else {
                throw CardError("File exceeds 64 KiB")
            }
            var card = try JSONDecoder().decode(Self.self, from: data)
            card.name = card.name.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !card.name.isEmpty, card.name.count <= 256,
                !card.name.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) })
            else {
                throw CardError("name must contain 1–256 characters without control characters")
            }
            return card
        } catch {
            throw CardError("Invalid \(url.path): \(error.localizedDescription)")
        }
    }

    /// Downloads initialize a card only when the publisher did not supply one.
    public static func createIfMissing(directory: URL, name: String) throws {
        let url = directory.appendingPathComponent(filename)
        guard !FileManager.default.fileExists(atPath: url.path) else {
            _ = try load(directory: directory.path)
            return
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        do {
            try encoder.encode(Self(name: name)).write(to: url, options: .withoutOverwriting)
        } catch CocoaError.fileWriteFileExists {
            _ = try load(directory: directory.path)
        }
    }
}

private struct CardError: LocalizedError {
    let errorDescription: String?
    init(_ message: String) { errorDescription = message }
}
