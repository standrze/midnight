import Foundation

/// Optional runner settings loaded from a model-stack configuration file.
/// Token limits, context length, and prefill step count token positions.
public struct ModelStackSettings: Decodable, Sendable {
    /// MLX runtime and listener overrides from the model stack.
    public struct MLXRunner: Decodable, Sendable {
        public let modelPath: String?
        public let servedModelName: String?
        public let engine: String?
        public let host: String?
        public let port: Int?
        public let maximumTokens: Int?
        public let contextLength: Int?
        public let prefillStepSize: Int?
        public let kvCompression: String?
        public let dflashModelPath: String?
        public let dflashBlockSize: Int?
        public let gemmaAssistantModelPath: String?
        public let gemmaAssistantBlockSize: Int?
        public let gemmaAssistantQuantizationBits: Int?
        public let autoAssistant: Bool?

        /// Apply checkpoint-local limits. Settings tied to another checkpoint must
        /// not follow a CLI model switch; host, port and engine remain shared.
        public func resolving(for selection: ResolvedModelSelection) throws -> Self {
            let local = try ModelLocalSettings.load(directory: selection.settingsDirectory)
            let matches =
                modelPath.map {
                    let configured = ModelCatalog.resolveMLX(model: $0)
                    return URL(fileURLWithPath: configured.modelPath).resolvingSymlinksInPath()
                        == URL(fileURLWithPath: selection.modelPath).resolvingSymlinksInPath()
                } ?? true
            return Self(
                modelPath: selection.modelPath,
                servedModelName: matches ? servedModelName : nil,
                engine: engine, host: host, port: port,
                maximumTokens: local?.maximumTokens ?? (matches ? maximumTokens : nil),
                contextLength: local?.contextLength ?? (matches ? contextLength : nil),
                prefillStepSize: local?.prefillStepSize ?? (matches ? prefillStepSize : nil),
                kvCompression: local?.kvCompression ?? (matches ? kvCompression : nil),
                dflashModelPath: matches ? dflashModelPath : nil,
                dflashBlockSize: matches ? dflashBlockSize : nil,
                gemmaAssistantModelPath: matches ? gemmaAssistantModelPath : nil,
                gemmaAssistantBlockSize: matches ? gemmaAssistantBlockSize : nil,
                gemmaAssistantQuantizationBits: matches ? gemmaAssistantQuantizationBits : nil,
                autoAssistant: matches ? autoAssistant : nil)
        }

        /// An MLX runner configuration with no overrides.
        public static var empty: Self {
            Self(
                modelPath: nil, servedModelName: nil, engine: nil, host: nil, port: nil,
                maximumTokens: nil, contextLength: nil, prefillStepSize: nil,
                kvCompression: nil, dflashModelPath: nil, dflashBlockSize: nil,
                gemmaAssistantModelPath: nil, gemmaAssistantBlockSize: nil,
                gemmaAssistantQuantizationBits: nil, autoAssistant: nil)
        }
    }

    public let mlxRunner: MLXRunner?

    /// Loads the selected model-stack file, or returns nil when none is found.
    public static func load(explicitPath: String?) throws -> Self? {
        guard let url = try SettingsFileLocator.find(explicitPath: explicitPath) else {
            return nil
        }
        do {
            return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
        } catch {
            throw ModelStackSettingsError.invalidFile(url.path, error.localizedDescription)
        }
    }
}

/// Optional midnight.json alongside a checkpoint, or at an adapter bundle root.
/// Keep runtime policy separate from the checkpoint's architectural config.json.
/// Token limits, context length, and prefill step count token positions.
public struct ModelLocalSettings: Decodable, Sendable {
    public let contextLength: Int?
    public let maximumTokens: Int?
    public let prefillStepSize: Int?
    public let kvCompression: String?

    /// Loads and validates checkpoint-local `midnight.json`, if present.
    public static func load(directory: String) throws -> Self? {
        let url = URL(fileURLWithPath: directory).appendingPathComponent("midnight.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        do {
            let data = try Data(contentsOf: url)
            guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw RequestAdmissionError.configuration("midnight.json must be an object")
            }
            let known: Set<String> = ["contextLength", "maximumTokens", "prefillStepSize", "kvCompression"]
            let unknown = Set(object.keys).subtracting(known)
            guard unknown.isEmpty else {
                throw RequestAdmissionError.configuration(
                    "unknown midnight.json keys: \(unknown.sorted().joined(separator: ", "))")
            }
            return try JSONDecoder().decode(Self.self, from: data)
        } catch {
            throw ModelStackSettingsError.invalidFile(url.path, error.localizedDescription)
        }
    }
}

/// A model-stack or checkpoint-local settings file is missing or invalid.
public enum ModelStackSettingsError: LocalizedError {
    case missingFile(String)
    case invalidFile(String, String)

    /// User-facing explanation for this error.
    public var errorDescription: String? {
        switch self {
        case .missingFile(let path):
            "Settings file does not exist: \(path)"
        case .invalidFile(let path, let detail):
            "Could not read settings file \(path): \(detail)"
        }
    }
}

private enum SettingsFileLocator {
    static func find(explicitPath: String?) throws -> URL? {
        let fileManager = FileManager.default
        if let explicitPath = explicitPath?.trimmingCharacters(in: .whitespacesAndNewlines),
            !explicitPath.isEmpty
        {
            let url = normalizedURL(explicitPath)
            guard fileManager.fileExists(atPath: url.path) else {
                throw ModelStackSettingsError.missingFile(url.path)
            }
            return url
        }

        if let environmentPath = ProcessInfo.processInfo.environment["MODEL_STACK_CONFIG"],
            !environmentPath.isEmpty
        {
            let url = normalizedURL(environmentPath)
            guard fileManager.fileExists(atPath: url.path) else {
                throw ModelStackSettingsError.missingFile(url.path)
            }
            return url
        }

        let workingDirectory = URL(
            fileURLWithPath: fileManager.currentDirectoryPath,
            isDirectory: true
        )
        let candidates = [
            workingDirectory.appendingPathComponent("model-stack.local.json"),
            workingDirectory
                .deletingLastPathComponent()
                .appendingPathComponent("model-stack.local.json"),
            fileManager.homeDirectoryForCurrentUser
                .appendingPathComponent(".midnight", isDirectory: true)
                .appendingPathComponent("model-stack.local.json"),
        ]
        return candidates.first(where: { fileManager.fileExists(atPath: $0.path) })
    }

    private static func normalizedURL(_ path: String) -> URL {
        URL(
            fileURLWithPath: NSString(string: path).expandingTildeInPath
        ).standardizedFileURL
    }
}
