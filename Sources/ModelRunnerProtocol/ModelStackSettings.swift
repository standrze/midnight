import Foundation

public struct ModelStackSettings: Decodable, Sendable {
    public struct MLXRunner: Decodable, Sendable {
        public let modelPath: String?
        public let servedModelName: String?
        public let engine: String?
        public let host: String?
        public let port: Int?
        public let maximumTokens: Int?
        public let dflashModelPath: String?
        public let dflashBlockSize: Int?
    }

    public let mlxRunner: MLXRunner?

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

public enum ModelStackSettingsError: LocalizedError {
    case missingFile(String)
    case invalidFile(String, String)

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
        ]
        return candidates.first(where: { fileManager.fileExists(atPath: $0.path) })
    }

    private static func normalizedURL(_ path: String) -> URL {
        URL(
            fileURLWithPath: NSString(string: path).expandingTildeInPath
        ).standardizedFileURL
    }
}
