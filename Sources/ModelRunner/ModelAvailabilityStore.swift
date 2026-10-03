import Foundation
import ModelRunnerProtocol

/// Stores availability by canonical checkpoint/bundle path so served aliases share the same policy.
/// The lifecycle actor owns mutation; a failed atomic write leaves the previous policy in effect.
struct ModelAvailabilityStore: Sendable {
    private struct Document: Codable {
        var unavailableModels: [String]
    }

    private let file: URL?
    private var unavailableModels: Set<String>

    init() {
        file = nil
        unavailableModels = []
    }

    init(file: URL) throws {
        self.file = file
        if FileManager.default.fileExists(atPath: file.path) {
            let document = try JSONDecoder().decode(Document.self, from: Data(contentsOf: file))
            unavailableModels = Set(document.unavailableModels)
        } else {
            unavailableModels = []
        }
    }

    static func configured(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser
    ) throws -> Self {
        let file =
            environment["MIDNIGHT_MODEL_AVAILABILITY_FILE"].map { URL(fileURLWithPath: $0) }
            ?? homeDirectory.appendingPathComponent(".midnight/config/model-availability.json")
        return try Self(file: file)
    }

    func isAvailable(_ request: ModelLoadRequest) -> Bool {
        !unavailableModels.contains(Self.key(for: request))
    }

    mutating func setAvailable(_ available: Bool, for request: ModelLoadRequest) throws {
        var updated = unavailableModels
        let key = Self.key(for: request)
        if available {
            updated.remove(key)
        } else {
            updated.insert(key)
        }
        if let file {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(Document(unavailableModels: updated.sorted()))
            try FileManager.default.createDirectory(
                at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: file, options: .atomic)
        }
        unavailableModels = updated
    }

    private static func key(for request: ModelLoadRequest) -> String {
        let selection = ModelCatalog.resolveMLX(model: request.model, adapter: request.adapter)
        return URL(fileURLWithPath: selection.settingsDirectory).standardizedFileURL.resolvingSymlinksInPath().path
    }
}

/// The console retains installed models even when they are hidden from API discovery.
struct ModelAvailabilityEntry: Sendable {
    let model: ModelLifecycleDescriptor
    let available: Bool
}
