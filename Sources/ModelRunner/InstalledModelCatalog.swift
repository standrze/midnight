import Foundation
import ModelRunnerProtocol

struct InstalledModelEntry: Sendable {
    let request: ModelLoadRequest
    let descriptor: ModelLifecycleDescriptor
}

extension ModelLoader {
    /// Metadata-only discovery. Never accepts an inference-supplied path.
    func installedModels(roots: [URL]? = nil) -> [InstalledModelEntry] {
        let environment = ProcessInfo.processInfo.environment
        let extra = (environment["MIDNIGHT_MODEL_DIRS"] ?? "").split(separator: ":")
            .map { URL(fileURLWithPath: String($0), isDirectory: true) }
        let directories =
            roots ?? [
                ModelCatalog.defaultDirectory(),
                URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                    .appendingPathComponent("Models", isDirectory: true),
            ] + extra
        var paths = Set<String>()
        var entries: [InstalledModelEntry] = []
        for directory in directories {
            for name in ModelCatalog.availableModels(modelsDirectory: directory) {
                let path = directory.appendingPathComponent(name).resolvingSymlinksInPath().path
                guard paths.insert(path).inserted,
                    let config = try? validate(ModelLoadRequest(model: path)),
                    let request = config.loadRequest
                else {
                    continue
                }
                let descriptor = ModelLifecycleDescriptor(
                    id: config.servedModelName, created: 0,
                    contextLength: nil, prefillStepSize: nil, kvCompression: nil,
                    memoryLimitBytes: nil,
                    maximumOutputTokens: config.modality == "text" ? config.tokenLimit.configuredMaximum : nil,
                    defaultOutputTokens: config.modality == "text" ? config.tokenLimit.defaultTokens : nil,
                    modality: config.modality, loadRequest: request, modelCard: config.modelCard,
                    nativeProtocol: nativeModelProtocol(at: path))
                entries.append(InstalledModelEntry(request: request, descriptor: descriptor))
            }
        }
        // Ambiguous IDs are not routable: configure unique servedModelName values.
        let groups = Dictionary(grouping: entries, by: { $0.descriptor.id })
        return groups.values.compactMap { $0.count == 1 ? $0[0] : nil }
            .sorted { $0.descriptor.id < $1.descriptor.id }
    }
}

/// Protocol capability comes from checkpoint metadata, never the served alias.
func nativeModelProtocol(at path: String?) -> String? {
    guard let path,
        let data = try? Data(contentsOf: URL(fileURLWithPath: path).appendingPathComponent("config.json")),
        let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
        let type = object["model_type"] as? String
    else {
        return nil
    }
    switch type {
    case "laguna": return "laguna"
    case "muse_glimmer": return "muse"
    default: return nil
    }
}
