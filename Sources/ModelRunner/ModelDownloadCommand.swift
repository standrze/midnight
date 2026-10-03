import ArgumentParser
import Foundation
import HuggingFace
import ModelRunnerProtocol

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

struct DownloadPreset: Codable, Sendable {
    let name: String
    let repo: String
    let note: String

    static let all: [Self] = ModelDownloadCatalog.entries.map { entry in
        return Self(name: entry.name, repo: entry.repository, note: entry.note)
    }

    static var catalogURL: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".midnight/config/downloads.json")
    }

    static func load(from url: URL = catalogURL) throws -> [Self] {
        if !FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            do {
                try encoder.encode(all).write(to: url, options: .withoutOverwriting)
            } catch CocoaError.fileWriteFileExists {
                // Another invocation initialized the catalog first; read its copy.
            }
        }
        do {
            let presets = try JSONDecoder().decode([Self].self, from: Data(contentsOf: url))
            var names = Set<String>()
            let allowed = CharacterSet(
                charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
            for preset in presets {
                guard !preset.name.isEmpty, !preset.name.hasPrefix("."), !preset.name.hasPrefix("-"),
                    preset.name.unicodeScalars.allSatisfy(allowed.contains), names.insert(preset.name).inserted
                else {
                    throw ValidationError("Preset names must be unique, safe folder names.")
                }
                _ = try DownloadPlan.repository(preset.repo)
            }
            return presets
        } catch {
            throw ValidationError("Invalid download catalog at \(url.path): \(error.localizedDescription)")
        }
    }

    static func printCatalog() throws {
        for preset in try catalog() {
            print("\(preset.name)  \(preset.repo)\n  \(preset.note)")
            do {
                _ = try ModelDownloadCatalog.selection(repository: preset.repo)
                print("  Download available")
            } catch {
                print("  Unavailable: \(ModelDownloadService.errorMessage(error))")
            }
        }
    }

    /// Always include the current scope, even when an older saved catalog omits it.
    static func catalog(from url: URL = catalogURL) throws -> [Self] {
        let saved = try load(from: url)
        let names = Set(all.map(\.name))
        let repositories = Set(all.map(\.repo))
        return all + saved.filter { !names.contains($0.name) && !repositories.contains($0.repo) }
    }
}

struct DownloadPlan: Sendable {
    let files: [String]
    let bytes: Int64

    static func repository(_ value: String) throws -> Repo.ID {
        let parts = value.split(separator: "/", omittingEmptySubsequences: false)
        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        guard parts.count == 2,
            parts.allSatisfy({
                !$0.isEmpty && $0 != "." && $0 != ".." && $0.unicodeScalars.allSatisfy(allowed.contains)
            })
        else {
            throw ValidationError("Use a preset name or an owner/model repository ID.")
        }
        return Repo.ID(namespace: String(parts[0]), name: String(parts[1]))
    }

    init(entries: [(String, Int?)], maxGB: Double, adapter: Bool = false) throws {
        guard maxGB.isFinite, maxGB > 0, maxGB <= 1_000_000 else {
            throw ValidationError("--max-gb must be greater than zero and at most 1000000.")
        }
        // Native checkpoints use root-level weights. Voxtral additionally
        // keeps its required preset voice embeddings in a single nested
        // directory. Do not download duplicate original/, GGUF, Python code,
        // or arbitrary repository artifacts.
        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        let selected = entries.filter { path, _ in
            let rootFile =
                !path.contains("/")
                && (path.hasSuffix(".safetensors") || path.hasSuffix(".json") || path.hasSuffix(".jinja")
                    || [
                        "tokenizer.model", "spiece.model", "merges.txt", "vocab.txt", "README.md", "LICENSE",
                        "LICENSE.txt",
                    ].contains(path))
            let voiceEmbedding =
                path.hasPrefix("voice_embedding/")
                && path.dropFirst("voice_embedding/".count).unicodeScalars.allSatisfy(allowed.contains)
                && path.hasSuffix(".safetensors")
            return rootFile || voiceEmbedding
        }
        guard selected.contains(where: { $0.0 == (adapter ? "adapter_config.json" : "config.json") }),
            selected.contains(where: { $0.0.hasSuffix(".safetensors") })
        else {
            throw ValidationError(
                "Repository needs root-level config.json and safetensors weights. GGUF and nested checkpoints are not supported by this command."
            )
        }
        var total: Int64 = 0
        var paths = Set<String>()
        for (path, size) in selected {
            let safePath =
                path.unicodeScalars.allSatisfy(allowed.contains)
                || (path.hasPrefix("voice_embedding/")
                    && path.dropFirst("voice_embedding/".count).unicodeScalars.allSatisfy(allowed.contains))
            guard safePath, !path.hasPrefix("."), paths.insert(path).inserted,
                let size, size >= 0
            else {
                throw ValidationError("Invalid or missing file metadata; download refused.")
            }
            let (next, overflow) = total.addingReportingOverflow(Int64(size))
            guard !overflow else {
                throw ValidationError("Repository size overflow.")
            }
            total = next
        }
        guard Double(total) <= maxGB * 1_000_000_000 else {
            throw ValidationError(
                String(
                    format:
                        "Download is %.2f GB, above the %.2f GB limit. Choose a smaller checkpoint or explicitly raise --max-gb.",
                    Double(total) / 1e9, maxGB))
        }
        files = paths.sorted()
        bytes = total
    }
}

struct DownloadCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "download", abstract: "Download a Hugging Face checkpoint into ~/.midnight/models.")
    @Argument(help: "Preset name or owner/model repository ID") var model: String?
    @OptionGroup var listing: ModelListOptions
    @Flag(name: .long, help: "Check metadata and size without downloading weights") var dryRun = false
    @Option(name: .long, help: "Branch, tag, or commit; resolved to a fixed commit before downloading") var revision =
        "main"
    @Option(name: .long, help: "Maximum selected download size in decimal GB") var maxGB: Double = 30

    @Flag(name: .long, help: "Download only the target, omitting its designated assistant") var targetOnly = false

    mutating func run() async throws {
        if listing.list || model == nil {
            try DownloadPreset.printCatalog()
            return
        }
        guard let model else {
            throw ValidationError("Choose a model, or use midnight download --list.")
        }
        let preset = model.contains("/") ? nil : try DownloadPreset.catalog().first { $0.name == model }
        let repository = preset?.repo ?? model
        let repo = try DownloadPlan.repository(repository)
        let name = preset?.name ?? "\(repo.namespace)--\(repo.name)"
        try await ModelDownloadService.download(
            repository: repository, name: name, revision: revision, maxGB: maxGB,
            dryRun: dryRun, targetOnly: targetOnly)
    }
}
