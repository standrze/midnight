import ArgumentParser
import Foundation
import HuggingFace
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

struct DownloadPreset: Codable, Sendable {
    let name: String
    let repo: String
    let note: String

    static let all: [Self] = [
        .init(name: "liquid-1.2b", repo: "LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit", note: "Publisher MLX 4-bit; LFM2.5"),
        .init(name: "liquid-8b", repo: "LiquidAI/LFM2.5-8B-A1B-MLX-4bit", note: "Publisher MLX 4-bit; ~4.8 GB weights"),
        .init(name: "liquid-24b", repo: "LiquidAI/LFM2-24B-A2B-MLX-4bit", note: "Publisher MLX 4-bit; LFM2, ~13.4 GB weights"),
        .init(name: "laguna-xs-2.1", repo: "poolside/Laguna-XS-2.1-NVFP4-mlx", note: "Publisher MLX NVFP4; ~21.6 GB; this exact quantization is not runtime-validated"),
        .init(name: "gpt-oss-20b", repo: "openai/gpt-oss-20b", note: "Publisher MXFP4; ~13.8 GB; raw weights expand in MLX memory, not an optimized MLX release"),
        .init(name: "gemma-e2b", repo: "google/gemma-4-E2B-it", note: "Publisher full precision; ~10.2 GB; Hugging Face license acceptance may be required"),
        .init(name: "gemma-e4b", repo: "google/gemma-4-E4B-it", note: "Publisher full precision; ~16.0 GB; Hugging Face license acceptance may be required"),
        .init(name: "gemma-12b", repo: "google/gemma-4-12B-it", note: "Publisher full precision; ~23.9 GB; unified architecture not runtime-validated"),
    ]

    static var catalogURL: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".midnight/config/downloads.json")
    }

    static func load(from url: URL = catalogURL) throws -> [Self] {
        if !FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
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
            let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
            for preset in presets {
                guard !preset.name.isEmpty, !preset.name.hasPrefix("."), !preset.name.hasPrefix("-"),
                      preset.name.unicodeScalars.allSatisfy(allowed.contains), names.insert(preset.name).inserted else {
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
        for preset in try load() {
            print("\(preset.name)  \(preset.repo)\n  \(preset.note)")
        }
    }
}

struct DownloadPlan {
    let files: [String]
    let bytes: Int64

    static func repository(_ value: String) throws -> Repo.ID {
        let parts = value.split(separator: "/", omittingEmptySubsequences: false)
        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        guard parts.count == 2, parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." && $0.unicodeScalars.allSatisfy(allowed.contains) }) else {
            throw ValidationError("Use a preset name or an owner/model repository ID.")
        }
        return Repo.ID(namespace: String(parts[0]), name: String(parts[1]))
    }

    init(entries: [(String, Int?)], maxGB: Double) throws {
        guard maxGB.isFinite, maxGB > 0, maxGB <= 1_000_000 else {
            throw ValidationError("--max-gb must be greater than zero and at most 1000000.")
        }
        // Native checkpoints use root-level weights. Do not download duplicate
        // original/, GGUF, Python code, or arbitrary repository artifacts.
        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        let selected = entries.filter { path, _ in
            !path.contains("/") && (path.hasSuffix(".safetensors") || path.hasSuffix(".json") || path.hasSuffix(".jinja") || ["tokenizer.model", "spiece.model", "merges.txt", "vocab.txt", "README.md", "LICENSE", "LICENSE.txt"].contains(path))
        }
        guard selected.contains(where: { $0.0 == "config.json" }), selected.contains(where: { $0.0.hasSuffix(".safetensors") }) else {
            throw ValidationError("Repository needs root-level config.json and safetensors weights. GGUF and nested checkpoints are not supported by this command.")
        }
        var total: Int64 = 0
        var paths = Set<String>()
        for (path, size) in selected {
            guard path.unicodeScalars.allSatisfy(allowed.contains), !path.hasPrefix("."), paths.insert(path).inserted,
                  let size, size >= 0 else {
                throw ValidationError("Invalid or missing file metadata; download refused.")
            }
            let (next, overflow) = total.addingReportingOverflow(Int64(size))
            guard !overflow else { throw ValidationError("Repository size overflow.") }
            total = next
        }
        guard Double(total) <= maxGB * 1_000_000_000 else {
            throw ValidationError(String(format: "Download is %.2f GB, above the %.2f GB limit. Choose a smaller checkpoint or explicitly raise --max-gb.", Double(total) / 1e9, maxGB))
        }
        files = paths.sorted()
        bytes = total
    }
}

struct DownloadCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "download", abstract: "Download a Hugging Face checkpoint into ~/.midnight/models.")
    @Argument(help: "Preset name or owner/model repository ID") var model: String?
    @Flag(name: .long, help: "Show publisher presets") var list = false
    @Flag(name: .long, help: "Check metadata and size without downloading weights") var dryRun = false
    @Option(name: .long, help: "Branch, tag, or commit; resolved to a fixed commit before downloading") var revision = "main"
    @Option(name: .long, help: "Maximum selected download size in decimal GB") var maxGB: Double = 30

    mutating func run() async throws {
        if list || model == nil {
            try DownloadPreset.printCatalog()
            return
        }
        guard let model else { throw ValidationError("Choose a model, or use midnight download --list.") }
        let preset = model.contains("/") ? nil : try DownloadPreset.load().first { $0.name == model }
        let repo = try DownloadPlan.repository(preset?.repo ?? model)
        let client = HubClient()
        let metadata = try await client.getModel(repo, revision: revision, filesMetadata: true)
        guard let sha = metadata.sha, sha.count == 40, sha.allSatisfy({ $0.isHexDigit }), let siblings = metadata.siblings else {
            throw ValidationError("Hugging Face returned incomplete commit or file metadata.")
        }
        let plan = try DownloadPlan(entries: siblings.map { ($0.relativeFilename, $0.size) }, maxGB: maxGB)
        let directoryName = preset?.name ?? "\(repo.namespace)--\(repo.name)"
        let root = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".midnight/models", isDirectory: true)
        let destination = root.appendingPathComponent(directoryName, isDirectory: true)
        print("Repository: \(repo.namespace)/\(repo.name)\nCommit: \(sha)")
        if let preset { print(preset.note) }
        print(String(format: "Download: %.2f GB (%d files)", Double(plan.bytes) / 1e9, plan.files.count))
        print("Destination: \(destination.path)")
        if dryRun { return }
        guard !FileManager.default.fileExists(atPath: destination.path) else {
            throw ValidationError("Destination already exists; refusing to replace it: \(destination.path)")
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let staging = root.appendingPathComponent(".download-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: staging) }
        print("Downloading with the Swift Hugging Face client (up to 8 concurrent files). Cached files can be reused after interruption.")
        _ = try await client.downloadSnapshot(of: repo, to: staging, revision: sha, matching: plan.files)
        for file in plan.files {
            let attrs = try FileManager.default.attributesOfItem(atPath: staging.appendingPathComponent(file).path)
            let expected = siblings.first { $0.relativeFilename == file }?.size
            guard attrs[.type] as? FileAttributeType == .typeRegular,
                  (attrs[.size] as? NSNumber)?.intValue == expected else {
                throw ValidationError("Downloaded file size/type mismatch: \(file)")
            }
        }
        let provenance = ["repository": "\(repo.namespace)/\(repo.name)", "revision": sha]
        try JSONEncoder().encode(provenance).write(to: staging.appendingPathComponent("midnight-download.json"))
        try FileManager.default.moveItem(at: staging, to: destination)
        print("Downloaded. Run with: midnight --model \(directoryName)")
        print("Loading still depends on the checkpoint architecture, quantization, and available RAM.")
    }
}

struct AuthCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "auth", abstract: "Manage the local Hugging Face token.", subcommands: [Login.self, Logout.self, Status.self])

    static var tokenURL: URL {
        let env = ProcessInfo.processInfo.environment
        if let path = env["HF_TOKEN_PATH"], !path.isEmpty { return URL(fileURLWithPath: path) }
        if let path = env["HF_HOME"], !path.isEmpty { return URL(fileURLWithPath: path).appendingPathComponent("token") }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/token")
    }

    struct Login: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Save a read token locally using Hugging Face's standard token file (0600).")
        mutating func run() async throws {
            guard isatty(STDIN_FILENO) != 0, let input = getpass("Hugging Face read token (hidden): ") else {
                throw ValidationError("Run midnight auth login in an interactive terminal, or set HF_TOKEN.")
            }
            let token = String(cString: input).trimmingCharacters(in: .whitespacesAndNewlines)
            memset(input, 0, strlen(input))
            guard token.hasPrefix("hf_"), !token.contains(where: { $0.isWhitespace }) else {
                throw ValidationError("Expected a Hugging Face token beginning with hf_.")
            }
            var request = URLRequest(url: URL(string: "https://huggingface.co/api/whoami-v2")!)
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            let (_, response) = try await URLSession.shared.data(for: request)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                throw ValidationError("Hugging Face did not accept the token; nothing was saved.")
            }
            let url = AuthCommand.tokenURL
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            let temp = url.deletingLastPathComponent().appendingPathComponent(".midnight-token-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: temp) }
            guard FileManager.default.createFile(atPath: temp.path, contents: Data(token.utf8), attributes: [.posixPermissions: 0o600]) else {
                throw ValidationError("Could not write token file.")
            }
            guard rename(temp.path, url.path) == 0 else { throw ValidationError("Could not install token file.") }
            print("Token verified and saved locally. HF_TOKEN takes precedence if set.")
        }
    }

    struct Logout: ParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Remove the shared Hugging Face token file; environment variables remain unchanged.")
        mutating func run() throws {
            let url = AuthCommand.tokenURL
            if FileManager.default.fileExists(atPath: url.path) { try FileManager.default.removeItem(at: url) }
            print("Local token file removed. Unset HF_TOKEN and HUGGING_FACE_HUB_TOKEN separately if used; other Hugging Face tools share this file.")
        }
    }

    struct Status: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Report whether a token is available without displaying it.")
        mutating func run() async throws {
            let token = await HubClient().bearerToken
            print(token?.isEmpty == false ? "Hugging Face token available (not validated)." : "No Hugging Face token found. Run midnight auth login.")
        }
    }
}
