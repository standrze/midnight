import ArgumentParser
import Foundation
import HuggingFace
import ModelRunnerProtocol

/// Shared native Swift transfer path for the CLI and loom console.
enum ModelDownloadService {
    typealias Transfer = @Sendable (Artifact, URL) async throws -> Void
    typealias Update = @MainActor @Sendable (String, Progress?) -> Void

    static func errorMessage(_ error: any Error) -> String {
        (error as? ValidationError)?.message ?? error.localizedDescription
    }

    struct Artifact: Sendable {
        let repository: Repo.ID
        let revision: String
        let files: [(String, Int?)]
        let plan: DownloadPlan
        let configuration: Data
    }

    static func download(
        repository: String, name: String, revision: String = "main", maxGB: Double = 30,
        dryRun: Bool = false, targetOnly: Bool = false,
        root: URL = ManagedModelDownloads().root, update: Update? = nil
    ) async throws {
        try ManagedModelDownloads.validateName(name)
        if repository == "bespokelabs/Bespoke-Nimble-9B" {
            guard !targetOnly else { throw ValidationError("Nimble requires its exact base and adapter together.") }
            try await downloadNimble(
                name: name, revision: revision, maxGB: maxGB, dryRun: dryRun, root: root, update: update)
            return
        }
        let selection = try ModelDownloadCatalog.selection(repository: repository)
        let client = HubClient()
        await update?("Checking publisher metadata…", nil)
        let target = try await prepare(repository: repository, revision: revision, maxGB: maxGB, client: client)
        try ModelDownloadCatalog.validateQuantization(target.configuration)
        let assistant: Artifact?
        if !targetOnly, let companion = selection.assistant {
            let prepared = try await prepare(repository: companion, revision: "main", maxGB: maxGB, client: client)
            try ModelLoader.validateDFlash(
                target: target.configuration, draft: prepared.configuration, blockSize: nil)
            assistant = prepared
        } else {
            assistant = nil
        }
        let total = target.plan.bytes + (assistant?.plan.bytes ?? 0)
        guard Double(total) <= maxGB * 1e9 else {
            throw ValidationError(
                String(
                    format: "Model and assistant total %.2f GB, above the %.2f GB limit.", Double(total) / 1e9, maxGB))
        }
        let destination = root.appendingPathComponent(name, isDirectory: true)
        print("Repository: \(repository)\nCommit: \(target.revision)\n\(selection.note)")
        if let assistant {
            print(
                "Assistant: \(assistant.repository.namespace)/\(assistant.repository.name)\nAssistant commit: \(assistant.revision)"
            )
        } else {
            print(
                targetOnly
                    ? "Target-only download requested."
                    : "No eligible publisher assistant is registered for this model.")
        }
        print(String(format: "Total download: %.2f GB", Double(total) / 1e9))
        print("Destination: \(destination.path)")
        await update?(String(format: "Prepared %.2f GB including selected companions", Double(total) / 1e9), nil)
        if dryRun { return }
        try await install(target: target, assistant: assistant, name: name, root: root) { artifact, directory in
            try await transfer(artifact, to: directory, client: client, update: update)
        }
        await update?("Downloaded \(name)\(assistant == nil ? "" : " with assistant").", nil)
        print("Downloaded. Run with: midnight --model \(name)")
        print("Runtime compatibility, memory and assistant performance still require validation.")
    }

    /// Nimble's official adapter is a model component, separate from speculative assistants.
    private static func downloadNimble(
        name: String, revision: String, maxGB: Double, dryRun: Bool,
        root: URL, update: Update?
    ) async throws {
        let client = HubClient()
        let adapter = try await prepare(
            repository: "bespokelabs/Bespoke-Nimble-9B", revision: revision,
            maxGB: maxGB, client: client, adapter: true)
        let schemaURL = try await client.downloadFile(
            at: "schema_config.json", from: adapter.repository, revision: adapter.revision)
        let temperatureURL = try await client.downloadFile(
            at: "temperature_config.json", from: adapter.repository, revision: adapter.revision)
        struct Temperature: Decodable {
            let temperature: Double
            // Keep the publisher's JSON field name stable.
            // swift-format-ignore: AlwaysUseLowerCamelCase
            let adapter_sha256: String
        }
        let temperature = try JSONDecoder().decode(Temperature.self, from: Data(contentsOf: temperatureURL))
        let contract = try DecisionModelContract.importNimble(
            schema: Data(contentsOf: schemaURL), temperature: temperature.temperature)
        guard contract.model == "Qwen/Qwen3.5-9B" else {
            throw ValidationError("Nimble base changed; this pairing requires explicit compatibility review.")
        }
        let base = try await prepare(
            repository: contract.model, revision: contract.revision, maxGB: maxGB, client: client)
        let total = base.plan.bytes + adapter.plan.bytes
        guard Double(total) <= maxGB * 1e9 else { throw ValidationError("Nimble base and adapter exceed --max-gb.") }
        print("Nimble adapter: bespokelabs/Bespoke-Nimble-9B @ \(adapter.revision)")
        print("Required base: \(contract.model) @ \(base.revision)")
        print(
            String(
                format: "Total download: %.2f GB (official full-precision base and LoRA adapter).", Double(total) / 1e9)
        )
        print("This scoped decision-model bundle requires Afterglow preparation before decision serving.")
        if dryRun { return }
        let destination = root.appendingPathComponent(name)
        guard !FileManager.default.fileExists(atPath: destination.path) else {
            throw ValidationError("Destination already exists: \(destination.path)")
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let staging = root.appendingPathComponent(".download-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: staging) }
        try await transfer(base, to: staging.appendingPathComponent("base"), client: client, update: update)
        try verify(base, directory: staging.appendingPathComponent("base"))
        try await transfer(
            adapter, to: staging.appendingPathComponent("publisher-adapter"), client: client, update: update)
        try verify(adapter, directory: staging.appendingPathComponent("publisher-adapter"))
        try Task.checkCancellation()
        try JSONEncoder().encode([
            "status": "downloaded_requires_afterglow_preparation",
            "base_revision": base.revision, "adapter_revision": adapter.revision,
            "adapter_sha256": temperature.adapter_sha256,
        ])
        .write(to: staging.appendingPathComponent("nimble-bundle.json"))
        try FileManager.default.moveItem(at: staging, to: destination)
        print("Downloaded complete source bundle: \(destination.path)")
        print(
            "Prepare with: afterglow convert --source \(destination.path)/publisher-adapter --revision \(adapter.revision) --output \(destination.path)/adapter"
        )
    }

    /// Publish one bundle only after every component succeeds. The transfer seam supports offline failure checks.
    static func install(target: Artifact, assistant: Artifact?, name: String, root: URL, transfer: Transfer)
        async throws
    {
        try ManagedModelDownloads.validateName(name)
        try Task.checkCancellation()
        let destination = root.appendingPathComponent(name, isDirectory: true)
        guard !FileManager.default.fileExists(atPath: destination.path) else {
            throw ValidationError("Destination already exists; refusing to replace it: \(destination.path)")
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let staging = root.appendingPathComponent(".download-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: staging) }
        try await transfer(target, staging)
        try verify(target, directory: staging)
        if let assistant {
            let companionDirectory = staging.appendingPathComponent("assistant", isDirectory: true)
            try await transfer(assistant, companionDirectory)
            try verify(assistant, directory: companionDirectory)
            try ModelCard.createIfMissing(directory: staging, name: name)
            var card = try ModelCard.load(directory: staging.path) ?? ModelCard(name: name)
            card.assistantModel = "assistant"
            try JSONEncoder().encode(card).write(
                to: staging.appendingPathComponent(ModelCard.filename), options: .atomic)
        } else {
            try ModelCard.createIfMissing(directory: staging, name: name)
        }
        try Task.checkCancellation()
        try FileManager.default.moveItem(at: staging, to: destination)
    }

    private static func prepare(
        repository: String, revision: String, maxGB: Double, client: HubClient, adapter: Bool = false
    ) async throws
        -> Artifact
    {
        let repo = try DownloadPlan.repository(repository)
        let metadata = try await client.getModel(repo, revision: revision, filesMetadata: true)
        guard let sha = metadata.sha, sha.count == 40, sha.allSatisfy(\.isHexDigit), let siblings = metadata.siblings
        else {
            throw ValidationError("Hugging Face returned incomplete commit or file metadata.")
        }
        let files = siblings.map { ($0.relativeFilename, $0.size) }
        let plan = try DownloadPlan(entries: files, maxGB: maxGB, adapter: adapter)
        let configName = adapter ? "adapter_config.json" : "config.json"
        guard let size = siblings.first(where: { $0.relativeFilename == configName })?.size, size <= 1_048_576 else {
            throw ValidationError("Model configuration exceeds 1 MiB.")
        }
        let config = try await client.downloadFile(at: configName, from: repo, revision: sha)
        let configuration = try Data(contentsOf: config)
        guard configuration.count <= 1_048_576 else { throw ValidationError("Model configuration exceeds 1 MiB.") }
        return Artifact(repository: repo, revision: sha, files: files, plan: plan, configuration: configuration)
    }

    private static func transfer(_ artifact: Artifact, to directory: URL, client: HubClient, update: Update?)
        async throws
    {
        try Task.checkCancellation()
        let label = "Downloading \(artifact.repository.namespace)/\(artifact.repository.name)"
        await update?(label, nil)
        _ = try await client.downloadSnapshot(
            of: artifact.repository, to: directory, revision: artifact.revision,
            matching: artifact.plan.files, progressHandler: { progress in update?(label, progress) })
        try Task.checkCancellation()
    }

    private static func verify(_ artifact: Artifact, directory: URL) throws {
        for file in artifact.plan.files {
            let attrs = try FileManager.default.attributesOfItem(atPath: directory.appendingPathComponent(file).path)
            let expected = artifact.files.first { $0.0 == file }?.1
            guard attrs[.type] as? FileAttributeType == .typeRegular, (attrs[.size] as? NSNumber)?.intValue == expected
            else {
                throw ValidationError("Downloaded file size/type mismatch: \(file)")
            }
        }
        let repository = "\(artifact.repository.namespace)/\(artifact.repository.name)"
        try JSONEncoder().encode(["repository": repository, "revision": artifact.revision])
            .write(to: directory.appendingPathComponent("midnight-download.json"))
    }
}
