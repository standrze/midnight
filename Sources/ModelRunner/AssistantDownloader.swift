import Foundation
import HuggingFace
import ModelRunnerProtocol

enum AssistantDownloader {
    typealias Validate = @Sendable (Data) throws -> Void
    typealias Transfer = @Sendable (String, URL, Validate) async throws -> Void

    /// Stage and validate before publishing. Never overwrite an installed assistant.
    static func install(
        _ selection: AssistantDiscovery.Selection, target: Data,
        blockSize: Int?, quantizationBits: Int?,
        transfer: Transfer = fetch
    ) async throws {
        guard let repository = selection.repository else {
            return
        }
        let validate: Validate = { assistant in
            try AssistantDiscovery.validate(
                kind: selection.kind, target: target,
                assistant: assistant, blockSize: blockSize, quantizationBits: quantizationBits)
        }
        let destination = selection.directory
        if FileManager.default.fileExists(atPath: destination.path) {
            try validate(ModelLoader.validateModelFolder(destination))
            return
        }
        let root = destination.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let staging = root.appendingPathComponent(".download-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: staging) }
        print("Downloading named assistant: \(repository)")
        try await transfer(repository, staging, validate)
        try Task.checkCancellation()
        try validate(ModelLoader.validateModelFolder(staging))
        try ModelCard.createIfMissing(directory: staging, name: repository)
        do { try FileManager.default.moveItem(at: staging, to: destination) } catch {
            // A concurrent process may have completed the same download first.
            guard FileManager.default.fileExists(atPath: destination.path) else {
                throw error
            }
            try validate(ModelLoader.validateModelFolder(destination))
        }
    }

    private static func fetch(_ repository: String, _ staging: URL, _ validate: Validate) async throws {
        let repo = try DownloadPlan.repository(repository)
        let client = HubClient()
        let metadata = try await client.getModel(repo, revision: "main", filesMetadata: true)
        guard let sha = metadata.sha, sha.count == 40, sha.allSatisfy(\.isHexDigit),
            let siblings = metadata.siblings
        else {
            throw ModelLoadingError("Assistant repository returned incomplete commit or file metadata")
        }
        let plan = try DownloadPlan(entries: siblings.map { ($0.relativeFilename, $0.size) }, maxGB: 30)
        // Reject the wrong architecture before downloading its weights.
        guard let configSize = siblings.first(where: { $0.relativeFilename == "config.json" })?.size,
            configSize <= 1_048_576
        else {
            throw ModelLoadingError("Assistant configuration exceeds 1 MiB")
        }
        let configURL = try await client.downloadFile(at: "config.json", from: repo, revision: sha)
        try validate(Data(contentsOf: configURL))
        _ = try await client.downloadSnapshot(of: repo, to: staging, revision: sha, matching: plan.files)
        for file in plan.files {
            let attrs = try FileManager.default.attributesOfItem(atPath: staging.appendingPathComponent(file).path)
            let expected = siblings.first { $0.relativeFilename == file }?.size
            guard attrs[.type] as? FileAttributeType == .typeRegular,
                (attrs[.size] as? NSNumber)?.intValue == expected
            else {
                throw ModelLoadingError("Downloaded assistant file size/type mismatch: \(file)")
            }
        }
        try JSONEncoder().encode(["repository": repository, "revision": sha])
            .write(to: staging.appendingPathComponent("midnight-download.json"))
    }
}
