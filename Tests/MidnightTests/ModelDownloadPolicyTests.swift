import ArgumentParser
import Foundation
import ModelRunnerProtocol
import Testing

@testable import Midnight

@Suite("Publisher download policy")
struct ModelDownloadPolicyTests {
    @Test func oldSavedCatalogCannotHideOrOverrideCurrentScope() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("downloads.json")
        let saved = [
            DownloadPreset(name: "gemma-31b", repo: "community/wrong-model", note: "Stale override"),
            DownloadPreset(name: "custom", repo: "custom/model", note: "Keep my entry"),
        ]
        let bytes = try JSONEncoder().encode(saved)
        try bytes.write(to: url)
        let catalog = try DownloadPreset.catalog(from: url)
        #expect(
            catalog.prefix(ModelDownloadCatalog.entries.count).map(\.name)
                == ModelDownloadCatalog.entries.map(\.name))
        #expect(catalog.first { $0.name == "gemma-31b" }?.repo == "google/gemma-4-31B-it")
        #expect(catalog.last?.name == "custom")
        #expect(try Data(contentsOf: url) == bytes)
        #expect(
            Set(
                catalog.filter { preset in
                    (try? ModelDownloadCatalog.selection(repository: preset.repo)) != nil
                }.map(\.repo)
            ) == [
                "bespokelabs/Bespoke-Nimble-9B", "openai/gpt-oss-20b", "openai/gpt-oss-120b",
                "poolside/Laguna-XS-2.1-NVFP4-mlx", "poolside/Laguna-S-2.1-NVFP4-mlx",
            ])
        #expect(!DownloadPreset.all.contains { $0.name.contains("talkie") || $0.name.contains("liquid") })
    }

    @Test func consolePreservesActionableDownloadErrors() {
        #expect(
            ModelDownloadService.errorMessage(ValidationError("Bundle exceeds the 30 GB limit."))
                == "Bundle exceeds the 30 GB limit.")
    }

    @Test func excludesCommunityAndDeferredMuse() {
        for repository in [
            "mlx-community/Muse-Glimmer-30B-4bit", "meta-models/Muse-Glimmer-30B", "z-lab/gpt-oss-20b-DFlash",
            "google/gemma-4-31B-it",
        ] {
            #expect(throws: (any Error).self) { try ModelDownloadCatalog.selection(repository: repository) }
        }
    }

    @Test func lagunaUsesTheNVFP4TargetAssistant() throws {
        let selection = try ModelDownloadCatalog.selection(repository: "poolside/Laguna-XS-2.1-NVFP4-mlx")
        #expect(selection.assistant == "poolside/Laguna-XS-2.1-DFlash-NVFP4")
        let gpt = try ModelDownloadCatalog.selection(repository: "openai/gpt-oss-20b")
        #expect(gpt.assistant == nil)
    }

    @Test func verifiesQuantizationMetadata() throws {
        try ModelDownloadCatalog.validateQuantization(Data(#"{"quantization":{"bits":4,"mode":"nvfp4"}}"#.utf8))
        try ModelDownloadCatalog.validateQuantization(Data(#"{"quantization_config":{"quant_method":"mxfp4"}}"#.utf8))
        for config in [#"{}"#, #"{"quantization":{"bits":8}}"#, #"{"quantization_config":{"quant_method":"fp8"}}"#] {
            #expect(throws: (any Error).self) { try ModelDownloadCatalog.validateQuantization(Data(config.utf8)) }
        }
    }

    @Test func deletionConfirmationKeepsTheRequestedIDAndCanBeCancelled() {
        var management = ConsoleModelManagement()
        management.requestDeletion("original-selection")
        management.downloadIndex = 4
        #expect(management.prompt?.contains("original-selection") == true)
        #expect(management.confirmDeletion() == "original-selection")
        #expect(management.confirmDeletion() == nil)
        management.requestDeletion("keep-this")
        management.cancelDeletion()
        #expect(management.confirmDeletion() == nil)
    }

    @Test func bundlePublishesBothComponentsWithProvenance() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let target = try artifact("poolside/target")
        let assistant = try artifact("poolside/assistant")
        try await ModelDownloadService.install(
            target: target, assistant: assistant, name: "bundle", root: root, transfer: Self.writeFixture)
        let destination = root.appendingPathComponent("bundle")
        #expect(
            FileManager.default.fileExists(
                atPath: destination.appendingPathComponent("assistant/model.safetensors").path))
        let card = try ModelRunnerProtocol.ModelCard.load(directory: destination.path)
        #expect(card?.assistantModel == "assistant")
        let managed = try ManagedModelDownloads(root: root).list(protectedDirectories: [])
        #expect(managed.data.map(\.id) == ["bundle"])
        let detached = try ManagedModelDownloads(root: root).detach(named: "bundle", protectedDirectories: [])
        _ = try detached.finish()
        #expect(!FileManager.default.fileExists(atPath: destination.path))
    }

    @Test func failedAssistantLeavesNoPublishedTarget() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        do {
            try await ModelDownloadService.install(
                target: artifact("poolside/target"), assistant: artifact("poolside/assistant"), name: "bundle",
                root: root
            ) { artifact, directory in
                if artifact.repository.name == "assistant" { throw CancellationError() }
                try await Self.writeFixture(artifact, directory)
            }
            Issue.record("Expected assistant transfer failure")
        } catch is CancellationError {}
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path).isEmpty)
    }

    @Test func wrongFileSizeAndExistingInstallAreRejected() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        do {
            try await ModelDownloadService.install(
                target: artifact("poolside/target"), assistant: nil, name: "bundle", root: root
            ) { artifact, directory in
                try await Self.writeFixture(artifact, directory)
                try Data().write(to: directory.appendingPathComponent("model.safetensors"))
            }
            Issue.record("Expected file-size validation failure")
        } catch {}
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("bundle").path))
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("bundle"), withIntermediateDirectories: true)
        do {
            try await ModelDownloadService.install(
                target: artifact("poolside/target"), assistant: nil, name: "bundle", root: root
            ) { _, _ in
                Issue.record("An existing installation must never trigger a transfer")
            }
            Issue.record("Expected refusal to replace installed model")
        } catch {}
    }

    private func artifact(_ repository: String) throws -> ModelDownloadService.Artifact {
        let data = Data("{}".utf8)
        let files: [(String, Int?)] = [("config.json", data.count), ("model.safetensors", 3)]
        return ModelDownloadService.Artifact(
            repository: try DownloadPlan.repository(repository), revision: String(repeating: "a", count: 40),
            files: files,
            plan: try DownloadPlan(entries: files, maxGB: 1), configuration: data)
    }

    private static func writeFixture(_ artifact: ModelDownloadService.Artifact, _ directory: URL) async throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try artifact.configuration.write(to: directory.appendingPathComponent("config.json"))
        try Data([1, 2, 3]).write(to: directory.appendingPathComponent("model.safetensors"))
    }
}
