import Foundation
import XCTest
@testable import Midnight

final class ModelDownloadTests: XCTestCase {
    func testCatalogInitializesAndPreservesUserEdits() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("config/downloads.json")
        XCTAssertEqual(try DownloadPreset.load(from: url).count, DownloadPreset.all.count)
        let custom = Data(#"[{"name":"my-model","repo":"my-account/optimized-mlx","note":"My release"}]"#.utf8)
        try custom.write(to: url)
        XCTAssertEqual(try DownloadPreset.load(from: url).map(\.name), ["my-model"])
        XCTAssertEqual(try Data(contentsOf: url), custom)
    }

    func testCatalogRejectsUnsafeNamesAndDuplicates() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("downloads.json")
        for name in ["../escape", ".hidden", "bad/name", "-option"] {
            try JSONEncoder().encode([DownloadPreset(name: name, repo: "owner/model", note: "")]).write(to: url)
            XCTAssertThrowsError(try DownloadPreset.load(from: url))
        }
        let preset = DownloadPreset(name: "duplicate", repo: "owner/model", note: "")
        try JSONEncoder().encode([preset, preset]).write(to: url)
        XCTAssertThrowsError(try DownloadPreset.load(from: url))
        try Data("invalid json".utf8).write(to: url)
        XCTAssertThrowsError(try DownloadPreset.load(from: url))
        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), "invalid json")
    }

    func testSelectsOnlyNativeRootFiles() throws {
        let plan = try DownloadPlan(entries: [("config.json", 10), ("model.safetensors", 20), ("original/model.safetensors", 900), ("model.gguf", 900), ("model.py", 900)], maxGB: 1)
        XCTAssertEqual(plan.files, ["config.json", "model.safetensors"])
        XCTAssertEqual(plan.bytes, 30)
    }

    func testRefusesMissingSizeAndOversize() {
        XCTAssertThrowsError(try DownloadPlan(entries: [("config.json", 10), ("model.safetensors", nil)], maxGB: 30))
        XCTAssertThrowsError(try DownloadPlan(entries: [("config.json", 10), ("model.safetensors", 31_000_000_000)], maxGB: 30))
        XCTAssertThrowsError(try DownloadPlan(entries: [("config.json", 10), ("model.safetensors", 20)], maxGB: .nan))
    }

    func testRefusesUnsafeAndUnsupportedInput() {
        for repo in ["../model", "owner/..", "owner/model/extra", "owner/model?token=x", "/model"] {
            XCTAssertThrowsError(try DownloadPlan.repository(repo))
        }
        XCTAssertThrowsError(try DownloadPlan(entries: [("config.json", 10), ("*.safetensors", 20)], maxGB: 30))
        XCTAssertThrowsError(try DownloadPlan(entries: [("config.json", 10), ("model.gguf", 20)], maxGB: 30))
    }

    func testSupportsFutureRepositories() throws {
        let repo = try DownloadPlan.repository("my-account/optimized-model-4bit")
        XCTAssertEqual(repo.namespace, "my-account")
        XCTAssertEqual(repo.name, "optimized-model-4bit")
        let download = try DownloadCommand.parse(["owner/model", "--dry-run"])
        XCTAssertTrue(download.dryRun)
    }
}
