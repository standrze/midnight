import Foundation
import ModelRunnerProtocol
import Testing

@Suite("Runner model catalog")
struct ModelCatalogTests {
  @Test("The default catalog lives below .runner in the user's home")
  func defaultCatalog() {
    let home = URL(fileURLWithPath: "/Users/tester", isDirectory: true)
    #expect(
      ModelCatalog.defaultDirectory(environment: [:], homeDirectory: home).path
        == "/Users/tester/.runner/models"
    )
    #expect(
      ModelCatalog.defaultDirectory(
        environment: [ModelCatalog.directoryEnvironmentKey: "/models/shared"],
        homeDirectory: home
      ).path == "/models/shared"
    )
  }

  @Test("A named MLX bundle resolves its base, adapter, and public name")
  func mlxBundle() throws {
    let root = temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let bundle = root.appendingPathComponent("gemma-4-e2b-it-cyber", isDirectory: true)
    try FileManager.default.createDirectory(
      at: bundle.appendingPathComponent("base-model", isDirectory: true),
      withIntermediateDirectories: true
    )
    try FileManager.default.createDirectory(
      at: bundle.appendingPathComponent("adapter", isDirectory: true),
      withIntermediateDirectories: true
    )

    let resolved = ModelCatalog.resolveMLX(
      model: "gemma-4-e2b-it-cyber",
      modelsDirectory: root
    )
    #expect(resolved.modelPath == bundle.appendingPathComponent("base-model").path)
    #expect(resolved.adapterPath == bundle.appendingPathComponent("adapter").path)
    #expect(resolved.servedModelName == "gemma-4-e2b-it-cyber")
  }

  @Test("A bare model name prefers the catalog over the current directory")
  func bareNameCatalogPrecedence() throws {
    let root = temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let catalog = root.appendingPathComponent("catalog", isDirectory: true)
    let catalogModel = catalog.appendingPathComponent("shared-name", isDirectory: true)
    try FileManager.default.createDirectory(at: catalogModel, withIntermediateDirectories: true)

    let named = ModelCatalog.resolveMLX(model: "shared-name", modelsDirectory: catalog)
    let explicit = ModelCatalog.resolveMLX(model: "./shared-name", modelsDirectory: catalog)
    #expect(named.modelPath == catalogModel.path)
    #expect(
      explicit.modelPath
        == URL(fileURLWithPath: "./shared-name").standardizedFileURL.path
    )
  }

  @Test("Explicit paths and adapter overrides remain authoritative")
  func explicitPaths() throws {
    let root = temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let model = root.appendingPathComponent("plain-model", isDirectory: true)
    let adapter = root.appendingPathComponent("custom-adapter", isDirectory: true)
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: adapter, withIntermediateDirectories: true)

    let resolved = ModelCatalog.resolveMLX(
      model: model.path,
      adapter: adapter.path,
      servedModelName: "custom-name",
      modelsDirectory: root
    )
    #expect(resolved.modelPath == model.standardizedFileURL.path)
    #expect(resolved.adapterPath == adapter.path)
    #expect(resolved.servedModelName == "custom-name")
  }

  @Test("Catalog listing includes direct models and bundles only")
  func availableModels() throws {
    let root = temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let direct = root.appendingPathComponent("direct-model", isDirectory: true)
    let bundle = root.appendingPathComponent("bundled-model/base-model", isDirectory: true)
    let incomplete = root.appendingPathComponent("incomplete", isDirectory: true)
    try FileManager.default.createDirectory(at: direct, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: incomplete, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: direct.appendingPathComponent("config.json"))
    try Data("{}".utf8).write(to: bundle.appendingPathComponent("config.json"))

    #expect(
      ModelCatalog.availableModels(modelsDirectory: root)
        == ["bundled-model", "direct-model"]
    )
  }

  private func temporaryDirectory() -> URL {
    FileManager.default.temporaryDirectory
      .appendingPathComponent("model-runner-catalog-\(UUID().uuidString)", isDirectory: true)
  }
}
