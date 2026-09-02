import Foundation

public struct ResolvedModelSelection: Equatable, Sendable {
  public let modelPath: String
  public let adapterPath: String?
  public let servedModelName: String

  public init(modelPath: String, adapterPath: String?, servedModelName: String) {
    self.modelPath = modelPath
    self.adapterPath = adapterPath
    self.servedModelName = servedModelName
  }
}

public enum ModelCatalog {
  public static let directoryEnvironmentKey = "MODEL_RUNNER_MODELS_DIR"

  public static func defaultDirectory(
    environment: [String: String] = ProcessInfo.processInfo.environment,
    homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser
  ) -> URL {
    if let configured = environment[directoryEnvironmentKey]?
      .trimmingCharacters(in: .whitespacesAndNewlines),
      !configured.isEmpty
    {
      return normalizedURL(configured)
    }
    return homeDirectory
      .appendingPathComponent(".runner", isDirectory: true)
      .appendingPathComponent("models", isDirectory: true)
      .standardizedFileURL
  }

  public static func resolveMLX(
    model: String,
    adapter: String? = nil,
    servedModelName: String? = nil,
    modelsDirectory: URL = defaultDirectory(),
    fileManager: FileManager = .default
  ) -> ResolvedModelSelection {
    let entry = resolveInput(model, modelsDirectory: modelsDirectory, fileManager: fileManager)
    let bundleBase = entry.appendingPathComponent("base-model", isDirectory: true)
    let bundleAdapter = entry.appendingPathComponent("adapter", isDirectory: true)
    let isBundle = directoryExists(bundleBase, fileManager: fileManager)
      && directoryExists(bundleAdapter, fileManager: fileManager)
    let resolvedAdapter = adapter.map {
      resolveInput($0, modelsDirectory: modelsDirectory, fileManager: fileManager).path
    } ?? (isBundle ? bundleAdapter.path : nil)

    return ResolvedModelSelection(
      modelPath: isBundle ? bundleBase.path : entry.path,
      adapterPath: resolvedAdapter,
      servedModelName: normalizedName(servedModelName) ?? entry.lastPathComponent
    )
  }

  public static func availableModels(
    modelsDirectory: URL = defaultDirectory(),
    fileManager: FileManager = .default
  ) -> [String] {
    guard
      let entries = try? fileManager.contentsOfDirectory(
        at: modelsDirectory,
        includingPropertiesForKeys: [.isDirectoryKey],
        options: [.skipsHiddenFiles]
      )
    else { return [] }

    return entries.compactMap { entry in
      guard directoryExists(entry, fileManager: fileManager) else { return nil }
      let directConfiguration = entry.appendingPathComponent("config.json")
      let bundledConfiguration = entry
        .appendingPathComponent("base-model", isDirectory: true)
        .appendingPathComponent("config.json")
      guard fileManager.fileExists(atPath: directConfiguration.path)
        || fileManager.fileExists(atPath: bundledConfiguration.path)
      else { return nil }
      return entry.lastPathComponent
    }.sorted()
  }

  private static func resolveInput(
    _ input: String,
    modelsDirectory: URL,
    fileManager: FileManager
  ) -> URL {
    let expanded = NSString(string: input).expandingTildeInPath
    let explicitlyPathed = NSString(string: expanded).isAbsolutePath
      || input.contains("/")
      || input.hasPrefix(".")
    if explicitlyPathed {
      return normalizedURL(expanded)
    }
    let catalogEntry = modelsDirectory
      .appendingPathComponent(input, isDirectory: true)
      .standardizedFileURL
    if fileManager.fileExists(atPath: catalogEntry.path) {
      return catalogEntry
    }
    return normalizedURL(expanded)
  }

  private static func normalizedURL(_ path: String) -> URL {
    URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
      .standardizedFileURL
  }

  private static func directoryExists(_ url: URL, fileManager: FileManager) -> Bool {
    var isDirectory: ObjCBool = false
    return fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
      && isDirectory.boolValue
  }

  private static func normalizedName(_ value: String?) -> String? {
    guard let value = value?.trimmingCharacters(in: .whitespacesAndNewlines),
      !value.isEmpty
    else { return nil }
    return value
  }
}
