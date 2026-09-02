import Foundation
import Testing

@Suite("Model loader resource guard placement")
struct ResourceGuardPlacementTests {
  @Test("Every Hugging Face container load is immediately guarded")
  func allLoadSitesAreGuarded() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
    let sources = packageRoot.appendingPathComponent("Sources", isDirectory: true)
    let enumerator = try #require(
      FileManager.default.enumerator(
        at: sources,
        includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles]
      )
    )
    var loadSiteCount = 0

    for case let file as URL in enumerator where file.pathExtension == "swift" {
      let lines = try String(contentsOf: file, encoding: .utf8)
        .split(separator: "\n", omittingEmptySubsequences: false)
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
      for index in lines.indices where lines[index].contains("#huggingFaceLoadModelContainer(") {
        loadSiteCount += 1
        let previousCodeLine = lines[..<index].last {
          !$0.isEmpty && !$0.hasPrefix("//")
        }
        #expect(previousCodeLine == "try MLXResourceGuard.apply(resourceLimits)")
      }
    }

    #expect(loadSiteCount > 0)
  }

  @Test("Serialized GPU paths reuse persistent default streams")
  func serializedGPUPathsReusePersistentStreams() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
    let core = packageRoot.appendingPathComponent("Sources/ModelRunnerCore")
    let runner = try String(
      contentsOf: core.appendingPathComponent("LocalModelRunner.swift"),
      encoding: .utf8
    )
    let voxtral = try String(
      contentsOf: core.appendingPathComponent("VoxtralTTS/VoxtralTTSSynthesizer.swift"),
      encoding: .utf8
    )

    #expect(!runner.contains("Stream.withNewDefaultStream(device: device)"))
    #expect(!voxtral.contains("Stream.withNewDefaultStream(device: device)"))
    #expect(runner.contains("Stream.withNewDefaultStream(device: .cpu)"))
  }
}
