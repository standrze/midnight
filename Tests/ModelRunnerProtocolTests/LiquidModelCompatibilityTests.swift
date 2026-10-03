import Foundation
import MLXLLM
import Testing

@testable import ModelRunnerCore

@Suite("Liquid checkpoint compatibility")
struct LiquidModelCompatibilityTests {
  private func fixture(_ name: String) throws -> Data {
    let tests = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    return try Data(contentsOf: tests.appendingPathComponent("Fixtures/Liquid/\(name).json"))
  }

  @Test("Official LFM2.5 Base configuration decodes through the dense loader")
  func denseBase() throws {
    let data = try fixture("LFM2.5-1.2B-Base")
    _ = try JSONDecoder().decode(LFM2Configuration.self, from: data)
    // Hybrid convolution state must not accidentally enter the transformer's hot-cache path.
    let capabilities = try ModelRuntimeCapabilities.decode(data)
    #expect(!capabilities.supportsHotConversationCache)
    #expect(capabilities.mistralFamily == nil)
  }

  @Test("Official MoE configurations preserve routing, attention layout and RoPE",
        arguments: ["LFM2.5-8B-A1B", "LFM2-24B-A2B"])
  func mixtureOfExperts(name: String) throws {
    let data = try fixture(name)
    let configuration = try JSONDecoder().decode(LFM2MoEConfiguration.self, from: data)
    let small = name == "LFM2.5-8B-A1B"
    #expect(configuration.modelType == "lfm2_moe")
    #expect(configuration.hiddenLayers == (small ? 24 : 40))
    #expect(configuration.numExperts == (small ? 32 : 64))
    #expect(configuration.numExpertsPerToken == 4)
    #expect(configuration.fullAttnIdxs == (small ? [2, 6, 10, 14, 18, 21] : [2, 6, 10, 14, 18, 22, 26, 30, 34, 38]))
    #expect(configuration.ropeTheta == (small ? 5_000_000 : 1_000_000))
    #expect(configuration.vocabularySize == (small ? 128_000 : 65_536))
    let capabilities = try ModelRuntimeCapabilities.decode(data)
    #expect(!capabilities.supportsHotConversationCache)
    #expect(capabilities.mistralFamily == nil)
  }
}
