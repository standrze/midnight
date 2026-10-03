import Foundation
import MLXLMCommon
import Testing

@testable import ModelRunnerCore

@Suite("Local Muse chat protocol")
struct MuseGlimmerChatConfigurationTests {
    @Test("Renamed local Muse checkpoints select the framed ATEM decoder")
    func localProtocol() {
        let configuration = LocalModelChatConfiguration.make(
            directory: URL(fileURLWithPath: "/tmp/arbitrary-local-model-name"),
            modelType: "muse_glimmer")
        #expect(configuration.toolCallFormat == .atem)
        #expect(configuration.extraEOSTokens.contains("<|eot|>"))
        #expect(configuration.extraEOSTokens.contains("<|end_of_text|>"))
        #expect(!configuration.extraEOSTokens.contains("<|eom|>"))
        #expect(configuration.reasoningConfig == nil)
    }

    @Test("Other model families retain registry inference and custom stop tokens")
    func otherModels() {
        for modelType in ["talkie", "gemma4", "qwen3_5"] {
            let configuration = LocalModelChatConfiguration.make(
                directory: URL(fileURLWithPath: "/tmp/muse-named-unrelated-model"),
                modelType: modelType, additionalEOSTokens: ["custom-eos"])
            #expect(configuration.toolCallFormat == nil)
            #expect(configuration.extraEOSTokens == ["<end_of_turn>", "<turn|>", "custom-eos"])
        }
    }
}
