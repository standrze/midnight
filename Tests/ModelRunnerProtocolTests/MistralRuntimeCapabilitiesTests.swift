import Foundation
import Testing

@testable import ModelRunnerCore

@Suite("Mistral runtime capabilities")
struct MistralRuntimeCapabilitiesTests {
  @Test("Every supported Mistral text architecture gets the shared runtime policy")
  func recognizesMistralFamilies() throws {
    let configurations: [(String, MistralRuntimeFamily)] = [
      (#"{"model_type":"mistral"}"#, .mistral),
      (#"{"model_type":"mixtral"}"#, .mixtral),
      (#"{"model_type":"mistral3"}"#, .mistral3),
      (#"{"model_type":"ministral3"}"#, .mistral3),
      (#"{"model_type":"mistral3_text"}"#, .mistral3),
    ]

    for (configuration, expectedFamily) in configurations {
      let capabilities = try ModelRuntimeCapabilities.decode(Data(configuration.utf8))
      #expect(capabilities.mistralFamily == expectedFamily)
      #expect(capabilities.supportsMistralConversationPrefixCache)
    }
  }

  @Test("Nested text metadata supports converted Mistral 3 checkpoints")
  func recognizesNestedTextConfiguration() throws {
    let capabilities = try ModelRuntimeCapabilities.decode(
      Data(
        #"{"model_type":"conditional_generation","text_config":{"model_type":"ministral3"}}"#
          .utf8
      )
    )

    #expect(capabilities.mistralFamily == .mistral3)
  }

  @Test("Architecture metadata is a fallback, not a folder-name heuristic")
  func recognizesArchitectureFallback() throws {
    let classic = try ModelRuntimeCapabilities.decode(
      Data(#"{"architectures":["MistralForCausalLM"]}"#.utf8)
    )
    let mixture = try ModelRuntimeCapabilities.decode(
      Data(#"{"architectures":["MixtralForCausalLM"]}"#.utf8)
    )
    let thirdGeneration = try ModelRuntimeCapabilities.decode(
      Data(#"{"architectures":["Mistral3ForConditionalGeneration"]}"#.utf8)
    )

    #expect(classic.mistralFamily == .mistral)
    #expect(mixture.mistralFamily == .mixtral)
    #expect(thirdGeneration.mistralFamily == .mistral3)
  }

  @Test("Laguna and non-Mistral checkpoints never inherit the Mistral capability")
  func excludesUnrelatedFamilies() throws {
    for configuration in [
      #"{"model_type":"laguna","architectures":["LagunaForCausalLM"]}"#,
      #"{"model_type":"gemma4"}"#,
      #"{"model_type":"llama"}"#,
    ] {
      #expect(
        try ModelRuntimeCapabilities.decode(Data(configuration.utf8)) == .none
      )
    }
  }

  @Test("Voxtral stays on its native audio route even with a Mistral text backbone")
  func excludesVoxtral() throws {
    for configuration in [
      #"{"model_type":"voxtral_tts","text_config":{"model_type":"mistral3"},"architectures":["VoxtralForConditionalGeneration"]}"#,
      #"{"model_type":"conditional_generation","text_config":{"model_type":"mistral3"},"architectures":["VoxtralForConditionalGeneration"]}"#,
    ] {
      #expect(
        try ModelRuntimeCapabilities.decode(Data(configuration.utf8)) == .none
      )
    }
  }

  @Test("Mistral prompt caching preserves every one-shot escape hatch")
  func promptCacheRouting() {
    let mistral = ModelRuntimeCapabilities(mistralFamily: .mistral)
    let unrelated = ModelRuntimeCapabilities.none

    #expect(
      LocalModelRunner.shouldUseMistralPromptCache(
        capabilities: mistral,
        enablePromptCache: true,
        normalizesGemma4Prompt: false,
        hasCustomStopStrings: false,
        usesDFlash: false
      )
    )
    #expect(
      !LocalModelRunner.shouldUseMistralPromptCache(
        capabilities: mistral,
        enablePromptCache: false,
        normalizesGemma4Prompt: false,
        hasCustomStopStrings: false,
        usesDFlash: false
      )
    )
    #expect(
      !LocalModelRunner.shouldUseMistralPromptCache(
        capabilities: mistral,
        enablePromptCache: true,
        normalizesGemma4Prompt: true,
        hasCustomStopStrings: false,
        usesDFlash: false
      )
    )
    #expect(
      !LocalModelRunner.shouldUseMistralPromptCache(
        capabilities: mistral,
        enablePromptCache: true,
        normalizesGemma4Prompt: false,
        hasCustomStopStrings: false,
        usesDFlash: true
      )
    )
    #expect(
      !LocalModelRunner.shouldUseMistralPromptCache(
        capabilities: mistral,
        enablePromptCache: true,
        normalizesGemma4Prompt: false,
        hasCustomStopStrings: true,
        usesDFlash: false
      )
    )
    #expect(
      !LocalModelRunner.shouldUseMistralPromptCache(
        capabilities: unrelated,
        enablePromptCache: true,
        normalizesGemma4Prompt: false,
        hasCustomStopStrings: false,
        usesDFlash: false
      )
    )
  }
}
