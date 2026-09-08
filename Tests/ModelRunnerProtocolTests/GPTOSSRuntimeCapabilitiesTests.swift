import Foundation
import Testing

@testable import ModelRunnerCore

@Suite("GPT-OSS runtime capabilities")
struct GPTOSSRuntimeCapabilitiesTests {
  @Test("GPT-OSS metadata enables hot sessions without Mistral KV capabilities")
  func recognizesGPTOSS() throws {
    for configuration in [
      #"{"model_type":"gpt_oss","architectures":["GptOssForCausalLM"]}"#,
      #"{"model_type":" GPT_OSS "}"#,
      #"{"architectures":["GptOssForCausalLM"]}"#,
    ] {
      let capabilities = try ModelRuntimeCapabilities.decode(Data(configuration.utf8))
      #expect(capabilities.isGPTOSS)
      #expect(capabilities.supportsGPTOSSConversationPrefixCache)
      #expect(capabilities.supportsHotConversationCache)
      #expect(capabilities.mistralFamily == nil)
      #expect(!capabilities.supportsMistralConversationPrefixCache)
    }
  }

  @Test("Audio, nested backbones, and unrelated model types do not enable GPT-OSS")
  func excludesUnrelatedModels() throws {
    for configuration in [
      #"{"model_type":"voxtral","text_config":{"model_type":"gpt_oss"}}"#,
      #"{"model_type":"gpt_oss","architectures":["AudioForCausalLM"]}"#,
      #"{"model_type":"conditional_generation","text_config":{"model_type":"gpt_oss"}}"#,
      #"{"model_type":"llama","architectures":["GptOssForCausalLM"]}"#,
      #"{"architectures":["NotGptOssForCausalLM"]}"#,
      #"{"model_type":"laguna"}"#,
      #"{"model_type":"gemma4"}"#,
    ] {
      let capabilities = try ModelRuntimeCapabilities.decode(Data(configuration.utf8))
      #expect(!capabilities.isGPTOSS)
      #expect(!capabilities.supportsGPTOSSConversationPrefixCache)
      #expect(!capabilities.supportsHotConversationCache)
    }
  }

  @Test("GPT-OSS retains all one-shot escape hatches")
  func promptCacheRouting() throws {
    let capabilities = try ModelRuntimeCapabilities.decode(Data(#"{"model_type":"gpt_oss"}"#.utf8))
    #expect(LocalModelRunner.shouldUseHotConversationCache(
      capabilities: capabilities, enablePromptCache: true,
      normalizesGemma4Prompt: false, hasCustomStopStrings: false, usesDFlash: false,
      hasTools: true))
    #expect(!LocalModelRunner.shouldUseHotConversationCache(
      capabilities: capabilities, enablePromptCache: true,
      normalizesGemma4Prompt: false, hasCustomStopStrings: false, usesDFlash: false))
    #expect(!LocalModelRunner.shouldUseMistralPromptCache(
      capabilities: capabilities, enablePromptCache: true,
      normalizesGemma4Prompt: false, hasCustomStopStrings: false, usesDFlash: false))

    for (enabled, normalizes, stops, speculative) in [
      (false, false, false, false),
      (true, true, false, false),
      (true, false, true, false),
      (true, false, false, true),
    ] {
      #expect(!LocalModelRunner.shouldUseHotConversationCache(
        capabilities: capabilities, enablePromptCache: enabled,
        normalizesGemma4Prompt: normalizes, hasCustomStopStrings: stops,
        usesDFlash: speculative, hasTools: true))
    }
  }

  @Test("The shared route preserves Mistral and rejects an unknown architecture")
  func sharedRoute() {
    #expect(LocalModelRunner.shouldUseHotConversationCache(
      capabilities: .init(mistralFamily: .mistral), enablePromptCache: true,
      normalizesGemma4Prompt: false, hasCustomStopStrings: false, usesDFlash: false))
    #expect(!LocalModelRunner.shouldUseHotConversationCache(
      capabilities: .none, enablePromptCache: true,
      normalizesGemma4Prompt: false, hasCustomStopStrings: false, usesDFlash: false))
  }

  @Test("Hidden Harmony history cannot carry a continuation beyond the context limit")
  func boundsLiveCacheTimeline() {
    #expect(LocalModelRunner.hotConversationFitsContext(
      processedTokenCount: 1_000, renderedPromptTokenCount: 900,
      maximumTokens: 100, contextLength: 2_000))
    #expect(!LocalModelRunner.hotConversationFitsContext(
      processedTokenCount: 1_001, renderedPromptTokenCount: 900,
      maximumTokens: 100, contextLength: 2_000))
    #expect(!LocalModelRunner.hotConversationFitsContext(
      processedTokenCount: nil, renderedPromptTokenCount: 900,
      maximumTokens: 100, contextLength: 2_000))
    #expect(!LocalModelRunner.hotConversationFitsContext(
      processedTokenCount: Int.max, renderedPromptTokenCount: 900,
      maximumTokens: 100, contextLength: 2_000))
  }
}
