import Testing

@testable import ModelRunnerCore

@Suite("Personal inference cache routing")
struct PersonalCacheRoutingTests {
    private func route(
        laguna: Bool = false,
        capabilities: ModelRuntimeCapabilities = .none,
        enabled: Bool = true,
        normalization: Bool = false,
        stops: Bool = false,
        speculation: Bool = false,
        tools: Bool = false,
        forcedPrefix: Bool = false,
        shared: Bool = true
    ) -> LocalModelRunner.PromptCacheRoute {
        LocalModelRunner.promptCacheRoute(
            supportsLagunaPromptCache: laguna, capabilities: capabilities,
            enablePromptCache: enabled, normalizesGemma4Prompt: normalization,
            hasCustomStopStrings: stops, usesDFlash: speculation, hasTools: tools,
            hasForcedToolPrefix: forcedPrefix, hasSharedPrefix: shared)
    }

    @Test("A long first turn establishes full conversation reuse")
    func conversationBeforeSharedCheckpoints() {
        #expect(route(laguna: true) == .lagunaConversation)
        #expect(route(capabilities: .init(mistralFamily: .mistral)) == .hotConversation)
        #expect(route(capabilities: .init(mistralFamily: nil, isGPTOSS: true), tools: true) == .hotConversation)
    }

    @Test("Models without a compatible conversation path retain shared reuse")
    func sharedFallback() {
        #expect(route() == .sharedPrefix)
        #expect(route(capabilities: .init(mistralFamily: nil, isGPTOSS: true)) == .sharedPrefix)
        #expect(route(shared: false) == .uncached)
        #expect(route(enabled: false) == .uncached)
        #expect(route(capabilities: .init(mistralFamily: .mistral), enabled: false) == .uncached)
    }

    @Test("Specialized generation cannot accidentally enter ordinary caching")
    func specializedPaths() {
        for laguna in [false, true] {
            #expect(route(laguna: laguna, speculation: true) == .uncached)
            #expect(route(laguna: laguna, forcedPrefix: true) == .uncached)
        }
    }

    @Test("Normalized Gemma prompts reuse exact token checkpoints without re-templating")
    func gemmaPreparedTokens() {
        for tools in [false, true] {
            #expect(route(normalization: true, tools: tools) == .sharedPrefix)
            #expect(route(normalization: true, stops: true, tools: tools) == .sharedPrefix)
            #expect(route(normalization: true, speculation: true, tools: tools) == .uncached)
            #expect(route(normalization: true, tools: tools, forcedPrefix: true) == .uncached)
            #expect(route(enabled: false, normalization: true, tools: tools) == .uncached)
            #expect(route(normalization: true, tools: tools, shared: false) == .uncached)
        }
        #expect(route(laguna: true, normalization: true) == .sharedPrefix)
    }

    @Test("Custom stops bypass sessions but can use independent prefix state")
    func customStops() {
        #expect(route(laguna: true, stops: true) == .sharedPrefix)
        #expect(route(capabilities: .init(mistralFamily: .mistral), stops: true) == .sharedPrefix)
        #expect(route(laguna: true, stops: true, shared: false) == .uncached)
    }

    @Test("Laguna preserves its session implementation with reuse disabled")
    func lagunaWithoutRetention() {
        #expect(route(laguna: true, enabled: false) == .lagunaConversation)
    }
}
