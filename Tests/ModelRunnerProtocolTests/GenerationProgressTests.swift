import Foundation
import MLXLMCommon
import Synchronization
import Testing

@testable import ModelRunnerCore

@Suite("Live generation progress")
struct GenerationProgressTests {
    @Test func liveRateExcludesPrefillAndFinalMetricsAreAuthoritative() throws {
        let monitor = GenerationProgressMonitor()
        #expect(!monitor.begin(now: 0))
        #expect(monitor.snapshot(now: 1) == nil)
        monitor.setEnabled(true)
        #expect(monitor.begin(now: 10))
        monitor.prepare(promptTokens: 100, maximumTokens: 64)
        monitor.record(tokenCount: 1, now: 12)
        #expect(monitor.snapshot(now: 12)?.tokensPerSecond == nil)
        monitor.record(tokenCount: 21, now: 13)
        let live = try #require(monitor.snapshot(now: 13))
        #expect(live.phase == .generating)
        #expect(live.tokensPerSecond == 20)
        #expect(live.firstTokenSeconds == 2)
        #expect(live.elapsedSeconds == 3)
        monitor.record(
            metrics: LocalModelRunnerMetrics(
                promptTokenCount: 100, cachedPromptTokenCount: 80, generationTokenCount: 21,
                promptTokensPerSecond: 10, tokensPerSecond: 18.5, stopReason: "stop"))
        monitor.finish(succeeded: true, cancelled: false, now: 14)
        let final = try #require(monitor.snapshot(now: 100))
        #expect(final.phase == .completed)
        #expect(final.tokensPerSecond == 18.5)
        #expect(final.elapsedSeconds == 4)
        #expect(final.generatedTokens == 21)
        #expect(final.maximumTokens == 64)
        #expect(final.cachedPromptTokens == 80)
        #expect(final.stopReason == "stop")
    }

    @Test func failuresCancellationAndTheNextRequestResetProgress() throws {
        let monitor = GenerationProgressMonitor()
        monitor.setEnabled(true)
        monitor.begin(now: 0)
        monitor.record(tokenCount: 3, now: 1)
        monitor.finish(succeeded: false, cancelled: true, now: 2)
        #expect(monitor.snapshot(now: 3)?.phase == .cancelled)
        monitor.begin(now: 4)
        let next = try #require(monitor.snapshot(now: 5))
        #expect(next.phase == .preparing)
        #expect(next.generatedTokens == 0)
        #expect(next.maximumTokens == nil)
        #expect(next.firstTokenSeconds == nil)
        monitor.finish(succeeded: false, cancelled: false, now: 6)
        #expect(monitor.snapshot(now: 7)?.phase == .failed)
    }

    @Test func observerCountsEmittedTokenIDsAndExcludesEOS() async throws {
        let counts = Mutex<[Int]>([])
        let result = await GenerationTokenObserver.$onToken.withValue(
            { count in
                counts.withLock { $0.append(count) }
            },
            operation: {
                var configuration = ModelConfiguration(id: "progress-fixture")
                configuration.eosTokenIds = [1]
                let (stream, producer) = MLXLMCommon.generateTask(
                    promptTokenCount: 5, modelConfiguration: configuration, tokenizer: ProgressTokenizer(),
                    iterator: ProgressTokenIterator(tokens: [4, 5, 6, 1, 7]))
                var text = ""
                var info: GenerateCompletionInfo?
                for await event in stream {
                    if let chunk = event.chunk {
                        text += chunk
                    }
                    if let completion = event.info {
                        info = completion
                    }
                }
                await producer.value
                return (text, info)
            })
        #expect(counts.withLock { $0 } == [1, 2, 3])
        #expect(result.0 == "456")
        #expect(result.1?.generationTokenCount == 3)
        #expect(result.1?.stopReason == .stop)
        #expect(GenerationTokenObserver.onToken == nil)
    }
}

private struct ProgressTokenIterator: TokenIteratorProtocol {
    let tokens: [Int]
    var tokenCount = 0
    var maxTokens: Int? { tokens.count }
    var promptPrefillTime: TimeInterval { 0 }

    mutating func next() -> Int? {
        guard tokenCount < tokens.count else {
            return nil
        }
        defer { tokenCount += 1 }
        return tokens[tokenCount]
    }
}

private struct ProgressTokenizer: MLXLMCommon.Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { tokenIds.map(String.init).joined() }
    func convertTokenToId(_ token: String) -> Int? { Int(token) }
    func convertIdToToken(_ id: Int) -> String? { String(id) }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
