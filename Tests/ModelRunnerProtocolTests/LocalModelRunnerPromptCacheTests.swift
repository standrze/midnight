import ModelRunnerProtocol
import Testing

@testable import ModelRunnerCore

@Suite("Local model runner prompt cache")
struct LocalModelRunnerPromptCacheTests {
    @Test("Retention admission reserves hot-session bytes before snapshot allocation")
    func retentionBudget() {
        var cache = CompletedMessagePrefixLRU<Int>(maximumEntries: 2, maximumBytes: 100)
        #expect(!cache.canStore(costBytes: 60, reservedBytes: 60))
        #expect(cache.canStore(costBytes: 50, reservedBytes: 50))
        #expect(!CompletedMessagePrefixLRU<Int>(maximumEntries: 0, maximumBytes: 100).canStore(costBytes: 1))
        let a = [OpenAIMessage(role: "assistant", content: "A")]
        let b = [OpenAIMessage(role: "assistant", content: "B")]
        cache.insert(1, committedMessages: a, costBytes: 40)
        cache.insert(2, committedMessages: b, costBytes: 40)
        cache.trim(toBytes: 40)
        #expect(cache.count == 1)
        #expect(cache.totalBytes == 40)
        #expect(cache.longestPrefix(of: a + [OpenAIMessage(role: "user", content: "next")]) == nil)
        cache.removeAll()
        #expect(cache.count == 0 && cache.totalBytes == 0)
    }

    @Test("A strict transcript extension starts after every committed message")
    func strictExtension() {
        let committed = [
            OpenAIMessage(role: "system", content: "Be concise."),
            OpenAIMessage(role: "user", content: "Name a prime number."),
            OpenAIMessage(role: "assistant", content: "Two."),
        ]
        let incoming =
            committed + [
                OpenAIMessage(role: "user", content: "Name another.")
            ]

        #expect(
            LocalModelRunner.cachedConversationSuffixStart(
                committed: committed,
                incoming: incoming
            ) == committed.count
        )
    }

    @Test("An identical transcript has no uncached continuation")
    func identicalTranscript() {
        let transcript = [
            OpenAIMessage(role: "user", content: "Hello"),
            OpenAIMessage(role: "assistant", content: "Hi"),
        ]

        #expect(
            LocalModelRunner.cachedConversationSuffixStart(
                committed: transcript,
                incoming: transcript
            ) == nil
        )
    }

    @Test("Editing an earlier message invalidates the cached transcript")
    func editedEarlierMessage() {
        let committed = [
            OpenAIMessage(role: "system", content: "Answer briefly."),
            OpenAIMessage(role: "user", content: "What is 2 + 2?"),
            OpenAIMessage(role: "assistant", content: "4"),
        ]
        let incoming = [
            OpenAIMessage(role: "system", content: "Explain every step."),
            OpenAIMessage(role: "user", content: "What is 2 + 2?"),
            OpenAIMessage(role: "assistant", content: "4"),
            OpenAIMessage(role: "user", content: "Why?"),
        ]

        #expect(
            LocalModelRunner.cachedConversationSuffixStart(
                committed: committed,
                incoming: incoming
            ) == nil
        )
    }

    @Test("A tool result continues directly after the committed tool call")
    func toolResultContinuation() {
        let toolCall = OpenAIToolCall(
            id: "call_weather_1",
            function: .init(
                name: "get_weather",
                arguments: #"{"city":"Boston"}"#
            )
        )
        let committed = [
            OpenAIMessage(role: "user", content: "What is the weather in Boston?"),
            OpenAIMessage(
                role: "assistant",
                content: nil,
                toolCalls: [toolCall]
            ),
        ]
        let incoming =
            committed + [
                OpenAIMessage(
                    role: "tool",
                    content: #"{"temperature":72}"#,
                    name: "get_weather",
                    toolCallID: "call_weather_1"
                )
            ]

        #expect(
            LocalModelRunner.cachedConversationSuffixStart(
                committed: committed,
                incoming: incoming
            ) == committed.count
        )
    }

    @Test("The deepest reusable prefix wins across cached branches")
    func deepestPrefixWins() {
        let root = [
            OpenAIMessage(role: "user", content: "Name a prime."),
            OpenAIMessage(role: "assistant", content: "Two."),
        ]
        let deeper =
            root + [
                OpenAIMessage(role: "user", content: "Another."),
                OpenAIMessage(role: "assistant", content: "Three."),
            ]
        let unrelated = [
            OpenAIMessage(role: "user", content: "Name a color."),
            OpenAIMessage(role: "assistant", content: "Blue."),
        ]
        let incoming = deeper + [OpenAIMessage(role: "user", content: "One more.")]

        #expect(
            LocalModelRunner.longestCachedConversationPrefixIndex(
                committed: [root, unrelated, deeper],
                incoming: incoming
            ) == 2
        )
    }

    @Test("An edited branch does not reuse a deeper incompatible checkpoint")
    func editedBranchUsesEarlierPrefix() {
        let root = [
            OpenAIMessage(role: "system", content: "Be concise."),
            OpenAIMessage(role: "user", content: "Name a prime."),
            OpenAIMessage(role: "assistant", content: "Two."),
        ]
        let deeper =
            root + [
                OpenAIMessage(role: "user", content: "Another."),
                OpenAIMessage(role: "assistant", content: "Three."),
            ]
        let incoming = root + [OpenAIMessage(role: "user", content: "Explain why.")]

        #expect(
            LocalModelRunner.longestCachedConversationPrefixIndex(
                committed: [deeper, root],
                incoming: incoming
            ) == 1
        )
    }

    @Test("Prefix cache limits are bounded and configurable")
    func prefixCacheLimits() {
        #expect(
            ConversationPrefixCacheLimits.resolve(environment: [:])
                == ConversationPrefixCacheLimits(
                    maximumEntries: 4,
                    maximumBytes: 2 * 1_024 * 1_024 * 1_024
                )
        )
        #expect(
            ConversationPrefixCacheLimits.resolve(
                environment: [
                    ConversationPrefixCacheLimits.entriesEnvironmentKey: "2",
                    ConversationPrefixCacheLimits.memoryEnvironmentKey: "768",
                ]
            )
                == ConversationPrefixCacheLimits(
                    maximumEntries: 2,
                    maximumBytes: 768 * 1_048_576
                )
        )
        #expect(
            ConversationPrefixCacheLimits.resolve(
                environment: [
                    ConversationPrefixCacheLimits.entriesEnvironmentKey: "1000",
                    ConversationPrefixCacheLimits.memoryEnvironmentKey: "broken",
                ]
            )
                == ConversationPrefixCacheLimits(
                    maximumEntries: 4,
                    maximumBytes: 2 * 1_024 * 1_024 * 1_024
                )
        )
    }
}
