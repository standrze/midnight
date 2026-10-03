import ModelRunnerProtocol
import Testing

@testable import ModelRunnerCore

@Suite("Completed message-prefix LRU")
struct CompletedMessagePrefixLRUTests {
  private func transcript(_ name: String) -> [OpenAIMessage] {
    [
      OpenAIMessage(role: "user", content: "Question \(name)"),
      OpenAIMessage(role: "assistant", content: "Answer \(name)"),
    ]
  }

  @Test("The deepest exact completed prefix is selected")
  func deepestPrefix() {
    var cache = CompletedMessagePrefixLRU<String>(maximumEntries: 4, maximumBytes: 100)
    let root = transcript("A")
    let deeper = root + transcript("B")
    cache.insert("root", committedMessages: root, costBytes: 10)
    cache.insert("deeper", committedMessages: deeper, costBytes: 10)

    let match = cache.longestPrefix(
      of: deeper + [OpenAIMessage(role: "user", content: "Next")]
    )
    #expect(match?.value == "deeper")
    #expect(match?.suffixStart == deeper.count)
  }

  @Test("Identical and edited transcripts are not cache hits")
  func misses() {
    var cache = CompletedMessagePrefixLRU<String>(maximumEntries: 4, maximumBytes: 100)
    let root = transcript("A")
    cache.insert("root", committedMessages: root, costBytes: 10)

    #expect(cache.longestPrefix(of: root)?.value == nil)
    #expect(
      cache.longestPrefix(of: [
        OpenAIMessage(role: "user", content: "Edited"),
        root[1],
        OpenAIMessage(role: "user", content: "Next"),
      ])?.value == nil
    )
  }

  @Test("Entry and byte limits evict least-recently-used values")
  func evictionAndPromotion() {
    var cache = CompletedMessagePrefixLRU<String>(maximumEntries: 2, maximumBytes: 25)
    let a = transcript("A")
    let b = transcript("B")
    let c = transcript("C")
    cache.insert("a", committedMessages: a, costBytes: 10)
    cache.insert("b", committedMessages: b, costBytes: 10)
    _ = cache.longestPrefix(of: a + [OpenAIMessage(role: "user", content: "Next")])
    cache.insert("c", committedMessages: c, costBytes: 10)

    #expect(cache.count == 2)
    #expect(cache.totalBytes == 20)
    #expect(
      cache.longestPrefix(of: b + [OpenAIMessage(role: "user", content: "Next")])?.value
        == nil
    )
    #expect(
      cache.longestPrefix(of: a + [OpenAIMessage(role: "user", content: "Next")])?.value
        == "a"
    )
  }

  @Test("Duplicates repair accounting and invalid entries are rejected")
  func replacementAndRejection() {
    var cache = CompletedMessagePrefixLRU<String>(maximumEntries: 3, maximumBytes: 20)
    let a = transcript("A")
    cache.insert("old", committedMessages: a, costBytes: 15)
    cache.insert("new", committedMessages: a, costBytes: 5)
    cache.insert("oversized", committedMessages: transcript("B"), costBytes: 21)
    cache.insert(
      "unfinished",
      committedMessages: [OpenAIMessage(role: "user", content: "No answer yet")],
      costBytes: 1
    )

    #expect(cache.count == 1)
    #expect(cache.totalBytes == 5)
    #expect(
      cache.longestPrefix(of: a + [OpenAIMessage(role: "user", content: "Next")])?.value
        == "new"
    )
  }
}
