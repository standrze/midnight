import ModelRunnerProtocol

/// A byte- and entry-bounded LRU of immutable completed conversation checkpoints.
struct CompletedMessagePrefixLRU<Value: Sendable>: Sendable {
  struct Match: Sendable {
    let value: Value
    let suffixStart: Int
  }

  private struct Entry: Sendable {
    let value: Value
    let committedMessages: [OpenAIMessage]
    let costBytes: Int
    var lastAccess: UInt64
  }

  let maximumEntries: Int
  let maximumBytes: Int
  private(set) var totalBytes = 0
  private(set) var count = 0
  private var accessClock: UInt64 = 0
  private var entries: [Entry] = []

  init(maximumEntries: Int, maximumBytes: Int) {
    self.maximumEntries = max(0, maximumEntries)
    self.maximumBytes = max(0, maximumBytes)
  }

  mutating func longestPrefix(of incoming: [OpenAIMessage]) -> Match? {
    let candidates = entries.indices.filter { index in
      let committed = entries[index].committedMessages
      return incoming.count > committed.count && incoming.starts(with: committed)
    }
    guard let index = candidates.max(by: { lhs, rhs in
      let lhsEntry = entries[lhs]
      let rhsEntry = entries[rhs]
      if lhsEntry.committedMessages.count != rhsEntry.committedMessages.count {
        return lhsEntry.committedMessages.count < rhsEntry.committedMessages.count
      }
      return lhsEntry.lastAccess < rhsEntry.lastAccess
    }) else { return nil }

    accessClock &+= 1
    entries[index].lastAccess = accessClock
    return Match(
      value: entries[index].value,
      suffixStart: entries[index].committedMessages.count
    )
  }

  mutating func insert(
    _ value: Value,
    committedMessages: [OpenAIMessage],
    costBytes: Int
  ) {
    let normalizedCost = max(0, costBytes)
    guard maximumEntries > 0, maximumBytes > 0,
      normalizedCost <= maximumBytes,
      committedMessages.last?.role == "assistant"
    else { return }

    if let duplicate = entries.firstIndex(where: {
      $0.committedMessages == committedMessages
    }) {
      totalBytes -= entries.remove(at: duplicate).costBytes
    }

    accessClock &+= 1
    entries.append(
      Entry(
        value: value,
        committedMessages: committedMessages,
        costBytes: normalizedCost,
        lastAccess: accessClock
      ))
    totalBytes += normalizedCost

    while entries.count > maximumEntries || totalBytes > maximumBytes {
      guard let oldest = entries.indices.min(by: {
        entries[$0].lastAccess < entries[$1].lastAccess
      }) else { break }
      totalBytes -= entries.remove(at: oldest).costBytes
    }
    count = entries.count
  }
}
