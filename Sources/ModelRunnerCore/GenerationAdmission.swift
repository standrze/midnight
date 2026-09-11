import Foundation
import Synchronization

/// FIFO ownership of model execution. Cancellation removes queued work without
/// releasing an active owner's slot before its GPU cleanup finishes.
final class GenerationAdmission: Sendable {
  enum Failure: Error { case full }
  private struct Waiter {
    let id: UUID
    let continuation: CheckedContinuation<Void, Error>
  }
  private struct State {
    var occupied = false
    var waiters: [Waiter] = []
  }
  private let state = Mutex(State())
  let capacity: Int
  init(capacity: Int = 64) { self.capacity = max(0, capacity) }

  func acquire() async throws {
    let id = UUID()
    try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
        state.withLock { state in
          if Task.isCancelled { continuation.resume(throwing: CancellationError()) }
          else if !state.occupied {
            state.occupied = true
            continuation.resume()
          } else if state.waiters.count >= capacity {
            continuation.resume(throwing: Failure.full)
          } else {
            state.waiters.append(Waiter(id: id, continuation: continuation))
          }
        }
      }
    } onCancel: {
      self.state.withLock { state in
        if let index = state.waiters.firstIndex(where: { $0.id == id }) {
          state.waiters.remove(at: index).continuation.resume(throwing: CancellationError())
        }
      }
    }
    // The slot may have been handed off concurrently with cancellation. The
    // caller owns it now and must release it through its normal defer.
  }

  func release() {
    state.withLock { state in
      precondition(state.occupied)
      if state.waiters.isEmpty { state.occupied = false }
      else { state.waiters.removeFirst().continuation.resume() }
    }
  }
  var queuedCount: Int { state.withLock { $0.waiters.count } }
}
