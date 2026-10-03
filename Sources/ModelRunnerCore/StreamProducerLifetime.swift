import Synchronization

/// Tracks producer tasks independently of their stream consumers. The observer
/// retains a task until it finishes; this object never retains a task or owner.
/// Register synchronously before returning a stream, and stop admitting streams
/// before waiting for idle.
final class StreamProducerLifetime: Sendable {
    private struct State {
        var activeCount = 0
        var waiters: [CheckedContinuation<Void, Never>] = []
    }

    private let state = Mutex(State())

    func track(_ task: Task<Void, Never>) {
        state.withLock { $0.activeCount += 1 }
        Task {
            await task.value
            let waiters = state.withLock { state in
                state.activeCount -= 1
                guard state.activeCount == 0 else {
                    return [CheckedContinuation<Void, Never>]()
                }
                let waiters = state.waiters
                state.waiters.removeAll()
                return waiters
            }
            for waiter in waiters {
                waiter.resume()
            }
        }
    }

    /// Cancellation of the waiter must not bypass producer teardown.
    func waitUntilIdle() async {
        await withCheckedContinuation { continuation in
            let isIdle = state.withLock { state in
                if state.activeCount == 0 {
                    return true
                }
                state.waiters.append(continuation)
                return false
            }
            if isIdle {
                continuation.resume()
            }
        }
    }
}
