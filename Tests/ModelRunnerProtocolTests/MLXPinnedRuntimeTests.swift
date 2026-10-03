#if os(macOS)
  import Darwin
  import Foundation
  import Synchronization
  import Testing

  @testable import ModelRunnerCore

  @Suite("Permanent MLX pinned runtime", .serialized)
  struct MLXPinnedRuntimeTests {
    @Test("Serial and task executors share one permanent pthread", .timeLimit(.minutes(1)))
    func executorPlacement() async throws {
      let runtime = MLXPinnedRuntime.shared
      let callerThreadID = currentTestThreadID()
      let actor = PinnedThreadProbe(runtime: runtime)

      let actorThreadIDs = await actor.threadIDsAcrossSuspension()
      let actorToTaskThreadIDs = try await actor.taskExecutorThreadIDs()
      let tasks = (0..<64).map { _ in
        Task(executorPreference: runtime.taskExecutor) {
          currentTestThreadID()
        }
      }
      var taskThreadIDs = [UInt64]()
      for task in tasks {
        taskThreadIDs.append(await task.value)
      }
      let runThreadIDs = try await runtime.run {
        let beforeYield = currentTestThreadID()
        await Task.yield()
        return [beforeYield, currentTestThreadID()]
      }

      let expected = runtime.threadIDForTesting
      let observed = actorThreadIDs + actorToTaskThreadIDs + taskThreadIDs + runThreadIDs
      #expect(Set(observed) == [expected])
      #expect(expected != callerThreadID)
    }

    @Test("The shared worker preserves FIFO submission order", .timeLimit(.minutes(1)))
    func fifoOrder() async {
      let runtime = MLXPinnedRuntime.shared
      let observed = PinnedRuntimeOrderRecorder()
      let tasks = (0..<128).map { value in
        Task(executorPreference: runtime.taskExecutor) {
          observed.append(value)
        }
      }

      for task in tasks {
        await task.value
      }

      #expect(observed.values == Array(0..<128))
    }

    @Test("A thrown operation does not poison the permanent worker", .timeLimit(.minutes(1)))
    func errorPropagation() async throws {
      let runtime = MLXPinnedRuntime.shared
      let before = try await runtime.run { currentTestThreadID() }

      do {
        let _: UInt64 = try await runtime.run {
          throw PinnedRuntimeProbeError.expected
        }
        Issue.record("Expected the pinned operation to throw")
      } catch let error as PinnedRuntimeProbeError {
        #expect(error == .expected)
      } catch {
        Issue.record("Unexpected error: \(error)")
      }

      let after = try await runtime.run { currentTestThreadID() }
      #expect(before == runtime.threadIDForTesting)
      #expect(after == before)
    }

    @Test(
      "Canceled executor jobs run on the worker and unwind cooperatively", .timeLimit(.minutes(1)))
    func cancellationUnwinds() async throws {
      let runtime = MLXPinnedRuntime.shared
      let gate = PinnedRuntimeBlockingGate()
      let cancellationThreadID = Mutex<UInt64?>(nil)
      let ranPastCancellationCheck = Mutex(false)

      let blocker = Task(executorPreference: runtime.taskExecutor) {
        let threadID = currentTestThreadID()
        gate.enterAndWait()
        return threadID
      }
      gate.waitUntilEntered()

      let canceled = Task(executorPreference: runtime.taskExecutor) {
        cancellationThreadID.withLock { $0 = currentTestThreadID() }
        try Task.checkCancellation()
        ranPastCancellationCheck.withLock { $0 = true }
      }
      canceled.cancel()
      gate.open()

      let blockerThreadID = await blocker.value
      do {
        try await canceled.value
        Issue.record("Expected cancellation to propagate")
      } catch is CancellationError {
        // Expected. The job still had to execute on the pinned worker to unwind.
      } catch {
        Issue.record("Unexpected cancellation error: \(error)")
      }

      let after = try await runtime.run { currentTestThreadID() }
      #expect(blockerThreadID == runtime.threadIDForTesting)
      #expect(cancellationThreadID.withLock { $0 } == blockerThreadID)
      #expect(!ranPastCancellationCheck.withLock { $0 })
      #expect(after == blockerThreadID)
    }
  }

  private actor PinnedThreadProbe {
    nonisolated let runtime: MLXPinnedRuntime

    nonisolated var unownedExecutor: UnownedSerialExecutor {
      runtime.unownedSerialExecutor
    }

    init(runtime: MLXPinnedRuntime) {
      self.runtime = runtime
    }

    func threadIDsAcrossSuspension() async -> [UInt64] {
      let beforeYield = currentTestThreadID()
      await Task.yield()
      return [beforeYield, currentTestThreadID()]
    }

    func taskExecutorThreadIDs() async throws -> [UInt64] {
      try await runtime.run {
        let beforeYield = currentTestThreadID()
        await Task.yield()
        return [beforeYield, currentTestThreadID()]
      }
    }
  }

  private final class PinnedRuntimeBlockingGate: @unchecked Sendable {
    private let condition = NSCondition()
    private var entered = false
    private var isOpen = false

    func enterAndWait() {
      condition.lock()
      entered = true
      condition.broadcast()
      while !isOpen {
        condition.wait()
      }
      condition.unlock()
    }

    func waitUntilEntered() {
      condition.lock()
      while !entered {
        condition.wait()
      }
      condition.unlock()
    }

    func open() {
      condition.lock()
      isOpen = true
      condition.broadcast()
      condition.unlock()
    }
  }

  private final class PinnedRuntimeOrderRecorder: @unchecked Sendable {
    private let storage = Mutex<[Int]>([])

    var values: [Int] {
      storage.withLock { $0 }
    }

    func append(_ value: Int) {
      storage.withLock { $0.append(value) }
    }
  }

  private enum PinnedRuntimeProbeError: Error, Equatable {
    case expected
  }

  private func currentTestThreadID() -> UInt64 {
    var identifier: UInt64 = 0
    precondition(pthread_threadid_np(nil, &identifier) == 0)
    return identifier
  }
#endif
