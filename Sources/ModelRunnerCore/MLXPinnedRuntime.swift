#if os(macOS)
  import Darwin
  import Foundation

  /// Process-lifetime execution context for MLX work that must remain on one
  /// macOS thread.
  ///
  /// Swift's serial executors guarantee ordering, but they do not otherwise
  /// guarantee that successive jobs use the same pthread. This runtime gives
  /// actor-isolated work and explicitly preferred tasks separate executor
  /// identities backed by one permanent FIFO worker thread.
  final class MLXPinnedRuntime: @unchecked Sendable {
    static let shared = MLXPinnedRuntime()

    let serialExecutor: MLXPinnedSerialExecutor
    let taskExecutor: MLXPinnedTaskExecutor

    private let worker: MLXPinnedThread

    var unownedSerialExecutor: UnownedSerialExecutor {
      serialExecutor.asUnownedSerialExecutor()
    }

    private init() {
      let worker = MLXPinnedThread()
      self.worker = worker
      self.serialExecutor = MLXPinnedSerialExecutor(worker: worker)
      self.taskExecutor = MLXPinnedTaskExecutor(worker: worker)
    }

    /// Runs an async operation with this runtime's task-executor preference.
    ///
    /// The explicit preferred executor is important: a plain unstructured
    /// `Task` created by a caller is not guaranteed to inherit the executor on
    /// which that caller happens to be running.
    func run<Result: Sendable>(
      _ operation: @escaping @Sendable () async throws -> Result
    ) async throws -> Result {
      try Task.checkCancellation()
      let task = Task(executorPreference: taskExecutor) {
        try await operation()
      }
      return try await withTaskCancellationHandler {
        try await task.value
      } onCancel: {
        task.cancel()
      }
    }

    /// Stable pthread identifier captured by the permanent worker at startup.
    /// Kept internal so focused tests can verify executor placement without
    /// exposing diagnostics through the runner's public API.
    var threadIDForTesting: UInt64 {
      worker.threadID
    }
  }

  final class MLXPinnedSerialExecutor: SerialExecutor, @unchecked Sendable {
    private let worker: MLXPinnedThread

    fileprivate init(worker: MLXPinnedThread) {
      self.worker = worker
    }

    func enqueue(_ job: consuming ExecutorJob) {
      let job = UnownedJob(job)
      let executor = asUnownedSerialExecutor()
      worker.enqueue {
        job.runSynchronously(on: executor)
      }
    }
  }

  final class MLXPinnedTaskExecutor: TaskExecutor, @unchecked Sendable {
    private let worker: MLXPinnedThread

    fileprivate init(worker: MLXPinnedThread) {
      self.worker = worker
    }

    func enqueue(_ job: consuming ExecutorJob) {
      let job = UnownedJob(job)
      let executor = asUnownedTaskExecutor()
      worker.enqueue {
        job.runSynchronously(on: executor)
      }
    }
  }

  private final class MLXPinnedThread: @unchecked Sendable {
    typealias Work = @Sendable () -> Void

    private final class WorkNode: @unchecked Sendable {
      let work: Work
      var next: WorkNode?

      init(_ work: @escaping Work) {
        self.work = work
      }
    }

    private let condition = NSCondition()
    private var head: WorkNode?
    private var tail: WorkNode?
    private var startedThreadID: UInt64?
    private lazy var thread = Thread { [weak self] in
      self?.runLoop()
    }

    init() {
      thread.name = "midnight.mlx-pinned"
      thread.qualityOfService = .userInitiated

      condition.lock()
      thread.start()
      while startedThreadID == nil {
        condition.wait()
      }
      condition.unlock()
    }

    var threadID: UInt64 {
      condition.lock()
      defer { condition.unlock() }
      // Initialization does not return until the worker publishes this value.
      return startedThreadID!
    }

    func enqueue(_ work: @escaping Work) {
      let node = WorkNode(work)
      condition.lock()
      if let tail {
        tail.next = node
      } else {
        head = node
      }
      tail = node
      condition.signal()
      condition.unlock()
    }

    private func runLoop() {
      condition.lock()
      startedThreadID = currentOSThreadID()
      condition.broadcast()
      condition.unlock()

      while true {
        condition.lock()
        while head == nil {
          condition.wait()
        }
        let node = head!
        head = node.next
        node.next = nil
        if head == nil {
          tail = nil
        }
        condition.unlock()

        autoreleasepool(invoking: node.work)
      }
    }
  }

  private func currentOSThreadID() -> UInt64 {
    var identifier: UInt64 = 0
    precondition(pthread_threadid_np(nil, &identifier) == 0)
    return identifier
  }
#endif
