#if os(macOS)
    import Cmlx
    import Darwin
    import Foundation
    import MLX
    import Testing

    /// These allocator assertions require a separate test process: serializing this
    /// suite does not prevent other suites from allocating model tensors.
    /// Run only with the named filter and MIDNIGHT_RUN_COMPILE_CACHE_LIFETIME=1.
    /// All tensor operations explicitly use the CPU; no model or GPU is required.
    @Suite(
        "Cross-thread MLX compile-cache lifetime", .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_RUN_COMPILE_CACHE_LIFETIME"] == "1"))
    struct MLXCompileCacheLifetimeTests {
        @Test(
            "Final release removes captured constants from two still-live compiler threads",
            .timeLimit(.minutes(1)))
        func finalReleaseAcrossThreads() throws {
            let first = CompileLifetimeThread("compile-cache-first")
            let second = CompileLifetimeThread("compile-cache-second")
            let release = CompileLifetimeThread("compile-cache-release")
            defer {
                first.stop()
                second.stop()
                release.stop()
            }
            let ids = try [
                first.run { compileLifetimeThreadID() },
                second.run { compileLifetimeThreadID() }, release.run { compileLifetimeThreadID() },
            ]
            try #require(Set(ids).count == 3)

            // Warm backend bookkeeping and the unaffected comparison function before
            // measuring the larger constant's lifetime.
            let unrelated = CompileLifetimeCapture(elements: 1024, value: 9)
            defer { unrelated.release() }
            #expect(try first.run { unrelated.evaluate() } == 11)
            #expect(try second.run { unrelated.evaluate() } == 11)
            try #require(
                unrelated.traceCount == 2,
                "Each alive worker must populate its own TLS compile cache")
            #expect(try first.run { unrelated.evaluate() } == 11)
            #expect(unrelated.traceCount == 2, "Compilation must really be cached")
            let before = try release.run { compileLifetimeMemory() }

            let elementCount = 1 << 20
            let capturedBytes = elementCount * MemoryLayout<Float>.stride
            let target = CompileLifetimeCapture(elements: elementCount, value: 3)
            defer { target.release() }
            #expect(try first.run { target.evaluate() } == 5)
            #expect(try second.run { target.evaluate() } == 5)
            try #require(target.traceCount == 2)
            #expect(try second.run { target.evaluate() } == 5)
            try #require(target.traceCount == 2)
            let populated = try release.run { compileLifetimeMemory() }
            try #require(
                populated >= before + capturedBytes,
                "The test must actually allocate and retain the 4 MiB captured tensor")

            // No result tensor or other strong Swift array owner escapes evaluate().
            // Only the compiler tapes can retain the buffer once this closure is gone.
            let released = try release.run {
                target.release()
                return target.swiftCaptureReleased
            }
            #expect(released, "The Swift array wrapper must be gone before judging C++ cache ownership")
            #expect(
                first.isRunning && second.isRunning,
                "Thread exit must not be the operation that frees the captured constants")
            let after = try release.run { compileLifetimeMemory() }
            let tolerance = 256 * 1024
            #expect(
                after <= before + tolerance,
                "Captured cache memory survived final release: baseline=\(before), populated=\(populated), after=\(after)"
            )
            #expect(
                populated - after >= capturedBytes - tolerance,
                "Final closure release must reclaim the target constant while both compiler threads stay alive")

            // Erasing one function must preserve all other function IDs and their warm
            // entries on both threads. Trace count is stronger than output equality.
            #expect(try first.run { unrelated.evaluate() } == 11)
            #expect(try second.run { unrelated.evaluate() } == 11)
            #expect(unrelated.traceCount == 2, "Unrelated cached functions must not be invalidated")
        }

        @Test(
            "A saved weak cache handle stays harmless after its owning thread exits",
            .timeLimit(.minutes(1)))
        func expiredCacheHandleAndThreadExit() throws {
            let exiting = CompileLifetimeThread("compile-cache-exit")
            let survivor = CompileLifetimeThread("compile-cache-survivor")
            let release = CompileLifetimeThread("compile-cache-exit-release")
            defer {
                exiting.stop()
                survivor.stop()
                release.stop()
            }
            let target = CompileLifetimeCapture(elements: 1024, value: 4)
            defer { target.release() }
            #expect(try exiting.run { target.evaluate() } == 6)
            let saved = try exiting.run { try CompileLifetimeWeakHandle() }
            let unrelated = CompileLifetimeCapture(elements: 1024, value: 12)
            defer { unrelated.release() }
            #expect(try survivor.run { unrelated.evaluate() } == 14)
            try #require(unrelated.traceCount == 1)

            exiting.stop()
            try #require(!exiting.isRunning, "The owner must exit before exercising its saved weak handle")
            let statuses = try release.run {
                // Keep the old handle-targeted API contract: an expired handle does not
                // clear another thread's cache. Final Swift release also visits a registry
                // containing a now-expired cache entry after the proposed fix.
                let values = saved.eraseAndClear()
                target.release()
                return values
            }
            #expect(statuses == [0, 0])
            #expect(target.swiftCaptureReleased)
            #expect(try survivor.run { unrelated.evaluate() } == 14)
            #expect(unrelated.traceCount == 1)
        }
    }

    private final class CompileLifetimeCapture: @unchecked Sendable {
        // Calls and release are explicitly sequenced by the test. The lock protects
        // only the trace counter; no caller can race evaluation against final release.
        private var compiled: (@Sendable (MLXArray) -> MLXArray)?
        private weak var capture: MLXArray?
        private let traces = CompileLifetimeCounter()

        init(elements: Int, value: Float) {
            let tensor = Device.withDefaultDevice(.cpu) {
                let array = MLXArray(Array(repeating: value, count: elements))
                eval(array)
                return array
            }
            capture = tensor
            let counter = traces
            compiled = MLX.compile { (input: MLXArray) in
                counter.increment()
                return add(input, tensor, stream: .cpu)
            }
        }

        var traceCount: Int { traces.value }
        var swiftCaptureReleased: Bool { capture == nil }

        func evaluate() -> Float {
            Device.withDefaultDevice(.cpu) {
                autoreleasepool {
                    let output = compiled!(MLXArray(Float(2)))
                    eval(output)
                    let result = output[0].item(Float.self)
                    Stream.cpu.synchronize()
                    return result
                }
            }
        }

        func release() { compiled = nil }
    }

    private final class CompileLifetimeCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0
        var value: Int { lock.withLock { count } }
        func increment() { lock.withLock { count += 1 } }
    }

    private final class CompileLifetimeWeakHandle: @unchecked Sendable {
        private var handle = mlx_compile_cache_new()
        init() throws {
            guard mlx_detail_compile_cache(&handle) == 0 else {
                throw CompileLifetimeFailure("Could not save the worker's weak compiler cache handle")
            }
        }
        deinit { mlx_compile_cache_free(handle) }
        func eraseAndClear() -> [Int32] {
            [mlx_detail_compile_erase(handle, UInt.max), mlx_detail_compile_clear_cache(handle)]
        }
    }

    private func compileLifetimeMemory() -> Int {
        Device.withDefaultDevice(.cpu) {
            autoreleasepool {
                Stream.cpu.synchronize()
                Memory.clearCache()
                return Memory.activeMemory
            }
        }
    }

    private func compileLifetimeThreadID() -> UInt64 {
        var identifier: UInt64 = 0
        precondition(pthread_threadid_np(nil, &identifier) == 0)
        return identifier
    }

    private struct CompileLifetimeFailure: Error {
        let message: String
        init(_ message: String) { self.message = message }
    }

    private final class CompileLifetimeResult<Value>: @unchecked Sendable {
        private let lock = NSLock()
        private var value: Result<Value, any Error>?
        func set(_ result: Result<Value, any Error>) { lock.withLock { value = result } }
        func get() throws -> Value {
            try lock.withLock {
                guard let value else {
                    throw CompileLifetimeFailure("Worker returned no result")
                }
                return try value.get()
            }
        }
    }

    /// A real persistent pthread, rather than a Dispatch queue which may change
    /// threads between calls. It keeps the upstream TLS compiler cache alive.
    private final class CompileLifetimeThread: @unchecked Sendable {
        private let condition = NSCondition()
        private var jobs: [@Sendable () -> Void] = []
        private var stopping = false
        private let done = DispatchSemaphore(value: 0)
        private var thread: Thread!

        init(_ name: String) {
            thread = Thread { [weak self] in self?.loop() }
            thread.name = name
            thread.start()
        }

        var isRunning: Bool { !thread.isFinished }

        func run<Value: Sendable>(_ operation: @escaping @Sendable () throws -> Value) throws -> Value {
            let result = CompileLifetimeResult<Value>()
            let complete = DispatchSemaphore(value: 0)
            condition.lock()
            guard !stopping else {
                condition.unlock()
                throw CompileLifetimeFailure("Work submitted after stopping thread")
            }
            jobs.append {
                result.set(Result { try operation() })
                complete.signal()
            }
            condition.signal()
            condition.unlock()
            guard complete.wait(timeout: .now() + 20) == .success else {
                throw CompileLifetimeFailure("Compiler worker timed out")
            }
            return try result.get()
        }

        func stop() {
            condition.lock()
            let alreadyStopping = stopping
            stopping = true
            condition.broadcast()
            condition.unlock()
            if !alreadyStopping {
                _ = done.wait(timeout: .now() + 5)
            }
            // The loop's completion signal precedes pthread TLS destruction. Wait for
            // Foundation's thread-finished state before asserting an expired cache.
            let deadline = Date().addingTimeInterval(5)
            while !thread.isFinished && Date() < deadline {
                usleep(1_000)
            }
        }

        private func loop() {
            defer { done.signal() }
            while true {
                condition.lock()
                while jobs.isEmpty && !stopping {
                    condition.wait()
                }
                if jobs.isEmpty && stopping {
                    condition.unlock()
                    return
                }
                let job = jobs.removeFirst()
                condition.unlock()
                autoreleasepool(invoking: job)
            }
        }
    }
#endif
