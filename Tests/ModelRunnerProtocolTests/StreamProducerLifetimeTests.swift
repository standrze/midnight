import Testing

@testable import ModelRunnerCore

@Suite("Stream producer drain", .timeLimit(.minutes(1)))
struct StreamProducerLifetimeTests {
    @Test("Consumer termination and waiter cancellation do not skip producer cleanup")
    func cancellationStillDrains() async {
        let lifetime = StreamProducerLifetime()
        let enteredCleanup = ProducerTestGate()
        let finishCleanup = ProducerTestGate()
        let cleanupFinished = ProducerTestCounter()
        let producer = Task {
            // A stream consumer may have already terminated at this point, while
            // its producer is still joining nested tasks and synchronizing MLX.
            await enteredCleanup.open()
            await finishCleanup.wait()
            await cleanupFinished.increment()
        }
        lifetime.track(producer)
        await enteredCleanup.wait()
        producer.cancel()

        let drainStarted = ProducerTestGate()
        let drain = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            await drainStarted.open()
            await lifetime.waitUntilIdle()
            return await cleanupFinished.value == 1
        }
        await drainStarted.wait()
        // Give the drain a chance to suspend while cleanup is deliberately held.
        for _ in 0..<20 {
            await Task.yield()
        }
        #expect(await cleanupFinished.value == 0)
        await finishCleanup.open()
        #expect(await drain.value)
    }

    @Test("Every waiter drains every producer")
    func multipleProducersAndWaiters() async {
        let lifetime = StreamProducerLifetime()
        let release = ProducerTestGate()
        let completed = ProducerTestCounter()
        for _ in 0..<8 {
            lifetime.track(
                Task {
                    await release.wait()
                    await completed.increment()
                })
        }
        let waiters = (0..<8).map { _ in
            Task {
                await lifetime.waitUntilIdle()
                return await completed.value
            }
        }
        await release.open()
        for waiter in waiters {
            #expect(await waiter.value == 8)
        }
        await lifetime.waitUntilIdle()
    }

    @Test("A producer that finishes before registration still reaches idle")
    func completedBeforeRegistration() async {
        let lifetime = StreamProducerLifetime()
        let producer = Task {}
        await producer.value
        lifetime.track(producer)
        await lifetime.waitUntilIdle()
        await lifetime.waitUntilIdle()
    }
}

private actor ProducerTestCounter {
    private(set) var value = 0
    func increment() { value += 1 }
}

private actor ProducerTestGate {
    private var isOpen = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        guard !isOpen else {
            return
        }
        await withCheckedContinuation { waiters.append($0) }
    }

    func open() {
        isOpen = true
        let waiters = waiters
        self.waiters.removeAll()
        for waiter in waiters {
            waiter.resume()
        }
    }
}
