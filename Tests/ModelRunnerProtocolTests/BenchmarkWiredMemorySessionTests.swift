#if os(macOS) && DEBUG
    import MLX
    import Testing

    @testable import ModelRunnerCore

    @Suite("Benchmark wired-memory session", .serialized)
    struct BenchmarkWiredMemorySessionTests {
        private func plan() -> MLXWiredMemoryPlan {
            MLXWiredMemoryPlan.measured(
                peakActiveBytes: 2_000, cacheReserveBytes: 100,
                allocatorLimitBytes: 10_000, recommendedWorkingSetBytes: 10_000,
                minimumHeadroomBytes: 500)!
        }

        private func manager() -> WiredMemoryManager {
            WiredMemoryManager.makeForTesting(
                configuration: .init(
                    shrinkThresholdRatio: 0, shrinkCooldown: 0,
                    policyOnlyWhenUnsupported: true, baselineOverride: 1_024,
                    useRecommendedWorkingSetWhenUnsupported: false))
        }

        @Test("Missing measurements and non-Metal engines are ineligible")
        func ineligiblePlans() {
            #expect(throws: BenchmarkWiredMemorySession.Failure.self) {
                try BenchmarkWiredMemorySession(engine: .metal, plan: nil)
            }
            #expect(throws: BenchmarkWiredMemorySession.Failure.self) {
                try BenchmarkWiredMemorySession(engine: .cpu, plan: plan())
            }
        }

        @Test("Nested fixed tickets retain one capacity until awaited cleanup ends")
        func sameCapAndCleanup() async throws {
            try await Device.withDefaultDevice(.cpu) {
                let manager = manager()
                let plan = plan()
                let initial = await manager.snapshot()
                #expect(initial.currentLimit == nil)
                #expect(initial.lastSuccessfulBackendLimit == nil)
                #expect(initial.backendSuccessCount == 0)
                let session = try BenchmarkWiredMemorySession(engine: .metal, plan: plan, manager: manager)
                let (value, report) = try await session.run(
                    requireBackendConfirmation: false,
                    cleanup: {
                        let state = await manager.snapshot()
                        #expect(state.activeTicketCount == 1)
                        #expect(state.currentLimit == plan.limitBytes)
                    },
                    operation: {
                        for _ in 0..<3 {
                            let inner = plan.makeTicket(manager: manager)
                            #expect(await inner.start() == plan.limitBytes)
                            #expect(await manager.snapshot().activeTicketCount == 2)
                            #expect(await inner.end() == plan.limitBytes)
                            #expect(await manager.snapshot().activeTicketCount == 1)
                        }
                        return 42
                    })
                #expect(value == 42)
                #expect(report.startReturnedLimitBytes == plan.limitBytes)
                #expect(report.started.currentLimitBytes == plan.limitBytes)
                #expect(report.endReturnedLimitBytes == 1_024)
                #expect(report.ended.currentLimitBytes == 1_024)
                #expect(report.ended.activeTicketCount == 0)
                #expect(report.ended.backendSuccessCount == 0)
                #expect(report.ended.lastSuccessfulBackendLimitBytes == nil)
                #expect(!report.ended.confirmsUnwiredBaseline)
            }
        }

        @Test("An old zero restoration cannot hide a later failed setter or active ticket")
        func baselineEvidence() {
            func state(current: Int, active: Int, succeeded: Bool, failures: Int) -> BenchmarkWiredMemoryState {
                BenchmarkWiredMemoryState(
                    baselineBytes: 0, activeBaselineBytes: nil, lastConfirmedRestoredBaselineBytes: 0,
                    currentLimitBytes: current, activeTicketCount: active, ticketCount: active,
                    backendSupported: true, lastAttemptedLimitBytes: 0,
                    lastAttemptSucceeded: succeeded, lastSuccessfulBackendLimitBytes: current,
                    backendSuccessCount: 2, backendFailureCount: failures)
            }
            #expect(state(current: 0, active: 0, succeeded: true, failures: 0).confirmsUnwiredBaseline)
            #expect(!state(current: 2_600, active: 0, succeeded: false, failures: 1).confirmsUnwiredBaseline)
            #expect(!state(current: 0, active: 0, succeeded: false, failures: 1).confirmsUnwiredBaseline)
            #expect(!state(current: 0, active: 1, succeeded: true, failures: 0).confirmsUnwiredBaseline)
        }

        @Test("Operation errors preserve the ticket through cleanup and return end diagnostics")
        func operationFailure() async throws {
            try await Device.withDefaultDevice(.cpu) {
                let manager = manager()
                let plan = plan()
                let session = try BenchmarkWiredMemorySession(engine: .metal, plan: plan, manager: manager)
                do {
                    let _: (Int, BenchmarkWiredMemorySessionReport) = try await session.run(
                        requireBackendConfirmation: false,
                        cleanup: {
                            let state = await manager.snapshot()
                            #expect(state.activeTicketCount == 1)
                            #expect(state.currentLimit == plan.limitBytes)
                        },
                        operation: { throw CancellationError() })
                    Issue.record("Expected cancellation failure")
                } catch let failure as BenchmarkWiredMemorySession.ExecutionFailure {
                    #expect(failure.underlying is CancellationError)
                    #expect(failure.report.ended.currentLimitBytes == 1_024)
                    #expect(failure.report.ended.activeTicketCount == 0)
                }
                #expect(await manager.snapshot().ticketCount == 0)
            }
        }

        @Test("A larger active policy prevents a falsely equal-cap session and still cleans up")
        func unequalAppliedLimit() async throws {
            try await Device.withDefaultDevice(.cpu) {
                let manager = manager()
                let small = plan()
                var larger = small
                larger.observe(peakActiveBytes: 6_000)
                let other = larger.makeTicket(manager: manager)
                _ = await other.start()
                let session = try BenchmarkWiredMemorySession(engine: .metal, plan: small, manager: manager)
                do {
                    let _: (Int, BenchmarkWiredMemorySessionReport) = try await session.run(
                        requireBackendConfirmation: false,
                        cleanup: { #expect(await manager.snapshot().activeTicketCount == 2) },
                        operation: {
                            Issue.record("An unequal applied limit must reject before operation")
                            return 0
                        })
                    Issue.record("Expected unequal-cap failure")
                } catch let failure as BenchmarkWiredMemorySession.ExecutionFailure {
                    #expect(failure.report.startReturnedLimitBytes == larger.limitBytes)
                    #expect(failure.report.ended.activeTicketCount == 1)
                }
                #expect(await other.end() == 1_024)
                #expect(await manager.snapshot().ticketCount == 0)
            }
        }
    }
#endif
