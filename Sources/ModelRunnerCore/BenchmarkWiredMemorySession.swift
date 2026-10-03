import Foundation
import MLX
import ModelRunnerProtocol

/// Benchmark-only view of the existing measured policy; never changes its capacity.
package struct BenchmarkWiredMemoryPlan: Encodable, Sendable {
    package let limitBytes: Int
    package let capBytes: Int
    package let cacheReserveBytes: Int

    init(_ plan: MLXWiredMemoryPlan) {
        limitBytes = plan.limitBytes
        capBytes = plan.capBytes
        cacheReserveBytes = plan.cacheReserveBytes
    }

    enum CodingKeys: String, CodingKey {
        case limitBytes = "limit_bytes"
        case capBytes = "cap_bytes"
        case cacheReserveBytes = "cache_reserve_bytes"
    }
}

/// Cached manager policy and setter results, not measured OS-resident bytes.
package struct BenchmarkWiredMemoryState: Encodable, Sendable {
    package let baselineBytes: Int?
    package let activeBaselineBytes: Int?
    package let lastConfirmedRestoredBaselineBytes: Int?
    package let currentLimitBytes: Int?
    package let activeTicketCount: Int
    package let ticketCount: Int
    package let backendSupported: Bool
    package let lastAttemptedLimitBytes: Int?
    package let lastAttemptSucceeded: Bool?
    package let lastSuccessfulBackendLimitBytes: Int?
    package let backendSuccessCount: Int
    package let backendFailureCount: Int

    package static func capture(manager: WiredMemoryManager = .shared) async -> Self {
        #if os(macOS)
            let state = await manager.snapshot()
            return Self(
                baselineBytes: state.baseline ?? state.lastConfirmedRestoredBaseline,
                activeBaselineBytes: state.baseline,
                lastConfirmedRestoredBaselineBytes: state.lastConfirmedRestoredBaseline,
                currentLimitBytes: state.currentLimit,
                activeTicketCount: state.activeTicketCount, ticketCount: state.ticketCount,
                backendSupported: state.backendSupported,
                lastAttemptedLimitBytes: state.lastAttemptedLimit,
                lastAttemptSucceeded: state.lastAttemptSucceeded,
                lastSuccessfulBackendLimitBytes: state.lastSuccessfulBackendLimit,
                backendSuccessCount: state.backendSuccessCount,
                backendFailureCount: state.backendFailureCount)
        #else
            return Self(
                baselineBytes: nil, activeBaselineBytes: nil, lastConfirmedRestoredBaselineBytes: nil,
                currentLimitBytes: nil, activeTicketCount: 0,
                ticketCount: 0, backendSupported: false, lastAttemptedLimitBytes: nil,
                lastAttemptSucceeded: nil, lastSuccessfulBackendLimitBytes: nil,
                backendSuccessCount: 0, backendFailureCount: 0)
        #endif
    }

    package var confirmsUnwiredBaseline: Bool {
        backendSupported && baselineBytes == 0 && currentLimitBytes == 0
            && activeTicketCount == 0 && ticketCount == 0
            && lastSuccessfulBackendLimitBytes == 0 && backendSuccessCount > 0
            && backendFailureCount == 0 && lastAttemptSucceeded == true
    }

    package func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(baselineBytes, forKey: .baselineBytes)
        try values.encode(activeBaselineBytes, forKey: .activeBaselineBytes)
        try values.encode(lastConfirmedRestoredBaselineBytes, forKey: .lastConfirmedRestoredBaselineBytes)
        try values.encode(currentLimitBytes, forKey: .currentLimitBytes)
        try values.encode(activeTicketCount, forKey: .activeTicketCount)
        try values.encode(ticketCount, forKey: .ticketCount)
        try values.encode(backendSupported, forKey: .backendSupported)
        try values.encode(lastAttemptedLimitBytes, forKey: .lastAttemptedLimitBytes)
        try values.encode(lastAttemptSucceeded, forKey: .lastAttemptSucceeded)
        try values.encode(lastSuccessfulBackendLimitBytes, forKey: .lastSuccessfulBackendLimitBytes)
        try values.encode(backendSuccessCount, forKey: .backendSuccessCount)
        try values.encode(backendFailureCount, forKey: .backendFailureCount)
    }

    enum CodingKeys: String, CodingKey {
        case baselineBytes = "baseline_bytes"
        case activeBaselineBytes = "active_baseline_bytes"
        case lastConfirmedRestoredBaselineBytes = "last_confirmed_restored_baseline_bytes"
        case currentLimitBytes = "current_limit_bytes"
        case activeTicketCount = "active_ticket_count"
        case ticketCount = "ticket_count"
        case backendSupported = "backend_supported"
        case lastAttemptedLimitBytes = "last_attempted_limit_bytes"
        case lastAttemptSucceeded = "last_attempt_succeeded"
        case lastSuccessfulBackendLimitBytes = "last_successful_backend_limit_bytes"
        case backendSuccessCount = "backend_success_count"
        case backendFailureCount = "backend_failure_count"
    }
}

package struct BenchmarkWiredMemoryRequest: Encodable, Sendable {
    package let requestedLimitBytes: Int
    package let startReturnedLimitBytes: Int
    package let activeState: BenchmarkWiredMemoryState

    enum CodingKeys: String, CodingKey {
        case requestedLimitBytes = "requested_limit_bytes"
        case startReturnedLimitBytes = "start_returned_limit_bytes"
        case activeState = "active_state"
    }
}

package struct BenchmarkWiredMemorySessionReport: Encodable, Sendable {
    package let requestedLimitBytes: Int
    package let startReturnedLimitBytes: Int
    package let endReturnedLimitBytes: Int
    package let startMilliseconds: Double
    package let endMilliseconds: Double
    package let before: BenchmarkWiredMemoryState
    package let started: BenchmarkWiredMemoryState
    package let ended: BenchmarkWiredMemoryState

    package var confirmsRestoredBaseline: Bool {
        ended.confirmsUnwiredBaseline
            && ended.backendSuccessCount == started.backendSuccessCount + 1
    }

    enum CodingKeys: String, CodingKey {
        case before, started, ended
        case requestedLimitBytes = "requested_limit_bytes"
        case startReturnedLimitBytes = "start_returned_limit_bytes"
        case endReturnedLimitBytes = "end_returned_limit_bytes"
        case startMilliseconds = "start_milliseconds"
        case endMilliseconds = "end_milliseconds"
    }
}

/// Holds one fixed-policy ticket until the caller has joined every producer.
package struct BenchmarkWiredMemorySession: Sendable {
    private let plan: MLXWiredMemoryPlan
    private let manager: WiredMemoryManager

    init(engine: ModelEngine, plan: MLXWiredMemoryPlan?, manager: WiredMemoryManager = .shared) throws {
        guard engine == .metal, let plan else {
            throw Failure.ineligible
        }
        self.plan = plan
        self.manager = manager
    }

    package func run<Result>(
        requireBackendConfirmation: Bool = true,
        cleanup: () async -> Void,
        operation: () async throws -> Result
    ) async throws -> (Result, BenchmarkWiredMemorySessionReport) {
        let ticket = plan.makeTicket(manager: manager)
        let before = await BenchmarkWiredMemoryState.capture(manager: manager)
        let clock = ContinuousClock()
        let startTime = clock.now
        let applied = await ticket.start()
        let startMilliseconds = Self.milliseconds(startTime.duration(to: clock.now))
        let started = await BenchmarkWiredMemoryState.capture(manager: manager)
        let result: Swift.Result<Result, Error>
        do {
            guard applied == plan.limitBytes else {
                throw Failure.notApplied(requested: plan.limitBytes, returned: applied)
            }
            if requireBackendConfirmation {
                guard before.confirmsUnwiredBaseline,
                    started.baselineBytes == 0,
                    started.currentLimitBytes == plan.limitBytes,
                    started.lastSuccessfulBackendLimitBytes == plan.limitBytes,
                    started.lastAttemptedLimitBytes == plan.limitBytes,
                    started.lastAttemptSucceeded == true,
                    started.backendSupported,
                    started.activeTicketCount == 1, started.ticketCount == 1,
                    started.backendFailureCount == 0,
                    started.backendSuccessCount == before.backendSuccessCount + 1
                else {
                    throw Failure.unconfirmedStart
                }
            }
            try Task.checkCancellation()
            result = .success(try await operation())
        } catch {
            result = .failure(error)
        }
        // Cancellation must not bypass producer joining or release this ticket early.
        await cleanup()
        let endTime = clock.now
        let returned = await ticket.end()
        let endMilliseconds = Self.milliseconds(endTime.duration(to: clock.now))
        let ended = await BenchmarkWiredMemoryState.capture(manager: manager)
        let report = BenchmarkWiredMemorySessionReport(
            requestedLimitBytes: plan.limitBytes, startReturnedLimitBytes: applied,
            endReturnedLimitBytes: returned, startMilliseconds: startMilliseconds,
            endMilliseconds: endMilliseconds, before: before, started: started, ended: ended)
        switch result {
        case .success(let value): return (value, report)
        case .failure(let error): throw ExecutionFailure(underlying: error, report: report)
        }
    }

    private static func milliseconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) * 1_000 + Double(parts.attoseconds) / 1e15
    }

    enum Failure: Error, LocalizedError {
        case ineligible
        case notApplied(requested: Int, returned: Int)
        case unconfirmedStart

        var errorDescription: String? {
            switch self {
            case .ineligible: "Session residency requires Metal and an available measured wired-memory plan."
            case .notApplied(let requested, let returned):
                "Session wired-memory limit was not applied exactly (requested=\(requested), returned=\(returned))."
            case .unconfirmedStart:
                "Session wired-memory setter history, baseline, or active ticket state could not be confirmed."
            }
        }
    }

    package struct ExecutionFailure: Error, LocalizedError {
        package let underlying: Error
        package let report: BenchmarkWiredMemorySessionReport

        package var errorDescription: String? {
            "Session residency ended after failure: \(underlying.localizedDescription); "
                + "requested=\(report.requestedLimitBytes), started=\(report.startReturnedLimitBytes), "
                + "end returned=\(report.endReturnedLimitBytes), "
                + "restore confirmed=\(report.ended.confirmsUnwiredBaseline)."
        }
    }
}
