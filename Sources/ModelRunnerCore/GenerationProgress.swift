import Dispatch
import Synchronization

/// Local console measurements; these are independent of the HTTP usage contract.
public struct GenerationProgressSnapshot: Sendable {
    /// Preparation, decoding, or the terminal outcome of the most recent monitored generation.
    public enum Phase: String, Sendable {
        case preparing, generating, completed, cancelled, failed
    }

    public let phase: Phase
    /// Emitted token IDs, including reasoning and tools, excluding discarded EOS tokens.
    public let generatedTokens: Int
    /// Effective output token budget after context and admission checks.
    public let maximumTokens: Int?
    public let promptTokens: Int?
    public let cachedPromptTokens: Int?
    /// Live decode throughput after the first token, or the final runner metric.
    public let tokensPerSecond: Double?
    /// Time since acquiring the generation slot, excluding queue wait.
    public let elapsedSeconds: Double
    /// Slot acquisition to first emitted token, including prompt rendering and prefill.
    public let firstTokenSeconds: Double?
    public let stopReason: String?
}

/// The decoding worker updates a small locked record; reading it never waits for the model actor.
final class GenerationProgressMonitor: Sendable {
    private struct State {
        var enabled = false
        var phase: GenerationProgressSnapshot.Phase?
        var started = 0.0
        var firstToken: Double?
        var ended: Double?
        var generatedTokens = 0
        var maximumTokens: Int?
        var promptTokens: Int?
        var metrics: LocalModelRunnerMetrics?
    }

    private let state = Mutex(State())
    static var now: Double { Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000_000 }

    func setEnabled(_ enabled: Bool) {
        state.withLock { $0.enabled = enabled }
    }

    @discardableResult
    func begin(now: Double = now) -> Bool {
        state.withLock { state in
            guard state.enabled else {
                return false
            }
            state = State(enabled: true, phase: .preparing, started: now)
            return true
        }
    }

    func prepare(promptTokens: Int, maximumTokens: Int) {
        state.withLock { state in
            state.promptTokens = promptTokens
            state.maximumTokens = maximumTokens
        }
    }

    func record(tokenCount: Int, now: Double = now) {
        state.withLock { state in
            guard state.phase == .preparing || state.phase == .generating else {
                return
            }
            state.phase = .generating
            state.firstToken = state.firstToken ?? now
            state.generatedTokens = tokenCount
        }
    }

    func record(metrics: LocalModelRunnerMetrics) {
        state.withLock { state in
            state.metrics = metrics
            state.generatedTokens = metrics.generationTokenCount
            state.promptTokens = metrics.promptTokenCount
        }
    }

    func finish(succeeded: Bool, cancelled: Bool, now: Double = now) {
        state.withLock { state in
            guard state.phase != nil else {
                return
            }
            if cancelled || state.metrics?.stopReason == "cancelled" {
                state.phase = .cancelled
            } else if succeeded {
                state.phase = .completed
            } else {
                state.phase = .failed
            }
            state.ended = now
        }
    }

    func snapshot(now: Double = now) -> GenerationProgressSnapshot? {
        state.withLock { state in
            guard let phase = state.phase else {
                return nil
            }
            let elapsed = max(0, (state.ended ?? now) - state.started)
            let rate: Double?
            if let metrics = state.metrics {
                rate = metrics.tokensPerSecond.isFinite ? metrics.tokensPerSecond : nil
            } else if let first = state.firstToken, state.generatedTokens > 1, (state.ended ?? now) > first {
                // The first token includes prefill. Measure decode intervals after it.
                rate = Double(state.generatedTokens - 1) / ((state.ended ?? now) - first)
            } else {
                rate = nil
            }
            return GenerationProgressSnapshot(
                phase: phase, generatedTokens: state.generatedTokens, maximumTokens: state.maximumTokens,
                promptTokens: state.promptTokens, cachedPromptTokens: state.metrics?.cachedPromptTokenCount,
                tokensPerSecond: rate, elapsedSeconds: elapsed,
                firstTokenSeconds: state.firstToken.map { max(0, $0 - state.started) },
                stopReason: state.metrics?.stopReason)
        }
    }
}
