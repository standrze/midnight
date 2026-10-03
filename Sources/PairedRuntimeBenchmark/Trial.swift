import ArgumentParser
import Foundation
import MLX
import ModelQualityCore
import ModelRunnerCore
import ModelRunnerProtocol

func runTrial(
    runner: LocalModelRunner, checkpoint: CheckedCheckpoint, messages: [OpenAIMessage], expectedPromptIDs: [Int],
    phase: String, pair: Int, order: String, position: Int, sequence: Int,
    experimentStart: ContinuousClock.Instant, memoryCeiling: Int
) async -> TrialRecord {
    // Reset and synchronization are symmetric and outside request timing. Reset
    // also clears the process-wide allocator cache, but never the loaded weights.
    await runner.waitUntilIdle()
    let clock = ContinuousClock()
    var record = TrialRecord(
        sequence: sequence, phase: phase, pair: pair, order: order, positionInPair: position,
        arm: checkpoint.record.arm, startedAt: ISO8601DateFormatter().string(from: Date()),
        startElapsedMilliseconds: elapsedMilliseconds(experimentStart.duration(to: clock.now)),
        thermalStateBefore: thermalStateValue())
    var requestStart: ContinuousClock.Instant?
    do {
        try await runner.resetKVCache()
        let required = checkpoint.profile.requestBytes(
            prompt: expectedPromptIDs.count, output: 512, prefillStepSize: 512, residentBytes: Memory.activeMemory)
        guard required < memoryCeiling else {
            throw ValidationError(
                "Combined resident allocation plus request estimate \(required) exceeds \(memoryCeiling).")
        }
        Memory.peakMemory = 0
        let startedAt = clock.now
        requestStart = startedAt
        let prepared = try await runner.preparePrompt(messages: messages, maximumTokens: 512)
        record.promptTokenCount = prepared.promptTokenCount
        record.promptTokenIDFingerprint = ModelQualityCore.tokenIDFingerprint(prepared.promptTokenIDs)
        guard prepared.promptTokenIDs == expectedPromptIDs else {
            throw ValidationError("Rendered prompt IDs changed; refusing a mismatched-input comparison.")
        }
        let events = await runner.stream(
            messages: messages, maximumTokens: 512, temperature: 0, topP: 1,
            enablePromptCache: false, enableSpeculativeDecoding: false, preparedPrompt: prepared)
        var metricsCount = 0
        for try await event in events {
            switch event {
            case .content(let text):
                if !text.isEmpty {
                    let elapsed = elapsedMilliseconds(startedAt.duration(to: clock.now))
                    if record.timeToFirstTokenMilliseconds == nil {
                        record.timeToFirstTokenMilliseconds = elapsed
                    }
                    if record.timeToFirstVisibleOutputMilliseconds == nil {
                        record.timeToFirstVisibleOutputMilliseconds = elapsed
                    }
                }
                record.content += text
            case .reasoning(let text):
                if !text.isEmpty, record.timeToFirstVisibleOutputMilliseconds == nil {
                    record.timeToFirstVisibleOutputMilliseconds = elapsedMilliseconds(startedAt.duration(to: clock.now))
                }
                record.reasoning += text
            case .toolCall:
                record.toolCallCount += 1
            case .metrics(let metrics):
                metricsCount += 1
                record.metrics = TrialMetrics(metrics)
            }
        }
        if metricsCount != 1 {
            record.validationIssues.append("Expected exactly one metrics event; got \(metricsCount).")
        }
        if record.toolCallCount != 0 {
            record.validationIssues.append("Unexpected tool-call output.")
        }
        if let metrics = record.metrics {
            if metrics.promptTokenCount != expectedPromptIDs.count
                || metrics.prefilledPromptTokenCount != expectedPromptIDs.count || metrics.cachedPromptTokenCount != 0
            {
                record.validationIssues.append("Fresh prompt accounting mismatch or unexpected cache reuse.")
            }
            if metrics.generationTokenCount != 512 {
                record.validationIssues.append(
                    "Early stop: generated \(metrics.generationTokenCount)/512 requested tokens.")
            }
            if (metrics.proposedDraftTokens ?? 0) != 0 || (metrics.acceptedDraftTokens ?? 0) != 0
                || metrics.speculativePassthroughReason != nil
            {
                record.validationIssues.append("Unexpected speculative metrics in target-only generation.")
            }
            if (metrics.tokensPerSecond ?? 0) <= 0 || (metrics.promptTokensPerSecond ?? 0) <= 0 {
                record.validationIssues.append("Missing, nonfinite, or nonpositive throughput.")
            }
        }
    } catch {
        record.error = error.localizedDescription
    }
    // A failed or cancelled producer must finish before another actor uses MLX.
    await runner.waitUntilIdle()
    if let requestStart {
        record.totalMilliseconds = elapsedMilliseconds(requestStart.duration(to: clock.now))
        record.peakActiveMemoryBytes = Memory.peakMemory
    }
    record.activeMemoryAfterBytes = Memory.activeMemory
    record.thermalStateAfter = thermalStateValue()
    if let peak = record.peakActiveMemoryBytes, peak >= memoryCeiling, record.error == nil {
        record.error = "Observed process-wide MLX peak reached the prototype's \(memoryCeiling)-byte ceiling."
    }
    return record
}

private func thermalStateValue() -> Int? {
    #if os(macOS)
        ProcessInfo.processInfo.thermalState.rawValue
    #else
        nil
    #endif
}
