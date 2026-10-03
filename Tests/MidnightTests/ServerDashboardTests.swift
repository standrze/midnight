import Testing
import loom

@testable import Midnight
@testable import ModelRunnerCore

@Suite("Terminal dashboard")
struct ServerDashboardTests {
    @Test @MainActor func dashboardShowsMeasurementsAndPreservesEarlyCompletion() throws {
        let monitor = GenerationProgressMonitor()
        monitor.setEnabled(true)
        monitor.begin(now: 0)
        monitor.prepare(promptTokens: 100, maximumTokens: 64)
        monitor.record(tokenCount: 16, now: 1)
        monitor.record(
            metrics: LocalModelRunnerMetrics(
                promptTokenCount: 100, cachedPromptTokenCount: 80, generationTokenCount: 16,
                promptTokensPerSecond: 100, tokensPerSecond: 20, stopReason: "stop"))
        monitor.finish(succeeded: true, cancelled: false, now: 2)
        var frame = Frame(width: 80, height: 24)
        ServerConsole.render(
            frame: &frame, endpoint: "local", listening: true, state: nil, models: [], selectedIndex: 0,
            message: "Ready", logs: [],
            dashboard: ServerDashboardSnapshot(
                generation: monitor.snapshot(now: 2), activeRequests: 2, queuedRequests: 1, uptimeSeconds: 3661))
        let text = frame.buffer.plainText
        #expect(text.contains("Completed · 16 / 64 token limit"))
        #expect(text.contains("20.0 tok/s · Prompt 100 · Cached 80"))
        #expect(text.contains("2 active · 1 queued · Up 01:01:01"))
        #expect(text.contains("Elapsed 2.0s · First token 1.00s"))
        #expect(text.contains("[######------------------] output limit · stop"))
        #expect(text.contains("Installed models"))
    }

    @Test @MainActor func barsClampInvalidValuesAndUnavailableMetricsAreExplicit() {
        #expect(ServerDashboard.bar(ratio: -1, width: 4) == "[----]")
        #expect(ServerDashboard.bar(ratio: 2, width: 4) == "[####]")
        #expect(ServerDashboard.bar(ratio: .nan, width: 4) == "[----]")
        var frame = Frame(width: 80, height: 24)
        ServerDashboard.render(frame: &frame, state: nil, snapshot: ServerDashboardSnapshot(activeRequests: 1))
        #expect(frame.buffer.plainText.contains("Request active · text metrics unavailable"))
        #expect(frame.buffer.plainText.contains("— tok/s"))
    }
}
