import Foundation
import ModelRunnerCore
import loom

struct ServerDashboardSnapshot: Sendable {
    var generation: GenerationProgressSnapshot? = nil
    var activeRequests = 0
    var queuedRequests = 0
    var uptimeSeconds = 0.0
}

/// Dashboard values have explicit units; bars measure a budget, not answer completeness.
@MainActor
enum ServerDashboard {
    static func render(
        frame: inout Frame, state: ModelLifecycleState?, snapshot: ServerDashboardSnapshot
    ) {
        let width = max(1, frame.area.width - 2)
        let expanded = frame.area.height >= 20
        func row(_ text: String, at y: Int, color: Int = 7) {
            frame.render(
                Paragraph(TextLayout(text, width: width), style: Style(foreground: .indexed(UInt8(color)))),
                in: Rect(x: 1, y: y, width: width, height: 1))
        }
        if let memory = state?.memory {
            row("MLX memory  \(memory.activeBytes / 1_048_576) MiB active", at: 4)
            row("Cache \(memory.cachedBytes / 1_048_576) MiB · Peak \(memory.peakBytes / 1_048_576) MiB", at: 5)
            if expanded, let budget = state?.loadedModel?.memoryLimitBytes, budget > 0 {
                let ratio = Double(memory.activeBytes) / Double(budget)
                row("\(bar(ratio: ratio, width: 16))  MLX budget \(budget / 1_048_576) MiB", at: 6, color: 6)
            }
        } else {
            row("MLX memory  —", at: 4)
        }
        let requestRow = expanded ? 7 : 6
        row(
            "Requests \(snapshot.activeRequests) active · \(snapshot.queuedRequests) queued · Up \(duration(snapshot.uptimeSeconds))",
            at: requestRow)
        guard let generation = snapshot.generation else {
            let activity =
                snapshot.activeRequests > 0 ? "Request active · text metrics unavailable" : "Awaiting text generation"
            row(activity, at: requestRow + 1, color: 6)
            row("— tok/s · 0 tokens", at: requestRow + 2)
            return
        }
        let phase =
            generation.phase == .preparing
            ? "\(spinner(snapshot.uptimeSeconds)) Preparing / prefill" : generation.phase.rawValue.capitalized
        let limit = generation.maximumTokens.map { " / \($0) token limit" } ?? " tokens"
        row("\(phase) · \(generation.generatedTokens)\(limit)", at: requestRow + 1, color: 6)
        let speed = generation.tokensPerSecond.map { String(format: "%.1f tok/s", $0) } ?? "— tok/s"
        let prompt = generation.promptTokens.map { " · Prompt \($0)" } ?? ""
        let cached = generation.cachedPromptTokens.map { " · Cached \($0)" } ?? ""
        row("\(speed)\(prompt)\(cached)", at: requestRow + 2)
        let first = generation.firstTokenSeconds.map { String(format: " · First token %.2fs", $0) } ?? ""
        row(String(format: "Elapsed %.1fs", generation.elapsedSeconds) + first, at: requestRow + 3)
        if let maximum = generation.maximumTokens, maximum > 0 {
            let ratio = Double(generation.generatedTokens) / Double(maximum)
            let reason = generation.stopReason.map { " · \($0)" } ?? ""
            let barWidth = max(2, min(24, width - 20))
            row("\(bar(ratio: ratio, width: barWidth)) output limit\(reason)", at: requestRow + 4, color: 6)
        }
    }

    static func bar(ratio: Double, width: Int) -> String {
        let cells = max(0, width)
        let bounded = ratio.isFinite ? min(1, max(0, ratio)) : 0
        let filled = Int((bounded * Double(cells)).rounded(.down))
        return "[" + String(repeating: "#", count: filled) + String(repeating: "-", count: cells - filled) + "]"
    }

    static func duration(_ seconds: Double) -> String {
        let total = max(0, Int(seconds))
        return String(format: "%02d:%02d:%02d", total / 3600, (total / 60) % 60, total % 60)
    }

    static func spinner(_ seconds: Double) -> String {
        ["|", "/", "-", "\\"][max(0, Int(seconds * 4)) % 4]
    }
}
