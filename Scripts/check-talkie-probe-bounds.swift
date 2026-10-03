// Pure Swift regression checks, without MLX, model loading, or GPU allocation:
// swiftc -parse-as-library Sources/TalkieContextProbe/ProbeBounds.swift \
//   Scripts/check-talkie-probe-bounds.swift -o /private/tmp/talkie-probe-bounds-check
// /private/tmp/talkie-probe-bounds-check

@main
private struct ProbeBoundsChecks {
    static func main() {
        var checked = 0
        for chunk in 1...64 {
            var processed = 0
            var capacity = 0
            while processed < 8_192 {
                let incoming = min(chunk, 8_192 - processed)
                // Simulate the pinned upstream caches' trim-then-grow rule, independently
                // of the diagnostic bound. This includes partial final chunks.
                if processed + incoming > capacity {
                    if processed % 256 != 0 {
                        capacity = processed
                    }
                    capacity += ((256 + incoming - 1) / 256) * 256
                }
                processed += incoming
                precondition(
                    capacity <= ProbeBounds.capacityUpperBound(after: processed),
                    "Underestimated cache capacity at chunk\(chunk), offset\(processed), capacity\(capacity)")
                if chunk == 33 && processed == 495 {
                    precondition(capacity == 718)
                    precondition(
                        ((processed + 255) / 256) * 256 < capacity,
                        "Regression fixture must distinguish the unsafe aligned estimate")
                }
                checked += 1
            }
        }
        precondition(ProbeBounds.seconds(.seconds(2) + .milliseconds(250)) == 2.25)
        precondition(ProbeBounds.seconds(.milliseconds(1)) == 0.001)
        precondition(ProbeBounds.seconds(.seconds(0)) == 0)
        print(
            "Passed \(checked) upstream-growth capacity checks (chunks1...64), odd-chunk33 regression, and monotonic-duration conversion checks."
        )
    }
}
