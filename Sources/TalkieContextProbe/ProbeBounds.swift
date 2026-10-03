/// Pure arithmetic used by the context diagnostic's allocation and time guards.
enum ProbeBounds {
    static func capacityUpperBound(after processedTokens: Int) -> Int {
        precondition(processedTokens >= 0)
        // Upstream cache growth can trim to the previous offset and append 256 rows.
        // For odd chunks its capacity is not a multiple of 256: chunk33 reaches
        // capacity718 at offset495. At most255 extra rows bound either cache type.
        return processedTokens + 255
    }

    static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }
}
