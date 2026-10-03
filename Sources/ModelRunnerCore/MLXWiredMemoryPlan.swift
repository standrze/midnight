import Foundation
import MLX
import MLXLMCommon

/// A conservative wired-memory limit learned from isolated and live peak measurements.
///
/// The value is an absolute process-residency budget, so it deliberately uses a fixed policy
/// rather than adding an unknowable process baseline. Live observations can only grow the
/// budget, and both the MLX allocator ceiling and Metal recommendation remain hard caps.
struct MLXWiredMemoryPlan: Equatable, Sendable {
  static let enabledEnvironmentKey = "MODEL_RUNNER_MLX_WIRED_MEMORY"
  static let tuningTokensEnvironmentKey = "MODEL_RUNNER_MLX_WIRED_TUNE_TOKENS"
  static let defaultTuningTokens = 513
  static let defaultMinimumHeadroomBytes = 512 * 1_048_576
  static let dflashMinimumHeadroomBytes = 1_024 * 1_048_576

  let capBytes: Int
  let cacheReserveBytes: Int
  let minimumHeadroomBytes: Int
  let headroomPercent: Int
  private(set) var limitBytes: Int

  static func measured(
    peakActiveBytes: Int,
    cacheReserveBytes: Int,
    allocatorLimitBytes: Int,
    recommendedWorkingSetBytes: Int?,
    minimumHeadroomBytes: Int = defaultMinimumHeadroomBytes,
    headroomPercent: Int = 10
  ) -> Self? {
    let recommended = recommendedWorkingSetBytes.map { max(0, $0) }
      ?? max(0, allocatorLimitBytes)
    let cap = min(max(0, allocatorLimitBytes), recommended)
    guard peakActiveBytes > 0, cap > 0 else { return nil }

    let plan = Self(
      capBytes: cap,
      cacheReserveBytes: max(0, cacheReserveBytes),
      minimumHeadroomBytes: max(0, minimumHeadroomBytes),
      headroomPercent: max(0, headroomPercent),
      limitBytes: 0
    )
    var measuredPlan = plan
    measuredPlan.limitBytes = measuredPlan.limit(forPeakActiveBytes: peakActiveBytes)
    return measuredPlan
  }

  static func isEnabled(environment: [String: String]) -> Bool {
    environment[enabledEnvironmentKey] != "0"
  }

  static func tuningTokenCount(environment: [String: String]) -> Int {
    guard let raw = environment[tuningTokensEnvironmentKey],
      let value = Int(raw),
      (128...4_096).contains(value)
    else { return defaultTuningTokens }
    return value
  }

  mutating func observe(peakActiveBytes: Int) {
    guard peakActiveBytes > 0 else { return }
    limitBytes = max(limitBytes, limit(forPeakActiveBytes: peakActiveBytes))
  }

  func makeTicket(manager: WiredMemoryManager = .shared) -> WiredMemoryTicket {
    WiredMemoryTicket(
      size: 0,
      policy: MLXLMCommon.WiredFixedPolicy(limit: limitBytes),
      manager: manager,
      kind: .active
    )
  }

  private func limit(forPeakActiveBytes peak: Int) -> Int {
    let percentHeadroom = Self.saturatingCeilingPercent(peak, percent: headroomPercent)
    let headroom = max(minimumHeadroomBytes, percentHeadroom)
    let desired = Self.saturatingAdd(
      Self.saturatingAdd(peak, cacheReserveBytes),
      headroom
    )
    return min(capBytes, desired)
  }

  private static func saturatingCeilingPercent(_ value: Int, percent: Int) -> Int {
    guard value > 0, percent > 0 else { return 0 }
    let product = value.multipliedReportingOverflow(by: percent)
    if product.overflow { return Int.max }
    return 1 + (product.partialValue - 1) / 100
  }

  private static func saturatingAdd(_ lhs: Int, _ rhs: Int) -> Int {
    let result = lhs.addingReportingOverflow(rhs)
    return result.overflow ? Int.max : result.partialValue
  }
}
