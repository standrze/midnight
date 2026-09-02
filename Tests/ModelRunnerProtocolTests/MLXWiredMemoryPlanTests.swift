import MLX
import Testing

@testable import ModelRunnerCore

@Suite("MLX wired-memory plan")
struct MLXWiredMemoryPlanTests {
  @Test("Measured peak receives cache reserve and conservative headroom")
  func measuredBudget() {
    let plan = MLXWiredMemoryPlan.measured(
      peakActiveBytes: 4_000,
      cacheReserveBytes: 500,
      allocatorLimitBytes: 10_000,
      recommendedWorkingSetBytes: 9_000,
      minimumHeadroomBytes: 700,
      headroomPercent: 10
    )

    #expect(plan?.capBytes == 9_000)
    #expect(plan?.limitBytes == 5_200)
  }

  @Test("Both allocator and Metal limits are hard caps")
  func caps() {
    let allocatorCapped = MLXWiredMemoryPlan.measured(
      peakActiveBytes: 8_000,
      cacheReserveBytes: 1_000,
      allocatorLimitBytes: 8_500,
      recommendedWorkingSetBytes: 20_000,
      minimumHeadroomBytes: 1_000
    )
    let metalCapped = MLXWiredMemoryPlan.measured(
      peakActiveBytes: 8_000,
      cacheReserveBytes: 1_000,
      allocatorLimitBytes: 20_000,
      recommendedWorkingSetBytes: 8_750,
      minimumHeadroomBytes: 1_000
    )

    #expect(allocatorCapped?.limitBytes == 8_500)
    #expect(metalCapped?.limitBytes == 8_750)
  }

  @Test("Live observations grow the plan but never shrink it")
  func monotonicObservation() {
    var plan = MLXWiredMemoryPlan.measured(
      peakActiveBytes: 2_000,
      cacheReserveBytes: 100,
      allocatorLimitBytes: 10_000,
      recommendedWorkingSetBytes: nil,
      minimumHeadroomBytes: 500
    )!
    let initial = plan.limitBytes
    plan.observe(peakActiveBytes: 1_000)
    #expect(plan.limitBytes == initial)
    plan.observe(peakActiveBytes: 5_000)
    #expect(plan.limitBytes == 5_600)
    plan.observe(peakActiveBytes: Int.max)
    #expect(plan.limitBytes == 10_000)
  }

  @Test("Invalid measurements fail closed and environment parsing is bounded")
  func invalidMeasurementAndEnvironment() {
    #expect(
      MLXWiredMemoryPlan.measured(
        peakActiveBytes: 0,
        cacheReserveBytes: 0,
        allocatorLimitBytes: 1_000,
        recommendedWorkingSetBytes: nil
      ) == nil
    )
    #expect(!MLXWiredMemoryPlan.isEnabled(environment: [
      MLXWiredMemoryPlan.enabledEnvironmentKey: "0"
    ]))
    #expect(
      MLXWiredMemoryPlan.tuningTokenCount(environment: [
        MLXWiredMemoryPlan.tuningTokensEnvironmentKey: "64"
      ]) == MLXWiredMemoryPlan.defaultTuningTokens
    )
    #expect(
      MLXWiredMemoryPlan.tuningTokenCount(environment: [
        MLXWiredMemoryPlan.tuningTokensEnvironmentKey: "1024"
      ]) == 1024
    )
  }

  @Test("Every request gets a unique absolute fixed-policy ticket")
  func freshAbsoluteTickets() {
    let plan = MLXWiredMemoryPlan.measured(
      peakActiveBytes: 2_000,
      cacheReserveBytes: 100,
      allocatorLimitBytes: 10_000,
      recommendedWorkingSetBytes: nil,
      minimumHeadroomBytes: 500
    )!
    let first = plan.makeTicket()
    let second = plan.makeTicket()

    #expect(first.id != second.id)
    #expect(first.size == 0)
    #expect(first.policy.limit(baseline: 99_000, activeSizes: [0]) == plan.limitBytes)
  }
}
