#if os(macOS)
import Foundation
import MLX
import Testing

@Suite("Sorted gather QMM NAX row bounds", .serialized,
  .enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_RUN_NAX_REGRESSION"] == "1"))
struct SortedGatherQMMNAXTests {
  // Adapted from MLX d73eb752ef2e6288fd95b032c0bff0a15a4a9e93.
  // Opt-in because a passing fallback on older hardware would not validate NAX.
  @Test("Large sorted expert batches match a dequantized FP32 oracle")
  func matchesReferenceAcrossSignedShortBoundary() throws {
    try #require(ProcessInfo.processInfo.isOperatingSystemAtLeast(
      OperatingSystemVersion(majorVersion: 26, minorVersion: 2, patchVersion: 0)))
    let architecture = GPU.deviceInfo().architecture
    let match = try #require(architecture.range(of: "g[0-9]+[a-z]$", options: .regularExpression))
    let suffix = architecture[match]
    let generation = try #require(Int(suffix.dropFirst().dropLast()))
    try #require(generation >= (suffix.last == "p" ? 18 : 17),
      "Run this regression on a NAX-capable GPU, such as M5 Max.")

    Device.withDefaultDevice(.gpu) {
      // Include upstream's group-32 case and Q4R8's 256-expert/group-64 geometry.
      for (experts, groupSize) in [(16, 32), (256, 64)] {
        let width = 64
        let weight = MLXRandom.normal([experts, width, width], scale: 0.1,
          key: MLXRandom.key(0)).asType(.float16)
        let quant = quantized(weight, groupSize: groupSize, bits: 4)
        let restored = dequantized(quant.wq, scales: quant.scales, biases: quant.biases,
          groupSize: groupSize, bits: 4).asType(.float32).swappedAxes(-1, -2)

        // 4096 is the default 512-token/top-8 chunk; 32776 is 4097 * 8.
        // Aligned controls exercise the branch that was never affected.
        for rows in [4096, 32767, 32768, 32769, 32776, 32832] {
          let x = MLXRandom.normal([rows, 1, width], scale: 0.1,
            key: MLXRandom.key(UInt64(rows))).asType(.float16)
          let indices = ((MLXArray(0..<rows) * experts).floorDivide(rows)).asType(.uint32)
          let reference = gatherMM(x.asType(.float32), restored,
            rhsIndices: indices, sortedIndices: true)
          eval(reference)
          #expect(abs(reference).max().item(Float.self) > 0.1,
            "The oracle must distinguish missing/zeroed outputs from the 0.05 tolerance.")
          Stream.defaultStream(.gpu).synchronize()

          // Missing writes must not pass because a recycled buffer was correct.
          for poison in [Float(-31), Float(47)] {
            poisonBuffer(shape: [rows, 1, width], value: poison)
            let actual = gatherQuantizedMM(x, quant.wq, scales: quant.scales,
              biases: quant.biases, rhsIndices: indices, transpose: true,
              groupSize: groupSize, bits: 4, sortedIndices: true)
            let error = abs(actual.asType(.float32) - reference).max().item(Float.self)
            #expect(error.isFinite && error < 0.05,
              "rows=\(rows), experts=\(experts), groupSize=\(groupSize), poison=\(poison), maxError=\(error)")
          }
        }
      }
    }
  }

  private func poisonBuffer(shape: [Int], value: Float) {
    let poison = MLXArray.full(shape, values: MLXArray(value), dtype: .float16)
    eval(poison)
    withExtendedLifetime(poison) { Stream.defaultStream(.gpu).synchronize() }
  }
}
#endif
