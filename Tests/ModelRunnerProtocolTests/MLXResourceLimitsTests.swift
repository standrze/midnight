import ModelRunnerProtocol
import Testing

@Suite("MLX resource limits")
struct MLXResourceLimitsTests {
  private let gibibyte = 1_073_741_824
  private let mebibyte = 1_048_576

  @Test("CUDA defaults stay below the 4090's usable VRAM")
  func cudaDefaults() throws {
    let limits = try MLXResourceLimits.resolve(
      for: .cuda,
      physicalMemoryBytes: 64 * UInt64(gibibyte),
      environment: [:]
    )

    #expect(limits.memoryLimitBytes == 18 * gibibyte)
    #expect(limits.maximumMemoryLimitBytes == 20 * gibibyte)
    #expect(limits.cacheLimitBytes == 128 * mebibyte)
    #expect(limits.maximumCacheLimitBytes == 1_024 * mebibyte)
  }

  @Test("CUDA overrides may not exceed the absolute bounds")
  func cudaBounds() throws {
    let accepted = try MLXResourceLimits.resolve(
      for: .cuda,
      physicalMemoryBytes: 64 * UInt64(gibibyte),
      environment: [
        MLXResourceLimits.memoryLimitEnvironmentKey: "20",
        MLXResourceLimits.cacheLimitEnvironmentKey: "1024",
      ]
    )
    #expect(accepted.memoryLimitBytes == 20 * gibibyte)
    #expect(accepted.cacheLimitBytes == 1_024 * mebibyte)

    #expect(throws: MLXResourceLimitError.self) {
      try MLXResourceLimits.resolve(
        for: .cuda,
        physicalMemoryBytes: 64 * UInt64(gibibyte),
        environment: [MLXResourceLimits.memoryLimitEnvironmentKey: "20.0001"]
      )
    }
    #expect(throws: MLXResourceLimitError.self) {
      try MLXResourceLimits.resolve(
        for: .cuda,
        physicalMemoryBytes: 64 * UInt64(gibibyte),
        environment: [MLXResourceLimits.cacheLimitEnvironmentKey: "1025"]
      )
    }
  }

  @Test("Metal is bounded by 32 GiB and half of physical RAM")
  func metalBounds() throws {
    let largeHost = try MLXResourceLimits.resolve(
      for: .metal,
      physicalMemoryBytes: 128 * UInt64(gibibyte),
      environment: [:]
    )
    #expect(largeHost.memoryLimitBytes == 24 * gibibyte)
    #expect(largeHost.maximumMemoryLimitBytes == 32 * gibibyte)
    #expect(largeHost.cacheLimitBytes == 256 * mebibyte)

    let smallHost = try MLXResourceLimits.resolve(
      for: .metal,
      physicalMemoryBytes: 32 * UInt64(gibibyte),
      environment: [:]
    )
    #expect(smallHost.memoryLimitBytes == 16 * gibibyte)
    #expect(smallHost.maximumMemoryLimitBytes == 16 * gibibyte)
    #expect(throws: MLXResourceLimitError.self) {
      try MLXResourceLimits.resolve(
        for: .metal,
        physicalMemoryBytes: 32 * UInt64(gibibyte),
        environment: [MLXResourceLimits.memoryLimitEnvironmentKey: "24"]
      )
    }
  }

  @Test("Malformed, non-finite, signed, and empty values fail closed")
  func malformedOverrides() {
    for value in ["", " ", "nan", "inf", "-1", "+1", "1e1", ".5", "5."] {
      #expect(throws: MLXResourceLimitError.self) {
        try MLXResourceLimits.resolve(
          for: .cuda,
          physicalMemoryBytes: 64 * UInt64(gibibyte),
          environment: [MLXResourceLimits.memoryLimitEnvironmentKey: value]
        )
      }
    }
  }

  @Test("A zero cache disables recycling but a zero memory limit is rejected")
  func zeroLimits() throws {
    let limits = try MLXResourceLimits.resolve(
      for: .cuda,
      physicalMemoryBytes: 64 * UInt64(gibibyte),
      environment: [MLXResourceLimits.cacheLimitEnvironmentKey: "0"]
    )
    #expect(limits.cacheLimitBytes == 0)
    #expect(throws: MLXResourceLimitError.self) {
      try MLXResourceLimits.resolve(
        for: .cuda,
        physicalMemoryBytes: 64 * UInt64(gibibyte),
        environment: [MLXResourceLimits.memoryLimitEnvironmentKey: "0"]
      )
    }
  }

  @Test("The cache may not exceed a lowered memory ceiling")
  func cacheBelowMemory() {
    #expect(throws: MLXResourceLimitError.self) {
      try MLXResourceLimits.resolve(
        for: .cuda,
        physicalMemoryBytes: 64 * UInt64(gibibyte),
        environment: [
          MLXResourceLimits.memoryLimitEnvironmentKey: "0.0625",
          MLXResourceLimits.cacheLimitEnvironmentKey: "128",
        ]
      )
    }
  }

  @Test("Auto must be resolved before resource policy selection")
  func rejectsAuto() {
    #expect(throws: MLXResourceLimitError.unresolvedEngine(.auto)) {
      try MLXResourceLimits.resolve(
        for: .auto,
        physicalMemoryBytes: 64 * UInt64(gibibyte),
        environment: [:]
      )
    }
  }
}
