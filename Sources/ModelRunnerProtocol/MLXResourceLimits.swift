import Foundation

/// Fail-closed MLX allocator limits resolved before a model is loaded.
public struct MLXResourceLimits: Equatable, Sendable {
  public static let memoryLimitEnvironmentKey = "MODEL_RUNNER_MLX_MEMORY_LIMIT_GIB"
  public static let cacheLimitEnvironmentKey = "MODEL_RUNNER_MLX_CACHE_LIMIT_MIB"

  public let memoryLimitBytes: Int
  public let cacheLimitBytes: Int
  public let maximumMemoryLimitBytes: Int
  public let maximumCacheLimitBytes: Int

  public static func resolve(
    for engine: ModelEngine,
    physicalMemoryBytes: UInt64,
    environment: [String: String] = ProcessInfo.processInfo.environment
  ) throws -> Self {
    let policy = try Policy(engine: engine, physicalMemoryBytes: physicalMemoryBytes)
    let memoryLimit =
      try parseOverride(
        key: memoryLimitEnvironmentKey,
        environment: environment,
        unitBytes: gibibyte,
        minimumBytes: 1,
        maximumBytes: policy.maximumMemoryBytes
      ) ?? policy.defaultMemoryBytes

    let effectiveCacheMaximum = min(policy.maximumCacheBytes, memoryLimit)
    let cacheLimit =
      try parseOverride(
        key: cacheLimitEnvironmentKey,
        environment: environment,
        unitBytes: mebibyte,
        minimumBytes: 0,
        maximumBytes: effectiveCacheMaximum
      ) ?? min(policy.defaultCacheBytes, effectiveCacheMaximum)

    return Self(
      memoryLimitBytes: memoryLimit,
      cacheLimitBytes: cacheLimit,
      maximumMemoryLimitBytes: policy.maximumMemoryBytes,
      maximumCacheLimitBytes: policy.maximumCacheBytes
    )
  }
}

public enum MLXResourceLimitError: LocalizedError, Equatable, Sendable {
  case unresolvedEngine(ModelEngine)
  case physicalMemoryUnavailable
  case physicalMemoryTooSmall(UInt64)
  case malformedOverride(key: String, value: String)
  case overrideOutOfRange(
    key: String,
    value: String,
    minimumBytes: Int,
    maximumBytes: Int
  )

  public var errorDescription: String? {
    switch self {
    case .unresolvedEngine(let engine):
      "Cannot select MLX resource limits for unresolved engine '\(engine.rawValue)'."
    case .physicalMemoryUnavailable:
      "Physical memory could not be determined; refusing to load an MLX model."
    case .physicalMemoryTooSmall(let bytes):
      "Only \(bytes) bytes of physical memory were reported; refusing to load an MLX model."
    case .malformedOverride(let key, let value):
      "\(key) must be a finite, plain decimal number; received '\(value)'."
    case .overrideOutOfRange(let key, let value, let minimum, let maximum):
      "\(key)=\(value) resolves outside the allowed range \(minimum)...\(maximum) bytes."
    }
  }
}

extension MLXResourceLimits {
  fileprivate static let mebibyte = 1_048_576
  fileprivate static let gibibyte = 1_073_741_824

  fileprivate struct Policy {
    let defaultMemoryBytes: Int
    let maximumMemoryBytes: Int
    let defaultCacheBytes: Int
    let maximumCacheBytes: Int

    init(engine: ModelEngine, physicalMemoryBytes: UInt64) throws {
      switch engine {
      case .cuda:
        self.init(
          defaultMemoryBytes: 18 * MLXResourceLimits.gibibyte,
          maximumMemoryBytes: 20 * MLXResourceLimits.gibibyte,
          defaultCacheBytes: 128 * MLXResourceLimits.mebibyte,
          // DFlash adds draft Q4/Q8 and verification kernels to Laguna's
          // target kernels. Keep the conservative default, but allow an
          // explicit larger cache on 24 GiB cards to avoid JIT churn.
          maximumCacheBytes: 1_024 * MLXResourceLimits.mebibyte
        )
      case .metal:
        let physicalMaximum = try Self.halfPhysicalMemory(physicalMemoryBytes)
        let maximum = min(32 * MLXResourceLimits.gibibyte, physicalMaximum)
        guard maximum >= MLXResourceLimits.gibibyte else {
          throw MLXResourceLimitError.physicalMemoryTooSmall(physicalMemoryBytes)
        }
        self.init(
          defaultMemoryBytes: min(24 * MLXResourceLimits.gibibyte, maximum),
          maximumMemoryBytes: maximum,
          defaultCacheBytes: 256 * MLXResourceLimits.mebibyte,
          maximumCacheBytes: 256 * MLXResourceLimits.mebibyte
        )
      case .cpu:
        let physicalMaximum = try Self.halfPhysicalMemory(physicalMemoryBytes)
        let maximum = min(20 * MLXResourceLimits.gibibyte, physicalMaximum)
        guard maximum >= MLXResourceLimits.gibibyte else {
          throw MLXResourceLimitError.physicalMemoryTooSmall(physicalMemoryBytes)
        }
        self.init(
          defaultMemoryBytes: min(18 * MLXResourceLimits.gibibyte, maximum),
          maximumMemoryBytes: maximum,
          defaultCacheBytes: 128 * MLXResourceLimits.mebibyte,
          maximumCacheBytes: 128 * MLXResourceLimits.mebibyte
        )
      case .auto:
        throw MLXResourceLimitError.unresolvedEngine(engine)
      }
    }

    init(
      defaultMemoryBytes: Int,
      maximumMemoryBytes: Int,
      defaultCacheBytes: Int,
      maximumCacheBytes: Int
    ) {
      self.defaultMemoryBytes = defaultMemoryBytes
      self.maximumMemoryBytes = maximumMemoryBytes
      self.defaultCacheBytes = defaultCacheBytes
      self.maximumCacheBytes = maximumCacheBytes
    }

    static func halfPhysicalMemory(_ physicalMemoryBytes: UInt64) throws -> Int {
      guard physicalMemoryBytes > 0 else {
        throw MLXResourceLimitError.physicalMemoryUnavailable
      }
      let half = physicalMemoryBytes / 2
      return Int(min(half, UInt64(Int.max)))
    }
  }

  fileprivate static func parseOverride(
    key: String,
    environment: [String: String],
    unitBytes: Int,
    minimumBytes: Int,
    maximumBytes: Int
  ) throws -> Int? {
    guard let rawValue = environment[key] else { return nil }
    guard isPlainDecimal(rawValue),
      let value = Double(rawValue),
      value.isFinite
    else {
      throw MLXResourceLimitError.malformedOverride(key: key, value: rawValue)
    }

    let bytesAsDouble = value * Double(unitBytes)
    guard bytesAsDouble.isFinite,
      bytesAsDouble >= Double(minimumBytes),
      bytesAsDouble <= Double(maximumBytes),
      bytesAsDouble <= Double(Int.max)
    else {
      throw MLXResourceLimitError.overrideOutOfRange(
        key: key,
        value: rawValue,
        minimumBytes: minimumBytes,
        maximumBytes: maximumBytes
      )
    }
    return Int(bytesAsDouble.rounded(.down))
  }

  fileprivate static func isPlainDecimal(_ value: String) -> Bool {
    guard !value.isEmpty else { return false }
    let pieces = value.split(separator: ".", omittingEmptySubsequences: false)
    guard pieces.count == 1 || pieces.count == 2,
      !pieces[0].isEmpty,
      pieces[0].allSatisfy(\.isNumber)
    else { return false }
    if pieces.count == 2 {
      guard !pieces[1].isEmpty, pieces[1].allSatisfy(\.isNumber) else { return false }
    }
    return true
  }
}
