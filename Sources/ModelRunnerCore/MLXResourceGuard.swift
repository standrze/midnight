import Foundation
import MLX
import ModelRunnerProtocol

public enum MLXResourceGuard {
  public static func apply(_ limits: MLXResourceLimits) throws {
    Memory.memoryLimit = limits.memoryLimitBytes
    Memory.cacheLimit = limits.cacheLimitBytes
    Memory.clearCache()

    let appliedMemoryLimit = Memory.memoryLimit
    let appliedCacheLimit = Memory.cacheLimit
    guard appliedMemoryLimit == limits.memoryLimitBytes,
      appliedCacheLimit == limits.cacheLimitBytes
    else {
      throw MLXResourceGuardError.applicationFailed(
        requestedMemoryBytes: limits.memoryLimitBytes,
        appliedMemoryBytes: appliedMemoryLimit,
        requestedCacheBytes: limits.cacheLimitBytes,
        appliedCacheBytes: appliedCacheLimit
      )
    }
  }
}

enum MLXResourceGuardError: LocalizedError {
  case applicationFailed(
    requestedMemoryBytes: Int,
    appliedMemoryBytes: Int,
    requestedCacheBytes: Int,
    appliedCacheBytes: Int
  )

  var errorDescription: String? {
    switch self {
    case .applicationFailed(
      let requestedMemory,
      let appliedMemory,
      let requestedCache,
      let appliedCache
    ):
      "MLX rejected the resource guard "
        + "(memory requested/applied: \(requestedMemory)/\(appliedMemory), "
        + "cache requested/applied: \(requestedCache)/\(appliedCache)); "
        + "refusing to load the model."
    }
  }
}
