import Foundation
import MLX
import ModelRunnerProtocol

/// Applies process-wide MLX memory and cache limits and verifies their values.
public enum MLXResourceGuard {
    /// Resolves backend limits with a fresh CUDA device-memory query before loading.
    public static func resolve(
        for engine: ModelEngine,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory,
        recommendedWorkingSetBytes: Int? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> MLXResourceLimits {
        try MLXResourceLimits.resolve(
            for: engine, physicalMemoryBytes: physicalMemoryBytes,
            recommendedWorkingSetBytes: recommendedWorkingSetBytes,
            cudaMemory: engine == .cuda ? CUDAMemoryRuntime.snapshot() : nil,
            environment: environment)
    }

    /// Installs process-wide byte limits and throws if MLX reports different values.
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
