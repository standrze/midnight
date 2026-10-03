import Foundation

/// Available memory on the CUDA device selected by MLX (logical device zero).
public struct CUDAMemorySnapshot: Equatable, Sendable {
    public let totalBytes: Int
    public let freeBytes: Int
    public let supportsLayerOffload: Bool

    public init(totalBytes: Int, freeBytes: Int, supportsLayerOffload: Bool) {
        self.totalBytes = totalBytes
        self.freeBytes = freeBytes
        self.supportsLayerOffload = supportsLayerOffload
    }

    /// Headroom for CUDA libraries, graphs, and allocations outside MLX.
    public var reserveBytes: Int { max(512 * 1_048_576, totalBytes / 20) }

    /// A fresh capacity ceiling; other processes' allocations reduce this value.
    public var usableBytes: Int { max(0, freeBytes - reserveBytes) }
}
