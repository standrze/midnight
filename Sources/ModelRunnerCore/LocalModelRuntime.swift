import MLX
import ModelRunnerProtocol

/// MLX allocator memory in bytes, excluding other process allocations.
public struct LocalModelMemorySnapshot: Codable, Equatable, Sendable {
    public let activeBytes: Int
    public let cachedBytes: Int
    public let peakBytes: Int
}

/// Process-wide MLX maintenance used between resident model lifetimes.
public enum LocalModelRuntime {
    /// MLX allocator measurements, which exclude other process/runtime memory.
    public static func memorySnapshot() -> LocalModelMemorySnapshot {
        let snapshot = Memory.snapshot()
        return LocalModelMemorySnapshot(
            activeBytes: snapshot.activeMemory,
            cachedBytes: snapshot.cacheMemory, peakBytes: snapshot.peakMemory)
    }

    /// Call after admission is closed, every backend has reached `waitUntilIdle`,
    /// and all backend references have been released. This releases unused MLX
    /// allocator buffers while preserving its worker threads and default streams.
    /// Cancellation does not interrupt this teardown barrier.
    public static func releaseUnusedMemory() async {
        #if os(macOS)
            if let runtime = MLXPinnedRuntime.existing {
                await runtime.runCleanup {
                    // Optional pinned text execution can use this worker's thread-local
                    // streams. Only visit it if a model used it.
                    Device.withDefaultDevice(.cpu) { Stream().synchronize() }
                    if CompiledMLXBackend.current != .cpu {
                        Device.withDefaultDevice(.gpu) { Stream().synchronize() }
                    }
                    synchronizeSharedStreams()
                    Memory.clearCache()
                }
            } else {
                synchronizeSharedStreams()
                Memory.clearCache()
            }
        #else
            synchronizeSharedStreams()
            Memory.clearCache()
        #endif
    }

    private static func synchronizeSharedStreams() {
        Stream.cpu.synchronize()
        if CompiledMLXBackend.current != .cpu {
            Stream.gpu.synchronize()
        }
    }
}
