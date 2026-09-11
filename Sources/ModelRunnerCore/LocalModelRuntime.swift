import MLX
import ModelRunnerProtocol

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
    return LocalModelMemorySnapshot(activeBytes: snapshot.activeMemory,
      cachedBytes: snapshot.cacheMemory, peakBytes: snapshot.peakMemory)
  }

  /// Call after admission is closed, every backend has reached `waitUntilIdle`,
  /// and all backend references have been released. This releases unused MLX
  /// allocator buffers while preserving its worker threads and default streams.
  /// Cancellation does not interrupt this teardown barrier.
  public static func releaseUnusedMemory() async {
    #if os(macOS)
      await MLXPinnedRuntime.shared.runCleanup {
        // Chatterbox and optional pinned text execution can use the permanent
        // worker's thread-local streams as well as Swift's shared streams.
        Device.withDefaultDevice(.cpu) { Stream().synchronize() }
        if CompiledMLXBackend.current != .cpu {
          Device.withDefaultDevice(.gpu) { Stream().synchronize() }
        }
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
    if CompiledMLXBackend.current != .cpu { Stream.gpu.synchronize() }
  }
}
