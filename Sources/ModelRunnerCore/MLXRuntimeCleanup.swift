import MLX

/// Synchronize and release MLX stream command encoders before process teardown.
///
/// Current MLX-CUDA requires explicit cleanup while the CUDA driver is still
/// alive. The Linux backport releases both the calling thread's encoders and
/// mlx-swift's cross-thread global encoders. Metal does not need this backport,
/// so the call is a no-op on Apple platforms.
public func clearModelRunnerMLXStreams() {
  #if os(Linux)
    clearStreams()
  #endif
}
