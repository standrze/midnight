# Experimental retained MLX CUDA replay sessions

This is a bounded session and allocation-ownership layer for replaying **MLX's
existing CUDA kernels**. It advances the September 13 scratch recorder into a
reproducible overlay and runnable operator probe. It is not full-model replay,
a second model implementation, or a production speed improvement. It is absent
from Midnight's default package, dependency preparation and installed runtime.

The overlay requires the reviewed Linux MLX core pin
`7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247` with the existing Midnight stream
cleanup overlays. `prepare.py` checks exact hashes of the two touched backend
files. It reads the supplied source tree and writes a new overlay directory;
it never edits that source tree. Compilation requires the explicit definition
`MIDNIGHT_MLX_CUDA_REPLAY=1`. Without it the device hooks compile away.

## Implemented behavior

`cu::ReplaySession` retains the MLX allocation owners of input/output arrays and
temporary workspaces. Aliases count once by allocation identity, using allocator
bytes rather than view sizes. Capture has explicit allocation, byte, graph-node,
chunk and nesting limits. Submitted MLX graphs are copied into ordered children
of one parent graph; their original kernels and addresses are reused.

Sessions bind a caller-supplied model epoch and an ordered list of storage
descriptors: allocation, view address, dtype, shape, strides and allocation size.
Changing either invalidates the executable. The caller must enumerate external
weights, inputs and cache bindings and increment the epoch before any in-place
weight mutation, adapter update or inspection change. This interface does not
detect unannounced writes to existing pointers.

The supported surface is deliberately narrow:

- One creator thread and one MLX CUDA encoder per capture/session; a second
  encoder or thread observed during capture rejects the capture.
- One process-wide capture at a time; nested capture is rejected.
- Only kernel, empty, memset and nested graph nodes. Host callbacks, borrowed
  events, memory-copy nodes, external semaphores and device allocations are
  rejected because their additional resource lifetimes are not owned here.
- **Replay is synchronous.** It flushes queued ordinary encoder work, launches
  the retained graph and drains that stream before returning. Raw CUDA replay
  does not refresh MLX's array completion events. `eval(existing_output)` is not
  a replay completion fence. Read outputs or schedule cross-stream consumers
  only after `replay` returns; do not access a session concurrently.
- Clear, invalidation, recapture and destruction drain GPU work before releasing
  graph objects and retained allocations. Encoder destruction marks borrowing
  sessions unusable after the encoder's own drain. CommandEncoder's layout is
  unchanged.

A callback exception drains ordinary work and invalidates its partial capture.
If that encoder drain itself fails, the entire session is quarantined for the
rest of the process. Its buffers may still be referenced by an unsubmitted MLX
graph; a raw stream wait alone cannot establish safety. Quarantined sessions
cannot clear, recapture or replay. This catastrophic backend-error path retains
memory intentionally; normal cancellation and cleanup release it.

## Local executable checks

```sh
bash Optional/MLXCUDAReplay/test.sh
```

Nine C++ suites execute the same session ownership/state implementation used by
the CUDA adapter with a fake backend. They check alias accounting, budget and
overflow rejection, invalidation, failed capture, thread affinity, release order,
failed-drain retry, and quarantine through destruction. The quarantine test
intentionally retains one small session for process lifetime. Four Python tests
verify source preservation, exact patch application/reversal and rejection of
unknown baselines or existing files. These are **CPU tests, not CUDA correctness
or performance results**.

Local ownership and patch tests passed. On September 26 the isolated CUDA 13
build and actual-MLX probe also passed on an RTX 4090. The probe retained one
graph chunk and 15 allocations; all exact output and lifecycle assertions
passed. The GPU was shared with an existing service, so this was correctness
validation only. See the [build and GPU evidence](../../benchmark-results/mlx-cuda-20260926/replay-build/README.md).

## Isolated CUDA build and probe

Use a complete disposable copy of the prepared Linux MLX source, never the
normal checkout or installed runtime. The output directories must be new.

```sh
MLX_REPLAY_SOURCE=/absolute/path/to/isolated/mlx
python3 Optional/MLXCUDAReplay/prepare.py \
  --mlx-source "$MLX_REPLAY_SOURCE" --output /tmp/midnight-mlx-replay-overlay
patch -d "$MLX_REPLAY_SOURCE" -p1 --batch --forward \
  -i /tmp/midnight-mlx-replay-overlay/overlay.patch
cmake -S Optional/MLXCUDAReplay -B /tmp/midnight-mlx-replay-build \
  -DMIDNIGHT_ENABLE_MLX_CUDA_REPLAY=ON \
  -DMLX_SOURCE="$MLX_REPLAY_SOURCE" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
  -DMLX_CUDA_ARCHITECTURES=89
cmake --build /tmp/midnight-mlx-replay-build --target midnight-mlx-replay-probe -j2
/tmp/midnight-mlx-replay-build/midnight-mlx-replay-probe
```

The CMake target uses MLX's normal CUDA/cuDNN dependencies and requires them to
be available. It is separate from SwiftPM and does not modify package manifests.
The recorded build explicitly selected the prepared runtime's cuDNN frontend
1.16.0 and CUTLASS 4.3.5. Its exact configuration is in the evidence directory.
For operators that invoke NVRTC, MLX also requires CUTLASS and CCCL headers in
its normal runtime include layout; an uninstalled CMake library alone does not
provide that layout. The evidence records the isolated header staging used for
the companion expert-kernel probe.
The probe compares ordinary and replayed MLX normalization/matmul/residual
operators exactly, including stable-address input replacement, interleaved
ordinary evaluation, mutation epochs, changed allocations, cancellation,
recapture, nested/other-encoder rejection and encoder destruction. It emits no
throughput claim and contains no language model or attention override.

## Synthetic performance experiment

`benchmark.cpp` and `run_benchmark.py` compare ordinary MLX evaluation with
retained replay in the same instrumented library. Configure the isolated build
above with `-DBUILD_SHARED_LIBS=ON`, then build and run:

```sh
cmake --build /tmp/midnight-mlx-replay-build --target midnight-mlx-replay-benchmark -j2
python3 Optional/MLXCUDAReplay/run_benchmark.py \
  --binary /tmp/midnight-mlx-replay-build/midnight-mlx-replay-benchmark \
  --library /tmp/midnight-mlx-replay-build/mlx/libmlx.so \
  --output /tmp/midnight-mlx-replay-timings-new
```

When adding this target to an existing build, rerun its CMake configuration
first. The output directory must be new. The runner requires one NVIDIA GPU
with at least 1 GiB free before each process; it leaves existing services alone.

Three workloads cover four width-128 FP32 stages and twenty-four width-896
FP32/FP16 stages. Each stage is RMS normalization, matrix multiplication and
residual addition. Every stage shares the same weights, making this a synthetic
submission-overhead test, not a transformer or memory-bandwidth benchmark.
Both arms include identical input-copy work and synchronize every iteration.
MLX `copy()` can share storage, so the benchmark explicitly allocates and checks
an independent mutable input buffer.

The runner uses three fresh processes per workload, checks exact outputs before
timing and after every block, warms both paths, and alternates five ABBA or BAAB
rounds of 100 samples per block. It preserves individual timings, binary/library
hashes, GPU snapshots and telemetry. The MLX allocator limit is 256 MiB, its
cache limit is 64 MiB, and retained capture storage is capped at 64 MiB.

On September 26, all nine final runs passed. The twenty-four-stage workloads
showed median paired elapsed-time reductions of **47.2% for FP16** and **49.0%
for FP32** on the shared RTX 4090. These compare with ordinary execution in the
same replay-instrumented library, whose capture hooks add overhead even when
inactive. They do not measure production Midnight tokens/s or establish the
benefit against an uninstrumented production library. See the
[full results and limitations](../../benchmark-results/mlx-replay-performance-20260926/README.md).

## Evidence preserved and remaining work

The [September 13 MLX integration proof](../../benchmark-results/mlx-replay-integration-20260913/README.md)
already replayed MLX kernels, including dynamic RoPE and slice indices. Its
synthetic timing is not a model result. The
[dynamic attention investigation](../../benchmark-results/mlx-replay-attention-20260913/README.md)
showed retained cuDNN attention reading changing device lengths, but its original
exact checks at lengths 511 and 2048 remain failed. This overlay neither applies
that diagnostic override nor changes those gates. The earlier
[standalone CUDA implementation](../../../midnight-diagnostics/experiments/CUDAReplay/README.md) is a separate path.

Full-model replay still needs a supported device-length attention input,
persistent KV feedback with bounded copying, integration with the actual model
and fast prefill, parameter/inspection invalidation wired into runtime ownership,
and real CUDA cancellation/cleanup and numerical validation. Retaining buffers
can prevent MLX donation, so capture memory and cache-copy behavior must be
measured before claiming a benefit. Only then should the complete serving path
be timed against the unchanged runtime. No default or deployment change is made.
