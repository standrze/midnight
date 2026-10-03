# Swift custom CUDA kernel binding

`mlx-swift-cuda-kernel.patch` exposes the CUDA custom-kernel interface already
present in both Midnight MLX-C pins. It adds `Source/MLX/MLXFastCUDAKernel.swift`
and excludes that file from Linux `SPM_CUDA=0` builds, which already exclude
`MLXFast.swift`, `MLXFastKernel.swift`, and MLX-C's fast-operation implementation.
It changes no existing Metal API or model execution path.

Supported dependency revisions:

| Platform | MLX Swift | MLX-C |
| --- | --- | --- |
| Darwin | `72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798` | `c74db5307cc8ce122f48d97ef951b30578674e7f` |
| Linux CUDA | `2d2724006b62855c6c2a71df633baf4ee4ad8a0f` | `fba4470b89073180056c9ea46c443051375f7399` |

Apply to the MLX Swift checkout with the ordinary dependency patch helper after
verifying its revision. No C header generation, MLX-C patch, or linker change is
required. The exported `mlx_fast_cuda_kernel_*` and `mlx_cuda_is_available`
signatures match on these pins.

## Usage

```swift
guard MLXFast.isCUDAKernelAvailable else {
    // Run the ordinary MLX implementation.
    return fallback(input)
}

let kernel = try MLXFast.cudaKernel(
    name: "copy_values", inputNames: ["input"], outputNames: ["output"],
    source: """
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < input_shape[0]) output[i] = input[i];
        """)
let output = try kernel(
    [input], grid: (input.size, 1, 1), threadGroup: (128, 1, 1),
    outputShapes: [input.shape], outputDTypes: [input.dtype], stream: .gpu)[0]
try withError { eval(output) }
```

The example assumes a one-dimensional, nonempty input. Grid dimensions count
threads, matching MLX's API, rather than CUDA blocks. MLX rounds the grid up to
whole blocks, so the source must check bounds. `sharedMemory` counts dynamic
shared-memory bytes per block. Template values may be `Bool`, `Int` within
Int32, or `DType`.

CUDA represents every input, including a Swift scalar, as a device pointer. For
example, a scalar input named `bias` is read with `bias[0]` inside the source.

The factory checks both CUDA backend support and a nonzero GPU device count,
reporting unavailable CUDA as a Swift error instead of calling a fatal stub on
Metal or trying to create a GPU stream without a device. A CPU stream is rejected
before creating a CUDA operation. Invalid
argument counts, dimensions, template values, and shared-memory values also throw.
Backend errors during construction are collected with MLX's scoped error handler.
Compilation and execution remain lazy; callers must wrap evaluation in `withError`
if they need recoverable NVRTC or execution errors. The binding cannot establish
that arbitrary kernel source accesses valid memory or produces correct results.

The kernel handle contains immutable construction state and each invocation owns
its own configuration. Stream synchronization remains the caller's responsibility,
as for the existing Metal interface.

## Verification

`Tests/ModelRunnerProtocolTests/MLXCUDAKernelTests.swift` verifies:

- Recoverable CUDA unavailability; on macOS, continued CPU use without changing
  the default CPU device.
- Metadata validation before any backend lookup.
- Existing Metal custom-kernel execution after a rejected CUDA request.
- On CUDA: scalar inputs, dtype/bool/int template arguments, two output dtypes,
  noncontiguous input normalization, output initialization, a partially filled
  final block, and continued kernel use after invalid launch requests.
- On CUDA: CPU-stream rejection and checked integer/count/shape conversions.

CUDA execution tests use explicit XCTest skips when CUDA is unavailable. The
test file is excluded through `MLX_CPU_BACKEND` for the existing CPU-only package
configuration. Run the existing test runner with filter `MLXCUDAKernelTests` on
Metal, and the same test filter on a prepared Linux CUDA build. Passing Metal
tests does not establish CUDA execution correctness or inference speed.

Both pristine Swift pins passed forward, reverse, and repeated patch checks.
The final tests passed on macOS (three passed, two CUDA skips) and on an RTX 4090
with Swift 6.3.3, CUDA 13.0, and the Linux pins above (three passed, one unavailable
backend skip). Linux validation compiled a fresh MLX Swift module including this
binding and linked existing pinned native backend objects, then executed the exact
saved XCTest source. It did not rebuild the entire application. CUDA allocations
were capped at 64 MiB, with at least 1 GiB of free device memory checked first.
See [CUDA validation evidence](../benchmark-results/cuda-kernel-binding-20260926/README.md).

With `CUDA_VISIBLE_DEVICES` empty, availability and factory rejection passed; the
two execution tests skipped. This does not establish a CPU fallback for a
CUDA-built Linux runtime: the pinned dependency can initialize its default GPU
stream or CUDA allocator even for a CPU operation or a memory-limit setter. Such
initialization fails without a visible device. The binding's availability probe
and unavailable error avoid those paths. A dedicated `SPM_CUDA=0` build has its
existing CPU-only behavior and excludes this API entirely.

The first GPU trial used invalid synthetic kernel source that treated a scalar
pointer as a value; the corrected tests use `bias[0]`. Recovery after a rejected
Swift configuration is covered, but recovery after an NVRTC compilation failure
is not claimed. These interface checks establish no inference throughput gain.
