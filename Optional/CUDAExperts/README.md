# Experimental CUDA expert GEMV

This default-off MLX backend candidate compiled on CUDA 13 and ran on an RTX
4090 on September 26, 2026, but **failed exact stock parity in all six eligible
fixtures. Do not enable it for ordinary inference.** All eight fallback fixtures
were exact, the fresh stock repeat matched all fourteen fixtures, and per-case
dispatch checks passed. No timing measurements were made and no installed
runtime was changed. See the [device report](../../benchmark-results/mlx-cuda-20260926/experts/README.md)
for raw outputs, binary/library identities, and numeric diagnostics.

`Patches/mlx-cuda-expert-qmv-fp32.patch` targets the pinned Linux MLX revision
`7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`. It adds a device header and uses
the already compiled `qmv.cu`, so no new CMake source entry is required. The
header here is the identical reviewable source embedded in the patch.
`source-provenance.json` records the original source and candidate hashes.

The existing dispatcher sends eight single-row expert selections to SM80 QMM,
whose minimum tile has sixteen M rows. Merely widening the ordinary QMV cutoff
would change partial accumulation to FP16/BF16. This candidate instead computes
one output row per warp, dequantizes with the existing narrow multiply-then-add
rounding semantics, and retains FP32 products and partial sums. It changes the
reduction order relative to tensor-core QMM. Equal precision is **not** a promise
of bitwise equality; the probe retains exact output as its correctness gate.

Eligibility requires all of:

- Explicit `MIDNIGHT_CUDA_EXPERT_QMV=1` at process start (default off).
- Compute capability 8.9, stock SM80 QMM support, transposed affine Q4/group64.
- FP16 or BF16 inputs, with matching contiguous scales and biases.
- 256 contiguous experts, one token, exactly eight selections.
- Projection `(N,K)` equal to `(512,2048)`, `(1024,2048)`, or `(2048,512)`.

All other requests use the unchanged stock dispatcher. Gather indices still
select input and weight rows separately, including duplicate expert selections.
Array lifetimes and execution remain owned by MLX's command encoder. This does
not implement grouped prefill, batching, speculative verification, or CUDA graph
replay.

## Validation when CUDA is available

Use an isolated checkout. Enable the dependency overlay with
`MODEL_RUNNER_ENABLE_CUDA_EXPERT_QMV_OVERLAY=1` during dependency preparation,
then perform a **clean CUDA build in a fresh scratch build directory**. The
Swift CUDA generator tracks `.cu` timestamps without transitive header changes;
an incremental build after editing `expert_qmv.cuh` can otherwise reuse stale
generated code. The source overlay itself remains off by
default until compiler and device checks pass. The runtime flag independently
selects the arm in a single patched binary.

For correctness on a host whose GPU is shared with a running service, build
`probe.cpp` against the isolated patched shared library:

```sh
cmake -S Optional/CUDAExperts -B /tmp/cuda-expert-probe-build \
  -DMLX_SOURCE=/path/to/isolated/mlx \
  -DMLX_LIBRARY=/path/to/isolated/build/libmlx.so
cmake --build /tmp/cuda-expert-probe-build -j2
PYTHONDONTWRITEBYTECODE=1 python3 Optional/CUDAExperts/run_correctness.py \
  --binary /tmp/cuda-expert-probe-build/midnight-cuda-expert-probe \
  --library /path/to/isolated/build/libmlx.so \
  --output /tmp/cuda-expert-correctness-new
```

This path requires only Python's standard library, runs fresh stock/candidate
processes, saves raw outputs and reference diagnostics, validates every fixture's
trace, and checks the linked library and binary identities. It makes no timing
measurements. The probe requires at least 1 GiB free GPU memory before each
fixture and limits the MLX allocator to 768 MiB; the largest fixture's persistent
device arrays occupy about 290 MiB. Coordinate with other GPU work and leave
existing services running. Exact output mismatches block promotion.
On a parity failure, the runner makes one fresh stock repeat after checking
free memory again; this distinguishes candidate differences from baseline
nondeterminism without relaxing the exact gate.
When the linked core also includes the experimental replay overlay, pass
`-DMIDNIGHT_MLX_CUDA_REPLAY=ON` to this probe's CMake configuration so its MLX
inline definitions match the library.
The core's runtime QMM compiler also needs the bundled CUTLASS and CCCL headers
under the library's parent `include` directory, as provided by an isolated MLX
installation. Building the library alone does not populate that runtime layout.
The runner places its JIT cache inside the new result directory.

For a CUDA Python MLX build made from that same patched core (plus NumPy), run:

```sh
python3 Optional/CUDAExperts/probe.py --output /tmp/cuda-experts-new
```

The output directory must be new. The harness runs separate fresh processes in
ABBA order, excluding warmup, for six eligible FP16/BF16 projections and eight
fallback cases. Flushed per-fixture trace markers require one matching-shape
specialized dispatch for every eligible case and zero for every fallback, so
misrouted cases cannot cancel out in an aggregate counter. A stale binary or
ineligible GPU cannot masquerade as a passing candidate. Timing runs disable
tracing. Packed codes, signed inputs and metadata
are deterministic; the probe includes repeated indices and distinct input rows.

Each process writes all output arrays, hashes, timings and independent FP64
reference errors using separately rounded dequantized weights. Exact stock
output is mandatory for every fixture, including fallbacks. A closer reference
error never waives a parity failure. Reported milliseconds include Python and
synchronization overhead; they are primitive diagnostics, not model throughput.
The harness records the loaded Python extension hash; retain the corresponding
core library, binary and build manifests as well before claiming reproducibility.

Even a passing probe requires real Laguna fixed-prefix/logit tests, greedy-text
comparisons, long-context and cancellation checks, and quiet alternating decode
benchmarks before any production promotion. No tolerances were loosened and no
GPU speedup is claimed here.

The host-side gate checks can run without CUDA:

```sh
python3 -m unittest discover -s Optional/CUDAExperts -p 'test_*.py'
```

These verify report rejection and exact-output enforcement only. They do not
validate the CUDA kernel. Local patch application/reversal was checked against
the exact pinned sources in a disposable directory, without editing the shared
dependency checkout.
