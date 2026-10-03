# TensorOps execution experiment — 2 October 2026

The affine Q4 cooperative-operand prototype works on this M5 Max running macOS
26.6.2. It removes the explicit decoded-weight threadgroup-memory round trip
without changing the stored quantization grid. It is an isolated runtime probe;
it does not replace Midnight's MLX dispatch.

## What was built

`Tests/Fixtures/TensorOpsQuantizedMatmul/affine-q4.metal` implements two matched
single-simdgroup kernels. Both use a 16×32×16 TensorOps multiply-accumulate,
FP16/BF16 operands, float accumulation, identical K order and output conversion.
The control decodes weights into 512 threadgroup elements and loads the right
operand. The candidate decodes directly into the right cooperative operand.
It queries operand coordinates rather than assuming a vendor lane map.

Weights are `[N,K/8]` packed uint32, with low-to-high consecutive nibbles;
metadata is `[N,K/group_size]`. Each operand value is
`Value(float(scale) * code + float(bias))`. No scale search, calibration,
activation quantization, checkpoint conversion or weight re-quantization occurs.
The supported probe inputs are contiguous, transposed-right affine Q4 matrices
with K divisible by the group size and groups of 32, 64 or 128. Batched/gathered
MoE layouts, other bit widths and float32 inputs are outside this probe.

The native Swift harness compiles Metal source through the runtime compiler;
the optional downloadable offline Metal toolchain is unnecessary. Its compiler
uses Metal 4.0, safe math and precise float functions. Swift formatting and lint
checks passed, and the shell launcher passed `bash -n`.

## Measured result

The authoritative run is
[run-v3/report.json](../benchmark-results/tensorops-quantized-matmul-20261002/run-v3/report.json).
Its adjacent `sha256.txt`, `toolchain.txt` and `run.log` record source/executable
provenance and Xcode 27.0 / SDK 27.0. Earlier directories are exploratory runs;
run-v1's sources were edited before the final report and its original hashes
must not be used as provenance for that report.

All 12 correctness fixtures passed: 7,008 scalar-reference entries across FP16
and BF16, including M/N tails, group boundaries, every Q4 code, zero and negative
scales, nonzero biases and K=2816. The reference reconstructs weights independently
on CPU, rounds operands to their stored precision and accumulates in double.
The report states the numerical tolerance. Candidate and control outputs were
bitwise identical; output guards remained intact. These are synthetic fixtures,
not accuracy results for a language model.

The 14 benchmark cases also passed complete candidate/control bitwise comparison
and sampled scalar-reference checks. Each case has three warmups and 20 paired
AB/BA trials with five dispatches per command. GPU durations exclude compilation
and allocation. Wall durations cover commit through completion, not a serving
request. Median ratios compare only these two kernels.

| Shape, group 64 | FP16 speedup vs staging | BF16 speedup vs staging |
| --- | ---: | ---: |
| M64, N2048, K2816 | 1.25× | 1.31× |
| M512, N2048, K2816 | 1.16× | 1.15× |

Across all cases the median ratios range from 1.03× to 1.43×. Smaller cases show
substantial timing variability; order-separated paired ratios are also recorded.
These results justify investigating direct operand filling. They do **not** show
a speedup over stock or locally patched MLX, and they do not establish faster
prefill, decode or time to first token.

## MXFP4 capability boundary

[Apple's TensorOps presentation](https://developer.apple.com/videos/play/wwdc2026/330/)
places the new MX scale-plane formats and E8M0 support in macOS 27. The SDK is
27, but the installed OS is 26.6.2. The harness records
`unsupported_os_requires_27`; `mxfp4-direct.metal` is explicitly an uncompiled
scaffold and is never selected by this probe.

The existing OS 26 operand-construction API is sufficient for the affine
experiment. This should not be confused with the newer ability to reuse arbitrary
cooperative results as subsequent operation inputs. Both the local MLX NAX
implementation and this native execution demonstrate the older operand path.

Before testing the native MX path, verify actual SDK/runtime/device support,
packed-nibble ordering, logical versus byte strides, scale-plane layout and
alignment, positive/negative zero, scale exponent range and tail behavior. Test
mixed FP16/FP4 operands rather than assuming that an FP4 input type guarantees
hardware acceleration. NVFP4's group-16 FP8/global scales are not interchangeable
with MXFP4's group-32 E8M0 scales.

## Reproduction and next gate

```bash
Scripts/benchmark-tensorops-quantized-matmul.sh /private/tmp/tensorops-q4-new-run --benchmark
```

The output path must be new. Native GPU access is required; a sandbox without
Metal access will fail clearly. The script does not load models or start servers.

The next implementation gate is an opt-in MLX NAX loader variant with MLX's
existing 32/64×64×64 tiling, reuse and split-K behavior, compared against the
actual current dependency kernels on real layer shapes. The dependency pin is
`mlx-swift` `72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798`; local patches must also be
recorded. Inspect `quantized_nax.h`, `fp_quantized_nax.h`, `steel/gemm/nax.h` and
`quantized.cpp` when integrating. Keep the existing matrix-vector decode path
unless that path independently loses a controlled benchmark.

Promotion additionally requires fixed-token logits/KL, generation quality,
long-context and hot/cold cache checks, followed by end-to-end AB/BA measurements.
The first target is affine Q4 prefill: it can use existing checkpoints and the
installed OS. [Disaggregated quantization](disaggregated-quantization-20261002.md)
is a separate format/quality experiment built on top of a validated execution path.
