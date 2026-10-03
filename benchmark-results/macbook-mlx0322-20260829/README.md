# MLX 0.32.2 MacBook backend validation

Date: 2026-08-29

## Provenance

- MLX Swift: `72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798`
- MLX core: `1f8e74e3f12f31365464a6867c6579f0e9b29d85` (v0.32.2)
- MLX-C: `c74db5307cc8ce122f48d97ef951b30578674e7f`
- Old control core: `ce45c52505c8158ea48d2a54e8caae05efd86bfe` (v0.31.1)
- Hardware: 40-GPU-core M5 Max MacBook
- Model: `/path/to/models/Laguna-XS-2.1-MLX-Q4R8-ScaleSearch-LS2`

The upgraded release build compiled its static Metal library directly from
the pinned MLX source checkout, including `dot.metal` and `fence.metal`.

## Full-model backend control

One loaded-model process was used per block. Every block had one warm-up and
three measured 512-token greedy generations. Execution order was balanced as
old → new → new → old.

| Backend block | Trial tok/s | Median tok/s |
| --- | --- | ---: |
| MLX 0.31.1, first | 150.78, 151.97, 149.28 | 150.78 |
| MLX 0.32.2, first | 151.99, 149.87, 145.42 | 149.87 |
| MLX 0.32.2, second | 151.85, 149.51, 146.79 | 149.51 |
| MLX 0.31.1, second | 145.85, 133.19, 135.51 | 135.51 |

The final old-core block collapsed as sustained device load accumulated. Heat
is larger than the backend delta in this sequence, so these results support
Q4R8 parity and adoption of the newer kernels, not a precise speedup claim.

## Laguna fusion regression

The MLX 0.32.2 binary alternated five unfused and five fused 512-token trials
in one loaded process:

| Mode | Trial tok/s | Median tok/s |
| --- | --- | ---: |
| Unfused | 125.02, 121.91, 112.35, 110.28, 114.64 | 114.64 |
| Compiled fusion | 135.36, 133.59, 121.28, 118.57, 115.16 | 121.28 |

The thermally stressed median gain was 5.80%, and every generated output
matched exactly. The fusion remains enabled.

## MLX 0.32.2 paired microbenchmarks

The raw output files use 32 warm-ups, queue depth 64, and 31 alternating
rounds. The `speedup` column is the second variant relative to the first.

| Test | Second variant | Total speedup | Decision |
| --- | --- | ---: | --- |
| Laguna dense QMV | NVFP4 g16 vs affine Q4 g64 | 0.939x | Keep Q4R8 |
| Laguna expert gate/up | NVFP4 g16 vs affine Q4 g64 | 0.999x | Tie |
| Laguna expert down | NVFP4 g16 vs affine Q4 g64 | 1.018x | Too small for a format change |
| Mistral SwiGLU | compiled elementwise vs eager | 0.983x | Keep current |
| Mistral gate/up | packed-row vs two QMMs | 0.722x | Reject |
| Mistral QKV | packed-row vs three QMMs | 0.955x | Keep current |
| Laguna router | compiled vs eager | 1.459x | Keep compiled |
| Laguna MoE reduction | compiled vs split | 1.320x | Keep compiled |

All graph comparisons validated matching outputs before timing. These are
shape probes, not substitutes for real-checkpoint Mistral or quality-matched
NVFP4 full-model benchmarks.
