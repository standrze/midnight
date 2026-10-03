# DFlash quantizer validation — RTX 4090

These reports validate the native Swift DFlash quantizer against Poolside's
full 0.5B-parameter checkpoints on the CUDA server. They are single-prompt,
single-sequence measurements, not a general quality or throughput claim.

## Setup

- Target: `Laguna-XS-2.1-MLX-Q4R8-ScaleSearch-LS2`
- GPU: RTX 4090, MLX CUDA, greedy generation
- Workload: the runtime benchmark's deterministic coding prompt, 128 output
  tokens per generation
- Quantizer profile: searched affine Q4 group-64 on 25 large draft matrices;
  standard affine Q8 group-64 on `fc` and five `self_attn.g_proj` gates
- DFlash BF16 source size: 882 MiB
- DFlash Q4R8 output size: 259 MiB

The official `DFlash-INT4` drafter accepted none of the custom MLX Q4R8
target's proposals in this workload (0/1,785). It must not be used with this
target. Poolside's BF16-target drafter was the better match.

## Results

| Pairing | Block | Cache | Acceptance | Median tok/s | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Target only | — | 1,024 MiB | — | 130.81 | Stable 129.12–131.43 control |
| BF16-target drafter, BF16 weights | 16 | 128 MiB | 72/773 (9.3%) | 72.15 | Too much rejected draft work |
| BF16-target drafter, Q4R8 weights | 16 | 128 MiB | 70/803 (8.7%) | 96.95 | Quantization retained most acceptance and reduced draft cost |
| BF16-target drafter, Q4R8 weights | 8 | 512 MiB | 70/387 (18.1%) | 109.42 | Better, still below target-only |
| BF16-target drafter, Q4R8 weights | 4 | 1,024 MiB | 63/191 (33.0%) | 140.43 | Median is 7.36% above matched target-only |

The block-4 median is encouraging but is not deployment-ready. Five of seven
measured trials ran at 139.45–141.50 tok/s; two fell to 30.84–32.68 tok/s even
though all trials emitted identical text and had identical acceptance. The
seven-trial mean was therefore 109.62 tok/s, 15.99% below the target-only mean
of 130.48 tok/s. Increasing the MLX cache from 512 MiB to 1,024 MiB did not
remove those DFlash-specific outliers.

## Decision

Keep the quantized BF16-target DFlash artifact as a research candidate: the
quantizer reduced storage by about 71%, preserved nearly all observed draft
acceptance, and can beat target-only throughput in the fast block-4 trials.
Do not enable it by default until the intermittent DFlash scheduling/JIT
slowdown is diagnosed. Use block size 4, not 16, for further work with this
Q4R8 target.

The canonical comparison reports are
`target-only-cache1024.json`, `q4r8-block4-cache1024.json`, and
`q4r8-block8-cache512-v2.json`. The remaining JSON files preserve earlier
pairing, quantization, block-size, and runtime-finalization diagnostics.
