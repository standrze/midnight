# Laguna Q4R8 versus ScaleSearch Q4R8 DFlash screening — 2026-08-29

## Question

Does affine ScaleSearch uniquely cause the deterministic greedy-output mismatch
between Laguna target-only decoding and block-3 DFlash verification on Metal?

## Controlled setup

- Host: 40-GPU-core M5 Max MacBook, 64 GiB unified memory.
- Runtime: native Swift runner, pinned MLX 0.32.2 Metal backend.
- Decode: greedy (`temperature: 0`, `topP: 1`), 512 requested tokens.
- Schedule: one warm-up per mode, then five alternating trials per mode in one
  loaded process.
- DFlash: the same BF16-target Q4R8 ScaleSearch drafter and block size 3 for
  both targets.
- Prompt/config control: the ordinary-Q4R8 view uses the ScaleSearch target's
  config, YaRN metadata, tokenizer, and chat template, so both runs receive the
  same prompt construction and model settings.

The ordinary control is the public MLX affine-Q4/group-64 checkpoint with 39
Q8/group-64 routers. It is a valid conventional-Q4R8 control, but it is not a
bit-identical copy of the ScaleSearch artifact's archived template. Roughly
1.2% of packed router/embedding words differ because the checkpoints were
produced by different conversion runs. The exact template remains
`/home/example/models/Laguna-XS-2.1-MLX-Q4R8-v1` on the currently unavailable
GPU host.

## Results

| Target | Target-only median | DFlash median | DFlash change | Acceptance | Exact output | First divergence |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Conventional Q4R8 control | 92.39 tok/s | 91.66 tok/s | -0.78% | 262/497 (52.7%) | No | UTF-8 byte 264 |
| ScaleSearch Q4R8 LS2 | 97.57 tok/s | 97.93 tok/s | +0.36% | 270/481 (56.1%) | No | UTF-8 byte 109 |

Every target-only trial was identical to the other target-only trials for its
checkpoint. Every DFlash trial was likewise identical to the other DFlash
trials. The mismatch is deterministic across execution modes, not random
sampling or run-to-run instability.

The opt-in first-rejection diagnostic was also identical at the beginning of
both runs: draft token 11119, verifier token 4601, top-two margin 1.125, cache
position 79. That diagnostic describes the first normal rejected proposal; it
does not locate the later target-only-versus-batched numerical divergence.

## Interpretation

The conventional Q4R8 control also fails exact greedy parity, so ScaleSearch is
not the unique cause of the DFlash correctness failure. ScaleSearch did not
reduce acceptance in this test; it raised it by about 3.4 percentage points.
Neither target produced a material DFlash speedup at block size 3.

The result supports the shape-dependent target-kernel hypothesis: ordinary
decode evaluates one row through QMV/QMV-fast, while block-3 verification uses
QMV-wide. A small numerical change can eventually alter an MoE route or token
argmax. ScaleSearch changes where the divergence appears, but this screening
test does not show that it creates the underlying failure.

A definitive ScaleSearch-only attribution still requires repeating the same
two commands with the exact archived standard template. No BF16 inference
control is required.

## Reports

- `standard-q4r8-ls2-metadata-dflash-ab-512x5.json`
- `scalesearch-q4r8-dflash-ab-512x5.json`
- `standard-ls2-metadata-smoke-32x1.json`
- `scalesearch-smoke-32x1.json`
