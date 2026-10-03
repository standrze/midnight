# Ministral 3 14B activation-weighted ScaleSearch — 2026-08-30

## Verdict

Activation-weighted ScaleSearch (AWSS) is a strong quality candidate for this
checkpoint. Against the existing LS2 Q4 model, it reduced exact
`KL(BF16 teacher || Q4 student)` by **45.76%** and improved held-out NLL by
**0.011384** across 30,118 WikiText-2 test tokens. It won the KL comparison on
all 64 chunks and the NLL comparison on 55 of 64 chunks.

AWSS retains affine Q4, group size 64, the same 927 indexed tensor entries, the same
7,597,762,560 indexed tensor bytes, and the same generation kernels. The
position-balanced runtime result was +1.88% for AWSS, but large within-arm
system-state drift makes that difference non-actionable. AWSS adds no new
format or kernel overhead; its empirical throughput effect is inconclusive.

The 698-token authored smoke corpus regressed, so AWSS should remain a measured
candidate until a larger instruction/code/math/tool-use suite also passes.

## System and checkpoints

- MacBook Pro, Apple M5 Max (18 cores), 64 GB unified memory
- macOS 26.6.2; Swift 6.3.3
- Repository commit `c4c47feb3672` plus the working-tree changes described here
- MLX Swift `72f3c3ad8aee`; MLX Swift LM `14414441fa44`
- BF16 source revision: `mistralai/Ministral-3-14B-Instruct-2512-BF16`
  at `3cea74c1ebaf5ce5f5a2553de470e2ceab825142`

| Artifact | Local directory |
| --- | --- |
| BF16 teacher/source | `tmp/models/Ministral-3-14B-Instruct-2512-BF16-3cea74c` |
| Standard Q4 | `tmp/models/Ministral-3-14B-Instruct-2512-MLX-Q4-Standard-3cea74c` |
| ScaleSearch LS2 Q4 | `tmp/models/Ministral-3-14B-Instruct-2512-MLX-Q4-ScaleSearch-LS2-3cea74c` |
| AWSS Q4 | `tmp/models/Ministral-3-14B-Instruct-2512-MLX-Q4-AWSS1-3cea74c` |

LS2 and AWSS have identical `config.json` SHA-256
`5fc4302f96fa357a453bc8f29f9d31a90c6bebff172cce099b71c73625bab1a7`
and identical index SHA-256
`328e5f68df312865774bb536392c6f18a6daae7677f18d5a6d9f0a080df65e0c`.
All tensor names, shards, shapes, dtypes, and payload lengths match. Physical
shard files differ by 28 bytes in aggregate because tensor ordering/data offsets
change the serialized header length; tensor payload byte totals are identical.

## Method

The collector records the actual input-channel second moment `E[x_j^2]` for
every dense Mistral linear, including the post-gating input to `down_proj` and
the final-normalized input to `lm_head`. AWSS starts every 64-value group from
the retained LS2 bytes and accepts a new affine grid only when:

1. the group-normalized diagonal activation-weighted error strictly improves
   on calibration data; and
2. the same error does not worsen on the separate dev corpus.

This is a conversion-time search. It does not add runtime operations or change
the Q4 representation.

The three splits were fingerprinted and had zero identical prepared chunk texts:

| Split | Purpose | Observed/scored tokens |
| --- | --- | ---: |
| MLX-LM calibration v5 | fit activation-weighted grids | 65,536 |
| WikiText-2 raw validation | per-group dev veto | 15,688 |
| WikiText-2 raw test | final BF16-teacher KL/NLL only | 30,118 |

Prepared-corpus SHA-256 values are
`8fe99d17de45a96e2da7b5aab82b08579276a41af01e4ad82f6d2c841f3b158e`
(calibration),
`34fb6e1a39a09adcd1791ce2ea167fbadf9ba87fd0543ecf6f3f8ecebca3d230`
(dev), and
`be40b2e92a40298a2e7c29b1b52844f4c57b096a27250ddcaf4733cc5dcba0e7`
(test).

Calibration metadata: [`calibration-stats-metadata.json`](calibration-stats-metadata.json).
Dev metadata: [`dev-stats-metadata.json`](dev-stats-metadata.json).

## Conversion diagnostics

The reported weighted MSE is a per-group mean-one normalized proxy. It preserves
each group's candidate ordering but is not the absolute global
`E[||x(W-Q)^T||^2]` magnitude.

| Metric | LS2 template | AWSS | Change |
| --- | ---: | ---: | ---: |
| Calibration weighted-MSE proxy | 2.6159993e-7 | 2.2986631e-7 | **-12.1306%** |
| Dev weighted-MSE proxy | 2.6142954e-7 | 2.2961706e-7 | **-12.1687%** |
| Raw weight MSE | 2.5961184e-7 | 2.6183840e-7 | +0.8576% |
| Changed rescored Linear Q4 groups | — | 142,325,354 / 200,540,160 | 70.9710% |
| Rescored groups with at least one calibration-improving proposal vetoed by dev | — | 44,956,830 | 22.4179% |

Full per-module provenance:
[`activation-scale-search-quantization.json`](activation-scale-search-quantization.json).

## Held-out quality

The benchmark computes exact full-vocabulary `KL(BF16 teacher || Q4 student)`
in FP32 against FP32-cached logits produced by the BF16 teacher. It also
computes ground-truth next-token NLL and teacher/student top-1 agreement. All
models used identical token IDs and the same 64 test chunks.

| Model | NLL | Perplexity | BF16-teacher KL | Teacher top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| BF16 teacher | 2.113990 | 8.281222 | 0 | 100% |
| Standard Q4 | 2.142824 | 8.523476 | 0.033313 | 90.27% |
| ScaleSearch LS2 Q4 | 2.136890 | 8.473043 | 0.032580 | 90.52% |
| **AWSS Q4** | **2.125506** | **8.377132** | **0.017672** | **92.90%** |

AWSS versus LS2:

- `KL(teacher || student)`: **45.7571% lower**; paired 50,000-resample chunk
  bootstrap 95% interval `[44.5445%, 46.9702%]`
- Token-weighted NLL: **0.011384 lower** (0.5327%); paired bootstrap 95%
  interval `[0.008441, 0.014287]`
- Top-1 agreement: **+2.387 percentage points**
- Chunk wins: **64/64 KL**, **55/64 NLL**

Raw report: [`teacher-kl-wikitext2.json`](teacher-kl-wikitext2.json).

Bootstrap details: `random.Random(20260830)` sampled the 64 paired chunks with
replacement 50,000 times. Each replicate recomputed token-weighted totals; the
reported interval endpoints are the sorted order statistics at
`int((N - 1) * 0.025)` and `int((N - 1) * 0.975)`.

The smaller authored smoke set remains a caution: LS2 NLL was 2.488547 while
AWSS was 2.514955 across only 698 scored tokens. Raw reports:
[`LS2 smoke`](../ministral3-scalesearch-20260830/quality-scalesearch-q4.json),
[`quality-awss-general-smoke.json`](quality-awss-general-smoke.json) and
[`teacher-kl-general-smoke.json`](teacher-kl-general-smoke.json).

## Runtime

Each process used two warmups and five measured 512-token generations. The
sequence was LS2 A1, AWSS B1, AWSS B2, LS2 A2. Every trial generated exactly
512 tokens, stopped on length, prefilled the same 570-token prompt, used no
prompt-cache tokens, and was internally deterministic.

| Block | Median decode rate |
| --- | ---: |
| LS2 A1 | 58.65 tok/s |
| AWSS B1 | 53.71 tok/s |
| AWSS B2 | 46.58 tok/s |
| LS2 A2 | 41.10 tok/s |

The position-balanced geometric means were 49.10 tok/s for LS2 and 50.02 tok/s
for AWSS (+1.88%). However, LS2's two blocks differed by 29.93% and AWSS's by
13.26%, so system-state drift dominates the apparent difference. No speed
advantage or regression is established.

Raw reports: [`runtime-ls2-a1-512.json`](runtime-ls2-a1-512.json),
[`runtime-awss-b1-512.json`](runtime-awss-b1-512.json),
[`runtime-awss-b2-512.json`](runtime-awss-b2-512.json), and
[`runtime-ls2-a2-512.json`](runtime-ls2-a2-512.json).

## Validation and isolation

- AWSS focused tests: 5/5 passed (layout/dtypes, strict LS2 fallback,
  activation-sensitive change, invalid-stat rejection, dev veto).
- Laguna and DFlash focused Swift tests: 19/19 passed.
- DFlash ScaleSearch conversion/reload XCTest: 1/1 passed.
- `LagunaQ4R8QuantizerTests.sh` passed.
- `GenericScaleSearchQuantizerTests.sh` passed.
- Full release builds passed for the collector, rescorer, teacher-KL evaluator,
  and runtime benchmark.
- No Laguna implementation, quantizer, runtime, or test source was changed.

## Commands originally executed

These commands assume the output paths do not already exist. The collectors
and evaluators also support `--overwrite`; the rescorer intentionally requires
a new destination checkpoint.

```sh
python3 Scripts/prepare-text-benchmark-corpus.py \
  tmp/calibration/mlx-lm-calibration-v5-571fda7.txt \
  tmp/calibration/mlx-lm-calibration-v5-awss.jsonl \
  --samples 128 --target-characters 3000 \
  --id-prefix mlx-cal-v5 --category mlx-lm-calibration

python3 Scripts/prepare-text-benchmark-corpus.py \
  tmp/calibration/wikitext-2-raw-v1-valid.txt \
  tmp/calibration/wikitext-2-raw-dev.jsonl \
  --samples 32 --target-characters 2200 \
  --id-prefix wikitext2-valid --category wikitext-2-raw-validation

python3 Scripts/prepare-text-benchmark-corpus.py \
  tmp/calibration/wikitext-2-raw-v1-test.json \
  tmp/calibration/wikitext-2-raw-eval.jsonl \
  --samples 64 --target-characters 2500 \
  --id-prefix wikitext2-test --category wikitext-2-raw-test

.build/release/model-runner-mistral-activation-stats \
  tmp/models/Ministral-3-14B-Instruct-2512-BF16-3cea74c \
  tmp/calibration/mlx-lm-calibration-v5-awss.jsonl \
  tmp/calibration/ministral3-awss-calibration-stats.safetensors \
  --maximum-total-tokens 65536

.build/release/model-runner-mistral-activation-stats \
  tmp/models/Ministral-3-14B-Instruct-2512-BF16-3cea74c \
  tmp/calibration/wikitext-2-raw-dev.jsonl \
  tmp/calibration/ministral3-awss-dev-stats.safetensors \
  --maximum-total-tokens 16384

.build/release/model-runner-mistral-awss-quantize \
  tmp/models/Ministral-3-14B-Instruct-2512-BF16-3cea74c \
  tmp/models/Ministral-3-14B-Instruct-2512-MLX-Q4-ScaleSearch-LS2-3cea74c \
  tmp/calibration/ministral3-awss-calibration-stats.safetensors \
  tmp/models/Ministral-3-14B-Instruct-2512-MLX-Q4-AWSS1-3cea74c \
  --validation-stats tmp/calibration/ministral3-awss-dev-stats.safetensors

.build/release/model-runner-teacher-kl-bench \
  tmp/models/Ministral-3-14B-Instruct-2512-BF16-3cea74c \
  tmp/calibration/wikitext-2-raw-eval.jsonl \
  benchmark-results/ministral3-awss-20260830/teacher-kl-wikitext2.json \
  --student tmp/models/Ministral-3-14B-Instruct-2512-MLX-Q4-Standard-3cea74c \
  --student tmp/models/Ministral-3-14B-Instruct-2512-MLX-Q4-ScaleSearch-LS2-3cea74c \
  --student tmp/models/Ministral-3-14B-Instruct-2512-MLX-Q4-AWSS1-3cea74c \
  --max-tokens-per-sample 512 \
  --position-chunk-size 16

.build/release/model-runner-quality-bench \
  tmp/models/Ministral-3-14B-Instruct-2512-MLX-Q4-AWSS1-3cea74c \
  Benchmarks/Quality/general-smoke.jsonl \
  benchmark-results/ministral3-awss-20260830/quality-awss-general-smoke.json \
  --max-tokens-per-sample 512

.build/release/model-runner-teacher-kl-bench \
  tmp/models/Ministral-3-14B-Instruct-2512-BF16-3cea74c \
  Benchmarks/Quality/general-smoke.jsonl \
  benchmark-results/ministral3-awss-20260830/teacher-kl-general-smoke.json \
  --student tmp/models/Ministral-3-14B-Instruct-2512-MLX-Q4-AWSS1-3cea74c \
  --max-tokens-per-sample 512 \
  --position-chunk-size 16

.build/release/model-runner-runtime-bench \
  tmp/models/Ministral-3-14B-Instruct-2512-MLX-Q4-ScaleSearch-LS2-3cea74c \
  benchmark-results/ministral3-awss-20260830/runtime-ls2-a1-512.json \
  --engine metal --tokens 512 --warmups 2 --trials 5
.build/release/model-runner-runtime-bench \
  tmp/models/Ministral-3-14B-Instruct-2512-MLX-Q4-AWSS1-3cea74c \
  benchmark-results/ministral3-awss-20260830/runtime-awss-b1-512.json \
  --engine metal --tokens 512 --warmups 2 --trials 5
.build/release/model-runner-runtime-bench \
  tmp/models/Ministral-3-14B-Instruct-2512-MLX-Q4-AWSS1-3cea74c \
  benchmark-results/ministral3-awss-20260830/runtime-awss-b2-512.json \
  --engine metal --tokens 512 --warmups 2 --trials 5
.build/release/model-runner-runtime-bench \
  tmp/models/Ministral-3-14B-Instruct-2512-MLX-Q4-ScaleSearch-LS2-3cea74c \
  benchmark-results/ministral3-awss-20260830/runtime-ls2-a2-512.json \
  --engine metal --tokens 512 --warmups 2 --trials 5
```
