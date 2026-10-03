# Ministral 3 14B ScaleSearch benchmark — 2026-08-30

## Verdict

The repository now has a retained, loadable ScaleSearch-quantized Mistral-family
checkpoint. The candidate passes the representative 15% reconstruction-MSE
gate, improves the fixed teacher-forced smoke corpus over a matched ordinary-Q4
control, and has indistinguishable storage geometry and fixed-token runtime.
It remains an experimental candidate pending a broader external quality suite.

## Checkpoints

All three artifacts derive from the official
`mistralai/Ministral-3-14B-Instruct-2512-BF16` checkpoint at immutable revision
`3cea74c1ebaf5ce5f5a2553de470e2ceab825142`:

| Artifact | Local directory | Size |
| --- | --- | ---: |
| Verified BF16 source | `tmp/models/Ministral-3-14B-Instruct-2512-BF16-3cea74c` | 26 GiB |
| Matched standard Q4 | `tmp/models/Ministral-3-14B-Instruct-2512-MLX-Q4-Standard-3cea74c` | 7.1 GiB |
| ScaleSearch LS2 Q4 | `tmp/models/Ministral-3-14B-Instruct-2512-MLX-Q4-ScaleSearch-LS2-3cea74c` | 7.1 GiB |

The source manifest SHA-256 is identical in all three directories:
`58f1b623481f3bb5f34b84cbc74305c2890c4ed6330ff6c4e0ca71cb39595308`.
Its six BF16 weight hashes match the pinned upstream manifest.

The Q4 artifacts use identical affine 4-bit, group-size-64 configuration,
sharding, index, tensor schema, and inference kernels. Their `config.json`
SHA-256 is identically
`5fc4302f96fa357a453bc8f29f9d31a90c6bebff172cce099b71c73625bab1a7`;
their 927-entry safetensors index SHA-256 is identically
`328e5f68df312865774bb536392c6f18a6daae7677f18d5a6d9f0a080df65e0c`.
The standard control has 282 ordinary-Q4 modules. The candidate has 281
ScaleSearch-Q4 linear modules and one ordinary-Q4 embedding. Their distinct
weight-shard hashes prove that the candidate is not a renamed control.

## Results

### Weight reconstruction

The expanded audit samples 22 representative tensors across the LM head and
attention/MLP projections in layers 0, 19, and 39: 1,583,349,760 weights total.

| Metric | Standard Q4 | ScaleSearch Q4 | Change |
| --- | ---: | ---: | ---: |
| Aggregate MSE | 2.5809896e-7 | 2.1927151e-7 | -15.0436% |
| Improved tensors | — | 22 / 22 | pass |
| Groups changed | — | 24,289,291 / 24,739,840 | 98.1789% |
| Stored bytes per tensor | matched | matched | 0 |

Raw report: [`weight-reconstruction-audit-expanded.json`](weight-reconstruction-audit-expanded.json).

### Teacher-forced quality smoke gate

All runs used FP32 logits for next-token cross-entropy, the same checkpoint
tokenizer, the same 10 authored samples, and the same 698 scored tokens.
Both fingerprints match across all checkpoints:

- Corpus: `fnv1a64:51d787ab821904d5`
- Evaluated token IDs: `fnv1a64:a4e0c038add8445c`

| Checkpoint | Token-weighted NLL | Perplexity | Peak MLX memory |
| --- | ---: | ---: | ---: |
| BF16 source | 2.509220 | 12.295333 | 27.09 GB |
| Standard Q4 | 2.497198 | 12.148408 | 7.81 GB |
| ScaleSearch Q4 | **2.488547** | **12.043762** | 7.81 GB |

Against the matched Q4 control, ScaleSearch lowers NLL by 0.008651 (0.3464%)
and perplexity by 0.8614%, improving 6 of 10 individual samples. Three runs
per Q4 artifact (the primary plus two repeats) reproduced the NLL values
exactly. The corpus is a small comparative smoke gate; its result does not
establish that Q4 is broadly better than BF16.

Raw reports: [`quality-bf16.json`](quality-bf16.json),
[`quality-standard-q4.json`](quality-standard-q4.json), and
[`quality-scalesearch-q4.json`](quality-scalesearch-q4.json).

### Runtime

The identical-token quality workload also serves as a path-matched timing
cross-check. Across three runs, median end-to-end time was 1.43845 seconds for
standard Q4 and 1.43541 seconds for ScaleSearch Q4, a 0.21% candidate advantage
that is below a meaningful noise threshold. Peak memory was the same.

Free-running 512-token generation was measured in ABBA blocks. Block medians
were 67.16 and 50.17 tok/s for standard Q4, versus 66.16 and 56.65 tok/s for
ScaleSearch. The large 67-to-49 tok/s sequence-wide performance/system-state
drift, consistent with thermal throttling, dominates the comparison, so these
data support no throughput claim in either direction.
ScaleSearch changes calibration, not storage layout or inference kernels.

Raw cold-runtime reports: [`runtime-standard-a1.json`](runtime-standard-a1.json),
[`runtime-scalesearch-b1.json`](runtime-scalesearch-b1.json),
[`runtime-scalesearch-b2.json`](runtime-scalesearch-b2.json), and
[`runtime-standard-a2.json`](runtime-standard-a2.json).

### Mistral hot conversation cache

The finalized five-trial paired probe used 256-token seed and continuation
generations. Every hot continuation reused the full 826-token seed ledger and
prefilled only the 24-token suffix; every cold control prefilled all 850 tokens
and reused zero.

| Mode | Median TTFT | Reused / prefilled prompt tokens |
| --- | ---: | ---: |
| Forced cold | 502.04 ms | 0 / 850 |
| Hot append-only cache | **83.67 ms** | **826 / 24** |

TTFT fell by 83.33%. Seed outputs were identical across all probes, and hot and
cold outputs were each internally deterministic. Hot versus cold continuations
were not bit-identical; the first UTF-8 divergence was byte 104. This is the
expected sensitivity of greedy generation to different valid numerical
schedules (tokenwise cached QMV/attention versus batched cold QMM/attention),
not a token-ledger mismatch. A deterministic synthetic affine-Q4 Mistral 3 test
measured about 0.00038 maximum final-logit drift on the pinned M5 Max run
against a 0.005 limit and verified all cache offsets and state lengths exactly.

Raw report: [`runtime-scalesearch-hot-cache-256.json`](runtime-scalesearch-hot-cache-256.json).

## Validation and scope

- Full `swift test`: 118 Swift Testing cases plus 2 conversion XCTest cases passed.
- Seven focused quantizer, dependency-patch, cache, and Laguna shell contracts passed.
- Full release build passed.
- `LagunaModel.swift` and `LagunaDFlashModel.swift` have zero diff; Laguna's
  prompt-cache, DFlash, quantizer, and runtime paths remain in place.
- This dense Mistral 3 checkpoint exercises ScaleSearch and hot conversation
  reuse. It does not exercise the separate classic-Mistral sliding-attention
  patch or Mixtral fused-router specialization; those still need representative
  full-checkpoint performance A/B runs before throughput claims.
