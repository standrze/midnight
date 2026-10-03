# Laguna compiled block tail A/B (2026-08-30)

## Scope

This experiment starts from the clean incremental ByteLevel candidate and
adds a guarded, decode-only compiled graph for the portion of each Laguna
decoder block after scaled-dot-product attention. The accepted production
branch and the incremental milestone remain unchanged.

The compiled path is eligible only for batch-one, one-token cached decode,
with no hidden-state capture and with the already-correct compiled attention
gate and MoE fusion enabled. Prefill, multi-token input, DFlash target-state
capture, missing caches, and offset-zero calls retain the existing path.

- Source commit: `b103114`
- Model: `Laguna-XS-2.1-Abliterated-Q4R8-ScaleSearch-LS2`
- Engine: Metal
- Prompt: 79 uncached tokens
- Generation: greedy, 128 requested tokens
- Warmups: two per mode
- Measured trials: six per mode, alternating `AB` / `BA`
- Release benchmark binary SHA-256:
  `2180fc576de2ba066b7d471d4d01fc25a58c5a04d5d1141b75ff6e138198217f`
- Metal library SHA-256:
  `903daf038bc9e65c6b77ccb3dc023df6435cf50d4d2dc78ed950a711f68be48c`
- Historical Ollama Q4R8 ScaleSearch median: `179.5662 tok/s`

## Result

| Mode | Median decode (tok/s) |
| --- | ---: |
| Existing eager block tail | `175.853778336442` |
| Compiled block tail | `177.499062770142` |

The same-loaded compiled arm is `+0.935598000375%` faster. It remains
`2.067137229858 tok/s` (`1.151183925404%`) below the historical Ollama
reference.

The first compiled warmup paid the expected one-time graph compilation cost
(`367.80 ms` TTFT). The second compiled warmup returned to `77.93 ms`, matching
the steady-state eager path, so compilation was complete before measurement.

## Correctness and evidence integrity

The focused GPU-backed `LagunaModelTests` suite passed all nine tests. New
coverage verifies the eligibility matrix, cached single-token eager/compiled
logit equivalence, KV-cache offset progression, and multi-token fallback.

All 12 measured generations produced exactly 128 tokens, stopped for `length`,
and emitted byte-identical 742-byte content. The report records
`outputs_match_exactly: true`.

- Raw report SHA-256:
  `1f56df17ae8e8a49b757be0da3418d73f065840c53c0ee979cda49dc9d1636f7`
- Aggregate measured-output SHA-256 from
  `jq -r '.trials[].content' ... | shasum -a 256`:
  `3e30da17e2ee690531192c9b06dfc827c1fc5276334a24358cc177374ab87d57`

## Decision

Preserve the candidate because it is correct and directionally useful, but do
not promote it. Its `+0.935598%` effect misses the predeclared `+1%` candidate
gate and does not beat Ollama. The next experiment should target the remaining
attention-side launch and graph-construction overhead while retaining this
tail compilation as an independently restorable milestone.
