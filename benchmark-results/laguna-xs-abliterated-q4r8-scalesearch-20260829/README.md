# Laguna-XS 2.1 abliterated Q4R8 ScaleSearch — 2026-08-29

This is the completed direct quantization of the abliterated BF16 checkpoint,
not the stock Laguna model and not a template-weight substitution.

- Source: `Laguna-XS-2.1-BF16-merged-qvo-r16-neg075-v4`
- Local artifact: `/path/to/models/Laguna-XS-2.1-Abliterated-Q4R8-ScaleSearch-LS2`
- Remote artifact: `/home/example/models/Laguna-XS-2.1-Abliterated-Q4R8-ScaleSearch-LS2`
- Indexed tensor payload: 18,821,963,264 bytes in four safetensors shards
- Policy: 359 ScaleSearch affine-Q4 linear/switch-linear modules, standard
  affine-Q8 for all 39 routed-expert gate projections, and standard MLX Q4
  handling for the embedding/custom module; zero skipped quantizable modules

## ScaleSearch source audit

The audit sampled 12 real tensors from the exact abliterated BF16 source. All
12 improved at identical stored byte cost.

| Metric | Standard Q4 | ScaleSearch Q4R8 |
| --- | ---: | ---: |
| Aggregate MSE | 0.0000084595791640 | 0.0000071867584998 |
| MSE change | — | -15.0459% |
| Groups selecting a searched scale | — | 4,398,159 / 4,489,216 (97.9717%) |

The result clears the experimental 15% MSE-reduction gate. It establishes a
weight-reconstruction improvement on this sample; it does not by itself prove
an end-task quality increase.

## Mac Metal runtime

Machine: 40-GPU-core M5 Max MacBook, 64 GiB unified memory. Runtime: native
Swift runner, explicit Metal engine, MLX 0.32.2. Generated-token rates exclude
prompt evaluation.

| Test | Median decode rate | Result |
| --- | ---: | --- |
| Q4R8 ScaleSearch, 3x512 after warmup | 159.74 tok/s | all trials completed at 512 tokens |
| Laguna compiled fusion off, 3x256 | 147.89 tok/s | control |
| Laguna compiled fusion on, 3x256 | 162.17 tok/s | +9.66%, outputs matched exactly |

The three 512-token runner trials were 159.79, 159.70, and 159.74 tok/s.

## Ollama runtime reference

Ollama 0.33.1's locally installed `laguna-xs-2.1:nvfp4` produced 151.88,
154.11, and 153.70 tok/s across three 512-token trials (median 153.70 tok/s)
with the same 79-token prompt and greedy settings. The measured runner median
was 3.93% higher in this snapshot.

This is not a strict quantization A/B: Ollama's checkpoint is stock Laguna,
whereas the runner checkpoint is the abliterated merge, and the formats are
NVFP4 versus Q4R8 ScaleSearch. Separate Metal processes can also change thermal
and model-residency conditions. Treat the comparison as a current Mac runtime
reference, not a universal winner claim.

A subsequent exact-value compatibility experiment made the same abliterated
Q4R8 arrays executable in Ollama without requantization. In the final clean
three-way block, Ollama Q4R8 measured 173.09 tok/s, the native Q4R8 runner
measured 161.16 tok/s, and Ollama stock NVFP4 measured 154.42 tok/s. See the
[exact Q4R8 Ollama/Metal comparison](../ollama-q4r8-exact-20260829/README.md).

## Integrity and behavior checks

- All four remote/local shard hashes matched exactly.
- Safetensors offsets, dtype lengths, index membership, and indexed total size
  were checked byte-for-byte: 1,397 tensors, no duplicates, gaps, or trailing
  data.
- The Q4 and Q8 module sets matched the policy and were disjoint; zero modules
  were skipped.
- Non-streaming and OpenAI-compatible SSE streaming requests both passed on the
  local Metal endpoint.
- A 20-case held-out behavioral sanity pass returned non-empty responses for
  every case. It classified 14 as comply and 6 as refusal, versus 17/3 for the
  saved distillation targets. That is not a BF16 regression measurement because
  the saved target text is not fresh BF16 inference. A direct BF16 control on
  the Linux server was blocked by an MLX CPU JIT `float16_t` compiler failure.

Raw evidence is in `scalesearch-source-audit-12.json`, `metal-512x3.json`,
`fusion-ab-256x3.json`, `metal-smoke.json`, and
`ollama-runtime-reference.json`.
