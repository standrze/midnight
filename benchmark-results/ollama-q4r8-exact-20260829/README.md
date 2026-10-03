# Exact Q4R8 Ollama/Metal comparison — 2026-08-29

Machine: 40-GPU-core M5 Max MacBook, 64 GiB unified memory, AC power. Both
runtimes used Metal. The original measured requests used the same prompt text,
reported 79 prompt tokens, greedy decoding, and 512 generated tokens after a
warm-up. A later token audit found that the final prompt ID was not identical:
native ended with `</think>` while Ollama defaulted to `<think>`.

## Corrected exact-prompt result

The corrected Ollama request explicitly sets `"think": false`. All 79 prompt
IDs then match native exactly. A cache-capacity-matched run generated 128 tokens
so prompt plus generation remained below both runners' first 256-token KV slab.
After two warmups, 12 adjacent measured trials produced:

| Deployment | Median decode | Relative result |
| --- | ---: | ---: |
| Ollama 0.33.1, exact-value Q4R8, identical prompt IDs | **179.57 tok/s** | **+8.11%** |
| Native Swift runner, exact-value Q4R8, identical prompt IDs | **166.09 tok/s** | control |

The full trial values and source/runtime audit are in
[`../ollama-optimization-audit-20260829`](../ollama-optimization-audit-20260829/README.md).
This corrected result confirms that prompt rendering, retained prefix state,
and first-slab KV capacity do not explain Ollama's steady decode lead.

The free-running completions still diverge after the same prompt and first
generated token. A teacher-forced identical output-token trace is required to
separate host submission gaps from different later MoE expert routes.

## Original 512-token deployment result

| Deployment | Measured trials | Median decode | Relative to Ollama stock NVFP4 |
| --- | --- | ---: | ---: |
| Ollama 0.33.1, exact-value abliterated Q4R8 compatibility view | 173.21, 173.04, 173.09 | **173.09 tok/s** | **+12.09%** |
| Native Swift runner, abliterated Q4R8 | 161.16, 161.30, 160.70 | **161.16 tok/s** | **+4.36%** |
| Ollama 0.33.1, stock `laguna-xs-2.1:nvfp4` | 154.68, 154.42, 154.19 | **154.42 tok/s** | control |

On the identical Q4R8 tensor values but the one-token-different rendered
prompts, Ollama's final clean median was 7.40% higher than the native runner's
recovered clean median. Earlier exact-Q4R8 Ollama blocks measured 169.76 and
170.58 tok/s, while clean native blocks measured 158.54 to 160.71 tok/s. The
corrected exact-prompt run above supersedes this block for runtime attribution;
this table remains useful as the original deployment record.

The deployment conclusion is two-part:

1. The project's Q4R8 package is faster than Ollama's stock NVFP4 package on
   this machine. The final same-Ollama-runtime comparison was +12.09%.
2. Ollama currently executes the project's Q4R8 values faster than the native
   Swift runner. The corrected identical-prompt, cache-fair comparison was
   +8.11%.

## Exact-artifact compatibility

Ollama's experimental importer accepted the original 1,397 tensors and
reported that it preserved source quantization, but Ollama 0.33.1 could not
execute the project's native namespace/layout directly. Its Laguna loader
expects the stock `model.*` namespace, a root `lm_head`, router name `mlp.gate`,
and separate gate/up tensors.

The working local compatibility model is:

```text
laguna-xs-abliterated-q4r8-ollama-compat:latest
```

The compatibility builder made no numerical conversion:

- tensor names were rewritten to Ollama's expected Laguna namespace;
- 79 fused gate/up arrays were split at the exact output-row midpoint;
- the imported headers for all 39 Q8 routers were corrected from the global
  INT4 default to their actual affine-INT8 geometry;
- packed weights, affine scales, and affine biases were not requantized.

The post-build audit compared all 599 source tensor layers, all 79 exact
gate/up splits, and all 39 Q8 routers. It verified all **18,821,963,264 tensor
payload bytes** against the source import.

The model loaded as 33.4B parameters, INT4 overall, 19 GB resident, 100% GPU,
and a 262,144-token context limit.

## Additional blocks and thermal boundary

Ollama stock NVFP4 produced these earlier medians:

- 152.18 tok/s in the first matched block;
- 143.85 tok/s in a later thermally affected block;
- 154.42 tok/s after recovery.

Ollama exact-value Q4R8 produced:

- 169.76 tok/s;
- 170.58 tok/s;
- 173.09 tok/s in the final block immediately after recovered stock NVFP4.

The native runner produced clean block medians of 160.71, 158.54, and 161.16
tok/s. One prolonged-session native block fell from 151.78 to 111.26 and 104.92
tok/s; it was classified as a thermal collapse and excluded from the clean
comparison. After a brief cooldown, the native median recovered to 161.16
tok/s.

## Evidence boundary

This establishes inference throughput, tensor preservation, loading, and
completion. It does not establish that Q4R8 has better end-task quality than
stock NVFP4 because the compared checkpoints do not share the same BF16 source:
the Q4R8 model is the abliterated merge and Ollama's NVFP4 model is stock
Laguna. A quality conclusion requires quantizing the same BF16 checkpoint into
both formats and running held-out retrieval, coding, instruction, and
perplexity evaluations.

The corrected throughput run establishes identical prompt IDs, but not an
identical free-running output stream. Its 8.11% result is therefore a rigorous
same-prompt deployment comparison, not yet a teacher-forced same-token-path
kernel comparison.

The compatibility and audit utilities used for this local experiment are in
`Scripts/build-ollama-q4r8-compat.py` and
`Scripts/audit-ollama-q4r8-compat.py`.
