# Talkie context extension: feasibility and evidence

**Status: deferred by user decision.** Production remains at 2,048 total tokens.
The [long-context PRD](talkie-long-context-prd.md) records the future requirements,
acceptance gates, and updated weight-memory budget. Current runtime verification
is complete; no further context experiments are scheduled.

Research and source inspection: September 13, 2026. Target: native Talkie on the
Apple M5 Max with 64 GiB unified memory. This document records a feasibility
assessment; it does not advertise a validated longer-context checkpoint.

**Unscaled 4K ran, but failed a retrieval task that passed at 2K. Useful 256K
context remains a separate model and runtime research project.** Changing the
configured token limit alone neither establishes that capability nor makes its
memory requirements fit.

## What the released model establishes

The BF16 source conversion declares `max_position_embeddings: 2048`. The authors'
[inference implementation](https://github.com/talkie-lm/talkie/blob/main/src/talkie/model.py#L132-L173)
defaults to `max_seq_len=4096`, which allocates RoPE cosine and sine buffers. Its
configuration does not declare a training sequence length, and the
[author report](https://talkie-lm.com/introducing-talkie) and
[model card](https://huggingface.co/talkie-lm/talkie-1930-13b-it) do not document a
trained context length or a long-context evaluation. The 4,096-position allocation
supports trying 4K inference; it is not evidence of preserved 4K quality.

Talkie uses 40 full attention heads in every one of its 40 layers, with 128 values
per head. Its rotary base is 1,000,000, with the inverse NeoX rotation and Q/K
normalization after rotation. Midnight implements those equations. A large RoPE
base by itself does not establish a usable long context. The requested 262,144
tokens are 128 times the converted checkpoint's declared 2,048-token context.

## Measured probes

The first Q8 experiment kept the positional equations unchanged and altered only
the experiment copy's `max_position_embeddings` from 2,048 to 4,096. All prompts
were fully prefilled, with no truncation or cache reuse, and all outputs stopped
at EOS before the 256-token limit. The four tasks used Metal, BF16 KV, and
128-token prefill chunks. [Exact outputs and review](../benchmark-results/talkie-20260913/context-extension/result-summary.md)
are preserved with source report checksums.

| Q8 context setting | Correct retrieval answers | Three core decisions in summary | All requested summary details | Peak MLX memory |
| --- | ---: | --- | --- | ---: |
| 2,048 | 3/3 | Fail | Fail | 14.78 GiB |
| 4,096 | 2/3 | Pass | Fail | 16.42 GiB |

At 4K, beginning retrieval returned `shortly` instead of `Copper Lantern`;
middle and end retrieval remained correct. The 2K summary omitted the 137-pound
appropriation, while the 4K summary included it. Both summaries omitted the cedar
detail in the key location, so both fail complete-detail scoring. The baseline
summary already fails; a summary failure at a larger context alone therefore
cannot establish an extension regression. These are four simple tasks over one
repetitive synthetic report, not a representative long-context benchmark.

A separate matched 2K diagnostic tested caches created in their requested format
from the first prefill chunk. Q8 weights, unchanged RoPE, the 1,782 prompt token
IDs, and 32-token chunks were identical across the runs:

| KV format | Beginning-retrieval output | Result | Stored KV bytes for 1,792 allocated tokens |
| --- | --- | --- | ---: |
| BF16 | `Copper Lantern` | Pass | 1,468,006,400 |
| Affine Q2/G64 | `are the words` | Fail | 229,376,000 |
| Affine Q4/G64 | `Copper Lantern` | Pass | 412,876,800 |

The quantized runs verified `uint32` packed values and BF16 metadata across all
40 layers, and their storage matched the formula below. Q2 reduces storage but
fails this task already within the declared context. Its slightly lower NLL on
the repetitive prompt did not preserve correct retrieval. A single Q4-cache
pass does not establish general quality. The diagnostic results are linked in
the [structured summary](../benchmark-results/talkie-20260913/context-extension/result-summary.json).

**No 256K allocation was attempted.** The failed checks did not justify proceeding
to 8K or 16K inference either. The largest tested context setting was 4,096,
with a maximum prompt length of 3,819 tokens. The peak-memory column above
reports MLX allocator peaks, not total process or system memory.

## Why weight Q4 does not solve 256K memory

Weight quantization and KV-cache quantization are independent. The model keeps
keys and values for every previous token in every attention layer. At batch one:

```
KV elements per token = 2 × 40 layers × 40 heads × 128 = 409,600
BF16 KV bytes         = tokens × 409,600 × 2
Affine KV bytes       = tokens × 409,600 × (bits/8 + 4/64)
```

The affine formula assumes group size 64 with one BF16 scale and one BF16 bias
per group. Inspection of the pinned MLX affine implementation confirms that
scales and biases retain the input dtype. Talkie's BF16 projection, normalization,
and gain path preserves BF16 activations. The matched cache diagnostic above
confirmed the predicted BF16 metadata on all 40 layers. A different execution
path must recheck its dtypes before relying on these capacity estimates.

| Total context | BF16 KV | Affine Q8/G64 KV | Affine Q4/G64 KV | Affine Q2/G64 KV |
| --- | ---: | ---: | ---: | ---: |
| 2,048 | 1.5625 GiB | 0.8301 GiB | 0.4395 GiB | 0.2441 GiB |
| 4,096 | 3.125 GiB | 1.6602 GiB | 0.8789 GiB | 0.4883 GiB |
| 8,192 | 6.25 GiB | 3.3203 GiB | 1.7578 GiB | 0.9766 GiB |
| 32,768 | 25 GiB | 13.2813 GiB | 7.0313 GiB | 3.9063 GiB |
| 65,536 | 50 GiB | 26.5625 GiB | 14.0625 GiB | 7.8125 GiB |
| 262,144 | **200 GiB** | **106.25 GiB** | **56.25 GiB** | **31.25 GiB** |

These are cache storage calculations, not measured peak memory. They exclude
weights, cache growth copies, activations, attention workspace, the operating
system, and other applications. The original approximately 7.11 GiB candidate
failed generation. Use the current compact `attn8` artifact instead:
10,533,124,415 weight bytes, or 9.80973655 GiB. With Q4 KV, the sum is about
66.06 GiB before those other costs, exceeding this machine's physical memory.
The Q8 reference occupies 13.14132286 GiB; with Q4 KV its sum is about 69.39 GiB.
The [PRD](talkie-long-context-prd.md) also records Q2 budget scenarios without
treating the failed Q2 quality probe as an accepted solution. The existing resource limit also
budgets at most 80% of physical memory by default, or 51.2 GiB on this machine.
The [reproducible calculations](../benchmark-results/talkie-20260913/context-extension/memory-estimates.json)
include exact byte counts and FP32-metadata alternatives.

Q2 cache storage is small enough to investigate, but it is not a validated
solution: the first matched retrieval test failed. The pinned MLX library
supports 2-bit affine operations; Midnight's
public cache options did not expose them at inspection time. If cache metadata
were promoted to FP32, that Q2 estimate would become 37.5 GiB. Two-bit cache error
also needs separate quality measurement; ScaleSearch on the model's weights
does not calibrate or validate its KV cache. [KIVI](https://arxiv.org/abs/2402.02750)
demonstrates a specifically designed asymmetric 2-bit cache method on other model
families. Its quality results cannot be transferred to generic affine Talkie KV.

## Runtime barriers in the inspected implementation

Two additional issues prevent treating the packed-cache estimate as a runnable
256K configuration:

1. **Initial prefill is uncompressed.** Talkie's `newCache` currently constructs
   ordinary attention caches. The generation path performs `model.prepare`
   before applying the cache quantization policy. A large prompt would therefore
   accumulate its full BF16 cache before compression. A viable long-context path
   must create compatible quantized caches from the beginning and enforce memory
   admission using their actual formats and transient costs.
2. **Quantized attention materializes scores.** The current implementation performs
   a quantized matrix multiplication, then masking, then softmax, then another
   multiplication. For 262,144 cached tokens and a 512-token query chunk, one
   BF16 score tensor alone is `40 × 512 × 262144 × 2` bytes: **10 GiB**. Scores,
   attention weights, and masking may require overlapping allocations. A chunk
   of 32 reduces one such tensor to 0.625 GiB, while increasing the number of
   chunks. Streaming or tiled attention compatible with the quantized cache
   would be the more scalable direction. [FlashAttention](https://arxiv.org/abs/2205.14135)
   provides a primary reference for exact attention without materializing the
   full score matrix.

Lower peak memory does not remove full attention's quadratic prefill work.
At 256K versus 2K, the number of token pairs grows by 16,384 times; that is a
comparison of the attention component, not a measured end-to-end latency ratio.

Source locations inspected: `Sources/ModelRunnerCore/TalkieModel.swift`,
`Sources/ModelRunnerProtocol/MLXResourceLimits.swift`, and the pinned
`mlx-swift-lm` revision `14414441fa44f45eee35a61e9fa0bab577cf9734`:
`Libraries/MLXLMCommon/CacheConfiguration.swift` (cache construction),
`Evaluate.swift` (prefill and policy application), and `KVCache.swift`
(quantized cache and attention). Dtype propagation was checked in the
`mlx-swift` revision `72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798` Cmlx
`mlx/mlx/ops.cpp` and `fast.cpp`. These observations describe the inspected
baseline; later runtime experiments must record their changed implementation.
The separate `model-runner-talkie-context-probe` diagnostic constructs quantized
caches before prefill for the small matched tests above; it does not change
normal production admission or assert a larger supported context.

## What useful 256K would require

Position interpolation and long-context adaptation are established research
directions, but their published success is not a Talkie guarantee.
[Position Interpolation](https://arxiv.org/abs/2306.15595) combines position
rescaling with fine-tuning; [YaRN](https://arxiv.org/abs/2309.00071) develops a
more efficient extension method. [LongRoPE](https://arxiv.org/abs/2402.13753)
uses positional search and progressive extension, including 256K fine-tuning
and short-context readjustment, on LLaMA2 and Mistral.

For Talkie, a credible 256K project needs all of the following:

- A positional extension implemented correctly for Talkie's rotation and
  normalization, evaluated against unchanged short-context behavior.
- Long-context adaptation on suitable documents, with enough training hardware
  and memory for the selected recipe. Historical documents are needed if
  preserving its pre-1931 character is part of the objective.
- An inference path that fits from the first prompt chunk: substantially larger
  memory hardware for BF16/Q8 caches, or validated aggressive KV compression,
  bounded workspace, and appropriate admission checks on this 64 GiB machine.
- Held-out long-document language modeling, retrieval at varied positions,
  multi-fact reasoning and summarization, and short-context regression tests.
  Finishing allocation or one successful retrieval is not sufficient.

Retrieval over a large external document collection or retaining a rolling
summary can be useful product features, but neither is equivalent to the model
attending to a true 256K-token context.

## Practical first milestones

The initial experiment used unchanged positional equations at 2K and 4K with
separate experiment metadata. It did not change the installed model's declared
capability. The measured failure above needs attention before proceeding.
The [paired smoke inputs](../benchmark-results/talkie-20260913/context-extension/README.md)
cover beginning, middle, and end retrieval and a three-fact summary. They use
1,782–1,792 prompt tokens at 2K and 3,809–3,819 at 4K, reserving 256 output tokens.
Matched 8K and 16K fixtures are also prepared at 7,896–7,906 and 16,070–16,080
prompt tokens respectively, with the same output reservation. The larger fixture
names do not assert model support. The generator also creates public JSONL files
for `model-runner-generation-bench`, with expected answers stored separately.
These are four simple tasks over one original synthetic report, with substantial
repeated distractor wording. They are smoke checks, not representative quality
tests; no inference was run while preparing them.

If a subsequent 4K candidate behaves acceptably, compare 8K with the same core tasks and held-out prose,
including native RoPE and an explicitly configured scaling candidate where
appropriate. Record loss, recall, summary completeness, repetition, peak memory,
prefill time, and decode speed. Preserve the 2K regression baseline. A 16K stage
can use the prepared fixtures after 8K passes; progress through larger lengths
only after each stage demonstrates useful behavior and
fits the measured memory budget. No stage in this plan implies 256K is currently
supported.
