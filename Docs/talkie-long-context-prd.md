# PRD: Talkie long-context support

**Status: Deferred by user decision — September 13, 2026.**

Keep Talkie's production context at **2,048 total tokens** and make the current
runtime reliable. Preserve the research and revisit context extension later.
This document schedules no additional experiments, training, or hardware work.
The future research target is **true 262,144-token context**, subject to quality
and resource feasibility; it is not a delivery commitment.

## Product goal and current scope

A future extended Talkie should use long historical documents and conversations
while preserving its useful short-context behavior. The total budget includes
chat formatting, previous messages, the current prompt, and generated output.
True context support must retain the supplied tokens and their accessibility to
attention. Silently truncating input, rotating away old tokens, retrieving a few
passages, or replacing history with a summary does not meet the 256K requirement.

Current production work remains limited to reliable operation at 2K: correct
model loading and tokenizer framing, normal and streamed chat, proper EOS stops,
multi-turn input, clear rejection of oversized requests, cancellation, and a
successful subsequent request. Context defaults, admission rules, and installed
model declarations must remain at 2,048. Weight-quantization choices are tracked
separately in [Talkie support](talkie.md) and the
[ScaleSearch evidence](../benchmark-results/talkie-20260913/scalesearch/README.md).

Current verification is complete: the compact `attn8` model is available in the
model list and passed four generation checks plus all seven installed-runtime
HTTP checks. The test server was local to this machine and was stopped afterward.
The Q8 checkpoint remains the higher-fidelity reference. See the
[HTTP evidence](../benchmark-results/talkie-20260913/scalesearch/http-installed-attn8/evidence.json).

## Evidence that led to deferral

The converted checkpoint declares 2,048 positions. The
[author implementation](https://github.com/talkie-lm/talkie/blob/main/src/talkie/model.py#L132-L173)
allocates 4,096-position RoPE buffers by default, but neither the
[report](https://talkie-lm.com/introducing-talkie) nor the
[model card](https://huggingface.co/talkie-lm/talkie-1930-13b-it) documents a trained
context length or validated long-context quality. An allocation limit is not a
capability claim.

The completed experiments used the same Q8 checkpoint and unchanged positional
equations. Only the experiment copy's declared context changed between 2K and
4K. All prompts were fully prefilled without truncation, and every response
stopped at EOS before its output budget.

| Experiment | Observed result |
| --- | --- |
| Q8 weights, BF16 KV, 2K, four synthetic tasks | 3/3 retrieval answers correct; summary omitted the 137-pound appropriation and cedar detail. |
| Q8 weights, BF16 KV, 4K, same tasks with more distractors | 2/3 retrieval answers correct; beginning answer was `shortly` instead of `Copper Lantern`. Summary covered all core decisions but omitted cedar. |
| Matched 2K beginning retrieval, BF16 KV | Correct: `Copper Lantern`. |
| Same 1,782 token IDs, affine Q2/G64 KV, 32-token chunks | Incorrect: `are the words`. |
| Same input and chunks, affine Q4/G64 KV | Correct: `Copper Lantern`; one successful task is not broad validation. |

The 2K summary already fails, so a summary failure at a larger context cannot
alone establish an extension regression. The 4K beginning-retrieval failure is
a regression on that matched task. These are simple questions about one
repetitive synthetic report; they do not estimate general performance.
The [reviewed results](../benchmark-results/talkie-20260913/context-extension/result-summary.md)
preserve exact outputs, explicit rubrics, report checksums, and limitations.

Quantized caches actually used BF16 scale/bias metadata and `uint32` packed
values across all 40 layers. At 1,792 allocated tokens, BF16, Q2, and Q4 caches
stored 1,468,006,400, 229,376,000, and 412,876,800 bytes respectively, matching
the analytical formula. Q2's slightly lower loss on the repetitive prompt did
not preserve correct retrieval. **No 8K/16K inference or 256K allocation was
attempted.** The largest tested context setting was 4,096, with 3,819 prompt
tokens at most. Prepared longer fixtures remain unused.

## Resource feasibility on the current 64 GiB machine

Talkie has 40 layers, 40 full attention heads per layer, and head dimension 128.
Keys and values therefore require 409,600 elements per token. At 262,144 tokens,
BF16 KV alone occupies **200 GiB**. Affine group-64 caches with BF16 metadata
require **106.25 GiB at Q8**, **56.25 GiB at Q4**, or **31.25 GiB at Q2**.

Use the current compact attention-protected weight candidate for planning,
not the failed approximately 7.11 GiB candidate:

| Weight artifact | Exact weight bytes | Weight GiB | With 256K Q4 KV | With 256K Q2 KV |
| --- | ---: | ---: | ---: | ---: |
| Compact `attn8` candidate | 10,533,124,415 | 9.80973655 | 66.0597 GiB | 41.0597 GiB |
| Q8/G64 fidelity reference | 14,110,387,976 | 13.14132286 | 69.3913 GiB | 44.3913 GiB |

These sums exclude activations, attention workspace, cache growth copies,
allocator overhead, runtime state, and the operating system. Q4 KV already
exceeds physical memory with either weight artifact. Q2 could fit its stored
tensors, but it failed the first quality probe and is not an accepted solution.
The normal default MLX budget is at most 51.2 GiB on this machine; after Q2 KV
and weights it leaves only about 10.14 GiB for the compact candidate or 6.81 GiB
for Q8 within that budget. Those residuals are not established usable headroom.
Any different dtype, group size, concurrent load, or cache scheme must be
recalculated and measured.

Production prefill presently starts with ordinary caches and applies compression
after preparation. A viable future path must avoid first accumulating the full
uncompressed long prompt. The separate diagnostic already creates compressed
caches from the first chunk, but does not change production behavior.

The current quantized-attention path materializes score and softmax tensors.
At 256K and 40 heads, one BF16 score tensor is 10 GiB for a 512-token query chunk,
or 0.625 GiB for a 32-token chunk; overlapping tensors and masks add cost.
Memory admission must cover transient allocations, not just final cache storage.
See the [implementation analysis](talkie-context.md) and
[exact calculations](../benchmark-results/talkie-20260913/context-extension/memory-estimates.json).

## Future work, in priority order

P0 is complete. P1–P5 are deferred until the user reopens context work. Retain
original artifacts and publish failed results alongside successes.

| Priority / milestone | Work | Acceptance gate before proceeding |
| --- | --- | --- |
| P0 — Complete: preserve the working 2K product | Compact `attn8` is available and passed four generation checks and seven installed-runtime HTTP checks. Q8 remains the fidelity reference. | Completed at 2K; normal/streamed chat, multi-turn input, context rejection, cancellation, and recovery passed. Context metadata and admission remain unchanged. |
| P1 — Establish a stronger evaluation baseline | Add diverse held-out historical prose and independently written retrieval, multi-fact reasoning, and summarization tasks. Retain beginning/middle/end placement and exact token accounting. Separate calibration, adaptation, and evaluation data. | Fixed corpus hashes, private answers, declared metrics, known baseline failures, and unchanged 2K reference results are recorded before tuning. Current smoke failures cannot be hidden by aggregation. |
| P2 — Investigate 4K positional adaptation | Compare unchanged positions with explicit scaling/adaptation candidates that preserve Talkie's inverse rotation and post-RoPE Q/K normalization. Evaluate language loss and task behavior within 2K and beyond it. | The existing beginning-retrieval regression is resolved; no previously passing 2K smoke task regresses; the expanded held-out suite meets the predeclared quality gate. Merely raising metadata is insufficient. |
| P3 — Validate cache compression and bounded attention | Evaluate a quality-preserving cache scheme from token zero, including layer-specific precision or better K/V quantization if needed. Implement bounded workspace and account for cache growth, masks, and simultaneous tensors. | Numerical/cache tests pass; no input is dropped; cache dtype, geometry, and stored bytes are verified across all layers. Matched short-context tasks remain correct and broader quality passes. Measured peaks stay under admission limits with host headroom and no sustained swapping. |
| P4 — Extend in stages | Advance through 8K, 16K, 32K, 64K, 128K, and finally 256K only after each preceding stage passes. Revisit adaptation data and kernels as evidence requires. | Every length has full-input proof, position-stratified task results, loss by position, short-context regression results, peak memory, prefill/decode measurements, and successful cancellation/recovery. Stop at the first failed gate; a smaller validated limit is an acceptable outcome. |
| P5 — Release an explicitly validated variant | Package exact positional/cache settings, training or adaptation provenance, model identity, and tested limit together. Preserve rollback to the current 2K model. | Advertised context equals the maximum tested and accepted length. The complete prompt plus output fits, oversize requests fail clearly, runtime checks pass, and known limitations are visible. No 256K label without 256K evidence. |

For P1, proposed quality criteria are: zero new failures on the existing passing
2K smoke cases; no more than a 3% relative short-context mean NLL increase; and
no more than a two-percentage-point decrease in short-context task success on a
substantially expanded held-out set. Long-context retrieval should cover at least
five position bands, with at least 95% exact-fact accuracy in every band and
separate multi-fact and summarization scoring. These are **proposed future gates**,
not measured achievements; finalize dataset size, uncertainty reporting, and
thresholds before tuning. Passing loss alone is never sufficient.

## Research choices and risks

[Position Interpolation](https://arxiv.org/abs/2306.15595),
[YaRN](https://arxiv.org/abs/2309.00071), and
[LongRoPE](https://arxiv.org/abs/2402.13753) provide primary precedents for positional
extension and adaptation. They were demonstrated on other model families;
Talkie's equations and short-context behavior require independent validation.
Training data and compute requirements must be estimated for an actual candidate
recipe. No suitable hardware, training budget, timeline, or final length is
promised by this PRD.

[KIVI](https://arxiv.org/abs/2402.02750) motivates investigating asymmetric key/value
quantization; its results do not validate generic affine Q2 for Talkie.
[FlashAttention](https://arxiv.org/abs/2205.14135) motivates tiled exact attention
with bounded intermediate storage. Applying those ideas to Talkie's quantized
MLX path remains engineering and quality work.

Principal risks are short-context regression, lost distant facts, repeated or
incomplete generations, historical-style drift from adaptation data, insufficient
memory during prefill, and impractical latency from full attention. A low NLL,
successful allocation, one correct retrieval, or an arbitrary declared token
limit can each give false confidence. Mitigate them through matched controls,
position-specific results, measured transient memory, and staged release gates.

## Architecture migration option: Talkie with grouped-query attention

The user's follow-up asks whether Talkie could move into an architecture with
a larger context. The most conservative candidate to investigate is a Talkie
derivative with **40 query heads and eight key/value heads**, retaining the
128-value head dimension. This is a trained model adaptation, not a lossless
file conversion or an automatic context extension. The user subsequently authorized
a bounded local GQA conversion and recovery experiment. Runtime support and the
training pilot are implemented; production weights and context limits remain
unchanged. See the [experiment record](../benchmark-results/talkie-20260913/gqa/README.md)
for measured recovery and generation results.

[GQA](https://arxiv.org/abs/2305.13245) establishes a precedent for converting
multi-head checkpoints and recovering quality through additional training.
Its reported training cost and quality on other models are not a Talkie budget
or guarantee. More recent [head-alignment work](https://github.com/fpcsong/mha2gqa)
provides an alternative to naive head averaging, but its published code targets
Llama-style models and would require adaptation to Talkie.

Calculated cache storage at 262,144 tokens, with group-64 affine quantization
and BF16 scale/bias metadata, excluding weights and all workspace:

| Key/value heads | BF16 KV | Affine Q8 KV | Affine Q4 KV |
| --- | ---: | ---: | ---: |
| Current 40 | 200 GiB | 106.25 GiB | 56.25 GiB |
| Proposed 8 | 40 GiB | 21.25 GiB | 11.25 GiB |
| More aggressive 4 | 20 GiB | 10.625 GiB | 5.625 GiB |

Eight KV heads would make higher-precision caches plausible within the current
machine's memory budget, avoiding dependence on the generic Q2 cache that failed
our retrieval probe. These are analytical estimates, not measurements of a
converted model. GQA retains all 40 query heads, so it does not by itself remove
the large attention-score workspace or quadratic prefill work described above.

Start from the verified BF16 Talkie checkpoint. Preserve its tokenizer,
embeddings, feed-forward blocks, query/output projections, per-query head gains,
inverse RoPE, and post-RoPE Q/K normalization. Initialize smaller K/V projections
from aligned or grouped original heads, then recover short-context behavior
through continued training or distillation from the original Talkie on suitable
historical data. Merging heads changes a nonlinear computation and cannot
preserve all outputs exactly. Requantize only after a trained candidate meets
the quality gates.

The underlying MLX attention and affine-cache paths support grouped query heads.
Midnight now decodes separate query/KV head counts, uses smaller K/V projections,
reports the corresponding cache geometry, and disables the equal-width projection
fusion for GQA. Tiny reference tests cover ordinary and quantized grouped caches
and cached decoding. First prove useful recovery at 2K; then
apply the staged positional adaptation, long-document training, and validation
milestones above. Smaller caches do not teach the model to use distant text.

A full transfer into a state-space or hybrid architecture is another research
route. [MOHAWK](https://arxiv.org/abs/2408.10189) demonstrates Transformer-to-SSM
distillation, with billions of training tokens in its reported experiments.
That precedent does not establish useful 256K recall for Talkie. It is a larger
behavior-transfer and runtime project than GQA, and recurrent state alone must
not be represented as full-token attention under this PRD's definition.
Starting with a modern pretrained long-context model would also bring its
existing modern knowledge; teaching it Talkie's prose would not establish the
same historical training boundary.

The first eight-KV-head pilot uses contiguous five-head averaging and trains
only K/V projections against the original BF16 teacher. This reduces total
parameters to 11,602,536,121 and cache storage by a factor of five at an unchanged
sequence length. It remains a dense model, not MoE. Passing numerical runtime
tests does not establish that the recovered model gives useful answers.

### Authorized GQA pilot result

The runtime and conversion/recovery tooling are implemented and tested. Naive
head averaging plus three bounded K/V recovery pilots failed the short-context
answer gate. The final pilot used 49 calibration samples / 6,405 unique token
positions, completing 408 updates within a five-minute training bound. Held-out
KL improved from 4.714229 to 1.735937, but the four final answers were repetitive,
irrelevant or incomplete. No derivative was promoted or quantized.

A broader recovery recipe remains research work. Do not advance to positional
extension or represent this candidate as a useful long-context Talkie model.
The complete outputs and measured limits are in the experiment record above.

## Decision record

September 13, 2026: the user chose to leave context as it is, document the future
work, and ensure the current model runs well. Production remains at 2,048 tokens.
Long-context extension remains deferred. A subsequent instruction authorized
short-context GQA recovery and answer testing, recorded above. Current-runtime
verification is complete; the compact model is available and the Q8 reference
is retained, without changing that decision.
