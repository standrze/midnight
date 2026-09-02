# vLLM-Metal and SGLang MLX optimization audit — 2026-08-29

## Executive answer

Neither current vLLM-Metal nor SGLang contains a hidden faster Laguna Q4
weight kernel that explains or supersedes this runner's Q4R8 path. Both load
normal MLX-LM quantized layers and use stock MLX quantized-matmul/QQMM paths for
the default weight path. Q4R8 is likewise a portable mixed-precision checkpoint
policy—affine Q4 group-64 with Q8 routers—not a separate runtime kernel. Its
format/quality tradeoff and our Laguna-specific graph reductions therefore
remain real differentiators. SGLang's narrow, default-off custom Q4 MoE gate
kernel is discussed below; its own batch-one measurement was within noise.

The important optimizations found upstream are mostly serving architecture:

- vLLM-Metal is the more feature-complete of the two audited MLX servers:
  native paged KV attention, fused K/V cache scatter, continuous mixed prefill/decode batching,
  automatic prefix caching, copy-on-write blocks, measured memory planning,
  TurboQuant KV compression, and generic speculative decoding.
- SGLang MLX has continuous batching and a global radix prefix cache, but its
  live attention cache is still per-request contiguous storage that it pads and
  batches. Its MLX radix reuse is disabled for models containing sliding-window
  attention—which includes this Laguna checkpoint. It does not have true MLX
  paged attention or MLX speculative decode.
- Both implement a one-step-ahead lazy-token decode pipeline. The native Swift
  runner already implements the same mechanism, so their published overlap
  gain is not an unclaimed speedup available to us.

For the existing short-context, batch-one Laguna test, the source audit found
no obvious large missing optimization. Code inspection alone cannot exclude an
execution-level win; a matched runtime benchmark is still required. The
remaining plausible direct candidates are a properly counterbalanced test of
`MLX_MAX_OPS_PER_BUFFER=2000`, a faster
bounded-candidate sampled-token path, and measured MLX buffer-cache/wired-memory
policy.
The highest-value missing work overall is paged KV plus continuous batching and
global content-addressed prefix reuse, but those raise server throughput, TTFT,
and long-context capacity more than isolated one-stream decode tokens/s.

## Source provenance

The audit pinned official sources rather than inspecting package summaries:

| Component | Audited revision |
|---|---|
| vLLM-Metal | [`a9184aa13ccd6ab7eb4278ec46d5d84876a39125`](https://github.com/vllm-project/vllm-metal/commit/a9184aa13ccd6ab7eb4278ec46d5d84876a39125), tag `v0.3.0.dev20260829020049` |
| vLLM paired release | [`2cf0a6915ce544dc493a0990f2ea38d81601128a`](https://github.com/vllm-project/vllm/tree/2cf0a6915ce544dc493a0990f2ea38d81601128a), tag `v0.28.0` |
| vLLM current main | [`680e2177e473ed8dfaa9773f7ead185b369cab46`](https://github.com/vllm-project/vllm/commit/680e2177e473ed8dfaa9773f7ead185b369cab46) |
| SGLang current main | [`9a489f8d2fc376fbe5594393af367fd99ada7aac`](https://github.com/sgl-project/sglang/commit/9a489f8d2fc376fbe5594393af367fd99ada7aac) |
| SGLang release | [`v0.5.18`](https://github.com/sgl-project/sglang/releases/tag/v0.5.18), commit `71de97b264b04dcd514cf904003028aefe9775c8` |
| Apple MLX used by vLLM-Metal | [`v0.32.0`](https://github.com/ml-explore/mlx/tree/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247) |
| MLX-LM pinned by vLLM-Metal | [`254d153fdeb6f150edd4fc5a54f9828638481fa8`](https://github.com/ml-explore/mlx-lm/commit/254d153fdeb6f150edd4fc5a54f9828638481fa8) |

The clean-room checkouts and detailed raw audits are currently under
`/private/tmp/vllm-mlx-audit/` and `/private/tmp/sglang-mlx-audit/`.

Unlike vLLM-Metal's pinned stack, SGLang declares `mlx>=0.32.0` and an
unversioned MLX-LM dependency. The source-level conclusions are pinned, but a
reproducible SGLang runtime or Laguna performance comparison requires locking
the actual MLX and MLX-LM revisions first.

The locally installed Docker vLLM-Metal backend is older: vLLM-Metal 0.2.0,
vLLM 0.19.0, MLX 0.31.1, and MLX-LM 0.31.2. That installed MLX-LM revision
does not include Laguna. The audited upstream vLLM-Metal alpha development
snapshot, `v0.3.0.dev20260829020049`, explicitly lists Laguna with paged GQA
and prefix caching and gives
`poolside/Laguna-XS-2.1-NVFP4-mlx` as its example
([support matrix](https://github.com/vllm-project/vllm-metal/blob/a9184aa13ccd6ab7eb4278ec46d5d84876a39125/docs/supported_models.md#L87-L110)).
Current upstream support for a standard Laguna checkpoint does not prove that
our exact mixed Q4R8 checkpoint will load unchanged; that needs an environment
built from the audited development snapshot and an actual load test.

vLLM-Metal describes itself as an alpha, community-maintained hardware plugin.
The feature comparison below is source inspection, not a claim that its overall
performance or production maturity is superior on every workload.

## Feature comparison

| Optimization | vLLM-Metal | SGLang MLX | Native Laguna Q4R8 | Meaning for us |
|---|---|---|---|---|
| Q4R8 mixed-precision policy | No named path | No named path | Yes: affine Q4 group-64 with Q8 routers | Our checkpoint policy remains relevant; it deliberately retains MLX's stock packed kernels |
| One-step lazy-token chaining | Yes, greedy paged path | Yes | Yes | Already matched; no extra gain to copy |
| True paged KV attention | Yes, native Metal | No | No | Largest serving/long-context architectural gap |
| Fused paged K/V scatter | Yes | No | Not applicable to contiguous cache | Useful with paged KV, not a standalone short-decode win |
| Continuous batching | Yes, unified varlen prefill/decode | Yes, padded batched decode | No; actor admits one generation | Major aggregate-throughput gap |
| Cross-request prefix reuse | vLLM automatic block cache | Unified radix tree, but not for models containing SWA | Hot exact session plus bounded completed-message LRU | Branches and interleaved chats now reuse exact completed transcripts; unrelated token prefixes still require radix/paged KV |
| Headless intermediate prefill graph | Yes, conditionally | When the model exposes a callable trunk | DFlash uses trunk; ordinary path builds full model but evaluates cache roots | Native benefit is unproven because unused logits remain lazy |
| Speculative decoding | Gemma4 MTP; compatible draft model or n-gram for other targets | Not on MLX | Laguna DFlash | No compatible Laguna draft is supplied upstream; our target-specific path may be stronger |
| KV compression | TurboQuant paged KV | No comparable paged path | TurboQuant code exists in dependency but server does not configure it | Capacity/long-context option is currently unwired |
| MLX-native sampling | Greedy only; sampled path bridges to CPU Torch | Yes, bounded candidate tail | Yes, but full-vocabulary top-p before top-k | SGLang/Ollama ordering ideas are relevant to sampled decode |
| Working-set/wired-memory planning | Measured and wired | Working-set-aware and wired | Measured adaptive fixed-policy ticket, allocator/Metal capped | Core mechanism is now matched; pressure and latency validation remain |
| Model/MLP compilation | Opt-in, only selected Qwen families | No whole-model compile | Laguna-specific compiled/fused islands | Current vLLM feature does not apply to Laguna |
| Custom speculation verify attention | Optional shared verification window | No | DFlash verification | Possible long-context DFlash research, not an automatic win |

## What vLLM-Metal actually adds

### Native paged KV attention

This is vLLM-Metal's most substantial MLX/Metal engineering. It has a block-
based KV cache, variable-length paged attention, mixed prefill/decode metadata,
online softmax, tiled prefill, partitioned split-KV decode for long contexts,
and an optional M5 NAX prefill path. Its non-quantized write path replaces two
K/V scatters with one Metal `reshape_and_cache` dispatch
([source](https://github.com/vllm-project/vllm-metal/blob/a9184aa13ccd6ab7eb4278ec46d5d84876a39125/vllm_metal/attention/impls/sdpa.py#L625-L649)).

That matters most when requests have heterogeneous lengths, prefixes are
shared, the server is concurrent, or context is long. The split-KV path needs
at least two 512-token partitions and therefore was not active in our roughly
200-token short benchmark
([dispatch gate](https://github.com/vllm-project/vllm-metal/blob/a9184aa13ccd6ab7eb4278ec46d5d84876a39125/vllm_metal/metal/paged_ops.cpp#L518-L541)).
Paged indirection may add overhead for one tiny request, so it should not be
treated as a guaranteed batch-one TPS improvement.

### Continuous batching and automatic prefix caching

Upstream vLLM schedules chunked prefill, decode, speculative tokens, and prefix
hits under one token budget. Its automatic prefix cache hashes KV blocks,
retains reusable blocks under an eviction policy, and applies copy-on-write.
The Metal runner packs variable-length work into a single forward. These are
the clearest capabilities our single-generation actor lacks.

References:

- [vLLM scheduler](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/core/sched/scheduler.py#L476-L520)
- [vLLM prefix lookup](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/core/kv_cache_manager.py#L232-L298)
- [Metal copy-on-write](https://github.com/vllm-project/vllm-metal/blob/a9184aa13ccd6ab7eb4278ec46d5d84876a39125/vllm_metal/attention/caches/kv_cache.py#L327-L355)

### Dead-compute elimination during chunked prefill

For a non-final prompt chunk, vLLM-Metal calls the transformer body without the
vocabulary projection because only the KV/state writes matter
([source](https://github.com/vllm-project/vllm-metal/blob/a9184aa13ccd6ab7eb4278ec46d5d84876a39125/vllm_metal/v1/model_runner.py#L1337-L1388)).
The optimization is conditional: there must be no decode requests in the
batch, every prefill item must be an intermediate chunk, no drafter may be
installed, and the model adapter must expose an intermediate forward.
Our Laguna DFlash prefill already calls the trunk directly. Ordinary Laguna
chunked prefill constructs the full model result for each intermediate chunk in
`Sources/ModelRunnerCore/LagunaModel.swift:812-840`, but evaluates only the
cache roots and discards `output.logits`. Because MLX is lazy, the LM-head node
is not expected to execute or allocate its full output merely because the graph
was constructed. Calling the trunk directly may still reduce host graph
construction or retention, but that is an A/B and trace candidate—not an
established TTFT or peak-memory win. It cannot affect one-token decode.

### Memory planning

vLLM-Metal sets MLX's wired limit to Metal's recommended working set, profiles
one configuration-sized dummy forward to measure allocator-cache overhead,
caps that cache to the measurement, and budgets remaining memory for paged KV
([wired policy](https://github.com/vllm-project/vllm-metal/blob/a9184aa13ccd6ab7eb4278ec46d5d84876a39125/vllm_metal/utils.py#L56-L74),
[profile](https://github.com/vllm-project/vllm-metal/blob/a9184aa13ccd6ab7eb4278ec46d5d84876a39125/vllm_metal/v1/model_runner.py#L782-L799)).

`MLXResourceGuard` still supplies hard allocator/cache ceilings. The runner now
adds an isolated 513-token calibration after model, DFlash, and adapter loading,
then creates a per-request absolute fixed-policy wired ticket capped by both
the allocator ceiling and Metal's recommended working set. The ticket remains
active until the generation producer synchronizes, and completed-request peaks
can only grow the next budget. On the measured Q4R8+DFlash checkpoint the
calibration observed a 19.61 GB active peak and selected a 21.84 GB wired limit
under the 25.77 GB allocator cap. Pressure testing is still needed; this is not
evidence of a clean batch-one TPS gain by itself.

### TurboQuant and speculative decoding

TurboQuant compresses KV state, not weights. vLLM-Metal fuses encode and paged
scatter and documents 2.56x KV reduction for its default q8-key/q3-value example
([documentation](https://github.com/vllm-project/vllm-metal/blob/a9184aa13ccd6ab7eb4278ec46d5d84876a39125/docs/turboquant.md#L47-L66)).
Our MLX-Swift-LM dependency already contains a substantial `TurboQuantKVCache`,
but `generationRequestSettings` does not set `kvScheme`/`kvBits`, so the server
does not currently expose it.

vLLM-Metal supports synchronous greedy MTP, draft-model, and n-gram speculation
on paged attention. MTP is currently Gemma4-only; draft-model and n-gram can use
Laguna as a target. A draft must share tokenizer/vocabulary, use full attention,
and run without pipeline parallelism, so a compatible Laguna draft cannot be
assumed to exist. Speculation also disables the ordinary one-step async decode
pipeline, meaning their gains are not cumulative. Its documented
Qwen3-8B/Qwen3-0.6B result on an M5 Pro 64 GB
is 1.36–1.48x better TPOT at one stream, but this is a different model pair and
algorithm—not a prediction for Laguna
([documentation](https://github.com/vllm-project/vllm-metal/blob/a9184aa13ccd6ab7eb4278ec46d5d84876a39125/docs/speculative_decoding.md#L144-L166)).
Our Laguna DFlash should be benchmarked directly against a compatible vLLM
draft rather than inferred from that number.

## What SGLang MLX actually adds

### It has overlap and batching, but not paged attention

SGLang builds step N+1 directly from step N's still-lazy sampled token, submits
it, and only then materializes the older token
([scheduler](https://github.com/sgl-project/sglang/blob/9a489f8d2fc376fbe5594393af367fd99ada7aac/python/sglang/srt/hardware_backend/mlx/scheduler_mixin.py#L115-L153),
[runner](https://github.com/sgl-project/sglang/blob/9a489f8d2fc376fbe5594393af367fd99ada7aac/python/sglang/srt/hardware_backend/mlx/model_runner.py#L1600-L1649)).
Its merged overlap PR reported +15.63% output TPS for one Qwen3-0.6B test. That
figure proves the mechanism can matter when absent; it cannot be added to our
expected TPS because `mlx-swift-lm` already performs the same lazy-token chain
in `.build/checkouts/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift:921-950`.
SGLang drains or breaks the chain when waiting prefill must run, a request
finishes or batch composition changes, or grammar/custom processing requires
the sampled token on the CPU. Its reported result was one historical
Qwen3-0.6B, 244-token PR run on a different runtime and machine.

For multiple requests, SGLang constructs a `[B,1]` decode input, pads each
request's contiguous K/V to a common length, concatenates it, and makes one
batched `mx.fast.scaled_dot_product_attention` call
([source](https://github.com/sgl-project/sglang/blob/9a489f8d2fc376fbe5594393af367fd99ada7aac/python/sglang/srt/hardware_backend/mlx/kv_cache/attention_wrapper.py#L303-L345)).
This is real batching, but not ragged/paged attention; heterogeneous context
lengths waste work. MLX extend/prefill still loops over requests individually
and queues their lazy graphs; only decode becomes one padded batched SDPA.
vLLM-Metal has the stronger design here.

### Global radix reuse is still valuable

SGLang's default unified radix tree matches token prefixes across requests,
supports eviction and reference locks, and can schedule longest-prefix-first.
The MLX runner stores reusable prefix K/V in a shared slot-indexed pool. An
upstream validation reused 3,661 of 3,662 tokens on the second identical prompt.

References:

- [radix cache](https://github.com/sgl-project/sglang/blob/9a489f8d2fc376fbe5594393af367fd99ada7aac/python/sglang/srt/mem_cache/unified_radix_cache.py#L149-L208)
- [prefix-aware scheduling](https://github.com/sgl-project/sglang/blob/9a489f8d2fc376fbe5594393af367fd99ada7aac/python/sglang/srt/managers/schedule_policy.py#L138-L197)
- [MLX shared KV pool](https://github.com/sgl-project/sglang/blob/9a489f8d2fc376fbe5594393af367fd99ada7aac/python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_pool.py#L19-L75)

The runner now keeps the preceding successful chat as a zero-copy hot session
and a byte/count-bounded LRU of immutable completed-message snapshots. It picks
the deepest exact strict prefix, clones cache storage on restore, and therefore
supports edited branches and interleaved conversations without sharing mutable
KV state. It still cannot reuse a common system prompt across unrelated users
or a repeated RAG template whose message transcript is not an exact completed
prefix. Token-radix lookup plus paged copy-on-write ownership remains the
larger architectural step.

A real Q4R8 sibling-branch probe forced this LRU restore path after moving the
hot session down a different branch. It reused 111 of 140 prompt tokens; after
one complete warm-up, three-trial median TTFT was 86.16 ms cached versus
180.95 ms forced-cold, a 52.39% reduction. The first unwarmed restore was
slower than cold and the greedy continuations were not bit-identical, so this
validates the mechanism—not a universal latency or reproducibility claim.
SGLang's reuse is not the same as vLLM's paged copy-on-write sharing: it gathers
the matched slots into a live per-request contiguous cache and can duplicate
active KV. Its MLX path also recomputes matched attention prefixes for models
that contain sliding-window attention. The exact Q4R8 benchmark checkpoint was
checked: its `config.json` declares 30 `sliding_attention` and 10
`full_attention` layers with a 512-token sliding window. It therefore hits this
SGLang limitation and should not be credited with cross-request attention-
prefix reuse for our direct Laguna comparison.

### Sampled decoding is the most portable SGLang detail

SGLang keeps sampling in the MLX graph. When `top_k <= 1024`, it restricts
subsequent top-p/min-p/log/Gumbel/argmax work to a `[B,K]` candidate tail
([source](https://github.com/sgl-project/sglang/blob/9a489f8d2fc376fbe5594393af367fd99ada7aac/python/sglang/srt/hardware_backend/mlx/sampling.py#L216-L320)).
It still performs a full-vocabulary FP32 softmax and full argsort first, and the
bounded tail is used only when every row in the batch has normalized
`top_k <= 1024`. This is therefore a useful tail reduction, not a true partial
top-K selection primitive. The current SGLang MLX sampler also omits frequency,
presence, and repetition penalties.

Our sampler currently computes full-vocabulary `logSoftmax`, applies top-p via
full sort, then applies top-k through `argPartition`, and finally samples over
the original vocabulary (`Evaluate.swift:342-423`). That matches Python MLX-LM
semantics, but it makes the default `topP=0.95`, `topK=64` order expensive.
An explicitly compatibility-tested top-K-first/bounded-candidate sampler is a
plausible sampled-decode optimization. It gives no benefit to the greedy TPS
benchmark.

### SGLang's custom Metal kernels are not priorities

SGLang has two opt-in, default-off kernels:

- AOT vanilla RoPE plus shared-pool scatter. Its own prior measurements were
  shape dependent: gains on tested LLaMA BF16 shapes and regressions on tested
  Qwen-like shapes.
- A group-64 affine-Q4 MoE gate QMV plus SiLU/multiply epilogue. Its source says
  batch-one end-to-end was only +0.4%, one quarter of its run-to-run noise band
  ([source](https://github.com/sgl-project/sglang/blob/9a489f8d2fc376fbe5594393af367fd99ada7aac/python/sglang/srt/hardware_backend/mlx/moe/fused_swiglu.py#L1-L42)).

Neither supersedes our Laguna-specific gate/up and MoE work. SGLang also has no
whole-model `mx.compile`, no MLX speculative decoding, and no MLX FlashInfer,
Triton, CUDA graph, or CUDA quantization path.

## Direct single-stream findings

### Already matched

1. One-step-ahead lazy sampled-token chaining.
2. Persistent MLX execution on one process-lifetime pthread and stream.
3. MLX-native sampling rather than vLLM-Metal's non-greedy CPU bridge.
4. Laguna-specific compiled/fused graph islands that neither upstream runtime
   currently applies to Laguna.
5. Laguna-specific DFlash speculation.

### Worth implementing or benchmarking

1. **Headless ordinary chunked-prefill A/B.** It may reduce graph construction
   or retention, but lazy cache-only evaluation should already prune the LM
   head. Trace and benchmark it before implementing; no decode-TPS change is
   expected.
2. **Top-K-first/bounded-candidate sampling.** Relevant to normal temperature
   generation at the server defaults; no greedy benefit. It requires an
   intentional compatibility policy because filter order can change outputs.
3. **Measured allocator-cache cap and wired-memory policy.** Implemented after
   this audit: Metal performs an isolated 513-token calibration, clamps the
   absolute wired limit to the allocator and recommended-working-set ceilings,
   holds a fresh ticket through producer synchronization, and grows the next
   budget monotonically from live peaks. Pressure and latency A/Bs are still
   required.
4. **`MLX_MAX_OPS_PER_BUFFER=2000`.** A first non-counterbalanced 12-trial pair
   measured 163.41 versus 161.75 tok/s, nominally +1.03%. Treat this only as a
   directional observation: test order and thermals were not controlled, and
   the reverse trial was invalidated by an unrelated Swift compiler consuming
   100% CPU. The reliable current exact-prompt native result remains 166.09
   tok/s. Run a clean interleaved A/B before adopting the setting, launching a
   fresh process for every arm because MLX caches this environment value at
   process initialization.
5. **TurboQuant KV exposure.** Useful for capacity and long context; benchmark
   quality and latency per scheme. It is not a model-weight optimization.

### Architectural work for a real server

1. Paged KV plus a fused K/V write primitive.
2. Continuous mixed prefill/decode batching.
3. A token-radix global prefix cache with content hashes and copy-on-write
   ownership. A bounded completed-message-prefix LRU is now implemented for
   exact interleaved conversations and branches; unrelated prompts sharing
   only a system prompt still require the fuller radix/paged design.
4. Working-set-derived KV pool sizing and request cache pooling.
5. Long-context split-KV attention and, on macOS 26.2 with supported dtype,
   head, and block-size gates, NAX prefill. Stock MLX NAX quantized QMM can also
   help prefill or sufficiently large batches; single-token decode remains QMV.

These are high-value features if the product needs many simultaneous clients,
large contexts, or repeated prefixes. They should not be justified as a way to
close Ollama's 8.11% gap on the short batch-one test.

## Things not to copy

- vLLM-Metal's CPU/Torch bridge for non-greedy sampling.
- SGLang's default-off Q4 SwiGLU kernel without a Laguna-specific benchmark.
- SGLang's padded contiguous-KV batching as the final cache architecture.
- `MLX_MAX_MB_PER_BUFFER=2000` on this 64-GiB Mac; vLLM-Metal only auto-enables
  it when usable memory is at least 90 GiB and other conditions hold.
- Periodic `mx.clear_cache()` on an assumed cadence without latency and thermal
  measurements.
- CUDA-oriented or otherwise unsupported vLLM-Metal features: CUDA graphs,
  FlashInfer/FlashAttention, Triton/CUTLASS, Marlin, tensor parallelism, and
  expert-parallel all-to-all. Some are hardware-agnostic concepts in other
  implementations; the point is that this Metal plugin does not provide them.

## Interpreting upstream headline numbers

vLLM-Metal's README claim of 83x TTFT and 3.6x throughput compares its v0.2
paged backend with its own v0.1 backend. It is not a comparison with this native
runner and not a batch-one Laguna result
([README](https://github.com/vllm-project/vllm-metal/blob/a9184aa13ccd6ab7eb4278ec46d5d84876a39125/README.md#L10-L15)).

Similarly, SGLang's +15.63% overlap number measures adding lazy-token chaining
to a path that did not have it, while our path already does. vLLM-Metal's
1.36–1.48x speculative number is for a different Qwen target/draft pair. None
of these percentages can be added to the native Q4R8 result.

## Recommended order

1. Prototype and trace headless ordinary Laguna chunked prefill on 4k, 16k, and
   32k prompts; retain it only if TTFT, host overhead, or peak memory improves.
2. Run clean interleaved sampled and greedy A/Bs for command-buffer sizing,
   allocator-cache limits, wired residency, and a bounded-candidate sampler.
3. Expose existing TurboQuant KV schemes experimentally with quality gates.
4. If multi-client serving matters, design paged KV, continuous batching, and a
   global prefix cache as one coordinated architecture rather than three local
   patches.
5. Benchmark the audited vLLM-Metal 0.3 development snapshot against the stock
   Laguna checkpoint first; then validate whether its loader accepts our exact
   Q4R8 checkpoint before claiming a direct runtime comparison.
