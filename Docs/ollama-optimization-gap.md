# Ollama MLX optimization gap audit — 2026-08-29

## Bottom line

The native Swift runner is **not missing a secret Ollama Q4 or Metal kernel**.
For the exact Q4R8 tensor payload, both runners reach the same MLX affine-Q4,
Q8-router, GatherQMM, SDPA, and NAX kernel families. The Swift Laguna graph is
actually leaner in several important places.

Ollama still decodes the compatibility model faster in the current deployment:

| Test | Native Swift Q4R8 | Ollama Q4R8 | Ollama advantage |
| --- | ---: | ---: | ---: |
| Exact-prompt, cache-capacity-matched 128-token block, 12 trials | 166.09 tok/s | 179.57 tok/s | **8.11%** |
| Earlier 512-token deployment block, later found to differ by one prompt ID | 161.16 tok/s | 173.09 tok/s | 7.40% |

This does **not** invalidate the Q4R8 format. In the same Ollama runtime, the
exact Q4R8 model reached 173.09 tok/s while Ollama's stock Laguna NVFP4 model
reached 154.42 tok/s, making Q4R8 12.09% faster in that deployment comparison.
The remaining problem is native runtime throughput, not the custom quantizer.

The corrected 128-token test explicitly set Ollama `think: false`. All 79
prompt IDs then matched native exactly, including the final
`<assistant></think>` token. Prompt plus generation stays below the first
256-token KV allocation in both runners. This rules out prompt rendering,
retained KV capacity, and prefix-cache allocation reuse as explanations for the
measured steady decode difference.

The synchronized cold-prompt Metal traces now locate the gap. Native and Ollama
perform essentially the same amount of GPU work, but native leaves materially
larger gaps between submissions. Native accumulated 813.86 ms of active GPU
work across a 1,000.78 ms span; Ollama accumulated 819.94 ms across 858.16 ms.
That is 81.32% versus 95.55% GPU duty. Native had 69 inter-command gaps above
1 ms while Ollama had two.

The corresponding CPU profile points to Swift frontend graph and ownership
work, not a missing Metal kernel. Two concrete source differences account for
most of the sampled budget: the generic Swift KV-cache subscript setter and the
cumulative text/tool decoding path. A direct KV slice-update prototype reached
168.60 and 168.64 tok/s around a 163.71 tok/s matched control, a repeatable
**3.0%** increase. Additional attempts after that were invalidated by unrelated
multi-gigabyte Git snapshot and PDF workloads and are retained only as
contaminated diagnostics.

## Cold Metal trace attribution

The cold-prompt traces used the same 79-token prompt and avoided Ollama prefix
reuse. Trace instrumentation depresses absolute throughput, so the useful
signals are work, duty cycle, gaps, and synchronized stack attribution.

| Metric | Native Swift | Ollama |
| --- | ---: | ---: |
| Active GPU union | 813.86 ms | 819.94 ms |
| Decode-span wall time | 1,000.78 ms | 858.16 ms |
| GPU duty | 81.32% | 95.55% |
| Gaps over 1 ms | 69 | 2 |
| p99 submission gap | 1,059 microseconds | 267 microseconds |

The near-equal active GPU time rules out a general kernel-throughput deficit.
The native submission thread instead performs more allocation and graph-wrapper
work. Time Profiler sampled about 151 ms in native malloc/free leaves versus
about 60 ms for Ollama over the traced request.

Within the 128-token native decode span, the source-attributable samples were:

| Native frontend component | Inclusive samples | Per token | Share of the 0.452 ms/token target |
| --- | ---: | ---: | ---: |
| Generic KV-cache `MLXArray` subscript setter | 29 ms | 0.227 ms | 50.1% |
| Text decoder/tool-handler envelope | 12–13 ms | 0.094–0.102 ms | 20.7–22.5% |
| Combined | 41–42 ms | 0.320–0.328 ms | 70.9–72.6% |

Sampling is directional rather than an exact stopwatch, but both categories
are independently visible in source and in the synchronized trace.

### Direct KV slice-update prototype

Laguna performs 80 K/V cache writes per token across 40 attention layers. The
Swift path writes each one through the general NumPy-style subscript setter.
For a decode update shaped `[1, H, 1, D]`, that path removes the leading
singleton, expands ellipsis/index metadata, constructs start/end/stride and
reshape arrays, and reshapes the update back to rank four before calling
`mlx_slice_update`. That can create 160 avoidable reshape/view nodes plus 80
metadata pipelines per token.

Ollama constructs the complete slice bounds and calls MLX slice update directly,
then updates the stable array wrapper with `mlx_array_set`. The first native
prototype bypassed the generic index translator. Its clean results were:

| Build | Median decode |
| --- | ---: |
| Matched current-tree control | 163.71 tok/s |
| Direct KV prototype | 168.60 tok/s |
| Direct KV confirmation | 168.64 tok/s |

Generated text was byte-identical to the control. The 3.0% delta saves about
0.178 ms/token in that bracket. A production patch should retain stable
`MLXArray` identity with `_updateInternal`, add graph-structure tests proving no
reshape/broadcast nodes, cover simple-cache growth and rotating-cache wrap, and
run a clean A/B once the machine is otherwise idle.

### Incremental ByteLevel detokenization

Swift's `NaiveStreamingDetokenizer` re-decodes the full newline-delimited token
segment and computes a common prefix after every token. Ollama maps only the
current vocabulary ID to bytes and buffers an incomplete UTF-8 suffix. The
named native text pipeline accounts for roughly 0.10 ms/token in the trace.

Laguna is eligible for a safe fast path because its tokenizer uses a top-level
ByteLevel decoder with cleanup disabled. The production design should expose a
certified per-ID ByteLevel capability and retain the current cumulative decoder
for SentencePiece, Metaspace, WordPiece, decoder sequences, and cleanup-enabled
tokenizers. Stop-string and tool-call processing remain downstream and need
differential streaming tests across arbitrary UTF-8 and tag boundaries.

## What the native runner is missing

### 1. Persistent compiled C closures

This is the clearest per-token optimization Ollama has and the native runner
does not.

The exact Laguna graph invokes 197 compiled islands for every decoded token:

- 119 calls with two inputs and one output;
- 39 calls with two inputs and two outputs;
- 39 calls with five inputs and one output.

Ollama creates each compiled MLX closure once with `sync.Once`, then reuses it.
The Swift `CompiledFunction` path takes a global evaluation lock and recreates
the callback handle and compiled-closure handle on every call. MLX's graph
compile itself is cached, but the wrapper work is still repeated.

An adjacent microbenchmark measured:

| Wrapper | Estimated overhead per Laguna token |
| --- | ---: |
| Current Swift | 128.31 microseconds |
| Ollama-style persistent closure | 104.82 microseconds |
| Difference | **23.48 microseconds** |

That is approximately **0.39% of native token time**. It is worth fixing, but
it explains only a small fraction of the observed gap.

Relevant code:

- Swift: `.build/checkouts/mlx-swift/Source/MLX/Transforms+Compile.swift`,
  `CompiledFunction.call` and `innerCall`.
- Ollama 0.33.1: `x/mlxrunner/mlx/compile.go`, `Compile`.

### 2. A permanent MLX worker pinned to one OS thread

Ollama loads the model and executes every request on one long-lived goroutine
that calls `runtime.LockOSThread`. This keeps MLX's thread-local stream,
command-encoder, and compile state attached to one pthread.

The native runner actor-serializes generation, but Swift tasks may resume on
different OS threads. It therefore uses MLX's global thread-unsafe stream.

A temporary native thread-local-stream probe was revealing:

- warm trials reached roughly 164.3–164.7 tok/s, no faster than the current
  global-stream path;
- trials that landed on a new OS thread collapsed to 14–28 tok/s while that
  thread rebuilt cold thread-local state.

The full implementation now uses a permanent Foundation pthread, Swift serial
and task executors backed by one FIFO, a normal thread-local MLX stream created
on that worker, and explicit executor preference in MLX-LM's two unstructured
producer tasks. Pinned tasks yield cooperatively once per token so streaming and
cancellation are not delayed until the completion ends.

The matched 128-token result was:

| Metric | Existing runtime | Pinned worker | Change |
| --- | ---: | ---: | ---: |
| Median decode | 166.09 tok/s | 166.56 tok/s | +0.28% |
| Median prompt | 1040.39 tok/s | 1040.22 tok/s | -0.02% |
| Median first token | 77.23 ms | 83.42 ms | +6.19 ms / 8.01% slower |

An immediate confirmation block was nonstationary and reached only 158.34
tok/s median. Therefore the small primary increase is not a credible throughput
win, and the worker is disabled by default. It remains reproducible with
`MODEL_RUNNER_PINNED_MLX=1` for tracing and further executor experiments.

The current global stream also checks a thread-local command-encoder map that
is guaranteed to miss before checking the global map. Reversing the lookup
order improved isolated native runs by roughly **0.5–0.9%**, with device-state
noise. A pinned worker would avoid needing this global-stream compromise.

Relevant code:

- Swift runner: `Sources/ModelRunnerCore/LocalModelRunner.swift`,
  `generateOnDevice`.
- Ollama 0.33.1: `x/internal/mlxthread/thread.go` and
  `x/mlxrunner/runner.go`.

### 3. A model-lifetime branching prefix cache

Ollama maintains a compressed token-prefix trie with per-layer snapshots,
branch switching, page-in/page-out, and an 8 GiB snapshot eviction threshold.
It can reuse arbitrary common prefixes across requests and conversation
branches.

The native runner retains a `ChatSession` only for a strict extension of the
previous committed transcript. It does not provide Ollama's general
model-lifetime prefix trie.

This can be a large real-world **time-to-first-token and multi-turn latency**
win, but it contributes zero to the cache-capacity-matched steady decode result.

Relevant code:

- Swift runner: `Sources/ModelRunnerCore/LocalModelRunner.swift`,
  `CachedConversation` and `generateWithLagunaPromptCache`.
- Ollama 0.33.1: `x/mlxrunner/prefix_cache.go`.

### 4. Top-K-first sampled decoding

For greedy decoding, both runners use `argmax`, so this did not affect the
benchmark.

For normal sampled decoding with `top_k = 64`, Ollama first partitions the
full vocabulary down to 64 candidates, then performs softmax and top-P on that
small set. The Swift MLX-LM sampler currently computes full-vocabulary
`logSoftmax`, performs full-vocabulary top-P sorting, and only then applies
top-K.

This is probably the largest missing **sampled-token** optimization. It needs a
separate benchmark because it also changes filter ordering and can change the
sample distribution. An Ollama-compatible fast sampler should therefore be an
explicit behavior choice, not a silent replacement.

Relevant code:

- Swift: `.build/checkouts/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift`,
  `TopPSampler.sample`.
- Ollama 0.33.1: `x/mlxrunner/sample/sample.go`, `sparseDistribution`.

### 5. Wired model residency

After loading and evaluating the weights, Ollama sets MLX's wired-memory limit
to the lesser of active model memory and Metal's recommended working-set size.
The current native tree now measures a model/KV/workspace plan and raises wired
residency per request. That source change landed after the original 166.09 tok/s
baseline. It was common-mode for the clean direct-KV bracket above, so it cannot
explain that bracket's 3.0% change; its independent contribution still needs an
explicit `MODEL_RUNNER_MLX_WIRED_MEMORY=0/1` A/B.

Wired residency should reduce paging variance under memory pressure and with
growing KV caches. It should not be credited with an unmeasured throughput win.

Relevant code:

- Swift: `Sources/ModelRunnerCore/MLXWiredMemoryPlan.swift` and
  `Sources/ModelRunnerCore/LocalModelRunner.swift`.
- Ollama 0.33.1: `x/mlxrunner/runner.go`, `configureWiredMemory`.

### 6. Larger prefill chunks

Ollama uses a 2,048-token prefill ceiling. Swift MLX-LM's generic default is
512 tokens with balanced chunking.

The benchmark prompt was only 79 tokens, so this cannot affect its decode
result. For long prompts, 2,048 may reduce submission overhead, but it also
increases peak attention and logits memory. This is a tuning opportunity, not
an unconditional optimization; benchmark 512, 1,024, and 2,048 by prompt
length.

### 7. Small graph-construction and handle cleanups

The Swift graph creates more temporary frontend objects in a few places:

- optional empty-array wrappers around GatherQMM arguments;
- an unused empty `inverseOrder` array in unsorted one-token MoE decode;
- BF16 correction biases that Ollama converts to FP32 once at load;
- a routed-scale scalar/cast that Ollama stores once.

The routed-scale cache was tested directly and changed median throughput by
**-0.11%**, effectively noise. These cleanups are worthwhile for simplicity,
but they are not credible explanations for the main gap.

### Related compatibility gap: routed NVFP4 global scales

Ollama's Laguna implementation supports per-expert global scales for the stock
NVFP4 routed gate, up, and down projections and folds selected scales into the
router/reduction path. The native `QuantizedSwitchLinear` does not currently
represent those global scales.

This does not affect the exact affine-Q4R8 comparison, and it is not an Ollama
speed optimization for Q4R8. It does mean a faithful native-versus-Ollama run
of the stock double-scaled NVFP4 model requires implementing that loading and
scale-folding path first. The existing 154.42 tok/s stock result was therefore
measured inside Ollama, where both stock NVFP4 and exact Q4R8 are supported.

## Optimizations already matched or better in Swift

The source comparison found all of the following at parity:

- standard MLX affine Q4/group-64 QMM and GatherQMM;
- Q8 expert routers;
- NAX quantized kernels on the M5 Max;
- fused SDPA, RoPE, RMSNorm, BF16 KV cache, and rotating/full cache behavior;
- 78 routed GatherQMMs per token;
- compiled router, routed SwiGLU, shared SwiGLU, and expert reduction islands;
- one-token-ahead asynchronous evaluation;
- greedy `argmax` sampling.

The native Laguna graph is leaner in several areas:

- it fuses all 40 always-active affine-Q4 gate/up projections, while Ollama
  deliberately excludes affine Q4 from that dense/shared fusion;
- it executes 398 quantized matmuls per token versus Ollama's 438;
- it compiles more of the attention-gate expression;
- its one-token MoE view plumbing is simpler.

This makes “missing model-graph fusion” an especially poor explanation for the
remaining difference.

## Controlled hypotheses that were ruled out

| Hypothesis | Test result | Conclusion |
| --- | --- | --- |
| Ollama has different/faster Metal quantization source | Exact MLX Metal trees match; the only `quantized.cpp` difference is host binding order | Ruled out |
| Ollama's large AOT `mlx.metallib` is faster | Native with Ollama's 174 MB AOT library was 1.12% slower than adjacent JIT control | Ruled out |
| `-O3 -DNDEBUG` explains the gap | Temporary native build was 0.8–0.9% slower than current `-O2` build | Ruled out |
| Metal 4 or NAX is absent in Swift | Both runners contain and select the same Metal 4/NAX kernel sources | Ruled out |
| Ollama uses a newer MLX backend | Native uses MLX 0.32.2; Ollama reports 0.32.1 plus 37 commits | Ruled out |
| Prompt-template mismatch creates the decode lead | Corrected all 79 prompt IDs and reran: 8.11% | Ruled out |
| Prefix/KV allocation reuse creates the decode lead | Sub-256-token exact-prompt test retained an 8.11% lead | Ruled out for steady decode |
| Caching the routed-scale tensor closes the gap | -0.11% A/B result | Ruled out |
| A permanent pinned pthread closes the gap | +0.28% primary decode, slower first token, and a 158.34 tok/s nonstationary confirmation | Ruled out as a production speed win; retained as opt-in instrumentation |
| Native Metal kernels are slower | Native used 813.86 ms active GPU time versus Ollama's 819.94 ms in cold-prompt traces | Ruled out; native loses time between submissions |
| The native 256 MiB MLX cache cap causes the gap | 256/1,024/4,096 MiB bracket converged around 163.2–163.5 tok/s | Ruled out |
| Nested autorelease pools cause the gap | One-pool median 165.13 tok/s; fully unpooled diagnostic 166.06 tok/s | Too small; unpooled cleanup is also deferred and unsafe long-term |
| Repeated scalar `eval` checks cause the gap | Ready-scalar microbenchmark estimated about 54 ns/token extra | Ruled out |

## Evidence boundary

The exact 18,821,963,264-byte Q4R8 tensor payload and all 79 prompt token IDs are
identical between the corrected deployments. The first generated token also
matches, but the free-running completions later diverge. That can arise from a
small frontend numerical difference or an early near-tie, after which this MoE
model may route through a different expert sequence. Consequently, this is now
a strict same-prompt deployment comparison, but not yet a strict same-generated-
token kernel A/B.

The cold-prompt Metal trace already establishes that the broad residual is host
submission work rather than GPU work. A strict per-route comparison should
still use identical prompt IDs and teacher-forced decode IDs, then collect:

1. command-buffer and encoder count per token;
2. individual kernel durations;
3. CPU gaps between GPU submissions;
4. allocations and cache misses on the submission thread.

That stricter trace will separate the remaining wrapper/ARC budget from any
route-dependent memory-address effect without confounding generated outputs.

## Recommended implementation order

1. Implement the exact axis/range KV slice-update fast path with stable wrapper
   identity, structural graph tests, cache-wrap tests, and a clean A/B.
2. Add the certified incremental ByteLevel detokenizer and differential stream
   tests; preserve cumulative decoding as the generic fallback.
3. Expose persistent compiled closures in the Swift/C binding and benchmark the
   measured ~0.39% opportunity independently.
4. Run a teacher-forced, route-hashed trace to apportion the smaller remainder.
5. Add an optional top-K-first Ollama-compatible sampler for sampled decoding.
6. Generalize the conversation cache into a branching prefix cache and tune
   long-prompt chunk sizes separately from steady decode.

Do not spend engineering time expanding the AOT metallib or changing the
release build to `-O3` for this gap; both were measured and lost.
