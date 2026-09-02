# MacBook MLX performance: Midnight Runner versus Ollama

Last updated: 2026-08-29

## Current conclusion

MLX is the correct backend for this project on Apple Silicon. It does not,
however, make every MLX frontend equally fast. Ollama's Laguna runner also uses
MLX and Metal, so this comparison is primarily about the MLX core revision,
quantized kernels, model graph, and scheduling—not Swift versus Go or Python.

The project now pins the official MLX Swift 0.32 update revision `72f3c3a`,
which synchronizes MLX Swift with MLX 0.32.2 (`1f8e74e`) and MLX-C
(`c74db53`). That core is newer than the `c793734` build observed in Ollama and
contains the audited M5, NAX, quantized-MoE, and GQA changes. Dependency
preparation verifies all three revisions before building.

The earlier compiled-Laguna change remains important. On the old core it
improved same-process median decode throughput by **8.40%**, from 139.46 to
151.17 tok/s. After the backend upgrade, a thermally stressed five-pair
alternating rerun measured 114.64 tok/s unfused and 121.28 tok/s fused,
**+5.80%**, with exact output agreement. The optimized path remains Laguna's
default and the native runtime benchmark retains an alternating A/B mode.

Two additional Laguna paths are now optimized. The server retains one
actor-isolated `ChatSession` and its KV cache when the next OpenAI transcript is
a strict continuation of the completed turn, so a multi-turn coding session
prefills only the new suffix. Metrics now distinguish total, cached, and
actually prefilled prompt tokens. Separately, Laguna's FP32 per-head
softplus-and-multiply epilogue is compiled once per concrete head layout. Two
release microbenchmark confirmations of that epilogue measured **1.06–1.32x**
faster Metal execution with equal output. A full-checkpoint decode and
multi-turn time-to-first-token A/B are still required before assigning either
change a whole-model percentage.

There is not yet a defensible claim that either complete runtime is always
faster. Independent process runs on this MacBook vary dramatically with device
state. Current Ollama controls using the same 512-token request produced 132.96
tok/s in one five-trial run and 149.27 tok/s in another. A standalone fused
Swift run immediately after the latter fell to 116.25 tok/s despite measuring
151.17 tok/s in the earlier interleaved test. Cross-process point estimates
must therefore be treated as thermally and residency sensitive.

## What the two runtimes actually execute

| Component | Midnight Runner | Ollama 0.33.1 Laguna runner |
| --- | --- | --- |
| Host implementation | Swift | Go |
| Mac compute backend | MLX + Metal | MLX + Metal |
| MLX core observed locally | 0.32.2, `1f8e74e` | 0.32.1-37, `c793734` |
| Test package | affine Q4 group-64 plus Q8 routers, ScaleSearch LS2 | NVFP4 group-16, BF16 routers, MXFP8 LM head |
| Laguna graph | Native project implementation | Native Ollama implementation |

The local Ollama server log explicitly reports `starting mlx runner subprocess`,
`MLX version=0.32.1-37-gc793734`, and `LagunaForCausalLM`. Ollama is not winning
these Laguna tests by silently switching to llama.cpp.

Ollama's runner also pins model arrays, evaluates them before serving, sets a
wired-memory limit, runs MLX work on a dedicated thread, enables graph
compilation, and maintains a prefix cache. Those are useful implementation
details, but the first-request prefix cache does not explain one-token decode
throughput by itself.

## Why Ollama had a lead before the upgrade

### 1. Its MLX core was materially newer

The project previously pinned `mlx-swift` 0.31.6, which embedded MLX 0.31.1.
Ollama 0.33.1 shipped a later development core. The intervening official MLX
history includes changes directly aligned with this M5 Max and Laguna's graph:

- `5a1e44c3`: optimize large NVFP4 QMV on M5 Max;
- `e7838d5e`: raise the QMV batch limit for large matrices on M5-class GPUs;
- `a076a632`: choose the gather-QMM block size from rows per expert;
- `c7ff35d9`: skip unused simdgroup work in quantized MoE matmuls on NAX;
- `fa0d4463`: read each K/V byte once in GQA-8 decode attention;
- `0ebcee8d`: add a wide GEMV path for a few BF16/FP16 rows;
- `a082cb91`: improve one-block transposed NAX QMM tiling.

Stopping at the original Swift 0.32.0 update would not have been sufficient:
the first listed M5/Laguna changes landed after its embedded `7a1d4f5c` core.
The selected official update revision now embeds MLX 0.32.2 (`1f8e74e`), which
descends from Ollama's observed `c793734` core and includes those changes.

### Backend-upgrade full-model control

The release binaries from immediately before and after the upgrade were run in
the balanced order old → new → new → old against the same external Q4R8
ScaleSearch checkpoint. Every block used one warm-up and three measured
512-token generations:

| Backend block | Trial tok/s | Median |
| --- | --- | ---: |
| MLX 0.31.1, first | 150.78, 151.97, 149.28 | 150.78 |
| MLX 0.32.2, first | 151.99, 149.87, 145.42 | 149.87 |
| MLX 0.32.2, second | 151.85, 149.51, 146.79 | 149.51 |
| MLX 0.31.1, second | 145.85, 133.19, 135.51 | 135.51 |

The late old-core collapse demonstrates that heat still overwhelms a small
backend delta. The justified conclusion is that MLX 0.32.2 preserves Q4R8
throughput and puts the Swift runtime in the same observed performance class;
this sequence does not support a precise percentage speedup. The upgrade is
still required because it unlocks the newer kernels and fixes on which the
next format and graph experiments depend.

### 2. Ollama's Laguna graph was more explicitly fused

The audited Ollama model compiles its sigmoid/top-8 router and compiles the
weighted expert reduction together with the routed scale, shared expert, and
residual. The old Swift graph compiled only the generic weighted expert sum;
router operations and the final additions remained separate.

The project now mirrors those two graph boundaries without requiring a new
checkpoint. On the 40-GPU-core M5 Max, three repeated fragment A/B runs showed:

| Decode fragment | Current split path | Compiled fused path | Execution speedup |
| --- | ---: | ---: | ---: |
| sigmoid top-8 router | 0.017 ms | 0.011 ms | 1.47–1.50x |
| routed sum + shared + residual | 0.008 ms | 0.006 ms | 1.42–1.43x |

Laguna-XS has 39 sparse layers, so small per-layer dispatch savings accumulate.
The full-model alternating result was larger than a simple sum of these
rounded fragment times:

| Mode | Five 512-token trials | Median |
| --- | --- | ---: |
| Unfused | 144.12, 141.02, 139.46, 134.33, 131.64 | 139.46 tok/s |
| Compiled fusion | 157.43, 155.24, 151.17, 149.33, 143.61 | 151.17 tok/s |

The median gain is 8.40%. Trial order alternated, both modes used one loaded
model, and generated text matched exactly across every trial.

Run the regression again with:

```bash
swift run -c release model-runner-runtime-bench \
  /absolute/path/Laguna-XS-2.1-MLX-Q4R8 \
  /absolute/path/laguna-fusion-ab.json \
  --engine metal --tokens 512 --warmups 1 --trials 5 \
  --laguna-fusion-ab
```

### 3. Ollama's package targets its newer NVFP4 kernels

Ollama's 19 GB package stores large weights as NVFP4 group-16. This project's
Q4R8 package stores most large weights as affine Q4 group-64 and its routers as
affine Q8. Both large-weight formats cost approximately 4.5 bits/value once
their block metadata is included, but their reconstruction and kernel paths
are different.

NVFP4 is not inherently faster on this MacBook after the backend upgrade. The
interleaved identical-tensor benchmark on MLX 0.32.2 produced the following
NVFP4 throughput relative to affine Q4 group-64:

| Laguna-shaped operation | NVFP4 relative throughput |
| --- | ---: |
| dense QMV | 0.939x |
| gathered expert gate/up | 0.999x |
| gathered expert down | 1.018x |

These sub-0.1-ms tests remain mixed and do not establish a stable format
winner. Earlier sequential measurements that appeared to show a large NVFP4
advantage were run-order/JIT artifacts. The M5-specific kernels are now
available in Swift, but changing the deployment format still requires a
quality-matched full-model A/B. The backend upgrade does not convert the
existing affine-Q4 checkpoint to NVFP4.

Q4R8 ScaleSearch remains useful for accuracy. It writes the same affine Q4
layout and invokes the same decode kernel, so it should not be described as a
kernel-speed optimization.

## DFlash result

DFlash remains implemented but should not be enabled for Mac deployment yet.
The same-loaded-model alternating benchmark measured 134.25 tok/s target-only
versus 128.54 tok/s with block-size-3 DFlash, a 4.26% regression. Acceptance
was 53.1%, and generated output did not match the target-only output. The
runtime keeps the implementation for later work, but compiled Laguna fusion is
the proven optimization.

## Ranked Laguna roadmap

1. **Keep persistent prompt/KV reuse enabled for ordinary Laguna turns.** It
   removes repeat prefill work from strict transcript continuations. Complete
   the real-checkpoint warm/cold TTFT acceptance run before publishing a
   percentage.
2. **Keep compiled MoE fusion enabled, and retain the compiled attention gate
   behind its A/B switch.** MoE fusion measured +8.40% on MLX 0.31.1 and +5.80%
   in the thermally stressed MLX 0.32.2 rerun. The gate epilogue is positive in
   microbenchmarks and logit tests but still needs the full-checkpoint A/B.
3. **Keep the synchronized MLX 0.32.2 backend pin.** Dependency preparation
   verifies the Swift, core, and C revisions, and the Metal build compiles the
   matching authoritative static kernel sources.
4. **Keep Q4R8 until a quality-matched full-model format A/B wins.** The new-core
   microbenchmark is mixed: affine Q4 wins dense QMV, expert gate/up ties, and
   NVFP4 narrowly wins expert down.
5. **If affine Q4 remains the accuracy choice, optimize its MLX kernels.** The
   useful target is the physical MLX affine group-64 QMV/gather-QMM dispatch for
   M=1 and Laguna's exact expert shapes—not a renamed checkpoint format.
6. **Evaluate long-context attention and residency independently.** Laguna's
   full layers use GQA-6, so test a GQA-6/D128 vector-attention specialization,
   plus array pinning, wired-memory limits, and a dedicated MLX execution
   thread. Keep only changes that improve full-model throughput or stability.
7. **Revisit DFlash only after correctness.** Require exact greedy equivalence,
   then sweep block size and acceptance. A draft path that changes output or
   loses throughput is not an optimization.

## Latest Laguna latency changes

### Persistent prompt/KV reuse

`LocalModelRunner` now keeps one Laguna conversation session behind its actor.
After a successful response, a subsequent request reuses the session only when
the incoming OpenAI messages strictly begin with the exact committed transcript.
The retained MLX session re-renders the conversation, proves the token prefix,
and feeds only the uncached suffix to the existing KV cache. Edited branches,
repeated original prompts, cancellations, custom stop strings, Gemma prompt
normalization, and active DFlash requests take the cold one-shot path instead.

The completion telemetry preserves OpenAI usage semantics while exposing the
optimization directly:

- `promptTokenCount`: complete rendered prompt;
- `cachedPromptTokenCount`: prefix served from KV cache;
- `prefilledPromptTokenCount`: suffix actually evaluated this turn.

This is primarily a time-to-first-token and prompt-energy optimization, not a
decode-tokens-per-second optimization. The full Laguna checkpoint was not
mounted for this change, so the acceptance run remains: identical greedy warm
and cold continuations, nonzero cached tokens on the warm turn, and a measured
TTFT reduction at several transcript lengths.

Run that acceptance probe with:

```bash
swift run -c release model-runner-runtime-bench \
  /absolute/path/Laguna-XS-2.1-MLX-Q4R8 \
  /absolute/path/laguna-prompt-cache.json \
  --engine metal --tokens 256 --warmups 1 --trials 5 \
  --prompt-cache
```

### Compiled per-head attention gate

Every Laguna layer applies an FP32 softplus gate to each attention head before
the output projection. The cast, softplus, cast-back, and broadcast multiply now
share a compiled graph boundary. Laguna alternates head counts, so the final
implementation intentionally uses a shape-specialized graph cache rather than
one shapeless graph; the latter failed the variable-head tiny-model test.

After that correction, two production-shaped release runs measured:

| Run | Eager execution | Compiled execution | Epilogue speedup |
| --- | ---: | ---: | ---: |
| 1 | 0.013 ms | 0.010 ms | 1.315x |
| 2 | 0.011 ms | 0.010 ms | 1.064x |

The deterministic tiny Laguna model produces equal logits with the eager and
compiled paths. This fragment is small, so these numbers must not be read as a
1.06–1.32x whole-model speedup.

Reproduce the fragment check with:

```bash
swift run -c release model-runner-metal-quant-bench \
  --laguna-graph-ab --warmup 20 --queue-depth 64 \
  --queue-rounds 15 --iterations 20
```

The corresponding whole-model alternating check is:

```bash
swift run -c release model-runner-runtime-bench \
  /absolute/path/Laguna-XS-2.1-MLX-Q4R8 \
  /absolute/path/laguna-attention-gate-ab.json \
  --engine metal --tokens 512 --warmups 1 --trials 5 \
  --laguna-attention-gate-ab
```

## Mistral and Mixtral roadmap

No full Mistral checkpoint is currently present on this MacBook, so this audit
makes no Mistral tok/s claim. The source audit still identifies concrete work:

- classic Mistral is currently routed through the upstream Swift `LlamaModel`;
- Mistral 3 uses the upstream `Mistral3TextModel`;
- the local pinned Swift patch restores classic Mistral's checkpoint-declared
  all-sliding or heterogeneous full/sliding cache layout;
- all three Mistral text families use a zero-copy hot conversation session,
  without Mistral branch snapshots or changes to Laguna's branchable LRU;
- both issue separate gate and up projections and separate Q, K, and V
  projections;
- Mixtral uses a `SwitchGLU` expert path plus a pinned single-row Metal
  top-k/gather router in GPU evaluation mode; CPU, training, and multi-token
  prefill keep the eager path. Its weighted reduction remains a
  separate compiled operation.

The implementation order should be:

1. Put one representative dense Mistral and one Mixtral checkpoint on the
   external drive. Record exact model revision, template, context, and format.
2. Establish native Swift and Ollama controls. Record whether Ollama used MLX
   or llama.cpp; a backend mismatch is a best-package comparison, not an
   orchestration comparison.
3. Keep the current projection graph, but retain the implemented
   architecture-metadata and hot-session optimizations. The checked-in graph
   A/B was rerun on MLX 0.32.2:
   compiled-only SwiGLU reached 0.983x current throughput, packed gate/up
   reached 0.722x, and packed QKV reached 0.955x end-to-end. All outputs matched,
   but none clears the performance gate.
4. If a future core or real-checkpoint profile changes the result, implement
   only the winning layout. Preserve independently quantized rows exactly by
   concatenating packed rows, scales, and biases; do not requantize merely to
   fuse execution.
5. Retain the implemented Mixtral decode-router specialization and benchmark
   whether further router/reduction fusion clears the end-to-end gate, with
   selected-logit softmax semantics preserved exactly.
6. Benchmark a real Mistral checkpoint before model-specific tuning. The newer
   GQA attention and wide-GEMV work is present now, but shape probes cannot
   establish full-model prompt or decode throughput.

Reproduce the checkpoint-free Mistral shape tests with:

```bash
Scripts/benchmark-metal-quantization.sh \
  --mistral-graph-ab --warmup 32 --queue-depth 64 --queue-rounds 31
```

## Performance and quality gates

Every proposed change must satisfy the relevant gates:

- same-process alternating order for graph changes;
- at least one warm-up and five 512-token trials per mode;
- exact greedy text match for graph-only transformations;
- logits `allClose` coverage on a tiny deterministic model;
- BF16-teacher KL/NLL and downstream quality checks for quantization changes;
- prompt and decode rates reported separately;
- temperature, power mode, model residency, runtime/core versions, and package
  format recorded with the result;
- long-context and repeated-request runs before calling an optimization stable.

## Primary implementation references

- [Ollama MLX runner](https://github.com/ollama/ollama/blob/v0.33.1/x/mlxrunner/runner.go)
- [Ollama Laguna MLX model](https://github.com/ollama/ollama/blob/v0.33.1/x/models/laguna/laguna.go)
- [MLX Swift 0.32 update](https://github.com/ml-explore/mlx-swift/pull/450)
- [MLX 0.32.2 release](https://github.com/ml-explore/mlx/releases/tag/v0.32.2)
- [MLX: optimize large NVFP4 QMV on M5 Max](https://github.com/ml-explore/mlx/commit/5a1e44c3bb991dab753cee394b0b1d889e2eb9a7)
- [MLX: raise the QMV limit on M5-class GPUs](https://github.com/ml-explore/mlx/commit/e7838d5e386d1159192eb959bee266dd65d27bb3)
- [MLX: optimize quantized MoE matmuls on NAX](https://github.com/ml-explore/mlx/commit/c7ff35d9714c78bcdf4620deb1d189f7ffb7c3b9)
- [MLX: GQA-8 decode attention read optimization](https://github.com/ml-explore/mlx/commit/fa0d4463e4616a0178c500b3e2211ad687f7ff86)
- [MLX core comparison used in this audit](https://github.com/ml-explore/mlx/compare/ce45c52505c8158ea48d2a54e8caae05efd86bfe...1f8e74e3f12f31365464a6867c6579f0e9b29d85)
