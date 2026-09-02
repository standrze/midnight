# Laguna Q4R8 for MLX and Metal

Last updated: 2026-08-29

## Decision

The deployment format for Laguna on Apple Silicon is:

- MLX affine Q4, group size 64, as the default weight format;
- MLX affine Q8, group size 64, only for calibration-selected modules;
- every MoE router in Q8;
- one precision for every fused gate/up pair, including dense, shared, and
  routed experts.

This policy is called **Q4R8**: Q4 weights with Q8 routers. Q4R8 is this
project's concise name for that mixed-precision policy, not a separate MLX
quantization mode. Formats that average roughly four bits by mixing 3, 4, 5,
and 6-bit tensors are useful sensitivity evidence, but they are not a
deployment candidate for this project.

The first custom quantization work should improve how optional Q8 promotions
and Q4 grids are derived. It should retain MLX's existing packed affine layout and
Metal kernels. A new physical bit layout is a later option only if profiling
shows a limit that cannot be removed with the native format.

## Why affine Q4 group-64 wins today

MLX packs eight Q4 values into each `UInt32` and provides native Metal QMV,
QMM, and gather-QMM kernels. Laguna decode selects 8 of 256 experts, so its hot
MoE path is gather-QMV rather than a dense matrix multiplication.

The project benchmark exercises Laguna-shaped dense and selected-expert
operations:

```bash
Scripts/benchmark-metal-quantization.sh \
  --warmup 8 --iterations 40 --queue-depth 32 --queue-rounds 9
```

On the local 40-GPU-core M5 Max, representative queued selected-expert results
were:

| Format | Fused gate/up | Down projection |
| --- | ---: | ---: |
| affine Q4 group-64 | 0.029 ms | 0.016 ms |
| affine Q4 group-128 | 0.026 ms | 0.015 ms |
| affine Q8 group-64 | 0.042 ms | 0.025 ms |
| MXFP4 group-32 | 0.028 ms | 0.015 ms |
| NVFP4 group-16 | 0.026 ms | 0.015 ms |

The sub-0.1-ms operations have normal run-to-run variance, so the small
differences among the Q4 formats are not evidence of a full-model win. Q8 is
consistently slower on the hot expert path because it moves twice as many
weight bits. Group-128 also gives up quantization granularity, while group-64
is already the checkpoint's established accuracy/performance point.

Poolside's NVFP4 quality results show that NVFP4 is a credible NVIDIA
deployment format. They do not establish a Metal advantage. MLX has an NVFP4
kernel, but Apple GPUs do not gain Blackwell's NVFP4 tensor-core behavior. The
interleaved format test has now been repeated on MLX 0.32.2, including its
large-NVFP4 QMV optimization for M5 Max. NVFP4 reached 0.939x affine-Q4
throughput for dense QMV, 0.999x for gathered expert gate/up, and 1.018x for
gathered expert down. That mixed microbenchmark is not a reason to replace the
accuracy-first Q4R8 checkpoint without a quality-matched full-model A/B.

## Laguna's Q8 bandwidth budget

The stock MLX Q4 checkpoint contains 33.44B logical parameters and 18.82 GB of
physical tensor data. It already follows the Q4R8 policy: the 39 router
projections are Q8 and the other quantized modules are Q4.

At one-token decode, the model reads approximately 1.59 GB of active quantized
weights per token before cache effects. Promoting a module from affine Q4 to
Q8 adds one-half byte per weight; scale and bias metadata are unchanged at the
same group size.

| Possible Q8 promotion | Extra checkpoint data | Approx. extra active weight data per token |
| --- | ---: | ---: |
| all routed experts | 15.70 GB | 490.7 MB |
| routed fused gate/up only | 10.47 GB | 327.2 MB |
| routed down only | 5.23 GB | 163.6 MB |
| all attention projections | 715.5 MB | 715.5 MB |
| attention Q and O only | 629.1 MB | 629.1 MB |
| attention K and V only | 83.9 MB | 83.9 MB |
| all shared-expert projections | 61.3 MB | 61.3 MB |
| LM head | 102.8 MB | 102.8 MB |
| token embedding | 102.8 MB | approximately one selected row |
| all routers, relative to all-Q4 | 10.2 MB | 10.2 MB |

This is why “make the important layers Q8” is not specific enough. Q8 for all
attention or all routed experts would consume most of the Q4 speed advantage.
The published same-hardware checkpoint figures point in the same direction:
126.0 tok/s for Q4 versus 95.4 tok/s for Q8 at a 1K prompt.

## Accuracy-first selection rule

The Q8 allowlist should come from a BF16-teacher calibration pass, not from a
hard-coded model-family guess:

1. Quantize every eligible module to affine Q4 group-64, with routers forced to
   Q8.
2. Measure next-token KL divergence between BF16 and Q4 over code, agent/tool,
   reasoning, prose, and long-context calibration inputs.
3. Measure the reduction in KL when each quantization unit is restored to Q8.
4. Rank promotions by quality recovered per active byte, with a separate
   checkpoint-size ceiling.
5. Evaluate candidate allowlists on perplexity plus Laguna-relevant coding,
   tool-calling, instruction-following, and retrieval tasks.
6. Benchmark tokens per second and long-generation stability in this runner.

The routed gate and up projections are one quantization unit. The runner fuses
them into one selected-expert Metal operation, so assigning different bit
widths would produce incompatible packed row widths and lose the optimized
path. Dense and shared-expert gate/up pairs are likewise indivisible. The
native converter canonicalizes either split name to `gate_up_proj`, and the
compact checkpoint packer rejects conflicting routed gate/up settings.

AWQ and GPTQ are plausible ways to derive more accurate Q4 weights while still
writing the ordinary MLX affine layout. They are calibration algorithms, not a
reason to add a new runtime format. Laguna's sparse experts need model-specific
calibration support; generic dense-layer AWQ code does not cover the routed MoE
path correctly. Until that implementation and its evaluations exist, the
converter uses MLX's standard affine grid and makes no unmeasured accuracy
claim.

The runner now also contains an experimental affine-Q4 scale search and a
Q4R8-only ScalePlan planner. Both preserve group-64 affine storage and existing
runtime kernels; neither is a validated quality claim. See
[ScalePlan and AffineScaleSearch for Q4R8](scaleplan-q4r8.md) for the exact
algorithm, preliminary isolated measurements, commands, and remaining gates.

## Native Swift Q4R8 converter

The runner includes a Swift/MLX converter for the original unquantized
Poolside checkpoint. Python is not used by the converter or by inference. It:

- reads the official per-expert BF16 layout;
- stacks 256 experts into MLX `SwitchLinear` tensors;
- fuses dense, shared-expert, and routed gate/up before quantization;
- emits affine Q4 group-64 by default;
- forces all router projections to Q8;
- accepts an exact Q8 allowlist and writes normal per-module MLX config
  overrides.

The unified `model-runner-quantize` executable keeps these Laguna-specific
rules rather than replacing them with filename heuristics. With `--template`,
it dispatches in-process to the bounded Laguna rescorer: the existing fused
gate/up ordering, expert batching, mandatory Q8 routers, standard-Q4 embedding,
template identity checks, and shard layout are preserved. Without a template,
the original `model-runner-laguna-quantize` executable remains available for
policy and ScalePlan-driven full-model conversion.

Inspect the mandatory policy without touching weights:

```bash
Scripts/quantize-laguna-q4r8.sh \
  /absolute/path/Laguna-XS-2.1-bf16 \
  /absolute/path/Laguna-XS-2.1-mlx-q4r8 \
  --dry-run
```

An allowlist file has this shape:

```json
{
  "format": 1,
  "q8_modules": [
    "language_model.lm_head"
  ]
}
```

Create the checkpoint after the allowlist has been calibrated:

```bash
Scripts/quantize-laguna-q4r8.sh \
  /absolute/path/Laguna-XS-2.1-bf16 \
  /absolute/path/Laguna-XS-2.1-mlx-q4r8 \
  --policy /absolute/path/laguna-q8-policy.json
```

The source must be unquantized safetensors. The current converter is not a
chunked low-memory converter; a 62 GB BF16 source requires substantial memory
headroom while the quantized graph and output are materialized. On a discrete
GPU host whose VRAM is smaller than the source, use `--cpu` so conversion uses
system RAM while preserving exactly the same MLX affine output:

```bash
Scripts/quantize-laguna-q4r8.sh \
  /absolute/path/Laguna-XS-2.1-bf16 \
  /absolute/path/Laguna-XS-2.1-mlx-q4r8 \
  --cpu
```

This flag affects offline conversion only. Serving the result still uses Metal
or CUDA normally.

## Built artifact and matched runtime validation

The converter has completed against the 62 GB Poolside BF16 checkpoint on the
project's 128 GB Ubuntu/RTX 4090 host. The deployed baseline is:

`/home/example/models/Laguna-XS-2.1-MLX-Q4R8-v1`

It contains 1,517 tensors in four safetensors shards, with 18,821,963,264 tensor
bytes. Its policy is affine Q4 group-64 everywhere eligible except exactly 39
affine Q8 group-64 router projections. The routed expert gate/up tensors are
already fused.

A matched A/B used one 256-token warm-up followed by five 256-token greedy
trials on the same Swift/MLX-CUDA binary. Median decode was 126.93 tok/s for the
new BF16-derived artifact and 127.45 tok/s for the prior compact Q4 checkpoint,
a 0.4% difference inside run-to-run variance. The artifacts have the same
1,517-key fused layout and effectively identical tensor size; the converter did
not introduce a runtime packing penalty.

This validates format and throughput, not model quality. No additional Q8
module has been promoted beyond the mandatory routers because the BF16-teacher
calibration described above has not yet been run. That is deliberate: the
baseline makes no unmeasured claim that a guessed Q8 allowlist improves
accuracy.

### Affine ScaleSearch LS2 candidate

The project also produced
`/home/example/models/Laguna-XS-2.1-MLX-Q4R8-ScaleSearch-LS2` by rescoring
the standard artifact's 399 Q4 modules against the original BF16 weights. For
each group it searches nine nearby scales, performs one least-squares bias
update and two joint affine fits, and retains only strict exact-MSE
improvements after stored-dtype rounding and requantization. It writes the same
ordinary MLX affine Q4 group-64 format; routers remain standard affine Q8.

A representative 48-tensor audit spanning 411,041,792 source values reduced
weighted reconstruction MSE by 15.0543%, from `8.6309273e-6` to
`7.3316020e-6`. A direct safetensors verifier found all 399 searched modules
changed, while all 320 preserved tensors (137,702,912 bytes), all copied
sidecars, and the index remained exactly identical.

Two matched execution orders gave ten measured 256-token trials per artifact.
Aggregate median decode was 126.815 tok/s standard versus 125.650 tok/s
searched, a 0.919% regression for this prompt. The format and kernels are the
same, but changed model trajectories and sparse expert choices can affect
whole-model timing. ScaleSearch LS2 is therefore an accuracy candidate, not a
throughput optimization. BF16-teacher KL, perplexity, coding, instruction, and
long-context evaluation are still required before preferring it for deployment.
A standalone local M5 Max check of the verified LS2 archive measured 150.0238
tok/s median across five 256-token trials, but the exact standard-checkpoint
Metal A/B was deferred; that number is not evidence of a relative Metal gain.
See [ScalePlan and AffineScaleSearch for Q4R8](scaleplan-q4r8.md) for commands,
integrity evidence, and the remaining validation gates.

## Fused always-active gate/up execution

Laguna has one dense first-layer MLP and 39 always-active shared experts in its
sparse layers. They previously issued separate affine-Q4 matrix-vector calls
for `gate_proj` and `up_proj`. The runner now concatenates their packed output
rows, scales, and affine biases at load time and executes one `gate_up_proj`
call followed by a split. Concatenation is exact because MLX quantizes each
output row independently; existing Q4R8 checkpoints are not requantized and
their weight-bit memory does not increase. This removes 40 quantized dispatches
per generated token.

The benchmark checks the fused and separate outputs with `allClose` before it
records timings. With affine Q4 group-64 and Laguna's 2,048-to-512 shared expert
shape, matched queued-operation measurements were:

| Backend | Separate gate/up | Fused gate/up | Isolated speedup |
| --- | ---: | ---: | ---: |
| M5 Max / Metal | 0.008 ms | 0.007 ms | 1.174x |
| RTX 4090 / MLX-CUDA | 0.007 ms | 0.006 ms | 1.166x |

The CUDA primitive win did not become a measurable whole-model win. With the
same Q4R8 artifact and request, one 512-token warm-up followed by three
512-token greedy trials produced a 124.82 tok/s pre-change median and a 124.61
tok/s post-change median. The -0.17% difference is run noise. The optimization
is retained because it is exact, removes redundant dispatches, improves the
isolated operation on both backends, and targets Metal without regressing CUDA;
it should not be advertised as a CUDA end-to-end speedup.

## MacBook Ollama comparison

The earlier 155.94 tok/s Ollama versus 124.27 tok/s Swift result is retained as
historical evidence, not the current performance conclusion. The audit found
that Ollama's Laguna subprocess also uses MLX and Metal. At the time it shipped
a newer MLX core, an NVFP4 package, and a more explicitly fused Laguna graph,
so that run did not isolate frontend overhead.

The graph gap has now been addressed directly. In a same-loaded-model,
alternating 5x512-token A/B, compiling Laguna's top-8 router and fusing the MoE
weighted reduction, routed scale, shared expert, and residual improved median
decode from 139.46 to 151.17 tok/s, or **8.40%**. Every generated output matched
exactly; median prompt processing improved 2.63%. The fused path is enabled by
default and can be retested with
`model-runner-runtime-bench --laguna-fusion-ab`.

The backend gap has also been addressed. macOS now pins official MLX Swift
revision `72f3c3a`, MLX 0.32.2 core revision `1f8e74e`, and MLX-C revision
`c74db53`; this core is newer than the `c793734` revision observed in Ollama.
A balanced old → new → new → old Q4R8 sequence put the new backend in
the same 149–152 tok/s observed class before sustained heat collapsed the final
old-core block. It proves parity, not a precise backend-only speedup. The
Laguna fusion remained exact on the new backend and measured +5.80% in a
thermally stressed alternating rerun.

Standalone runtime controls remain too state-sensitive to name a durable
winner. Two fresh Ollama 0.33.1 five-trial controls on this MacBook measured
132.96 and 149.27 tok/s medians with the same request, while a subsequent
standalone fused Swift run measured 116.25 tok/s after the machine had been
under sustained load. These swings are much larger than the residual runtime
difference. Cross-process comparisons now require recorded thermal/power state
and bracketed runs; same-process alternating A/B is the acceptance test for
internal graph changes.

An exact-artifact comparison was attempted through Ollama's experimental local
safetensors import, which reported that it preserved source quantization. The
current Ollama Laguna MLX path did not execute this Q4R8 checkpoint: the two
native namespace variants failed on the MoE router or LM-head names, and a
metadata-only router-name compatibility shim progressed to an MLX runtime
index panic. The temporary imported model and compatibility clone were removed.
Until Ollama can execute the artifact unchanged, none of the best-package
numbers should be interpreted as proving a frontend-orchestration advantage.
See [MacBook MLX performance](macbook-mlx-performance.md) for the current
version audit, controlled results, and ranked Laguna/Mistral roadmap.

## Measured Metal runtime improvement

Quantization is only one part of decode. The Swift token iterator previously
synchronized each Metal decode step too aggressively. The dependency patch now
queues the next token asynchronously on macOS while retaining a strict
token-plus-cache realization boundary on Linux/CUDA, where the asynchronous
cache mutation path had caused corruption.

With the stock Q4 checkpoint, a roughly 990-token prompt, and 100 generated
tokens on the local M5 Max:

| Runtime | Median decode |
| --- | ---: |
| Swift runner before scheduling patch | 115.99 tok/s |
| Swift runner after scheduling patch | 124.70 tok/s |
| same-machine `mlx-vlm` control | 135.99 tok/s |

The implemented Metal change is a 7.5% median improvement. A 512-token
generation completed at 120.99 tok/s without corruption. The remaining
gap to the Python-fronted MLX reference was a runtime graph/integration target,
not evidence that Python executes the model faster; both paths execute the
model through MLX and Metal.

The subsequent compiled-MoE change removed another verified graph gap. It
compiles Laguna's sigmoid top-8 router and folds the routed reduction, 2.5x
scale, shared expert, and residual into one compiled fragment. The 5x512-token
alternating full-model benchmark measured 139.46 tok/s unfused and 151.17 tok/s
fused, an 8.40% median improvement with exact generated-text agreement. A tiny
native Laguna regression also checks fused and unfused logits with `allClose`.

On Linux, short-lived Swift concurrency tasks can finish on a thread other than
the one that created mlx-swift's global stream. The runner therefore backports
an explicit terminal cleanup that releases both the calling thread's encoders
and the global cross-thread encoders before CUDA unload. A loopback inference
followed by SIGTERM now exits with status 0 instead of aborting from a late
`cudaStreamSynchronize` during static destruction.

## Primary references

- [MLX quantization modes and packing](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
- [Poolside Laguna XS 2.1](https://huggingface.co/poolside/Laguna-XS-2.1)
- [MLX Laguna Q4 checkpoint and performance](https://huggingface.co/mlx-community/Laguna-XS-2.1-4bit)
- [Poolside Laguna NVFP4 quality results](https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4)
- [MLX Swift 0.32 update](https://github.com/ml-explore/mlx-swift/pull/450)
- [MLX 0.32.2 release](https://github.com/ml-explore/mlx/releases/tag/v0.32.2)
- [Ollama safetensors import](https://docs.ollama.com/import)
- [AWQ paper](https://arxiv.org/abs/2306.00978)
- [GPTQ paper](https://arxiv.org/abs/2210.17323)
