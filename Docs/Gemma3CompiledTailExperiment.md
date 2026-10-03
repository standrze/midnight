# Gemma 3 compiled post-attention tail

`MIDNIGHT_GEMMA3_COMPILED_TAIL=1` enables an experimental inference path for
Gemma 3 text models. It is **off by default** and changes no checkpoint tensors
or quantization. Stage 5 Metal checks and a 270M fixed-prefix pilot preserve
the recorded numerical results exactly. Four alternating timing pairs show
no established decode improvement and higher observed TTFT and peak active
memory. The experiment has not earned default promotion.

The initial scope is Metal, batch one, a single cached token, BF16 activations,
and stock affine Q4/G64 `QuantizedLinear` gate/up/down projections with BF16
scales and no global scale. Prefill, other shapes/precisions, training, QLoRA and
custom projection subclasses use the original eager path. The first one-token
prompt is also eager because eligibility snapshots the cache offset before
attention updates it.

## What is compiled

The tail keeps the existing mathematical sequence:

1. Post-attention Gemma RMSNorm, then the attention residual addition.
2. Pre-MLP Gemma RMSNorm.
3. Separate gate/up projections, approximate GELU times up, and down projection.
4. Post-MLP Gemma RMSNorm, then the final residual addition.

Attention output projection, Q/K/V projections, RoPE, cache updates, masks and
SDPA remain outside this graph. The two activation inputs have fixed
`[1, 1, hiddenSize]` shapes. Growing or rotating a cache therefore does not
change the compiled tail's shapes. A new prefill preserves an already warmed
tail for the next decode request.

The pinned compiler fuses elementwise operations, including the GELU/product
chain. Its fusion whitelist does **not** include the RMSNorm or quantized-matmul
primitives. Compiling a norm-plus-residual graph does not mean that those two
operations become one Metal shader. MLX also instantiates nodes from the cached
tape on each invocation; this is not Metal command-buffer replay.

## Parameter, training and ownership safety

The three norm modules and the MLP are explicit `compile(inputs:)` state inputs.
MLX gathers their current arrays on every invocation. `Module.update(parameters:)`
therefore changes the values passed to a warm graph, including normalization
weights and quantized scales/biases; they are not hidden captured constants.
The existing `1 + norm weight` expression is retained inside the body.

An opaque non-`Module` cache holds strong identity snapshots of all three norms,
the MLP, and its three projections. Replacing any of those modules rebuilds the
compiled closure. Tracking the projections catches replacement *within* an
otherwise unchanged MLP. Keeping this record outside MLX module reflection avoids
registering cached children as duplicate parameters. Strong snapshots prevent
an old object's address from being reused as a false identity match.

The closure captures child modules, never the owning decoder layer. Entering
training mode clears the cached closure. Each call also checks child training
modes. QLoRA/custom projections invalidate the cache and execute eagerly, so
adapter enable flags, dropout and other mutable Swift behavior are not frozen
inside a trace. As with the surrounding MLX module API, concurrent parameter
mutation and inference on the same model are not supported.

Independent norm-scale caching is deliberately deferred. Retaining a derived
`1 + weight` without a reliable update generation would risk stale values after
fine-tuning or loading new parameters. This experiment observes current weights
instead of adding that separate optimization.

## Validation and promotion

CPU preparation checks cover the pinned revision, repeated application,
round-trip restoration, compatibility with the existing Gemma 3 layout patch,
partial installations, unexpected source changes, and revision rejection:

```sh
bash Tests/Shell/Gemma3CompiledTailPatchTests.sh
```

The separate `Gemma3CompiledTailTests` Metal suite is opt-in through
`MIDNIGHT_RUN_GEMMA3_COMPILED_TAIL_TESTS=1`. Its six tests pass in the
[repaired stage 5 run](../benchmark-results/gemma-performance-20260929/root-tests-stage5-fixture-repair.log).
Run future checks against a coordinated rebuilt binary when the device is
available. The suite covers:

- Fixed trace counts through sliding-cache rollover and a new request's prefill.
- Warm norm/Q4 parameter updates without stale graph values.
- Projection replacement inside an existing MLP and corresponding retracing.
- Real QLoRA replacement, mutable adapter enable state, and return to stock Q4.
- Training-mode gradients after warm inference, then fresh inference traces.
- Decoder/projection deallocation and unchanged registered parameter keys.

The quality harness accepts `--model gemma3-270m --candidate gemma3-compiled`
in NLL mode. Native reports include `gemma3_compiled_tail_trace_counts` and their
sum, `gemma3_compiled_tail_trace_count`, read through the test SPI after scoring.
The runner rejects a baseline with any traces, a candidate with an untraced
layer, inconsistent totals, missing fields or mismatched runtime flags. These
are compilation counts, not token or kernel counts. `MIDNIGHT_GEMMA3_*` settings
are scrubbed in every arm; only this candidate sets the opt-in flag explicitly.
`MLX_DISABLE_COMPILE` is removed, including a value of `0`, because its mere
presence would bypass compilation while still executing the wrapper body.
Task-mode reports do not yet provide this activation proof, so this candidate
is restricted to NLL. Earlier harness bytes are preserved by SHA256 under
`benchmark-results/gemma-performance-20260929/quality/harness-source-by-sha256/`.

The tiny fixture's approximate parity tolerance is not a model-quality claim.
Use the exact 270M checkpoint and fixed prefixes for NLL/task-quality comparison,
including uncertainty. Different generated tokens alone are not grounds for
rejection: evaluate the quality/speed/memory tradeoff. Measure cold trace latency
separately from warmed alternating decode runs. The observed-state traversal and
identity checks add CPU work, and compiled graph retention can add memory; both
must earn their cost in a full-model measurement.

Command-timing observations can compare late submissions and queue-local gaps.
They cannot establish shader utilization or physical memory bandwidth. Keep Q4
tail, cache and other experiment flags fixed when attributing this graph change.

## Stage 5 measured result

The [fixed-prefix ABBA pilot](../benchmark-results/gemma-performance-20260929/quality/runs/stage5-270m-compiled-tail-pilot/nll-analysis.json)
uses the pinned Gemma 3 270M Q4/G64 checkpoint, 24 prose/code/math records,
up to 128 tokens per record and prefill step one. Both baseline processes
report zero compiled traces; both candidate processes report 18, exactly one
per layer. All four reports have exactly equal per-sample NLL and retained
token diagnostics across 2550 scored positions, with token-weighted NLL
4.635712748. [Fixed-prefix analysis](../benchmark-results/gemma-performance-20260929/quality/runs/stage5-270m-compiled-tail-pilot/fixed-prefix-analysis.json)
finds zero winner changes and zero recorded same-winner logit change;
[within-arm repeats](../benchmark-results/gemma-performance-20260929/quality/runs/stage5-270m-compiled-tail-pilot/repeatability.json)
also match exactly. This covers the retained NLL and winner/runner-up diagnostics,
not every vocabulary logit, all prompts or generated-task quality.

The [runtime screen](../benchmark-results/gemma-performance-20260929/stage5-270m-compiled-tail-four-pair/summary.json)
uses four alternating AB/BA process pairs, one excluded warmup and two measured
256-token greedy trials per process. Exact outputs and retained artifact
identities match across all eight processes. Other experimental flags and
prefix caching are fixed; `MLX_DISABLE_COMPILE` is absent. These are warmed
measurements on one short-prompt workload, not isolated cold trace latency.

| Metric | Eager median | Compiled median | Observed change |
|---|---:|---:|---:|
| Decode | 545.720 tok/s | 546.377 tok/s | +0.1204% |
| TTFT | 6.880 ms | 8.005 ms | +16.3494% |
| Peak active memory | 166.313 MB | 167.249 MB | +0.5628% |

The [paired sensitivity analysis](../benchmark-results/gemma-performance-20260929/stage5-270m-compiled-tail-four-pair/paired-gate-analysis.json)
applies the existing campaign's 10% drift/order gates **after** the screen;
it is not a preregistered acceptance test. Individual decode changes are
−5.2404%, +1.9627%, −0.5617% and +1.3266%. The paired median is +0.3825%,
with a 2000-draw paired-bootstrap 95% interval of [−5.2404%, +1.9627%]. Decode
and memory pass measurement eligibility, which does not establish an
improvement. TTFT fails eligibility because baseline drift is 10.89% and its
AB/BA order effect is 11.06%; the observed +16.35% remains diagnostic. Four
pairs give weak uncertainty estimates and do not remove temporal correlation.

The repaired integrated run reports 528 passing Swift Testing tests and nine
passing XCTest tests, with two CUDA tests skipped. The original
[failed run](../benchmark-results/gemma-performance-20260929/root-tests-stage5.log)
is retained: a projection-replacement fixture's final-logit magnitude assertion
was insensitive to a scale change after normalization. The repair uses a
projection sign reversal and checks the replacement's observable effect.
The six Metal lifecycle tests then pass. None of these results establishes
a quality improvement that could justify extra latency or memory.

## Durable sources

- `Patches/mlx-swift-lm-gemma3-compiled-tail.patch`
- `Scripts/gemma3-compiled-tail-patch.sh`
- `Tests/Shell/Gemma3CompiledTailPatchTests.sh`
- `Tests/ModelRunnerProtocolTests/Gemma3CompiledTailTests.swift`

The preparation call follows `mlx-swift-lm-gemma3-attention-layout.patch` and
accepts only MLX-LM revision `14414441fa44f45eee35a61e9fa0bab577cf9734` on Darwin.
The existing layout overlay remains independently reversible.
