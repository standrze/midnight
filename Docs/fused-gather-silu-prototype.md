# Fused affine-Q4 gathered gate/up prototype

This opt-in synthetic benchmark evaluates one Metal kernel for Laguna's
256-expert, top-8, input-width-2048, hidden-width-512 affine-Q4/group-64 graph.
The same kernel source and wrapper are used by `model-runner-metal-quant-bench`
and a narrowly scoped runner experiment. Default execution remains unchanged;
`MODEL_RUNNER_LAGUNA_FUSED_GATHER_SILU=1` opts Metal inference into the experiment.
The public `LagunaRuntimeTuning.useFusedGateUpSilu` optional override takes precedence,
so a same-loaded control can explicitly disable it even when the environment is set.

The baseline is the existing packed `gatherQuantizedMM` followed by
`compiledSiluProduct`. The prototype computes two gate/up pairs per SIMDgroup,
retains four FP32 accumulators (matching the stock four-output QMV tile), rounds
the projections to BF16 as the baseline does, and writes only the activated
512-element hidden vector per selected expert. It can save the intermediate
1024-element gate/up write/read and a dispatch. Whether that offsets register,
occupancy, and compiler effects must be measured.

The dot-product input scaling and packed-nibble decomposition follow MLX's
MIT-licensed `qmv_fast_impl`, `load_vector`, and `qdot` at pinned MLX revision
`1f8e74e3f12f31365464a6867c6579f0e9b29d85`. This is a shape-specific experiment,
not a replacement for generic quantized matrix multiplication. The model still
needs its downstream expert down-projection and weighted reduction.

After building the Metal benchmark in release mode, run the two order controls
in separate processes with other GPU work stopped:

```sh
.build/release/model-runner-metal-quant-bench \
  --fused-gather-silu-ab --warmup 16 --iterations 20 \
  --queue-depth 32 --queue-rounds 9 \
  --fused-gather-silu-output /private/tmp/fused-gather-silu-ab.json

.build/release/model-runner-metal-quant-bench \
  --fused-gather-silu-ab --fused-gather-silu-candidate-first \
  --warmup 16 --iterations 20 --queue-depth 32 --queue-rounds 9 \
  --fused-gather-silu-output /private/tmp/fused-gather-silu-ba.json
```

The JSON contains both first-call latencies (including JIT), every paired warm
synchronized trial, and alternating queued timings. It separates graph-building
plus evaluation time from evaluation time, retains/evaluates every output, and
uses distinct inputs within each queue so repeated lazy expressions cannot be
removed. First-call comparisons remain startup/order sensitive and must be
reported separately from warmed performance.

Before timing, eight deterministic fixtures compare the prototype with both
the stock BF16 pipeline and a dequantized FP32 matmul/SiLU oracle. Selections
include expert IDs 0 and 255, unsorted indices, and duplicate indices. The report
includes baseline/candidate full-output hashes, maximum absolute differences,
RMS difference, each path's FP32-reference error, and a separate untimed kernel's
projection-stage parity to isolate matmul errors from activation errors. The initial prototype gate
is `allClose(rtol: 0.01, atol: 0.02)` versus the stock path, plus finite oracle
error. This does not imply token-exact model equivalence. A failed gate emits
`numerical_gate_failed`, skips timed trials, and exits nonzero.

Promotion requires a repeatable primitive benefit, stricter full-model numerical
and generated-output checks, and an end-to-end workload win. These results alone
cannot establish a decode speedup, DFlash benefit, or reduced energy per token.

The runner experiment requires BF16 activations, 256 experts, top-8 routing,
K=2048, hidden=512, and affine Q4/group-64 gate/up tensors with no learned bias.
Packed weights and quantization metadata are referenced directly during tracing.
MLX ensures contiguous inputs at dispatch, copying unusual strided views when
necessary. The original down projection and residual path are retained. A separate
compiled decode closure prevents the first A/B arm from freezing the other arm's
TaskLocal selection. Prefill, multirow verification, hidden-state capture,
calibration, training, unsupported quantization, and CPU execution use stock paths.
No diagnostic projection buffers are allocated by the production primitive.

The first v2 primitive runs matched projection and final output hashes exactly
on all eight fixtures in both process orders. Queued evaluation speedups were
1.247x and 1.401x; synchronized speedups were 1.005x and 1.018x. These are primitive
results on the tested M5 Max, not an end-to-end decode improvement. Raw reports:
`/private/tmp/midnight-optimization-20260904/fused-gather-v2-ab.json` and
`/private/tmp/midnight-optimization-20260904/fused-gather-v2-ba.json`.

Run the opt-in full routed-projection regression separately from other GPU work:

```sh
MIDNIGHT_RUN_FUSED_GATHER_REGRESSION=1 swift test -c release \
  --filter LagunaFusedGatherSiluParityTests
```

The ordinary `LagunaFusedGatherSiluEligibilityTests` exercise fallback metadata
without evaluating large arrays. The Metal regression checks bit-exact output
through the original down projection, strided activation/index inputs,
broadcast packed weights, distinct expert scaling, and duplicate expert IDs.
Use the runtime benchmark's `--laguna-gather-silu-ab` mode for the same-loaded
end-to-end gate; it requires complete checkpoint coverage before measuring.
Do not enable this by default without generated-token, numerical-quality, and
end-to-end latency/throughput evidence across representative prompts.
