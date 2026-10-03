# ScalePlan and AffineScaleSearch for Q4R8

Last updated: 2026-08-28

## Status and boundary

This is a **measured experimental candidate**. Real-weight reconstruction,
artifact integrity, native loading, and matched CUDA throughput have been
measured; end-task model quality has not. It is not a tokens-per-second
optimization.

The implementation preserves this project's established Q4R8 policy:

- affine Q4 group-64 for ordinary quantized weights;
- affine Q8 group-64 for every router and any calibration-selected promotion;
- the existing MLX packed arrays and Metal/CUDA inference kernels;
- one precision and calibration for each fused gate/up quantization unit.

It rejects MXFP4, NVFP4, mixed 3/4/5/6-bit recipes, and other group sizes.
ScaleSearch-NVFP4 motivated the conversion-time search pattern, but that
paper's E4M3-scale/E2M1-weight algorithm is not being represented as an affine
Q4 result.

## AffineScaleSearch-Q4R8

MLX affine quantization stores a signed scale and affine bias for each group,
then reconstructs a packed code as:

```text
reconstructed = code * scale + bias, where code is 0...15
```

For each 64-value Q4 group, the `q4r8_affine_scale_search_ls2` converter:

1. obtains MLX's ordinary packed Q4 result, scale, and bias;
2. treats that result as the mandatory fallback;
3. searches scale factors `0.75...1.25` in `1/16` steps while retaining the
   baseline grid center;
4. rounds each candidate scale and bias through the tensor's stored dtype,
   requantizes to codes `0...15`, and measures exact reconstruction MSE;
5. performs one least-squares bias update for the assigned codes, rounds it
   through the stored dtype, requantizes, and retains it only if exact MSE is
   lower;
6. performs two coordinate-descent slope/intercept fits, each followed by
   stored-dtype rounding, requantization, and an exact-MSE acceptance test;
7. rejects zero-variance fits and slopes whose sign disagrees with MLX's
   baseline scale;
8. replaces a group only when the best final candidate is strictly better
   than ordinary MLX Q4, then writes the ordinary packed weights, scales, and
   biases.

The search is conversion-only. It adds no inference operation, metadata, model
bytes, dispatch, or alternate runtime format. Q8 modules always use standard
affine Q8. Linear and stacked `SwitchLinear` expert weights are searched;
embeddings remain on standard MLX Q4 because the pinned `QuantizedEmbedding`
initializer does not accept precomputed packed arrays.

## Standalone architecture-aware Swift converter

The conversion algorithm is available as a direct Swift executable rather
than only through a Laguna shell workflow:

```bash
MODEL_RUNNER_BUILD_CONFIGURATION=release \
WICK_BUILD_PRODUCT=wick \
../wick/build.sh

../wick/.build/release/wick \
  /absolute/path/unquantized-mlx-compatible-safetensors \
  /absolute/path/q4r8-scale-search-candidate \
  --dry-run
```

For an ordinary-affine-Q4 control with matching storage geometry, run the same
source through the generic converter with a separate destination:

```bash
../wick/.build/release/wick \
  /absolute/path/unquantized-mlx-compatible-safetensors \
  /absolute/path/q4-standard-control \
  --standard-q4
```

This changes only eligible Q4 calibration from ScaleSearch to standard MLX
affine Q4. Group size, architecture sanitizer, mandatory Q8 overrides, output
sharding, and runtime format remain the same. Provenance is kept distinct as
`standard-q4-quantization.json`; Laguna's bounded `--template` path does not
accept this generic-control flag.

The executable reads `model_type` and `architectures`, asks the pinned MLX
Swift LLM or drafter registry to instantiate the real architecture, enumerates
its quantizable leaf modules, and uses the architecture's sanitizer during
conversion. This avoids guessing semantics from safetensors filenames. It
currently provides mandatory router
profiles for Laguna, Mixtral, GPT-OSS, supported Qwen MoE variants, Phi MoE,
MiniMax, and Jamba; callers can add exact or globbed Q8/skip paths. Built-in Q8
router requirements cannot be weakened.

For Mixtral, the profile protects every
`model.layers.*.block_sparse_moe.gate` in Q8 and searches the stacked expert
`SwitchLinear` projections in Q4. A two-layer integration fixture completed an
actual safetensors conversion and verified expert `weight/scales/biases`, Q8
router config overrides, and output shards. Dense registered architectures
such as Mistral and Llama have no mandatory Q8 routers and use searched Q4 for
eligible matrices.

Poolside `DFlashLagunaForCausalLM` checkpoints take a distinct drafter path
despite sharing `model_type: laguna` with the target. The DFlash profile forces
the shared `fc` target-context bottleneck and every tiny
`layers.*.self_attn.g_proj` gate to Q8, then applies affine ScaleSearch Q4 to
the large draft projections. Its model sanitizer splits upstream fused QKV
weights and fuses upstream SwiGLU gate/up weights before quantization, so the
saved names and mixed-precision configuration reload directly in the Swift
runtime. Each fresh artifact is deliberately labeled unbenchmarked at
conversion time until that exact output is paired with a target and measured.

```bash
../wick/.build/release/wick \
  /absolute/path/Laguna-XS-2.1-DFlash-INT4 \
  /absolute/path/Laguna-XS-2.1-DFlash-INT4-MLX-Q4R8-ScaleSearch \
  --dry-run
```

Full-checkpoint validation subsequently confirmed that this path converts and
reloads both Poolside XS drafters. The better-matched BF16-target drafter
shrank from 882 MiB to 259 MiB and retained nearly all measured acceptance
after quantization.
At block size 4 its median exceeded target-only by 7.36% on the RTX 4090, but
intermittent DFlash slow trials made its seven-trial mean 15.99% worse. The
quantizer result is therefore useful enough to retain, not yet safe to select
by default. See the [raw DFlash validation](../benchmark-results/dflash-quantizer-20260829/README.md).

The support boundary is registered MLX Swift text-model and drafter
architectures with unquantized safetensors weights. GGUF, PyTorch `.bin`, already-quantized
checkpoints, unregistered custom models, and VLM-only architectures fail
closed. Other `Quantizable` module classes can still use standard MLX Q4, but
only `Linear` and `SwitchLinear` currently receive ScaleSearch-derived arrays.

Create a complete experimental candidate by adding one flag to the normal
Swift conversion:

```bash
../wick/Scripts/quantize-laguna-q4r8.sh \
  /absolute/path/Laguna-XS-2.1-bf16 \
  /absolute/path/Laguna-XS-2.1-mlx-q4r8-scale-search \
  --q4-scale-search \
  --cpu
```

The converter keeps all mandatory routers in Q8 even when scale search is the
Q4 default.

For a large BF16 source and an existing standard Q4R8 layout, the bounded
rescorer is the practical full-checkpoint path:

```bash
../wick/Scripts/rescore-laguna-q4r8.sh \
  /absolute/path/Laguna-XS-2.1-BF16 \
  /absolute/path/Laguna-XS-2.1-MLX-Q4R8-standard \
  /absolute/path/Laguna-XS-2.1-MLX-Q4R8-ScaleSearch-LS2 \
  --expert-batch 16
```

It uses the standard checkpoint read-only for shard placement and preserved
Q8/unquantized arrays, verifies representative standard Q4/Q8 tensors against
the BF16 source, rescales all eligible Q4 modules on the selected MLX device,
streams four output shards, and refuses an existing destination. The template
does not supply the searched Q4 values.

## Real-checkpoint audit

Before spending the space and conversion time on a complete checkpoint, audit
a deterministic Laguna sample directly from the BF16 safetensors:

```bash
../wick/Scripts/audit-laguna-q4-scale-search.sh \
  /absolute/path/Laguna-XS-2.1-BF16 \
  /absolute/path/laguna-q4-scale-search-audit.json
```

The default sample spans layers 0, 1, 20, and 39; attention; the dense MLP;
shared experts; routed experts 0, 127, and 255; and `lm_head`. Pass repeated
`--tensor` options to replace that sample with exact source checkpoint keys.
`--limit` provides a deterministic smoke-test subset, and `--cpu` keeps all
quantization operations off the GPU.

For every tensor, the JSON report records standard and searched MSE, relative
MSE, changed-group fraction, stored bytes, and conversion time. It fails if
searched MSE increases or if the ordinary affine-Q4 storage size changes. Its
15% field is the research candidate gate, not permission to skip model-level
quality evaluation.

On Linux/CUDA, both the audit and conversion scripts use the same isolated
SwiftPM scratch path and verified CUDA runtime environment as the server. The
converter may still use `--cpu` for a BF16 model larger than VRAM; that affects
conversion placement only, not the resulting Q4R8 checkpoint.

The final LS2 audit sampled 48 real Laguna tensors and 411,041,792 BF16 values
across attention, dense, shared-expert, routed-expert, and output projections.
All 48 tensors improved. Weighted reconstruction MSE fell from
`8.6309273e-6` to `7.3316020e-6`, a **15.0543% reduction**, and 98.0091% of
6,422,528 groups changed. Stored bytes were identical. This passes the
research report's 15% reconstruction gate, but reconstruction MSE remains a
proxy rather than an end-task quality score.

The archived audit is
`laguna-xs-2.1-q4r8-scalesearch-ls2-audit-20260828.json`, SHA-256
`d7981a8d986abae8fcb427c64773024bf3b43aeab7d69e88b70731727bc3c73b`.
On the RTX 4090, searched conversion of the sampled values took about 21.4x
the standard offline quantization time. That cost is paid once and is not in
the inference path.

## Preliminary isolated Metal result

The dedicated A/B uses the same affine-Q4 group-64 `quantizedMM`
specialization for both arrays and warms them together before ABBA timing:

```bash
swift run --package-path ../wick -c release wick-metal-quant-bench \
  --scale-search-only \
  --warmup 8 \
  --iterations 25 \
  --queue-depth 32 \
  --queue-rounds 9
```

On the local M5 Max, a synthetic BF16 normal matrix of shape `6144 x 2048`
produced:

| Metric | Standard affine Q4 | Searched affine Q4 | Observation |
| --- | ---: | ---: | --- |
| Reconstruction MSE | 3.314103e-6 | 2.812670e-6 | 15.13% lower |
| Stored bytes | 7,077,888 | 7,077,888 | identical |
| Warm queued QMV | 0.012 ms | 0.012 ms | 1.034x ratio; noise-level |
| Warm conversion | 0.933 ms | 203.018 ms | 217.6x offline overhead |

This synthetic result proves the implementation can find a lower-error
same-format grid. Together with the real-weight audit, it passes the 15%
reconstruction candidate gate; neither result establishes a Laguna task-quality
gain. The tiny QMV timing difference is noise, not an inference speedup claim:
both arrays dispatch the same kernel and store the same number of bytes.

## Full Laguna LS2 artifact

The completed candidate is:

`/home/example/models/Laguna-XS-2.1-MLX-Q4R8-ScaleSearch-LS2`

It has the same four-shard, 1,517-tensor index and the same 18,821,963,264
tensor payload bytes as the standard Q4R8 artifact. ScaleSearch rescored 399
Q4 modules. Exactly 39 Q8 routers and the standard Q4 embedding remained on
the template path.

Run the backend-independent structural verifier with:

```bash
../wick/Scripts/verify-laguna-q4r8.sh \
  /absolute/path/Laguna-XS-2.1-MLX-Q4R8-standard \
  /absolute/path/Laguna-XS-2.1-MLX-Q4R8-ScaleSearch-LS2 \
  --report /absolute/new/path/q4r8-safetensors-verification.json
```

It parses safetensors headers directly and compares tensor payload ranges. For
the built candidate, all 399/399 searched Q4 modules differ from the standard
template, while all 320 preserved tensors (137,702,912 bytes) and all nine
copied sidecars are byte-for-byte identical. The index is byte-identical. The
artifact's 17-file `SHA256SUMS` manifest hashes to
`59e5d6d2c37439472547a5e7d15b23371efdf97c7b7b08197566bbcd1e85c0bf`.

Native Swift/MLX-CUDA smoke generation passed. A balanced A/B used two
execution orders; each checkpoint received one 256-token warm-up and five
measured 256-token greedy generations. The ten-trial aggregate median was
126.815 tok/s for standard Q4R8 and 125.650 tok/s for LS2, or **-0.919%** for
the searched candidate on this prompt. Both blocks independently showed about
a 0.9-1.0% difference. The storage and kernels are unchanged, but changed
hidden states, generated tokens, and sparse expert choices can still change a
whole-model trajectory. This is measured as a small regression, not advertised
as neutral or as a speedup.

For a single checkpoint, the backend-neutral direct harness records
`LocalModelRunner` metrics without HTTP timing or log parsing:

```bash
Scripts/benchmark-runtime-model.sh \
  /absolute/path/Laguna-XS-2.1-MLX-Q4R8 \
  /absolute/new/path/runtime-benchmark.json \
  --engine metal \
  --tokens 256 \
  --warmups 1 \
  --trials 5
```

Use separate new report paths for standard and searched checkpoints. The CUDA
A/B automation in `Scripts/benchmark-cuda-laguna-ab.sh` additionally balances
server startup order and records both artifacts together.

A standalone run loaded the hash-verified LS2 archive on the local M5 Max and
completed one warm-up plus five measured 256-token generations. All generations
reached exactly 256 tokens with `length` stops; measured decode rates ranged
from 149.90 to 150.23 tok/s with a 150.0238 tok/s median. The raw report's
SHA-256 is
`186ceaa6c99b127a27bbbedc4695abfd1dd6ae1b1d5122b4a5241a3150c97a29`.
Because the exact standard Q4R8 Metal run was deferred, this is a successful
end-to-end runtime measurement, not a candidate-versus-baseline speed claim.

## ScalePlan-Q4R8

`wick-scale-plan` consumes a measured, per-unit JSON ledger and emits
separate decode-first and prefill-first plans. Each candidate records:

- layer path, tensor shape, affine format, bit width, group size, and
  calibration;
- packed payload bits and metadata bytes;
- reconstruction MSE, relative MSE, and optional teacher KL;
- warm decode, verification, and prefill nanoseconds;
- peak bytes, hidden contiguity-copy bytes, and cold compile milliseconds;
- chip, OS, MLX commit, converter commit, and calibration seed.

The quality objective is either a HIGGS-style
`quality_alpha * relative_mse` sum or a measured teacher-KL forecast. By
default, planning stops unless that additive forecast passed a held-out
full-model check. `--allow-unvalidated-proxy` exists only for plumbing work and
does not remove the experimental label.

[The illustrative ledger](../../wick/Examples/scaleplan-q4r8-ledger.example.json)
shows the complete format but contains fabricated costs and an intentionally
unvalidated proxy. It is a schema example, not input for a deployable plan.

The planner performs a multiple-choice dynamic program with conservative byte
and latency quanta. It selects exactly one candidate per unit, filters routers
to Q8, minimizes predicted quality loss under both budgets, and fails closed
if its state safety limit is exceeded. It never truncates the frontier and
pretends the result is optimal.

Generate both profiles:

```bash
swift run --package-path ../wick wick-scale-plan \
  /absolute/path/laguna-measured-ledger.json \
  /absolute/path/laguna-scale-plan.json \
  --max-stored-bytes 18821963264 \
  --max-decode-ns 10000000 \
  --max-prefill-ns 100000000
```

Apply one profile during conversion:

```bash
../wick/Scripts/quantize-laguna-q4r8.sh \
  /absolute/path/Laguna-XS-2.1-bf16 \
  /absolute/path/Laguna-XS-2.1-mlx-scaleplan \
  --scale-plan /absolute/path/laguna-scale-plan.json \
  --scale-plan-profile decode-first \
  --cpu
```

A Q4 ledger candidate may set `calibration` to `standard` or
`q4_affine_scale_search`. Q8 candidates must use `standard`. The Laguna
converter validates every selected path and independently refuses a Q4 router,
so a malformed plan cannot weaken the mandatory router policy.

## Completed and remaining evidence

Completed for the LS2 candidate:

- representative per-tensor/per-group reconstruction distributions on actual
  Laguna BF16 weights;
- exact storage, index, preserved-tensor, and sidecar verification;
- native Swift/CUDA load and coherent generation;
- balanced matched whole-model CUDA prompt/decode measurements.

Still required before calling it quality-validated:

- BF16-teacher KL and held-out forecast error;
- perplexity, coding, tool-calling, instruction-following, retrieval, and
  long-context evaluations;
- answer flips and long-generation stability;
- whole-model Metal A/B and longer CUDA stability measurements;
- peak memory and hidden-copy accounting;
- a decision threshold showing that recovered quality justifies the measured
  approximately 0.9% CUDA decode cost on the current prompt.

Only after those checks should the ledger status move from
`experimental_unbenchmarked` to `measured_candidate` or `validated`.
