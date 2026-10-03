# Native affine quantization challengers — 2026-09-04

The next promising quantizer change is to use input correlations while retaining MLX's existing affine storage. A bounded GPTQ-style probe reduced development projection error substantially, but this is one attention projection, not a complete-model result. G128 standard and LS2 conversion are implemented as separate challengers so grouping and scale search can be measured independently.

## G128 conversion

`model-runner-quantize --template <standard-Q4R8-G64> --source <BF16> --output <new-directory> --group-size 128` applies LS2 to Q4 linear/expert modules. Add `--standard-q4` for native affine quantization without scale search. Both preserve the template's Q8/G64 routers and Q4/G64 embedding exactly. The default remains LS2/G64. Activation-weighted rescoring currently requires G64 and rejects `--standard-q4`.

Run `./prepare-dependencies.sh` before building. The added patch parameterizes the existing public LS2 helper with a default group size of 64; other generic conversion policies retain G64. The bounded Laguna converter validates source geometry and performs the existing exact representative/router identity checks, releases BF16 source arrays after their final consumer is evaluated, and writes the actual tensor-byte total into the index and provenance. Its config explicitly records every module's geometry in both quantization aliases. Native fused gate/up paths inherit the Q4/G128 default after sanitation; explicit router/embedding overrides retain G64.

G64 uses 4.5 bits/weight for affected Q4 tensors with 16-bit scales and biases; G128 uses 4.25. That reduces their tensor bytes by 5.56%, with a smaller whole-checkpoint saving because routers and embedding stay unchanged. It is not a predicted throughput improvement. Compare complete checkpoint bytes, reference likelihood, generated-task quality, and matched prefill/decode timing before promoting it.

Focused fixtures cover standard/LS2 conversion, exact preserved arrays, smaller metadata shapes, native config resolution after fusion, source geometry, byte totals, and invalid activation-stat combinations. Real G128 conversion and model-level evaluation are tracked by the main campaign rather than inferred from these fixtures.

## Bounded GPTQ-style experiment

[The script](../../midnight-quantization/Scripts/probe-laguna-affine-gptq.py) reads only selected source tensor rows. For Laguna layer 0, the input to `q_proj` is exactly RMSNorm of the BF16 token embedding. It samples 128 evenly distributed output rows from the 6144-row projection and retains all 2048 input columns. Calibration uses 4096 tokens from eight local MLX calibration records; development uses 4064 tokens from eight separate WikiText validation records. Token hashes reject exact duplicate full samples and selected prefixes; they do not establish absence of shared substrings or tokens. These are development comparisons; the existing AWSS reference already used this development corpus for its validation gate.

The eight calibration records contain only 756 distinct token inputs, so the empirical 2048-dimensional covariance has rank at most 756 before damping. The 128 output rows follow the structured schedule 0,48,…; they are not a random sample of all model channels. The probe forms the full 2048×2048 Float32 covariance, with 1% diagonal damping, and performs blocked error compensation with 128-column blocks and contiguous G64/G128 groups. Scale/bias parameters are rounded to the source BF16 storage before selecting codes. No activation-order permutation or dynamic group map is introduced. The equations follow the [original GPTQ implementation](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py); this is a native-affine adaptation, not a bit-exact reproduction of that quantizer.

The measured objective is Float32 projection error on the exact stored affine grid, divided by BF16-teacher-weight projection power. It excludes subsequent attention, BF16 matmul rounding, later layers, routing, and generation behavior.

| Method | Group | Development relative output MSE | Selected tensor bytes |
|---|---:|---:|---:|
| Native affine, CPU |64|0.00426506|147,456|
| GPTQ-style native affine, CPU |64|0.00106066|147,456|
| Native affine, CPU |128|0.00424868|139,264|
| GPTQ-style native affine, CPU |128|0.00129298|139,264|
| Existing LS2 checkpoint, converted on GPU |64|0.00361599|147,456|
| Existing AWSS checkpoint, converted on GPU |64|0.00149617|147,456|

GPTQ-style reduced this development proxy by 75.1% at G64 and 69.6% at G128 versus the corresponding native CPU baseline. Weight MSE increased: compensating correlated errors can improve output reconstruction while worsening independent weight error. The GPU-converted references are additional context; native CPU/GPU quantization is not necessarily bit-identical.

The final checked eight-record run used Python 3.14.6, MLX 0.31.2, tokenizers 0.22.2, CPU only, and finished in 2.46 s. Its full covariance occupied 16 MiB; MLX reported 276,287,248 peak bytes and process maximum RSS was 515,096,576 bytes. These are observed peaks for this small probe, not a bound for a full model. The script caps MLX allocation at 2 GiB, selectively reads payloads of at most 32 MiB, validates offsets and finite values, and fingerprints every selected source payload. It does not claim whole-checkpoint identity from those subset hashes.

Six synthetic CPU tests cover protection against overwriting an existing report, source-array lifetime, invalid offsets, zero/constant groups, unobserved inputs, finite positive damping, independent diagonal-covariance behavior, and exact agreement between blocked and unblocked error compensation across two blocks/four groups. Packing is also checked against native affine dequantization during the real run. [Raw evidence and conversion-reference provenance](../benchmark-results/next-priorities-20260904/laguna-affine-gptq-probe/) are archived.

A fixed-parameter 32-record expansion, including the original eight records, used 15,760 calibration tokens with 2,932 distinct inputs and 15,465 development tokens with 3,841 distinct inputs. That removes the structural vocabulary-size bound below 2048, although full empirical rank was not measured. Damping, row selection, grouping and algorithms were unchanged after viewing the initial result. This expansion is not an independent dataset replication.

| Method | Group | Expanded development relative output MSE |
|---|---:|---:|
| Native affine, CPU |64|0.00425157|
| GPTQ-style native affine, CPU |64|0.000907852|
| Native affine, CPU |128|0.00425825|
| GPTQ-style native affine, CPU |128|0.00110160|
| Existing LS2 checkpoint, converted on GPU |64|0.00361161|
| Existing AWSS checkpoint, converted on GPU |64|0.00150530|

The expanded proxy improved by 78.65% at G64 and 74.13% at G128 versus the respective CPU baselines. It completed in 4.41 s, with 1,050,054,672 MLX peak bytes and 861,929,472 maximum RSS bytes. These are aggregate descriptive results from the selected projection, with no confidence interval or population-level accuracy claim. Both reports preserve corpus hashes, sample hashes, token counts, row indices, source payload hashes and fixed damping.

Reproduce with existing local source/corpus paths:

```sh
/opt/homebrew/bin/python3 ../midnight-quantization/Scripts/probe-laguna-affine-gptq.py \
  --source <Laguna-BF16-directory> \
  --calibration <mlx-lm-calibration-v5-awss.jsonl> \
  --development <wikitext-2-raw-dev.jsonl> \
  --reference ls2-g64-gpu=<Laguna-LS2-directory> \
  --reference awss-g64-gpu=<Laguna-AWSS-directory> \
  --device cpu --output <new-report.json>
```

Add `--samples 32` and use another new output path for the expansion. Existing report paths are rejected before model loading and written with exclusive creation.

## Why the official full-model GPTQ path is not the next local run

The inspected [MLX-LM GPTQ source](https://github.com/ml-explore/mlx-lm/blob/32bb4e68791c941db382d6fc8fa5b35ba9f3d98b/mlx_lm/quant/gptq.py) installs catchers on all quantizable linears and gathers all Hessians before quantizing them. On the existing Ministral 14B configuration, seven split linears per block require `40 × (6 × 5120² + 16384²) × 4 = 68,115,496,960` bytes, or 63.44 GiB, for Float32 Hessians alone. BF16 weights and work buffers come in addition. A tiny CPU check on installed MLX 0.31.2 verified the exact catcher expression: although BF16 `x.T @ x` produces BF16, adding it to the initial `mx.array(0.0)` produces and retains a Float32 Hessian. Thus this estimate matches that observed mechanism; it is not a measured full-model memory peak. A different implementation using half-precision Hessians would halve their storage but require separate numerical validation. Fusing gate/up only modestly reduces this cost.

That implementation also accumulates a shared `x.T @ x` for `SwitchLinear`; its catcher does not condition the Hessian on the selected expert IDs. A second tiny check reproduced the group-loop indexing concern: writing `err[...,64:65]` into a local buffer of width 64 silently writes no values on installed MLX. The inspected loop uses that global index in its second group. This isolates a concrete empty-write mechanism, not a complete execution or quality comparison of upstream GPTQ. Validate or correct the group loop before adoption; it should not replace the existing expert-conditioned pipeline without reviewing routing. The [mechanism report](../benchmark-results/next-priorities-20260904/laguna-affine-gptq-probe/official-gptq-mechanism-check.json) records versions and source hash. The practical next implementation is layer-major: compute shared attention covariances once, capture selected-expert input and post-SwiGLU covariances, quantize that block, and discard its statistics. Start with sufficiently observed experts and preserve the existing quantizer for low coverage. Avoid saving every layer's expert Hessians as a giant calibration artifact.

## AutoRound feasibility

AutoRound's [documented native MLX export](https://github.com/intel/auto-round/blob/7d8905efd0de6a29b038b24eeffa2a49c148f72d/docs/step_by_step.md) is experimental and supports mixed per-layer bits/group sizes. It emits native affine storage, so using its rounding optimizer need not require a new MLX inference kernel. Its current source includes a [first-class MPS device implementation](https://github.com/intel/auto-round/blob/7d8905efd0de6a29b038b24eeffa2a49c148f72d/auto_round/utils/device_manager.py#L574), including memory accounting and BF16 capability. The documentation's hardware table is narrower than the source; it would be incorrect to declare MPS unavailable solely from that table.

The [MLX exporter](https://github.com/intel/auto-round/blob/7d8905efd0de6a29b038b24eeffa2a49c148f72d/auto_round/export/export_to_mlx/export.py#L507) converts its learned scale/zero-point representation to packed UInt32 codes plus FP16 scales and affine biases. That stored precision differs from our BF16 metadata; evaluate its final packed grid. Exporting linears does not establish complete Laguna compatibility. Preserve our exact router/embedding policy, audit raw-to-fused module names, and validate native Swift loading. Embedding quantization is not supported by this exporter, so retaining the existing template embedding requires explicit integration.

There is also a concrete runtime integration hazard. The runner's pinned MLX core [promotes affine quantized-matmul inputs and metadata to a common dtype](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/ops.cpp#L4794); [gather_qmm uses the same promotion](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/ops.cpp#L5584). Tiny CPU checks on both MLX 0.31.2 and 0.32.2 confirmed:

| Activations | Affine scales/biases | Dense and gathered output dtype |
|---|---|---|
| BF16 | BF16 | BF16 |
| BF16 | FP16 | FP32 |
| BF16 | FP32 | FP32 |

Simply retaining AutoRound's FP16 metadata in an otherwise BF16 model therefore changes the operation to FP32 under this dispatch policy and may lose the intended BF16 fast path. No GPU latency penalty was measured here. Casting those parameters to BF16 after fitting changes the affine grid, so it requires fresh reconstruction and model-quality checks. If FP16 metadata proves materially better, an explicit mixed-metadata kernel/dispatch path with controlled BF16 activation/output precision becomes a specific MLX optimization candidate. Otherwise, fit and validate on the BF16 grid already used by this runner. [The reproducible dtype check](../benchmark-results/next-priorities-20260904/laguna-affine-gptq-probe/check-affine-metadata-dtype.py) and both versioned results are archived.

Neither a full AutoRound model nor a full GPTQ model was trained or evaluated in this work. The local interpreters have no PyTorch/AutoRound installation. A bounded real-input AutoRound projection/block experiment is the next comparison after environment setup and MPS compatibility verification; loading the entire 62 GiB Laguna BF16 model into a conventional PyTorch calibration flow is not a viable starting assumption.
