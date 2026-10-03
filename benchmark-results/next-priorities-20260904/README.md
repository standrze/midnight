# Generated quality, G128 and runtime profiling — 2026-09-04

This continuation tests the priorities following v0.2.0-beta.2. Results distinguish
whole-model generated answers, reference likelihood, local projection error and
runtime measurements. These are different objectives.

## Fixed generated evaluation

The question-only suite contains 64 MBPP code tasks (IDs 11–74), 64 GSM8K test
problems and nine seeded key/value retrieval probes. All 64 original MBPP
reference solutions pass the exact scoring tests in the same pinned Python Docker
container used for generated code. No reference implementations or answers enter
code/math generation prompts. The retrieval answer appears in the supplied
context by design.

The model loads once, uses its chat template with a single user message, generates
greedily with a 1,024-token output ceiling, and starts each request without prompt
cache reuse or speculative decoding. Prefill chunks are 512 tokens. Exact prepared
prompt fingerprints, stop reasons, output limits, memory and raw generated text
are retained. The scripts reject silent prompt truncation and incompatible paired
settings. Code is executed only in an unprivileged, network-disabled, resource-
bounded Docker container with read-only input and root filesystem.

Retrieval context budgets are estimates. Report actual prepared token counts;
these tasks land around 4.3K, 17K and 34K tokens on Laguna. Strict exact output is
the primary retrieval score. A correct value repeated around a `</think>` tag is a
format failure under that rule. Optional value-presence diagnostics are kept
separate and are not relabeled as retrieval accuracy.

This is a finite regression suite, not an uncontaminated general benchmark.
Public training overlap is unknown, the code/math references were used in the
previous likelihood evaluation, and MBPP has only the supplied public tests.
Paired bootstrap intervals describe variation over these records; they do not
establish universal rankings. Generated-evaluation timings are diagnostic: CPU
profiling and CPU-only research/scoring work occurred during this campaign.

## Whole-model quality and storage

| Checkpoint | Tensor bytes (GB, decimal) | Code /64 | Math /64 | Strict retrieval /9 | NLL (lower is better) |
|---|---:|---:|---:|---:|---:|
| Standard G64 |18.822|39|61|6|2.435966|
| ScaleSearch G64 |18.822|41|63|7|2.452241|
| Activation-weighted ScaleSearch G64 |18.822|40|61|5|2.426263|
| Standard G128 |17.784|42|61|8|2.482633|
| ScaleSearch G128 |17.784|40|62|6|2.483369|

All five completed the same 137 tasks without prompt truncation. Standard G64
hit the output ceiling on one math task; the other four had no output-limit
stops. The generated-task paired intervals do not establish an overall winner.
NLL uses the same 192 records and 50,096 scored tokens for each checkpoint;
its ordering differs from generated-answer accuracy. These metrics should not
be substituted for one another. Standard G128 minus Standard G64 NLL is +0.04667
(paired 95% interval +0.03873 to +0.05503); ScaleSearch G128 minus G64 is +0.03113
(+0.02255 to +0.03992). Both likelihood regressions are established on this corpus.

Both G128 checkpoints save exactly 1,038,014,464 tensor bytes (5.515%). The
independent auditor verifies every one of 320 retained tensor payloads by SHA256,
1,314 regrouped Q4 tensor headers, tokenizer sidecars, architecture, and both raw
and native fused-module quantization defaults. G128 remains opt-in because its
likelihood is worse here and its generated-task results do not establish a
compensating universal benefit.

Standard G128 conversion took 18.66 seconds with 5,734,290,432 peak MLX bytes;
G128 ScaleSearch took 812.99 seconds with 17,451,614,244 peak MLX bytes. Both
converted 438 Q4 modules and released all 30,513 BF16 source tensors. These are
observed conversion resource figures, not controlled inference speed comparisons.

## Profile-guided command-buffer experiment

The optimized native runner was profiled by attaching Instruments after normal
launch. Direct Instruments launch stalled inside Metal initialization and yielded
only startup samples; it is excluded from decode findings.

A short runtime trace had 3,812 symbolized samples. A longer 20-second selected
interval during real generated tasks had 26,309 symbolized samples. Approximately
67% of the latter sampled CPU weight included `TokenIterator.next()`, and 57%
included MLX asynchronous evaluation. Allocation/free and Metal submission/
completion were prominent leaf costs. Inclusive percentages overlap and are not
wall-time fractions or GPU utilization. This evidence motivates reducing graph
and submission overhead; it does not by itself prove the end-to-end workload is
CPU-bound. A Metal System Trace attempt did not finish finalizing and is excluded
from GPU claims.

The targeted experiment raised `MLX_MAX_OPS_PER_BUFFER` to 200 and
`MLX_MAX_MB_PER_BUFFER` to 256. Four fresh-process, counterbalanced pairs on a
short 512-output-token workload initially yielded a 1.0166 median paired decode
ratio and about 249 MB additional peak MLX allocation. All output strings matched.

A separate four-pair confirmation per workload did not establish a repeatable
benefit. Short-prompt measurements failed the 10% drift/order gates. Longer-prompt
decode had a paired ratio of 1.0750 but an interval spanning 0.9826–1.2071; its
prefill/TTFT measurements failed a drift gate. All outputs still matched. Stock
command-buffer defaults remain unchanged. The first small gain is not promoted
as a production speedup.

## Quantization challengers

G128 conversion regenerates Q4 projections directly from the pinned BF16 source.
Q8/G64 routers and the original Q4/G64 embedding remain byte-identical. Explicit
configuration overrides and actual tensor byte counts are written. Both ordinary
MLX affine and ScaleSearch use the same output policy. G64 activation-weighted
fallback arrays cannot be reused with G128 metadata shapes, so that unsupported
combination is rejected.

The GPTQ-style experiment is deliberately a bounded real-input projection probe,
not a converted model. Its calibrated full covariance can test an optimization
that diagonal weighting cannot represent. Whole-model quality conclusions require
a subsequent bounded converter and generated evaluation; local error reductions
alone are insufficient.

See [the generated evaluation workflow](../../Docs/generated-evaluation.md),
[profiling workflow](../../Docs/runtime-profiling.md), and
[quantization research](../../Docs/mlx-quantization-research-20260904.md).

## G64/G128 runtime checks

Four fresh-process, counterbalanced pairs per quantizer used a fixed short prompt,
512 generated tokens and three warmups. No builds, profiling, other native GPU
jobs or Docker scoring ran during these comparisons. Natural greedy trajectories
can differ between checkpoints, so these are runner measurements rather than
identical-work kernel timings.

| G128 versus G64 | Median peak active MLX: G64 → G128 (GB) | Diagnostic decode ratio | Validity |
|---|---:|---:|---|
| Standard affine |19.029 →18.036|0.7982|Candidate drift failed|
| ScaleSearch |19.029 →18.036|0.9977|Candidate drift and order gates failed|

All prefill and TTFT comparisons also failed a drift or order gate. The diagnostic
decode intervals were 0.6374–1.0339 and 0.8129–1.0454, respectively. No speedup or
slowdown is established. Peak active allocations were about 0.992 GB lower at
G128 in these runs, consistent with its smaller tensor payload; active allocation
is not total process RSS or a guarantee for other workloads. Runtime defaults
remain unchanged.

## KV compression and prefill peak

The three cache modes use the same AWSS G64 checkpoint and nine identical
retrieval prompts.

| KV mode | Strict retrieval /9 | Maximum per-request MLX peak bytes |
|---|---:|---:|
| none |5|20,881,835,488|
| affine8 |5|20,881,842,000|
| turbo8v4 |4|20,881,889,972|

All prompt fingerprints and settings match except the explicitly permitted KV
mode. There were no output-limit stops or prompt truncations. Affine8 minus none
accuracy has a paired 95% interval of −33.33 to +33.33 percentage points;
turbo8v4 minus none is −11.11 points with interval −33.33 to 0. Nine probes cannot
establish equivalence or a broad quality regression. The recorded peak resets
before each request; it is not a retained-cache-size measurement.

Compression is eligible for ten full-attention layers; thirty
sliding-window layers remain uncompressed. In the pinned implementation, Laguna
processes the normal prefill chunks using the original cache arrays. The token
iterator applies compression after the final prompt forward pass. Consequently,
compression cannot eliminate the uncompressed cache peak already reached during
prefill. This source mechanism can explain similar measured high-water marks,
but does not identify exactly which allocation produced each measured peak.

Admission remains conservative: the runner budgets full-precision KV even when
compression is selected, including native sliding windows and prefill workspace.
No compressed-cache admission underbudget was found. Measure memory by phase before changing when compression runs. Moving the current
compression callback into the chunk loop is insufficient: affine cached attention
uses explicit QK scores and softmax, whereas uncompressed attention uses MLXFast
SDPA. At 512 query tokens × 34,000 keys × 48 heads, one BF16 score array alone is
about 1.67 GB (3.34 GB in FP32). That theoretical intermediate could erase cache
savings; it was not measured in this campaign. A blockwise/fused compressed
prefill path, or bounded layerwise dequantization into fast SDPA, needs quality
and peak-memory validation before enabling earlier compression.
[Source review and provenance](../../Docs/kv-prefill-findings-20260904.md) retain
the exact implementation locations and patched dependency hashes.

## Calibration optimizer findings

The expanded GPTQ-style probe used real layer-0 inputs, 128 selected output rows,
15,760 calibration tokens and 15,465 development tokens. Relative output MSE fell
from 0.00425157 to 0.000907852 at G64 (78.65%) and from 0.00425825 to
0.00110160 at G128 (74.13%). This is a selected-projection reconstruction result,
not generated accuracy or a full GPTQ checkpoint. The expansion includes the
original smaller sample and is not an independent replication.

A bounded layer-major converter is the next justified implementation: share
attention covariance within a block, condition expert statistics on routing,
quantize and release each block, and retain a fallback for low-coverage experts.
The inspected upstream all-layer Hessian approach would require 63.44 GiB for
Ministral 14B statistics alone before loading weights, so it cannot simply be
adopted for this 64 GiB machine.

AutoRound's source supports MPS and experimental native MLX export, but its FP16
metadata deserves explicit integration work. With BF16 activations, the inspected
MLX dispatch promotes FP16 scales/biases to FP32 computation; CPU checks on MLX
0.31.2 and 0.32.2 reproduce the output dtype. No GPU slowdown is claimed. Fitting
on the final BF16 affine grid is the first candidate. A mixed-metadata kernel is
worth considering only if retained FP16 metadata delivers meaningful whole-model
quality gains. Neither a full AutoRound model nor a full GPTQ model was evaluated.
See [source-pinned research and probe evidence](../../Docs/quantization-challengers-20260904.md).

## Next implementation decisions

1. Keep native affine storage and the existing G64 default. G128 is an explicit
   memory/quality tradeoff that callers can choose and evaluate for their model.
2. Implement bounded layer-major, expert-conditioned GPTQ on the final BF16
   affine grid. Start with attention and sufficiently covered experts; measure
   complete checkpoint likelihood and generated tasks before expanding coverage.
3. Compare a bounded AutoRound MPS block on the same inputs and packed grid.
   Validate export metadata and native fused-module resolution before attempting
   a complete checkpoint.
4. Extend generated evaluation with more independently selected code/math tasks,
   stronger code tests, and longer/more varied retrieval contexts. Nine synthetic
   probes are a regression check, not a general long-context quality guarantee.
5. Obtain a successful GPU trace and repeat controlled runtime experiments on
   additional workloads. Optimize the dominant native kernel or graph only after
   identifying its contribution to end-to-end latency. The current CPU trace
   does not establish that a new quantization format or an MLX fork would help.

## Validation and retained artifacts

The final optimized test build passed 171 Swift Testing cases in 36 suites and
two XCTest cases, including the enabled NAX/fused-gather GPU regressions and the
new native fused-module policy fixture. All 78 Python tests passed, including
real Docker isolation checks and CPU MLX quantizer tests. All 64 MBPP reference
solutions passed before generated answers were graded. All 30 shell CLI/policy scripts also passed; their logs
and the normal optimized release rebuild are retained under `validation/`.

The archive includes raw native outputs, private scoring keys kept separate from
public prompts, paired analyses, conversion reports, preserved-tensor audits,
full model-hash provenance, runtime schedules/logs, compressed CPU XML exports,
and frozen reproduction scripts. `artifact-index.json` lists SHA256 and byte
counts for every other file in this archive. Absolute paths in raw provenance
identify the original local run; use the documented CLIs with local paths to
reproduce it. Model weights, raw Instruments trace bundles, and the unfinished
Metal trace are deliberately excluded. This is a source prerelease; the local
G128 checkpoints are not GitHub release assets.
