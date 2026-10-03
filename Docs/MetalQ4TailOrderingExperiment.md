# Opt-in Q4 tail kernel: eight-value revision

The current `MLX_METAL_AFFINE_Q4_QMV_TAIL=1` implementation uses **eight values
per lane and 256-column tiles**. It replaces the initial sixteen-value,
512-column experiment while preserving its flags and eligibility: affine
Q4/G64, FP16/BF16, output width divisible by eight, input width at least 512
and divisible by 64 but not 512. `MLX_METAL_AFFINE_Q4_QMV_TAIL_SCOPE` remains
`all`, `dense`, or `gather`. The default is off. No weights, precision, cache
representation, or ordinary aligned fast kernels change.

## Why the variant changed

The original stage3 sixteen-value variant passed primitive numerical tests
but changed fixed-prefix model predictions: 107/2550 scored positions on
Gemma 3 270M and 808/2540 on Gemma 4 A4B. Those differences invalidate exact
preservation claims even where short throughput screens looked favorable.
They do not, by themselves, establish worse task quality. No quality benefit
has been demonstrated for that variant.

Real 270M activation captures localized the initial changes. All 19 changed
output channels among 18 projection calls with identical inputs differ by
exactly one BF16 ULP. Subsequent layers amplify those changes under identical
supplied tokens. At captured position 11, final-head input differs by up to
5.375, logits by up to 2.375, and the top prediction flips with a 0.25 margin in
both arms. See the saved
[CPU amplification analysis](../benchmark-results/gemma-performance-20260929/quality/projection-diagnostics/gemma270-tail-amplification-notes.md).

Real A4B layer 0 expert projections also show sparse changes: 16 of 30 cases are
exactly equal, with 99 differing values out of 1,182,720 overall. At most 31
values differ in one case; the largest cross-arm difference is 0.25. Every
case retains the same maximum error against its independent FP32 oracle.
Those primitive results do not cancel the full-model preservation failure.

The exact stage3 source files, original tail patches, and SHA256 manifest are
preserved under
`benchmark-results/gemma-performance-20260929/stage3-q4-tail-16value-source/`.
The existing stage3 binaries and benchmark evidence remain untouched. Their
results apply to the sixteen-value variant, not the new eight-value revision.

## Execution order

The revised tail template keeps the generic QMV lane mapping and uses the
same typed input sums, `qdot`, FP32 result accumulators, metadata conversions,
and final `simd_sum`. It retains the generic loop condition `k<K-256`, then
executes one separate final tile. Because K is divisible by 64, each active
final lane contains exactly eight valid values. The tail needs one lane
predicate but no clamp, partial-vector zero fill, or safe-helper loops.
Output alignment removes the generic row minimum and row bounds checks.

For K=2816, ten main-loop tiles precede a final full 256-column tile; all 32
lanes participate. Final widths 64/128/192 enable 8/16/24 lanes respectively.
The order is unchanged even when K is divisible by 256. This is distinct from
the old sixteen-value scheme's 512-column distribution and affine cancellation
grouping. No input-sum widening, weight pre-dequantization, split accumulators,
or combined adjacent groups are introduced.

Compiled and JIT pipelines use the new `_tail_g8` kernel-name suffix. This
makes the variant visible in captures and separates it from older tail
pipeline identities. The shader and host sources must be rebuilt together.
Source-level arithmetic order is preserved. The stage 4 GPU fixtures below
check compiled behavior, including compiler reassociation, FMA contraction
and safe-helper replacement, on the measured inputs.

The expected savings are narrower than the old variant's: row/tail addressing
and dynamic safe-loop overhead. Metadata-load frequency remains the same as
generic QMV. A short A4B screen shows an approximately 9% gain. The subsequent
longer paired recheck fails the existing drift and order-effect gates, so a
repeatable throughput improvement has not been established.

## Durable preparation and validation

The ordered patch stack is specialization, original tail, then:

- `Patches/mlx-affine-q4-qmv-tail-order.patch`
- `Patches/mlx-swift-affine-q4-qmv-tail-order-jit.patch`

`Scripts/affine-q4-qmv-tail-patch.sh` owns all three overlays per checkout. It
copies affected files, peels exact applied overlays in reverse order, and
replays the full stack before modifying either repository. Clean, base-only,
old sixteen-value, current eight-value, and partially upgraded source/generated
states are handled. Unexpected revisions and source drift fail without
mutating either checkout. The existing dependency preparation call is unchanged.

CPU checks passed on September 29, 2026:

```bash
bash Tests/Shell/MLXAffineQ4QMVTailPatchTests.sh
python3 -m unittest discover -s Tests/Python -p test_q4_tail_lane_order.py
bash Tests/Shell/MLXAffineQ4QMVSpecializationPatchTests.sh
```

These cover patch replay/idempotence/conflict handling, address-sanitized
extracted kernel control flow, strict selector boundaries, core/generated/JIT
consistency, and symbolic equality of ordered affine contributions for every
eligible width 576–8192. Mutation cases detect changed tile order, sixteen-value
grouping, permuted SIMD lanes, and wrong metadata indexing even when channel
coverage remains complete. Original arithmetic helpers and generic QMV source
are checked against the pinned revision.

The coordinated stage 4 native and Metal rebuild passes the full suite:
522 Swift Testing tests and nine XCTest tests, with two CUDA tests skipped.
Separate baseline/candidate GPU processes then produce exact outputs across
all 132,306 synthetic values and 1,182,720 real-A4B-weight fixture values.
The captured 16-token 270M projection trace also has zero differing eligible
projection calls. These finite fixtures do not establish exactness for every
future input or shape.

The [A4B whole-model fixed-prefix comparison](../benchmark-results/gemma-performance-20260929/quality/runs/stage4-a4b-g8-fixed-prefix/nll-analysis.json)
uses `--prefill-step-size 1`: 24 records, 2540 scored positions, 24 tied record
losses, and exactly equal pooled NLL 6.612685821938703. The companion
[winner analysis](../benchmark-results/gemma-performance-20260929/quality/runs/stage4-a4b-g8-fixed-prefix/fixed-prefix-analysis.json)
reports zero winner changes and zero maximum change to the recorded winning
logit. This is a single A/B fixed-input comparison, not a task-accuracy result
or proof about all logits.

The [31B fixed-prefix comparison](../benchmark-results/gemma-performance-20260929/quality/runs/stage4-31b-g8-fixed-prefix/nll-analysis.json)
also matches exactly at prefill step one: 24 records, 2540 positions, 24 tied
record losses and pooled NLL 8.965165077419732 in each arm. Its
[winner analysis](../benchmark-results/gemma-performance-20260929/quality/runs/stage4-31b-g8-fixed-prefix/fixed-prefix-analysis.json)
records zero flips and zero maximum same-winner logit change. Thirteen records
reach the 128-token sample cap. This is one A/B pair on the convenience corpus;
it supplies neither repeatability evidence nor generated-task accuracy, and the
retained winner diagnostics do not compare the full vocabulary.

Fresh-process ABBA screens use one warmup and two measured 256-token greedy
generations per process, with all other experimental paths disabled:

| Model and retained report | Baseline | Eight-value tail | Change |
|---|---:|---:|---:|
| [Gemma 4 A4B](../benchmark-results/gemma-performance-20260929/stage4-a4b-g8/summary.json) | 134.28 tok/s | 146.35 tok/s | +8.99% |
| [Gemma 3 270M](../benchmark-results/gemma-performance-20260929/stage4-270m-g8/summary.json) | 561.33 tok/s | 557.91 tok/s | −0.61% |

Both preserve exact generated output and artifact identities. The A4B result
is promising for that prompt; it is not a broad speed guarantee or a 270M
improvement.

The longer follow-up retains two measured trials after one warmup in each
fresh process. A4B has four alternating AB/BA pairs and 512 output tokens;
31B has two pairs and 256 tokens. Both preserve exact output. Applying the
existing campaign `metric_summary` after the run gives:

| Model and analysis | Diagnostic median paired change | Pair bootstrap 95% interval | Baseline / candidate drift | AB/BA order effect | Gate result |
|---|---:|---:|---:|---:|---|
| [A4B](../benchmark-results/gemma-performance-20260929/stage4-a4b-g8-four-pair/paired-gate-analysis.json) | +8.10% | −8.67% to +9.97% | 21.91% / 27.84% | 10.10% | Rejected |
| [31B](../benchmark-results/gemma-performance-20260929/stage4-31b-g8/paired-gate-analysis.json) | +16.58% | +2.81% to +30.35% | 27.42% / 0.51% | 26.78% | Rejected; two pairs only |

The drift/order threshold is 10%; accepted estimates and intervals are null.
A4B's four adjacent-pair changes are +6.31%, +9.88%, −8.67% and +9.97%. Its
unpaired pooled medians, 85.00 versus 85.51 tok/s, hide the large absolute
drift and should not be substituted for paired analysis. These gates were
applied after the screen, not preregistered for it. Four or fewer pairs provide
weak uncertainty estimates, especially with serial correlation.

[Background activity and memory observations](../benchmark-results/gemma-performance-20260929/stage4-timing-drift-observations.md)
are context, not a demonstrated cause of the variation. No unrelated process
was stopped. Repeatable throughput improvement remains unestablished; keep
the default off while preserving the positive numerical evidence.
