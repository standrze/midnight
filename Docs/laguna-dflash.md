# Native Laguna DFlash

The runner implements Poolside's `Laguna-XS-2.1-DFlash` checkpoint directly
in MLX Swift. It does not require a Python sidecar and it does not change the
target model's Metal kernels. DFlash reduces the number of expensive 40-layer
target forwards by proposing a block with a five-layer draft model and
verifying that block in one target forward.

## Supported pairing

The implementation validates the checkpoint/target contract at startup:

| Property | Laguna-XS-2.1 DFlash |
| --- | --- |
| Target depth | 40 layers |
| Target hidden taps | `[1, 13, 25, 33, 39]` (zero-based, post-layer) |
| Draft depth | 5 dense layers |
| Draft hidden / FFN | 2,048 / 8,192 |
| Query / KV heads | 64 / 8, head dimension 128 |
| Attention | causal sliding window 512 |
| RoPE | theta 500,000, non-traditional MLX layout |
| Vocabulary / mask | 100,352 / token 12 |
| Maximum block | 16 rows: one anchor plus 15 proposals |

The draft checkpoint contains no token embedding or output head. Both are
borrowed from the loaded Laguna target, so a mismatched vocabulary, hidden
width, or target depth is rejected rather than producing low-acceptance output.

For every target row, the five tapped hidden slices are independently RMS
normalized, concatenated, projected from 10,240 to 2,048 dimensions, and RMS
normalized again. Each draft layer applies its input RMS norm to that context
before projecting context K/V. The block path uses Q/K RMS normalization,
split-half RoPE, per-head `softplus` output gates, and SwiGLU exactly as the
Poolside checkpoint expects.

## Cache and verification behavior

MLX Swift's staged MTP verifier owns the target-cache transaction. A round
evaluates `[bonus, draft_1, ...]`, keeps the bonus plus the accepted prefix,
and drops rejected rows without trying to rewind a wrapped rotating cache.
The DFlash block K/V is ephemeral; only target-derived context is committed to
the drafter's five rotating caches.

All draft layers have a 512-token window. A small pinned dependency patch lets
the drafter advertise this bound so prompt prefill retains at most 512 rows of
the 10,240-wide target feature tensor. Without the bound, long prompts would
retain roughly 20 KiB of auxiliary BF16 state per token even though those rows
could never affect a proposal.

## Current status (22 September 2026)

Laguna DFlash remains opt-in. When a drafter is explicitly loaded and no block
size is supplied, Midnight now uses **3**, capped by the checkpoint maximum.
An explicit `--dflash-block-size` / `dflashBlockSize` still overrides this.
The old implicit maximum of 16 accepted only 7.8% of proposals on the installed
Q4R8 target and ran 56.6% slower than target-only decoding in the fresh Metal
screen. Block 3 accepted 60.1% and was within 0.9% of target-only median speed
on a 512-token tutorial. These are shared-machine measurements, not a general
speed guarantee. CUDA has not been revalidated with this new default.

The output divergence is reproducible without a drafter or speculative cache
rollback: replaying the same token prefixes through ordinary multi-row target
forwards changes logits relative to one-token forwards. Enabling hidden-state
capture alone produced exactly identical logits across 256 positions. For
blocks 2 and 4, the first differing prediction was at a BF16 top-two tie:
` structure` versus ` representation`, position 136 of the generated sequence.
This explains the same divergence at byte 745 in the full DFlash run; it does
not establish that every possible cache/verification issue is absent.

An FP32 diagnostic that preserves packed quantized weights had no block-2 or
block-4 token mismatches over its own 159 replayed positions (maximum logit
error about 0.000019). Block 16 still differed. Widening parameters changes the
reference computation and its trajectory, so this is evidence of numerical
sensitivity, not a production fix or an exact-equivalence guarantee.

Laguna DFlash now supports nonzero-temperature requests. The drafter proposes
greedy tokens without consuming sampling RNG; the verifier samples each emitted
position from the target using the requested temperature and `top_p`. Only
matching draft prefixes are accepted. This replaces the earlier greedy-only
fallback. Greedy behavior is unchanged. Higher sampling entropy can reduce
acceptance, so support does not imply acceleration at every temperature. Structured-output and forced-tool
requests retain the target-only path. Native protocol requests also use the
target-only path. New DFlash A/B reports use `status: failed_output_parity`
whenever exact output comparison fails, even if throughput improves.

A separate 258 MiB Q4/Q8 drafter was created from the 882 MiB BF16 source. Its
five-pair block-3 screen did not establish a speed benefit: median throughput
was 3.1% below its target-only control, with the same first divergence. That
run overlapped a CPU build and had timing drift; the artifact is experimental
and was not installed or enabled automatically.

See [fresh reports and reproducible diagnostics](../benchmark-results/laguna-dflash-20260922/README.md).
The results below are historical and use different checkpoints/builds.

## Follow-up: production controls and verification fusion

The follow-up source review found no architecture mismatch with oMLX's Laguna
adapter for the installed checkpoint. A compiled verification-tail experiment
matched logits and captured features exactly across 130 real-model batches,
but did not show a consistent speed benefit across tutorial, coding and
reasoning prompts. It remains disabled by default; developers can reproduce
it with `MODEL_RUNNER_LAGUNA_COMPILED_VERIFY_TAIL=1`.

DFlash A/B now retains the ordinary target's production single-token
optimizations. Earlier controls disabled those optimizations, understating
DFlash's overhead. With the corrected control, block-3 DFlash ranged from
about 13% slower to 3.5% faster across the small shared-machine screen; no
general speed win or quality guarantee is established. Exact-output failure
is reported separately from the question of algorithmic correctness.

See [follow-up results and reference comparison](../benchmark-results/laguna-dflash-20260922/followup/README.md).

## Running and measuring

```bash
./run.sh \
  --model /models/Laguna-XS-2.1 \
  --dflash-model /models/Laguna-XS-2.1-DFlash \
  --dflash-block-size 16
```

The equivalent settings keys are:

```json
{
  "mlxRunner": {
    "modelPath": "/models/Laguna-XS-2.1",
    "dflashModelPath": "/models/Laguna-XS-2.1-DFlash",
    "dflashBlockSize": 16
  }
}
```

Use `model-runner-runtime-bench` for target-only versus DFlash A/B tests. The
report records `proposed_draft_tokens` and `accepted_draft_tokens`; always
compare end-to-end tokens per second as well as acceptance. Smaller blocks can
win when a target/checkpoint pairing accepts only short prefixes, so test 4,
8, and 16 rather than assuming the largest block is fastest.

Each report also records the effective `dflash_block_size` and any
`speculative_passthrough_reason`. A benchmark requested with DFlash fails if
the iterator enters target-only passthrough, preventing fallback throughput
from being mislabeled as a speculative result. `--dflash-ab` records
`first_output_divergence_utf8_offset` when output differs. For a one-time
internal record of the first normal draft/verifier rejection—including token
IDs, cache position, and the verifier's top-two logit margin—run with
`MODEL_RUNNER_DFLASH_FIRST_REJECTION_DIAGNOSTIC=1`.

Poolside publishes a BF16-target drafter and an INT4-target drafter. A custom
affine Q4R8 target is not identical to either training target. Start with the
INT4-trained draft, then A/B it against the BF16-trained draft on representative
prompts. Keep the pairing with the higher measured throughput and stable greedy
parity; published BF16 speedups do not automatically transfer to Q4R8.

## Quantizing the drafter in Swift

Both published XS drafters are BF16 models; `DFlash-INT4` describes the target
precision used during drafter training, not the drafter's own stored precision.
The unified Swift quantizer recognizes `DFlashLagunaForCausalLM` separately
from the Laguna target and converts it directly:

```bash
../wick/.build/release/wick \
  /models/Laguna-XS-2.1-DFlash-INT4 \
  /models/Laguna-XS-2.1-DFlash-INT4-MLX-Q4R8-ScaleSearch
```

The experimental profile uses searched affine Q4 group-64 for the large draft
projections and Q8 group-64 for the shared target-context projection plus all
per-head attention gates. The native sanitizer handles the checkpoint's fused
QKV and separate SwiGLU gate/up tensors; the emitted checkpoint has the names
and per-layer quantization metadata expected by `--dflash-model`.

Do not discard the BF16 drafter after conversion. Quantizing the speculator can
change its proposals and acceptance length even though target verification
still checks every accepted token against the target. Batched target arithmetic
can nevertheless differ from sequential target-only decoding. Benchmark BF16 versus Q4R8 with the same
target, prompts, block size, and generation length, then keep Q4R8 only if its
end-to-end tokens per second improve without pathological acceptance loss.

## First full-checkpoint Q4R8 result

The RTX 4090 validation used the custom
`Laguna-XS-2.1-MLX-Q4R8-ScaleSearch-LS2` target and 128-token greedy coding
generations. Poolside's INT4-target drafter was incompatible with this custom
target on the measured prompt: both its BF16 and quantized forms accepted
0/1,785 proposed tokens. The BF16-target drafter was the better pairing.

Quantizing that drafter reduced its checkpoint from 882 MiB to 259 MiB. At
block 16, acceptance changed only from 72/773 (9.3%) for BF16 to 70/803 (8.7%)
for Q4R8, while median throughput increased from 72.15 to 96.95 tok/s. Reducing
the block to 4 raised measured acceptance to 63/191 (33.0%). With an explicit
1,024 MiB MLX cache, its seven-trial median was 140.43 tok/s versus 130.81
target-only, a 7.36% median gain.

That median is not yet a deployment win. Two of the seven DFlash trials fell
to 30.84–32.68 tok/s despite identical text and acceptance, making mean DFlash
throughput 15.99% lower than target-only. The default CUDA cache remains 128
MiB; the runtime now permits explicit cache tuning up to 1,024 MiB, but the
larger cache did not eliminate the outliers. Keep the quantized BF16-target
artifact and use block size 4 for the next experiments. The verifier and
passthrough scheduling fixes are now implemented, but this historical CUDA
result has not been rerun; leave DFlash disabled by default until a fresh A/B
removes the outliers and proves acceptable output behavior.

Raw reports and the decision table are in
[`benchmark-results/dflash-quantizer-20260829`](../benchmark-results/dflash-quantizer-20260829/README.md).

## M5 Max Metal result

The MacBook benchmark alternated target-only and DFlash generations on one
loaded Q4R8 ScaleSearch target, which removes model-load and most run-order
effects. With block size 3 and five 512-token trials per mode, target-only
median decode was 134.25 tok/s and DFlash median decode was 128.54 tok/s, a
4.26% regression. DFlash accepted 263/495 proposals (53.1%), but its generated
text differed from the target-only continuation.

The first post-fix same-process probe used three 128-token trials per mode.
Target-only median decode was 145.58 tok/s and block-3 DFlash was 157.36 tok/s,
an 8.09% gain with 201/357 proposed tokens accepted (56.3%). The generated
continuation still diverged at UTF-8 byte 109. A separate block-2 probe diverged
at the same byte, ruling out the new multi-row greedy argmax width as the cause.
The DFlash warm-up also ran at only 101.29 tok/s, so this small probe does not
prove that the historical outliers are gone. Keep DFlash opt-in until the
real-checkpoint forward/cache parity issue is understood and a longer
counterbalanced run passes the chosen output-quality gate.

Raw post-fix reports are in
[`benchmark-results/serving-optimizations-20260829`](../benchmark-results/serving-optimizations-20260829/README.md).

## Sampled requests and workload effects

Use the normal OpenAI-compatible `temperature` and `top_p` request fields.
No new HTTP field or load setting is required. Laguna remains explicit opt-in;
Muse's separate greedy-only restriction is unchanged. Structured-output,
forced-tool and native-protocol routing restrictions still apply.

The benchmark now accepts `--temperature` and `--top-p` and records both in
its JSON report. Defaults remain 0 and 1. For stochastic runs, independent A/B
outputs are not expected to match, and an exact-output comparison is not a
sampling-distribution test. Prefer a single DFlash run to check participation
and use seeded fixed-model tests for sampling correctness.

Prompt content affects acceptance; higher-entropy continuations are usually
harder to predict. Longer outputs amortize setup costs, while longer prompts
increase prefill and context work. Sending concurrent requests does not by
itself improve single-request draft acceptance. Shared GPU load and changing
machine conditions can materially affect throughput measurements.

## Poolside reference follow-up (22 September 2026)

A review of Poolside's `llama.cpp` Laguna branch at
`06f8cebd7fe728687be3d19f8bdedb70d75883af` agrees with the current greedy-draft,
target-sampled verification approach. Its F16 feature-overflow workaround did
not reproduce on this BF16 path: captured features were large but finite.
The official INT4-specific assistant accepted 0/251 proposals against the
custom Q4R8 target and was not adopted. A compiled drafter-tail experiment
showed no established gain and was removed.

An opt-in synchronized profile attributed 78% of measured decode-stage time
to target verification, 18% to drafting and 4% to draft-context preparation.
This identifies an optimization target, not a proven throughput fix.
Use `MODEL_RUNNER_LAGUNA_DFLASH_STAGE_TIMING=1` for stage timings or
`MODEL_RUNNER_LAGUNA_DFLASH_FEATURE_DIAGNOSTIC=1` for feature finiteness and
magnitude. Both are developer diagnostics, default off, and add synchronization;
do not compare their throughput directly with uninstrumented runs.

[Reports and limitations](../benchmark-results/laguna-dflash-20260922/poolside-followup/README.md).

## Official INT4 pairing comparison (22 September 2026)

A three-prompt screen compared both targets with both assistants using identical
prompt token IDs, temperature 0, block size 3, 128 output tokens, and two paired
trials per prompt after warmup. The custom Q4R8 target with the BF16 assistant
accepted 420/674 proposals (62.3%); the official INT4 weights with their matching
assistant accepted 400/722 (55.4%). Both crossed pairings accepted 0/1,506.
The official checkpoint uses rotated weights and a different mixed-precision
layout. The assistants are not interchangeable merely because both targets
are described as four-bit.

The official pair was 5–32% faster than its own target-only control, but still
ran at 101–125 tok/s versus 155–175 tok/s for custom target-only decoding.
The custom pair remained 4–17% slower than its own control. Keep the installed
model and explicit opt-in policy; this does not establish custom quantization
as the cause of the remaining slowdown. Exact output parity still failed on
two of three prompts with either matched pairing.

The official weights used an experimental layout adapter preserving packed
codes, scales and rotations; no requantization was applied. Both comparisons
used BF16 KV cache, not the official FP8 cache. Full reference-backend parity
and model-quality equivalence are untested. Loading also exposed a macOS
MLX-Swift sparse module-update crash when layer 0 is unquantized; the pinned
dependency patch now handles that layout. Direct compressed-tensors loading
remains unsupported.

[Full comparison and reproduction details](../benchmark-results/laguna-dflash-20260922/official-pair/README.md).
