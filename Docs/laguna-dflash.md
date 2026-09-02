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

## Current decoding boundary

DFlash is enabled only when the request has `temperature: 0`. The verifier is
designed to preserve ordinary greedy generation, and tiny-model tests satisfy
that contract. The iterator now batch-materializes greedy verifier rows once
per round and uses the ordinary asynchronous Metal pipeline when speculation
falls back to target-only decoding. The earlier real Laguna/Metal run produced
a different 512-token continuation, and that A/B has not yet been repeated
after these fixes. Greedy equivalence therefore remains a required deployment
gate, not a claimed result. Nonzero-temperature requests use the existing
target-only path. Exact sampled speculative decoding needs probability-ratio
acceptance and residual sampling; equality of two independently sampled tokens
is not distribution preserving.

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
.build/release/model-runner-quantize \
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
keeps greedy output token-identical. Benchmark BF16 versus Q4R8 with the same
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
