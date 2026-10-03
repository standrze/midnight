# Experimental grouped expert verification

`MIDNIGHT_METAL_GROUPED_EXPERT_VERIFY=1` enables a custom Metal projection for
Gemma 4 A4B-shaped affine Q4/G64 expert weights during inference. It is off by
default. The gate admits BF16 activations and affine metadata, 128 experts,
eight selected experts per token, and 2–8 flattened token positions. Supported
input/output widths are 2816/704, 2816/1408, and 704/2816. Other precisions,
formats, shapes, CPU execution, and training retain `gatherQuantizedMM`.

The shape gate identifies the Gemma expert layout; it does not inspect a model
name or distinguish a short final prefill chunk from target verification. A
batch of two sequences with three positions each is represented by six rows,
as in the existing Gemma expert wrapper. Single-token decode is unchanged.
Router scores, selected expert IDs, weighted reduction, and quantized weights
are unchanged. This first implementation supports Q4 only; Q8 retains the
existing path, and no format below four bits is admitted.

## Why this path exists

The pinned dense quantized matmul can reuse weights across several input
vectors through `qmv_wide`. The expert wrapper instead presents one matrix row
per token/expert assignment to gathered matmul. Even when several positions
choose the same expert, the small gathered path treats those assignments
independently. Its sorted QMM threshold is unreachable for 16–64 assignments
spread across 128 experts.

The candidate retains the existing output layout while grouping repeated IDs
inside its own kernel. One threadgroup owns each consecutive group of up to
four occurrences of an expert, including unsorted occurrences. It decodes an
eight-value packed subchunk once and multiplies it by up to four input vectors.
Each member receives its own output; groups beyond four occurrences elect a
new leader. All members of an inactive threadgroup return before the barrier.

There is no extra sorting dispatch, CPU readback, atomic output reduction, or
change to routing. The eight input lanes per output row handle 44 quantization
groups for width 2816 and 11 groups for width 704. The final groups are covered
without reading a padded width. The kernel uses FP32 accumulation and produces
BF16 output. MLX makes noncontiguous input views contiguous when necessary.

This trades extra membership scans, inactive threadgroups, and register use
for fewer repeated weight reads. It can be slower when reuse is low. Its
floating-point operation order also differs from stock gathered QMV; numerical
agreement is tolerance-based. Neither a lower theoretical byte count nor a
passing primitive test establishes end-to-end speculative decoding speed or
model quality.

## Integration and validation

The durable overlay is
`Patches/mlx-swift-lm-gemma-grouped-expert-verification.patch`. Add these lines
to dependency preparation, after the existing gate/up slices preparation:

```bash
source "$PACKAGE_ROOT/Scripts/gemma-grouped-expert-patch.sh"
model_runner_prepare_gemma_grouped_expert "$HOST_OS" "$PACKAGE_ROOT" "$MLX_SWIFT_LM_CHECKOUT"
```

The helper verifies MLX-LM pin `14414441fa44f45eee35a61e9fa0bab577cf9734`,
is a strict no-op outside Darwin, and checks the complete patch before applying
it. Its hunks do not overlap the gate/up view or quantization constructor
changes. `bash Tests/Shell/GemmaGroupedExpertPatchTests.sh` verifies pinned and
current overlay round trips, repeat preparation, drift refusal without
mutation, and continued reverse applicability of those existing overlays.
These shell checks, Swift parsing, and formatting passed September 29, 2026.

The overlay and numerical test are integrated into the dependency checkout
and `Tests/ModelRunnerProtocolTests/GemmaGroupedExpertVerificationTests.swift`.
Dependency preparation replays the reviewed overlay; no manual copy is needed.

After a coordinated Swift rebuild, run:

```bash
MIDNIGHT_RUN_GROUPED_EXPERT_REGRESSION=1 \
  swift test -c release --filter GemmaGroupedExpertVerification
```

The default CPU eligibility test does not evaluate tensor data. The optional
GPU suite invokes the candidate through its test SPI, independently of the
model-wide switch. It covers no reuse, mixed unsorted IDs, more than four
repeated occurrences, all-same IDs, sorted eight-position inputs, flattened
batched positions, each projection width, and strided input/index views.
NaN-initialized outputs expose missing writes. Expert-specific metadata and
assignment-specific inputs expose incorrect expert selection and scattering.
An independent FP32 dequantization plus gathered matmul is the numerical
oracle; stock gathered QMV error is reported separately. GPU numerical checks
are pending at this staging milestone.

Before deployment, compare stock and candidate in separate processes on the
same fixed token prefixes and expert routes, then measure useful output tokens
per second with the same assistant, context, and draft policy. Include a
no-reuse case and preserve the stock fallback if reuse does not repay the
membership overhead. Keep the switch off until those measurements support it.
