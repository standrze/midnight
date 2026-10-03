# Experimental Metal D512 decode

`MIDNIGHT_METAL_SDPA_D512=1` opts into fused attention for one query token,
batch size one, matching 512-wide Q/K/V heads, FP16 or BF16, GQA up to eight,
and no attention mask. Other shapes retain the existing routing. The switch is
off by default. This changes attention execution, not weight or KV precision.

The initial target is Gemma 4 global attention: the 31B configuration has 32
query heads and four KV heads; the 26B A4B configuration has 16 query heads and
two KV heads. Models with smaller head dimensions keep their existing kernels.
Prefill, multi-token verification, batches larger than one, FP32, and explicit
or causal masks do not enter this experiment.

The patch instantiates the existing two-pass vector templates for D512 in both
the core Metal source and Swift's generated Metal source. The first pass uses
32 threads per query head; the second reduces partial outputs using the
existing 1,024-thread kernel. No D512 single-pass variant is instantiated. The
first pass accumulates in FP32 and stores partial outputs in the input dtype,
as for existing dimensions. Softmax sums and maxima remain FP32. Attention
sinks retain the existing two-pass handling.

Two passes avoid the materialized attention-score tensor but add a reduction
dispatch and temporary buffers. Short contexts may lose to the fallback; the
switch must remain off until model-level measurements establish useful routing
thresholds. Numerical agreement is tolerance-based, not bitwise. A passing
numerical test alone does not establish a throughput gain.

## Reproducibility and validation

The durable helper is `Scripts/metal-sdpa-d512-patch.sh`, called as:

```bash
source "$PACKAGE_ROOT/Scripts/metal-sdpa-d512-patch.sh"
model_runner_prepare_metal_sdpa_d512 "$HOST_OS" "$PACKAGE_ROOT" "$MLX_SWIFT_CHECKOUT"
```

It preflights the complete D512/D256 patch stack before writing either checkout and checks the pinned MLX
core `1f8e74e3f12f31365464a6867c6579f0e9b29d85` and MLX Swift
`72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798` revisions. It is a no-op outside
Darwin. The core patch only modifies the SDPA dispatch and Metal instantiation
file; it does not change the quantized matrix-multiply kernels.

The CPU-only patch check:

```bash
bash Tests/Shell/MetalSDPAD512PatchTests.sh
```

This checks exact patch targets, clean application to pinned source,
reversibility, and equivalent core/generated Metal contents after normalizing
include paths. The patch check, helper's already-applied path, Swift formatting,
and whitespace checks passed on September 29, 2026.

After rebuilding both the native executable and Metal library, the opt-in
numerical suite is:

```bash
MIDNIGHT_RUN_SDPA_D512_REGRESSION=1 MIDNIGHT_METAL_SDPA_D512=1 \
  swift test -c release --filter MetalSDPAD512Tests
```

The test calls the public MLXFast API with `forceFused: true`, so a silently
selected fallback cannot satisfy the fused-path cases. It compares to explicit
FP32 QK, softmax, and PV operations using deterministic inputs, and covers:

- FP16 and BF16; GQA factors one, four, and eight.
- KV lengths 1, 31, 33, 1,023, 1,024, and 4,097, including empty partitions.
- Padded and offset cache slices with non-packed head strides.
- Concentrated attention scores and attention sinks.
- Rejection of batch two, query length two, FP32, GQA sixteen, and array masks,
  with successful ordinary fallback for those shapes.

The initial error limits are maximum absolute error 0.004 for FP16 and 0.025
for BF16, plus relative RMS limits 0.004 and 0.018 respectively. Do not relax a
failed threshold without examining the outputs and the independent oracle.
After the coordinated rebuild with a matched Metal library on September 29,
2026, all three GPU numerical tests passed. Model throughput and routing
thresholds still require separate measurement; the switch remains opt-in.

## D256 sliding-window prefill investigation

The pinned backend already builds a head-dimension-split NAX D256 prefill
kernel, including bool and additive masks. Its automatic routing admits only
causal attention without an array mask, at least 1,024 query rows, and a
supported NAX dtype/device. It otherwise prefers decomposed D256 prefill.
Gemma's sliding-window mask becomes an array mask after the history exceeds
the configured window, so it does not satisfy that fast-path rule.

Changing the routing rule alone does not implement window-aware attention.
In `steel_attention_nax.h`, `attention_nax_dsplit` starts its KV tile loop at
zero. Causal mode can lower the ending tile; an array mask is applied after
the QK product and does not prune fully excluded KV tiles. The kernel still
loads, multiplies, and masks history outside the window.

A native window implementation needs explicit window/position metadata, a
per-query-block lower KV tile bound, correct mask handling at both window
edges, and offset-aware behavior for chunked prefill. That requires host/API
and kernel work, numerical tests at tile/window boundaries, and a separate
performance decision. The separate [D256 masked-prefill experiment](MetalSDPAD256MaskedPrefillExperiment.md)
uses bounds derived from the actual mask and remains independently opt-in.

A smaller independent optimization is to crop a prefill chunk's K/V and mask
to the union of its causal windows before attention. For query length L,
post-update key length N, and window W, the first potentially visible key is
`max(0, N - L - W + 1)`, assuming the usual inclusive causal position and W
total visible keys. The exact mask convention and shared/cache offsets must
be verified in the model before implementing that bound. This can reduce
fallback work without requiring a new NAX kernel.
