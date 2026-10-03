# Experimental D256 masked prefill

`MIDNIGHT_METAL_SDPA_D256_MASKED=1` opts into the head-dimension-split NAX
attention kernel for batch one, exactly 512 query rows, matching 256-wide
Q/K/V heads, BF16, 512 through 1,536 key rows, GQA up to eight, and an array
mask. It requires a NAX-capable GPU. All other shapes keep existing routing.
The experiment is off by default and changes no weight or KV precision.

The pinned implementation admits causal D256 NAX attention automatically only
at 1,024 or more query rows without an array mask. Its upstream commit
`714a7efcb83c1424b4ade9226a5fd810fa184b73` describes the query threshold as
providing enough blocks to fill the GPU; it does not identify a hard limit
preventing masked 512-row execution. The existing split kernel uses 256
threads and divides the head dimension across two SIMD groups per row group.
Whether 512 rows outperform decomposed attention requires measurement.

## Work avoided

The experimental kernel reads the actual bool or additive mask on the GPU
and computes conservative first/last occupied KV tile bounds for each
64-row query block. It skips excluded leading and trailing tiles before
QK and PV multiplication. It preserves the existing element mask for edge
tiles and interior holes. GQA still maps each query head to its own KV head;
per-head mask offsets and broadcast mask row strides remain honored.

This does not infer a window size from tensor dimensions and does not read
mask data back to the CPU. Each SIMD group scans complete query rows, followed
by one block-wide reduction. Added threadgroup storage is eight first-tile
indices, eight last-tile indices, and eight flags. The scan has a cost, so the
improvement must be compared with the same NAX kernel without pruning.

For a 512-query chunk with 1,535 keys and a 1,024-key causal window, the exact
mask bounds retain 272 of 384 32-key tile visits across the eight query blocks,
29.2% fewer QK/PV tile visits. That is an operation-count calculation, not a
measured speedup. Cache cropping and this kernel compose: cropping reduces the
shared input range, while each query block can still omit unused portions of
that range.

## Mask edge cases

- Bool `false` and additive negative infinity are excluded when deriving bounds.
  Finite negative biases, positive infinity, and NaNs are retained.
- If any real query row in a block has no active key, that block uses the full
  key range. Bool-masked real scores retain the backend's finite-minimum
  convention, which yields a uniform value mean for a completely masked row.
- An experiment-only padding correction runs after the array mask, before
  softmax. It assigns negative infinity to nonexistent key positions in the
  final 32-key tile. Real masked keys retain their normal values. This prevents
  an all-masked row at an uneven key length from counting padded keys in the
  softmax denominator.

The padding correction is present in both experimental NAX modes. It uses
function constant 304; pruning uses function constant 303. These constants are
included in the pipeline cache key. Core Metal, generated Metal, and the
generated embedded JIT source carry the same complete kernel body.

## Validation and comparison

The existing `Scripts/metal-sdpa-d512-patch.sh` helper now owns the ordered
D512 then D256 overlays. Its function name and preparation call are unchanged.
It first copies the affected files, removes already-applied overlays in
reverse order, and replays the full stack. Only after both repository
preflights succeed does it apply missing patches to the checkouts.

`bash Tests/Shell/MetalSDPAStackPatchTests.sh` verifies clean, D512-only, and
fully applied states, repeated preparation, second-checkout conflict
atomicity, and core/generated Metal/JIT equivalence. These checks and Swift
formatting passed on September 29, 2026. After the coordinated rebuild,
all three GPU numerical tests also passed in each arm (`PRUNE=0` and `PRUNE=1`).
These numerical checks do not establish a throughput benefit.

After a coordinated native and Metal rebuild:

```bash
MIDNIGHT_RUN_SDPA_D256_REGRESSION=1 MIDNIGHT_METAL_SDPA_D256_MASKED=1 \
  MIDNIGHT_METAL_SDPA_D256_PRUNE=1 \
  swift test -c release --filter MetalSDPAD256MaskTests
```

Repeat with `MIDNIGHT_METAL_SDPA_D256_PRUNE=0` to validate the NAX control.
The suite calls the public MLXFast API and compares against explicit FP32
QK, softmax, and PV operations. Cases include real Gemma GQA geometry, cache
head padding and slice offsets, uneven key lengths, per-head masks, interior
holes, entirely masked query blocks and individual rows, negative-infinite
masks, and finite negative biases. Fully masked rows at key length 1,057 are
also compared directly with `mean(V)` so a denominator rounded up to 1,088
cannot hide behind a global output tolerance.

Benchmark three separate processes using the same inputs and fresh state:

1. `MIDNIGHT_METAL_SDPA_D256_MASKED=0`: existing dispatch.
2. `MIDNIGHT_METAL_SDPA_D256_MASKED=1 MIDNIGHT_METAL_SDPA_D256_PRUNE=0`: NAX control.
3. `MIDNIGHT_METAL_SDPA_D256_MASKED=1 MIDNIGHT_METAL_SDPA_D256_PRUNE=1`: NAX with tile bounds.

Report prefill time, peak memory, output correctness, and end-to-end model
throughput. The second-to-third comparison isolates the value of mask-derived
tile bounds. The first-to-second comparison measures the changed dispatch.
Keep both experiments opt-in until these tests justify broader routing.
