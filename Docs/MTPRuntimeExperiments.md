# MTP runtime experiments

Both experiments are disabled by default. They retain 4-bit and higher weight formats.
The draft-width experiment changes verification scheduling; the mask experiment changes
the representation of an all-zero additive attention mask.

`MIDNIGHT_MTP_ADAPTIVE_DRAFTS=1` enables per-stream adaptation only for `ArgMaxSampler`
with no logit processor. The gate is checked again each round. The public configured
`blockSize` stays unchanged; actual verification width never exceeds
`min(blockSize, 8)` or the remaining output budget. Draft counts come from 1, 3, 7 and
the applicable maximum. One warmup observation at each width is excluded. Subsequent
observations use `(acceptedDrafts + 1) / elapsedSeconds`, with a 0.25 update weight.
Every eighth measured post-warmup round schedules a deterministic probe, rotating
through candidate widths. Ties favor smaller widths. Budget-clipped tails and invalid
durations do not train the controller.

A drafter that caps its effective block size below two keeps the existing target-only
behavior; adaptation is not initialized. Run `singlePositionDrafter` with
`MIDNIGHT_MTP_ADAPTIVE_DRAFTS=1` to cover that explicit environment-gate regression.

The monotonic timer starts before cache staging and draft graph construction and ends
after the existing greedy verifier `eval` and host readback. There are no new evals,
GPU synchronizations, random draws, or consumer-drain delays in the measurement.
This measures draft/verify latency: lazy cache reconciliation after the readback can
still contribute work to a following round. Compilation at a previously unseen context
shape can also affect later observations. End-to-end generation throughput decides
whether this experiment should be enabled; its controller score alone is insufficient.

All cache operations, hidden-state selection and telemetry still use the actual
`numDraft`. Early finalization now records whether the last round used native hybrid
rewind, rather than inferring that mechanism from the configured maximum width.

`MIDNIGHT_GEMMA_ASSISTANT_UNMASKED=1` omits explicit zero attention masks in both Gemma
text and VLM assistants. Full shared-K/V attention is bidirectional. Sliding shared
K/V is sliced to the accepted active window before drafting, and the existing
`slidingKvLen <= slidingWindow` precondition remains before mask selection. Under that
invariant the old sliding helper returns all zeros too. `.none` therefore preserves
which keys are visible; it may select a different Metal kernel and reduction order.
CPU tests compare each implementation's explicit and omitted mask paths independently,
including Q4 weights, multiple query lengths, partial/full windows and absolute offsets.
These tests do not prove BF16 D512 GPU parity or a performance improvement.

## Integration

The patches target mlx-swift-lm `14414441fa44f45eee35a61e9fa0bab577cf9734`.

`prepare-dependencies.sh` sources `Scripts/mtp-adaptive-drafts-patch.sh` and calls
`model_runner_prepare_mtp_adaptive_drafts "$PACKAGE_ROOT" "$MLX_SWIFT_LM_CHECKOUT"`
in place of the three earlier MTP patch calls. The helper owns the prompt-window,
scheduling, diagnostic, and adaptive overlays and validates their complete ordered
stack on copied source before applying one combined diff.

The zero-mask patch is included at the end of the existing Gemma model stack in
`Scripts/gemma4-window-cache-patches.py`, preserving replay of the added assistant
source. `MTPAdaptiveDraftPolicyTests.swift`, `MTPAdaptiveIteratorTests.swift`, and
`GemmaAssistantUnmaskedTests.swift` are installed under `Tests/ModelRunnerProtocolTests`.
`Patches/midnight-mtp-runtime-tests.patch` retains their reproducible source additions.
The CPU patch replay checks pass; coordinated numerical tests are still required.

Do not enable either experiment by default before coordinated numerical and throughput
validation. The application API and configured block-size setting are unchanged.
