# Native decision models

Midnight serves bounded decision models. Afterglow (`../afterglow`) owns training,
adapter conversion, calibration, evaluation, export, and quantization. Wick and
Training are preserved as the original projects; Midnight contains no training
launcher or forwarding CLI.

## Nimble baseline

The first supported publisher release is `bespokelabs/Bespoke-Nimble-9B` at
`bd792f44ec8e265be861bfcdf4e05967ffe0e858`, paired with `Qwen/Qwen3.5-9B` at
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. The native downloader retrieves the
publisher's BF16 base and PEFT adapter as a source bundle (19.52 GB), rather than
claiming a community quantization is publisher-owned. See the scoped exception
in [model-support-scope.md](model-support-scope.md).

Convert the publisher adapter with Afterglow, retaining its pinned schema,
temperature configuration, and source provenance. Load the base with the
converted adapter through Midnight's existing `--adapter` option. The adapter's
`decision-model.json` enables discovery's `capabilities.decisions` and the
Midnight-specific `POST /v1/decisions` route. See the [API Field Guide](api-field-guide/index.html#decisions).

Each requested field receives the publisher's complete prompt, preserving schema
order and candidate codes. The model scores the final-position logits of allowed
candidate tokens, applies the artifact's temperature, and returns probabilities
and a selected value. Boolean choices are false then true. Integer-valued enum
choices can opt into an expected score through `score_fields`. Requests are
limited to 64 KiB and the artifact's context length; excess input is rejected,
never truncated. Separate field caches prevent cross-request conversation state.
Decision scoring does not sample generated text or use speculative decoding.

## Training and release sequence

1. Validate JSONL labels and disjoint train/development/calibration source families.
2. Train the candidate-token cross-entropy objective in Afterglow, with frozen
   base weights, native LoRA updates, FP32 optimizer state, and atomic checkpoints.
3. Resume only when base, initialization adapter, data, contract, and options match.
4. Evaluate a held-out split and, separately, fit temperature on calibration data.
5. Export a base/adapter bundle; verify native scoring after reloading the staged
   export before committing it to its destination.
6. Treat local four-bit quantization as a separate candidate artifact. Compare its
   decisions and probabilities against the BF16 baseline before promoting it.

Publisher prompt-token parity, native gradients, deterministic tiny-model
checkpoint recovery, real-model scoring, and HTTP checks form the acceptance
suite. The two authored return-policy examples in
`benchmark-results/nimble-20261002` are smoke fixtures, not a representative
quality benchmark. CUDA training and broad task-quality evaluation remain
separate validation work; the implementation targets native Swift/MLX on Apple
silicon.
