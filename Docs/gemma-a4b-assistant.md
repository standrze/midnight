# Gemma 4 26B-A4B assistant decoding

Midnight automatically uses a compatible installed assistant for Gemma 4 on
Metal. It checks the model card, embedded `assistant/` or `drafter/` folders,
and `~/.midnight/drafters` (override with `MODEL_RUNNER_ASSISTANTS_DIR`).
A named Hugging Face assistant in the model card is downloaded if missing.
This remains experimental. It uses the text-only MLX implementation; no vision module is linked
into the runner for this feature.

Use `--no-auto-assistant`, `autoAssistant: false` in a model-load request, or
`mlxRunner.autoAssistant: false` in model-stack settings to disable discovery
and automatic downloads. Explicit assistant paths still take priority. See
[model cards](model-cards.md#automatic-assistants) for pairing and download rules.

These options are in the workspace build, not the published installer release.
From the Midnight repository, build the executable:

```sh
swift build -c release --product midnight
```

With the target and matching assistant already downloaded, selecting the target
is enough. To select a particular assistant and quantization explicitly:

```sh
.build/release/midnight \
  --model "$HOME/.midnight/models/mlx-community--gemma-4-26B-A4B-it-qat-4bit" \
  --gemma-assistant-model "$HOME/.midnight/drafters/google--gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant" \
  --gemma-assistant-block-size 4 \
  --gemma-assistant-quantization-bits 4
```

The assistant is available at every supported `temperature` (including the
default 1.0). Greedy requests use its normal block verifier. For sampled
requests, each verifier position is sampled sequentially from the target and a
draft token is retained only when it matches that sample. This keeps sampling
target-defined, but acceptance rates can be lower than greedy decoding.
Structured-output requests use ordinary target decoding.
`"include_reasoning": true` still enables live reasoning separately from the
answer in Chat Completions and Responses. Without that option, reasoning is
disabled for supported Gemma templates.

The quantization option converts only the assistant in memory, leaving the
target and files on disk unchanged. It accepts 4 or 8 bits and requires an
unquantized assistant source. Omit it to retain the source precision. This
path requires uncompressed KV caches and cannot be combined with DFlash.

The same options are available in the model-load API as
`gemmaAssistantModel`, `gemmaAssistantBlockSize`, and
`gemmaAssistantQuantizationBits`. Model-stack JSON uses these keys under
`mlxRunner`, except that the path key is `gemmaAssistantModelPath`. Settings
tied to one target do not follow a switch to another target. Invalid assistant
metadata is rejected before unloading the current model.

The native Q4, block-4 implementation measured 16.1% median paired throughput
gain on a tutorial workload and 9.7% on prose on this M5 Max. A coding run showed
19.3% but had more timing drift, so its magnitude needs confirmation. Eight small correctness
cases passed, including retrieval beyond the sliding window. These are limited
results, not a general quality guarantee or an RTX 4090 measurement.

Batched target verification can change floating-point results and generated
wording. Long free-running comparisons did not maintain exact output parity;
the benchmark records these as parity failures. Do not use this option where
identical target-only output is required. Shared prompt-cache reuse is disabled
for speculative requests, so gains depend on workload.

See `benchmark-results/gemma4-a4b-reasoning/README.md` for the paired measurements,
quality screen, numerical diagnostics, and model revisions. This feature is in
the workspace build; it has not been published in a new release.

For target-only decoding, a separate experimental optimization combines the
experts' packed gate/up projections without requantizing. It measured +3.65%
on the tutorial and +3.44% on prose, with exact output matches in both screens:

```sh
MIDNIGHT_GEMMA4_EXPERT_GATE_UP=1 .build/release/midnight \
  --model "$HOME/.midnight/models/mlx-community--gemma-4-26B-A4B-it-qat-4bit"
```

This remains off by default. Preflight restricts it to the validated A4B
affine-Q4/group-64 layout on Metal without an adapter, including compatible
expert quantization overrides. While the environment option is enabled,
incompatible model loads are rejected before replacing the active model.
Its extra gain with the assistant was inconclusive; no additive speedup is
claimed. See `benchmark-results/gemma4-a4b-reasoning/expert-gate-up-screen/`.
