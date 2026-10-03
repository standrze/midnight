# Changelog

## Unreleased

- Avoid an unnecessary GPU token-ID conversion during text prompt preparation,
  preserving exact token values and the fallback for other token dtypes.
- Add explicit layer, site, and token selections with bounded raw activation
  export for supported inspection models and the ABSlayer adapter.
- Manage the optional Apple-silicon FastVLM worker through normal model loading
  and unloading, with independent model-file protection and text restoration.
- Expose replayable model selections and optional instance/generation guards for
  Lowlight and Lowlight-browser's managed vision controls.
- Release Mac compiled-model constants across tracing threads and avoid retained
  gate/up graphs during unload, while keeping compiled inference enabled.

## 0.2.0-beta.5

- Add OpenAI Responses, structured-output controls, and live model loading.
- Queue overlapping text requests with cancellation and bounded admission.
- Reuse bounded, exact shared prompt prefixes and report cached-token usage.
- Publish a Linux x86-64 CUDA 13/sm_89 archive with runtime resources and an installer.
- Keep embeddings and retrieval shelved; preserve existing OpenAI chat support.

## 0.2.0-beta.3

- Add a native load-once generated evaluation tool with greedy independent
  requests, exact prompt fingerprints, admission checks and atomic partial reports.
- Add pinned question-only code/math and synthetic retrieval tasks, isolated
  Docker code scoring, reference-solution validation and paired accuracy analysis.
- Add bounded standard-affine and ScaleSearch Q4/G128 Laguna conversion while
  preserving Q8/G64 routers and Q4/G64 embeddings, with explicit module metadata.
- Add a bounded, source-fingerprinted native-affine GPTQ projection probe and
  source-verified AutoRound/MLX metadata compatibility research.
- Add weighted Instruments CPU profile analysis and command-buffer experiments.
  Runtime defaults remain evidence-gated; profiling timings are diagnostic.

## 0.2.0-beta.2

- Add source-verified, layerwise Laguna activation calibration with bounded
  tensor and Foundation-buffer lifetimes.
- Support public split gate/up Q4R8 templates in ScaleSearch and optional
  activation-weighted refinement, retaining insufficiently covered experts.
- Validate template identity on the selected quantization device and release
  BF16 tensors after their final conversion consumer.
- Add reproducible benchmark campaigns, pinned reference corpora, chunked NLL
  scoring, paired uncertainty analysis, and drift/identity checks.
- Backport the upstream M5 NAX sorted-gather row-bound fix to offline and JIT
  Metal sources, with long-row GPU regression coverage.
- Add an opt-in fused Laguna gate/up-SiLU experiment and same-loaded A/B tools.
  It remains off by default; experiments did not establish a decode speedup.

The prefill default remains 512 tokens. Held-out quantization results vary by
model and domain; neither ScaleSearch nor activation weighting is promoted as
a universal quality winner. See `Docs/benchmarking.md` and
`benchmark-results/quantization-20260904/README.md` for evidence and limitations.

## 0.2.0-beta.1

- Admit text requests using exact prompt tokens, requested output, conservative
  KV/workspace estimates, and the process memory budget before prefill.
- Expose context length and prefill chunk size through CLI/settings and model
  discovery; preserve the model's declared context ceiling.
- Scale Metal memory limits with physical RAM, host reserve, and the device's
  recommended working set; retain explicit bounded overrides.
- Budget hot conversations and branch snapshots together, skip snapshots that
  cannot be retained, and evict before copying or admitting a request.
- Integrate opt-in affine8, affine4, and turbo8v4 KV compression for compatible
  Laguna/Mistral-family layers, preserving native sliding windows.
- Fix macOS release launch selection and make optimized release the default.
- Add memory, context, cache, compression, and launcher regression tests.

This remains a beta. KV compression is experimental and does not imply a
quality or speed improvement. Paged attention and continuous batching are not
included. See `Docs/memory-and-context.md` for configuration and boundaries.
