# Personal-runner performance audit

This audit follows the [product requirements](product-requirements.md): single-user responsiveness decides default behavior. It supersedes historical feature-gap lists where the implementation has changed. September 11, 2026.

## Decode overhead: current state

The August 29 Ollama comparison is a historical baseline, not a current speed ranking. Its two largest concrete candidates are already present in Midnight's dependency preparation:

| Candidate | Current evidence | Disposition |
| --- | --- | --- |
| Direct KV slice updates | `mlx-swift-direct-slice-update.patch` and `mlx-swift-lm-direct-kv-slice-update.patch` are wired in `prepare-dependencies.sh` | Already implemented; patch persistence checks pass for pinned Mac/Linux sources. |
| Incremental ByteLevel text decoding | `swift-transformers-incremental-bytelevel-decoder.patch` and `mlx-swift-lm-incremental-bytelevel-streaming.patch` are wired in dependency preparation | Already implemented with tokenizer capability checks and fallback; patch persistence checks pass. |
| Persistent stateless compiled-function handles | An isolated prototype passed nine CPU tests and all 18 paired answer/usage comparisons across 36 measured LFM requests | Deferred. Median paired generation changes were only +0.31% to +0.78%, with inconsistent wins and slightly worse short-prompt first text. The prototype is absent from the production candidate. |
| Asynchronous token evaluation | Metal uses one-token pipelining; CUDA deliberately realizes token/cache state before mutation | Keep CUDA's correctness boundary. Removing it to improve a stopwatch result is not an acceptable optimization. |

Do not reimplement optimizations already applied or interpret the old 8% Ollama gap as a current result. The controlled competitor runs, where available, determine the next profile target.

The [compile-handle experiment](../benchmark-results/personal-prd-20260911/stateless-compile-prototype/README.md) preserves the patch, tests, raw measurements, and limitations for later profiling. Keeping ordinary MLX graph compilation enabled is separate from enabling this experimental handle-retention change.

## Cache policy

Full conversation reuse takes precedence over partial shared-prefix checkpoints for eligible models from the first turn. Shared reuse remains available for eligible requests without that conversation path. A zero shared-cache budget bypasses shared preparation.

The decision is captured by `LocalModelRunner.promptCacheRoute` with regression coverage for long first turns, LFM-style fallback, GPT-OSS tools versus ordinary text, custom stops, forced prefixes, normalization, disabled reuse, and speculation. This is request-level routing, not extra per-token machinery.

## Default-setting gates

- Prefill size changes require matched long-prompt measurements as well as short-prompt controls, memory observations, and generation-rate checks.
- Compression stays opt-in until a model/hardware workload demonstrates an acceptable quality, memory, and latency tradeoff.
- Cache and speculation decisions must include tail latency and total request time; a better median with severe slow runs is insufficient.
- Explicit context, output limits, and user-selected settings must remain authoritative. A default must not silently shorten the task to appear faster.
- Tuning is an offline benchmark activity. No background autotuner or optional draft model may compete with ordinary inference by default.

The September 11 Laguna/RTX 4090 trial with prefill 2,048 and context 8,192 failed memory admission at the matched 20 GiB ceiling: estimated 21,598,119,160 bytes exceeded 21,474,836,480 bytes. It produced no valid speed measurement. The default remains 512; matching another runner's larger prefill default is not sufficient justification to increase ours.

## Speculative decoding

The existing Laguna DFlash path is opt-in. Laguna now supports target-verified sampling with deterministic draft proposals. The recorded performance experiments here use greedy requests; their speed results do not establish sampled-request performance. A compatible draft checkpoint, sufficient memory, token correctness, first-text latency, total latency, and slow-run behavior all matter.

Broader upstream draft-model support is an implementation possibility, not proof of a personal-use speed gain. Historical DFlash measurements included regression and severe outliers. Keep speculative defaults off unless a fresh compatible target/draft experiment passes the performance gates. A failed experiment is a reason to retain the target-only default, not to enable more speculation machinery.

The fresh Linux follow-up with `MLX_CUDA_SDPA_CACHE_SIZE=1024` completed 20 measured requests without the earlier attention-cache-thrashing abort. Target-only generation had a median of 141.1 tokens/s, compared with 45.0 for DFlash; median completed-answer time was 1.91 seconds versus 5.77 seconds. None of the ten paired answers matched exactly. Increasing the attention-cache capacity addressed that observed abort, but established neither a speed win nor output equivalence. DFlash remains off by default.

The [September 22 Metal investigation](../benchmark-results/laguna-dflash-20260922/README.md) identifies shape-dependent target arithmetic as a reproducible source of output divergence and reduces the implicit Laguna block size from 16 to 3. The new default remains slower than target-only on the tested workload, so Laguna stays opt-in. This is not a new CUDA result.

## Evidence location

The current campaign's harnesses, raw measurements, and focused reports are under `benchmark-results/personal-prd-20260911/`. Those reports distinguish controlled fixed histories from natural conversations and document unavailable or incompatible competitor environments. Inspection and deletion validation are reported separately there or in their focused documentation.
