# Inspection isolation and performance evidence

Midnight’s inspection API installs observation wrappers only for an explicit trace or an armed recording of the next ordinary Chat Completions/Responses request. It does not install them at model load or for unarmed generation. This document separates what code and tests establish from what performance measurements establish.

The public [Midnight Activation Capture](activation-capture.md) feature documents the selected-layer/raw-vector interface and its consumer contract. Historical measurements below remain scoped to the source revisions they actually tested.

## Execution and lifetime

- `/v1/inspector/model` describes the loaded graph and tensor shapes only when requested. It acquires the same runner admission slot as generation.
- `/v1/inspector/trace` also owns that slot for its entire operation, so inspection and ordinary model execution cannot overlap on the same runner.
- Trace requests are limited to a 16 KiB question, 256 rendered prompt tokens, and at most 64 output tokens. Memory admission accounts for retained chat state and inspection scratch space.
- The trace installs temporary `RMSNorm` wrappers for supported residual observation sites. They call the original normalization object and record summaries of its input. No trained weight is substituted in the normalization calculation.
- Every trace uses a fresh KV cache. It does not replace the retained conversation cache with inspection state.
- Completion, model failure, and cancellation leave through `defer` cleanup. The stream synchronizes before the original normalization objects are restored and temporary observers are released. Laguna compiled block tails are disabled only in the inspection task, avoiding compiled traces that capture temporary wrappers.

An unarmed HTTP generation installs no recording wrappers and performs no activation-to-CPU capture or inspector serialization. The normal path checks whether a recording was explicitly armed after execution admission. Merely registering an HTTP route does not collect activations.

`POST /v1/inspector/recordings` arms bounded summary capture for the next admitted text generation. The captured request uses its actual uncached prompt suffix, with token coordinates mapped back to the full prompt. It disables speculative decoding and Laguna compiled block tails for observation, while preserving the client’s sampling, parser and response limit. The recorder passes logits through unchanged and calls the original normalization and projection objects. Passive recordings also reduce actual attention and dense/shared MLP projection outputs, and retain bounded selected expert IDs/normalized weights. Laguna observes router results; GPT-OSS mirrors the pinned router selection on bounded rows of actual router logits. This does not measure expert output vectors or causal importance. Inactive Laguna eager expert execution has a nullable recording observer check; no tensor capture occurs unless armed. It captures the last evaluated prompt token and a bounded decode window without an extra inference request. Polling and cancellation use the listener’s in-memory store without acquiring the model lease; cancellation stops collection without cancelling the conversation. Original modules are restored after the generation producer has joined.

The historical timing evidence below predates passive recording. It does not establish the overhead of the new inactive check or of an active recording.

## Remaining inactive runtime support

Laguna's public library also supports an explicitly installed routed-activation observer for external calibration tools. This is separate from the HTTP residual-summary trace. An inactive model retains nullable calibration fields, an observer check in eager expert execution, and a flag controlling compiled-tail eligibility. A compiled block-tail graph traces the inactive expert branch once; the callback check is not itself an additional GPU tensor operation.

These facts establish structural isolation, not a numerical zero-cost claim. Removing observer support solely because a branch exists would not establish a user-visible speed improvement. The PRD instead requires a matched inactive-versus-absent experiment.

## Verified cleanup

`swift test --filter ModelInspectionTests --jobs 4` passed all 11 tests on September 11, 2026. The tests run small models on the CPU and cover:

- Request bounds, metadata, supported residual geometry, and exact known-value summaries.
- Exact quantized logits during observation and after restoration, plus original module object identity.
- Invalid installation leaving the original graph untouched.
- Successful tracing and failure cleanup.
- Cancellation restoring original modules before subsequent ordinary inference.
- Observer release after restoration, including repeated cleanup.

These are lifecycle and correctness checks. They do not measure real-model GPU performance, total process memory release, or end-to-end HTTP disconnect timing.

## Reproducible absent comparison

The benchmark tools are in `benchmark-results/personal-prd-20260911/`:

- `prepare-inspection-absent.py` transforms only an expendable source checkout without `.git`. It strips the HTTP inspection implementation and normal-path Laguna calibration state/checks. It preserves generation, speculative-decoding capture, and ordinary kernels. It records before/after source hashes and refuses mismatched source boundaries. This is a benchmark control, not a supported production build mode.
- `inspection-comparison.py` compares that binary with the matching unstripped binary. It alternates arm order, starts fresh loopback servers, excludes warmups, and records identical prompts, input/output hashes, output lengths, first-text latency, total latency, and client-observed generation speed. All started processes are cleaned up by the harness.

Measure the two arms on one otherwise idle device, without concurrent compilation or other inference benchmarks. Do not replace installed binaries to conduct this comparison. The summary must include variation, output equivalence, and which hardware/model was tested; a small inconclusive delta is not proof of universal zero overhead.

## Selected capture and remaining verification

The HTTP trace now requires explicit layer and token-position selections, supports selected sites, and exports bounded float32 residual tensors as well as summaries. Negative prefill positions and prefill-only captures support external model tooling without requiring a decode step. Capture budgets are checked before wrappers are installed; selected rows alone are reduced or copied to the CPU. Arbitrary intermediate tensor capture is not supported. Model support remains limited to the supported native architectures advertised by `/v1/inspector/model`. No abliteration algorithm or training workflow is implemented here. The new implementation has its own integration tests and matched performance controls, recorded below; prior measurements do not automatically validate this larger source revision.

## Final selected-capture validation, September 11, 2026

The selected interface passed all 17 CPU inspection tests, including exact raw
float bits, sparse layer/site selection, token identity, memory preflight,
prefill-only execution, early-stop reporting, cancellation/failure restoration,
and exact ordinary logits after cleanup. The complete Midnight suite passed
with 381 registered Swift Testing tests (opt-in GPU/lifetime tests excluded from
the default invocation) and six XCTest tests. Separate focused CPU/Metal tests
cover the Mac gate/up cleanup workaround, and a deterministic cross-thread test
covers compiled-function ownership. Real Mac and CUDA capture checks passed;
the real ABSlayer bridge validated four selected vectors and its artifact receipt.
[Functional and cleanup evidence](../benchmark-results/inspection-vision-20260911/README.md).

The new fixed-forward comparison holds complete prompt/decode token arrays,
model settings, backend resources and dependency sources constant. Independent
unversioned copies differ in exactly three inspection-removal files. It times
native forwards with fresh KV state, without HTTP, sampling, tokenization or
startup. Every process performs two warmups and five measured passes; process
medians, rather than individual passes, are the independent paired observations.

On the M5 Max, all 32 process pairs (16 short, 16 long) passed exact full-logit
checks throughout and every owned process exited. Long decode and total latency
passed the declared 3% nonregression criterion. Short total paired change was
+0.25%, with a 95% interval [−0.24%, +0.61%], but common timing drift exceeded
the original 10% limit. Long prefill also had drift and a wider interval. Thus
the overall gate is **inconclusive**, not proof of universal zero cost.
[Complete Mac result](../benchmark-results/inspection-vision-20260911/inspection/fixed-forward/mac-results.md).

On the RTX 4090, all 16 short pairs completed but four had differing full-logit
hashes in one pass. Twelve valid pairs give a descriptive total-latency change
of −0.02%, interval [−0.28%, +0.30%]; the numerical failures disqualify acceptance.
Long inputs varied within a single build despite matching token counts/cache
lengths, including with CUDA graphs disabled. Their paired timing run was not
launched. The numerical cause remains unresolved; no exact-output criterion or
CUDA default was relaxed. Independent source review found no proven harness
lifetime error. [Complete CUDA result](../benchmark-results/inspection-vision-20260911/linux/fixed-forward/README.md).

These complete the measurement attempts and leave the strict acceptance gate
open. The measured implementation performed no recording or activation copies in ordinary
generation. Current unarmed generation also collects no activations, but structural
isolation and near-neutral paired estimates must not
be presented as a passed universal speed guarantee. The earlier measurements
below describe their own prior source revision.

## September 11, 2026: measured Linux comparison

The inactive-versus-absent experiment completed 48 valid requests on the RTX 4090 with Laguna XS 2.1 Q4R8: six alternating paired trials, fresh servers, two excluded warmups per server, temperature zero, context 8192, prefill 512, no KV compression, and a 20 GiB MLX memory ceiling. Both arms used the same published-release-derived benchmark build with the frozen personal-conversation cache route; they differ only by the recorded benchmark-only inspection transformation. This does not benchmark every later workspace change.

Values below are median [minimum–maximum] across six trials. First-text latency is client-observed request-to-first-content time.

| Workload | Inspection inactive, ms | Inspection absent, ms |
| --- | ---: | ---: |
| Short prompt, 64 output tokens | 100.3 [94.0–104.1] | 104.2 [95.7–108.3] |
| Short prompt, 256 output tokens | 92.7 [88.8–98.0] | 97.1 [94.5–102.2] |
| Fresh long prompt, 64 output tokens | 1267.8 [1237.6–1571.5] | 1252.0 [1109.0–1258.8] |
| Exact repeated long prompt, 64 output tokens | 1165.6 [1065.7–1249.9] | 1240.9 [1061.3–1271.1] |

For 256-token answers, post-first-text speed was 135.45 [121.08–137.65] tokens/s inactive and 136.45 [135.07–136.77] absent. The median paired difference was −0.73%, with a descriptive 95% bootstrap interval spanning −6.13% to +0.35%. All requested output lengths were reached. All 24 paired inputs were identical, but only 14/24 generated texts matched. Each arm also matched its own exact cold/repeat answer in only 2/6 trials, showing output variation within both implementations.

**The PRD performance gate remains inconclusive.** Removing inspection did not produce a consistent material generation-speed improvement. Fresh long-prompt first-text latency did favor absence (median paired difference +1.56% inactive, bootstrap interval +0.24% to +20.45%); this must not be described as proof of zero overhead. Other first-text medians favored the inactive implementation, and the first inactive trial had large latency/rate outliers. Six trials with variable output paths cannot establish a universal cost bound.

No production inspection implementation or settings were changed. The measured result supports retaining the existing isolated implementation while completing a stronger fixed-token and steady-state comparison before claiming the strict performance acceptance gate has passed. Eleven CPU lifecycle tests passed, including cancellation during the first observed forward evaluation and subsequent exact ordinary logits. They complement these measurements rather than substitute for them.

All own benchmark servers exited; GPU snapshots returned to 0% utilization and 33 MiB occupied. [The complete report](../benchmark-results/personal-prd-20260911/inspection-linux/README.md) contains total-latency/rate ranges, paired deltas, binary/source hashes, raw prompts and answers, and methodology. No active-inspection timing or Metal absent comparison was performed.

## Upstream comparison from downloaded source

Reviewed vLLM commit `c191787a6861868069bc4f6ed6f842af541de23a` and vLLM Metal commit `b7d419b253089cffa4e4791177ffd4227bca5ce9` on September 11, 2026. These are source findings, not runtime benchmark results.

| Surface | Upstream behavior | Implication for Midnight |
| --- | --- | --- |
| vLLM hidden-state extraction | An explicit `extract_hidden_states` mode accepts selected layer IDs and exports tensors plus token IDs to safetensors. Requests can include output-token states; prompt-only capture is the default. | This is a real capability gap relative to Midnight's summarized residual trace. A future external-tool interface should expose bounded selected tensors without making it normal-serving work. |
| vLLM extraction isolation | The extraction proposer and connector are created only when configured. That mode allocates capture buffers, copies selected tensors to pinned CPU memory, and uses writer threads. Its documentation says chunked prefill is incompatible. | Keep any equivalent as a dedicated inspection operation or worker, because its active execution and memory policy differ from the fastest ordinary path. |
| vLLM performance diagnostics | Profiler selection defaults to none; detailed timing, layerwise NVTX, KV residency, and CUDA graph metrics default off. NVTX module hooks are registered only when requested and have compilation limitations. | Model-level markers and request metrics are performance diagnostics. They do not by themselves provide activations to an abliteration tool. Defaults alone do not prove absent-versus-inactive parity. |
| vLLM Metal | A Metal capture profiler is created lazily on explicit start. The inspected speculative-method factory supports Gemma4 MTP, draft models, and n-grams; it rejects other methods, including the upstream extraction method. | Do not assume upstream CUDA inspection or speculative support automatically exists in the Metal plugin. |

Source evidence: [vLLM extraction guide](https://github.com/vllm-project/vllm/blob/c191787a6861868069bc4f6ed6f842af541de23a/docs/features/speculative_decoding/extract_hidden_states.md), [capture buffer implementation](https://github.com/vllm-project/vllm/blob/c191787a6861868069bc4f6ed6f842af541de23a/vllm/v1/spec_decode/extract_hidden_states.py), [connector implementation](https://github.com/vllm-project/vllm/blob/c191787a6861868069bc4f6ed6f842af541de23a/vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py), [observability defaults](https://github.com/vllm-project/vllm/blob/c191787a6861868069bc4f6ed6f842af541de23a/vllm/config/observability.py), [profiler defaults](https://github.com/vllm-project/vllm/blob/c191787a6861868069bc4f6ed6f842af541de23a/vllm/config/profiler.py), [Metal profiler lifecycle](https://github.com/vllm-project/vllm-metal/blob/b7d419b253089cffa4e4791177ffd4227bca5ce9/vllm_metal/v1/worker.py), and [Metal method dispatch](https://github.com/vllm-project/vllm-metal/blob/b7d419b253089cffa4e4791177ffd4227bca5ce9/vllm_metal/v1/model_runner.py).

## Model information telemetry

Managed `GET /v1/runtime` exposes optional `generationMetrics`, the latest text backend completion report for the current loaded model. The runner retains a small locked scalar snapshot when a metrics event arrives; reading it does not acquire the MLX execution lease or enable per-token monitoring. Invalid/nonfinite rates are not published. No token text, logits, or model arrays are stored in this snapshot. Unload/replacement removes it with the runner. This additional metrics-event bookkeeping has not been separately benchmarked. Allocator active/cache/peak measurements remain process-wide MLX figures, separate from checkpoint storage and configured memory admission limits. Studio’s Model Information view refreshes runtime data every two seconds and validates model identity/generation before reusing architecture metadata.
