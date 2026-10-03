# Personal-runner workstream results

September 11, 2026. This records the earlier seven workstreams against the [product requirements](product-requirements.md); their measurements below are historical. The subsequent [CUDA and modality-isolation update](../benchmark-results/cuda-vision-20260911/README.md) records the newer builds, optional vision work and validation. The overriding decision rule remains single-user response speed with usable answers. Completing an experiment does not imply its candidate should ship.

The newer [P0 performance pass](../benchmark-results/p0-performance-20260911/README.md)
adds a repeatable regression harness and tests shared-cache and Laguna CUDA
optimizations. The cache prototype improves fresh-long Mac latency by about 4%,
but neither candidate clears its complete gate; both remain shelved and production
is unchanged. The original runtime implementations were restored after testing.

The completed [selected inspection and managed vision implementation](../benchmark-results/inspection-vision-20260911/README.md)
provides bounded, caller-selected residual vectors and normal model selection
for the separate Mac FastVLM worker. Lowlight/Lowlight-browser switch to vision
and restore text configuration; ABSlayer has a typed selected inspection client,
and Studio supplies explicit selections. Final functional, cancellation, file
protection, and real-model cleanup checks passed on their supported platforms.
The full Midnight/client suites and locally built candidates are recorded in
the report. This campaign has not changed production installations.

Two Mac compiled-model retention issues were corrected without disabling
compilation. The final large-text → vision → text/unload sequence returned to
17,376 active MLX bytes and zero allocator cache. Focused CPU/Metal tests confirm
matched output parity and repeated model cleanup.

Performance measurements are complete, but their strict overall acceptance
gates remain **inconclusive**. All 352 measured ordinary Mac requests succeeded,
with all 176 paired responses identical. LFM's principal inference paths and
all tested Laguna inference latencies passed the 3% criterion; some startup and
conversation intervals and an unchanged LFM marker-format failure do not clear
the complete harness gate. The Mac inactive-versus-absent run matched all full
logits, but common timing drift exceeded the declared limit. CUDA repeatability
failures disqualified its corresponding gate. No criteria were relaxed and no
universal zero-cost claim follows. The historical deployment below does not
represent these new builds.

| Workstream | Result | Product decision |
| --- | --- | --- |
| 1. Active-conversation caching | On the RTX 4090 with Laguna XS 2.1 Q4R8, controlled follow-up first-text medians fell from about 1.05–1.12 seconds to 0.15–0.23 seconds. Natural Responses conversations showed the same large benefit. | Ship conversation-first routing for eligible models. Preserve LFM's shared-prefix fallback and its measured fresh-versus-repeat tradeoff. |
| 2. Fresh Ollama/vLLM comparison | Same canonical Qwen 2.5 0.5B F16 values, verified tensor conversion, matched request settings and one active request. Mac long-output speed: Midnight about 395 versus Ollama 312 tokens/s. Linux: Midnight about 358, Ollama 547, vLLM 569. | Midnight is not universally fastest. The measured CUDA gap is the strongest new P0 profiling target. These small-model results do not rank unsupported Laguna/LFM checkpoints. The six simple quality probes passed only 4/6 for Midnight, 3/6 for Ollama, and 4/6 for vLLM; these timings are not a quality gate. |
| 3. Host work between tokens | Direct KV slice updates and incremental ByteLevel decoding already exist. An isolated persistent-compile-handle prototype passed correctness checks but gained only about 0.3–0.8% in median paired decode measurements, with inconsistent wins. | Keep existing optimizations. Shelve the prototype; it is absent from the installed source and dependency preparation. |
| 4. Model/hardware defaults | Laguna prefill 2,048 exceeded the same 20 GiB memory admission ceiling at context 8,192, producing no valid speed sample. | Keep prefill 512, explicit context/output choices, and existing memory defaults. No background autotuner or silent reduction in requested work. |
| 5. Proven speculation | The completed 20-request DFlash follow-up ran at about 45 tokens/s versus target-only 141, with slower tails and no exact paired-answer matches. | DFlash remains opt-in and off by default. Broader speculation is not justified by this result. |
| 6. Inspection isolation | Eleven lifecycle tests pass. A 48-request inactive-versus-absent Linux comparison did not show a consistent material speed benefit from deleting inspection, but the strict zero-overhead gate remains inconclusive. | Retain the existing explicit inspection path. Do not claim proven zero cost. Selected raw tensors/layers were a gap in this measured revision; the newer implementation has passed functional validation; its completed performance measurements leave the strict acceptance gate inconclusive. |
| 7. Model removal and Lowlight | Managed-download listing, unload, and removal are implemented. Updated processes hold lifetime file leases; removal rejects active model/dependency files. Lowlight exposes `/models downloads`, `/models unload`, and `/models remove NAME`. | Ship both sides together. Preserve `/model NAME`. File-lease work occurs during model lifecycle/management, with no added per-token work. |

## Earlier validation and rollout

The earlier Mac source passed **352 Midnight Swift Testing tests in 62 suites plus 6 XCTest tests**, and **152 Lowlight tests in 7 suites**. Both optimized binaries are installed; the selected model and settings on port 8080 were restored, and the secondary listener on port 18080 remains empty. Every known Mac Midnight listener was updated to participate in the advisory lease protocol.

Eight installed API/client checks passed, including Chat Completions, Responses creation and recall, and Lowlight using each API. The actual Lowlight TUI also passed continuation, expired-response recovery, and session resume. A separate installed TUI observation measured three turns and successfully used `/models downloads`; it retained an incorrect backward-spelling answer as a quality limitation. It is an absolute observation, not a before/after speed comparison.

Linux passed **15 model-management/lease tests in two suites**, including a directory URL-hint regression discovered during validation. These used a temporary focused package graph because an unrelated benchmark target references an API unavailable in the pinned Linux MLX. The exact original manifest was restored before the optimized production rebuild; this is not a full Linux suite pass.

The final lease cleanup fix also passed **four standalone Linux scenarios** using the exact production source, with a failing-before/passing-after reproduction. It explicitly releases an owner's lock even while a duplicate or inherited descriptor remains open; forked-child destruction cannot unlock the parent's active lease. The final Mac full suite includes a regression for this behavior. No per-token work was added.

Both optimized Linux binaries are packaged and installed. All eight installed API/client checks and both listing-command spellings passed on Mac and Linux. Linux's temporary verification server was stopped, preserving the machine's previous idle state; the final check showed no Midnight process and GPU utilization 0%. See the [deployment record and installable archives](../benchmark-results/personal-prd-20260911/deployment/README.md).

These are local builds of the existing version lines, identified by their source and binary hashes. This workstream does not create a new public GitHub tag or release. Earlier installed payloads remain available for rollback. Advisory leases protect cooperating updated Midnight processes; older binaries and unrelated applications must be upgraded or stopped before relying on cross-process removal protection.

## Next priorities

Planned September 15, 2026: implement nonzero-temperature speculative sampling and establish properly validated Laguna DFlash/assistant support. The [PRD acceptance criteria](product-requirements.md#planned-sampled-speculative-decoding-and-laguna-assistants) require distribution-correct sampling, compatible checkpoint pairing, actual-use diagnostics, Lowlight verification, and measured benefit over target-only decoding. These remain pending; the earlier negative Laguna benchmark is not superseded by this plan.

1. Follow up the completed CUDA Qwen trace: investigate matrix-vector execution and graph preparation/submission. Larger graph budgets and disabling graphs both regressed measured decoding, so current defaults remain. Profile a larger supported checkpoint before generalizing from a 0.5B model; use controlled quality checks alongside timing.
2. Investigate cached-first-text overhead: Ollama beat Midnight on the Mac's repeated/shared Qwen prompts even though Midnight's decode was faster. Keep the existing LFM fresh-prompt cost visible when evaluating cache policy.
3. Resolve the observed CUDA numerical repeatability and benchmark timing drift before claiming the strict cross-platform inspection performance gate passes. The selected activation export, managed vision lifecycle, and Mac cleanup corrections are implemented and functionally validated; their complete performance acceptance is still unproven.

Images remain secondary. The user's clarified isolation rule applies to both voice and vision: explicitly loading those models may load their required components and replace the text model. An ordinary text model must perform no image or speech processing and must not initialize unused modality workers. Simultaneous residency is not required. The earlier modality-isolation campaign installed updated Midnight and Lowlight builds on Mac and Linux and validated separate FastVLM screenshot requests and voice-to-text switching. The current campaign implements normal FastVLM model selection with explicit worker startup, independent file leases, cancellation/drain, and text restoration; final cleanup passed; the complete performance gate remains inconclusive for the reasons above. The worker remains a separate Mac build. CUDA vision, additional vision architectures, and vision Responses/tool/structured-output support are deferred. RAG, team scheduling, and Anthropic compatibility remain outside this phase.

## Evidence

- [Current selected inspection and managed vision campaign](../benchmark-results/inspection-vision-20260911/README.md)
- [Complete benchmark report and raw records](../benchmark-results/personal-prd-20260911/README.md)
- [Compile-handle experiment](../benchmark-results/personal-prd-20260911/stateless-compile-prototype/README.md)
- [Inspection measurements](../benchmark-results/personal-prd-20260911/inspection-linux/README.md)
- [Model-removal lease validation](../benchmark-results/personal-prd-20260911/model-removal-leases/README.md)
- [Installed Lowlight TUI observations](../benchmark-results/personal-prd-20260911/mac-installed-lowlight-tui/README.md)
- [Installation manifests and checks](../benchmark-results/personal-prd-20260911/deployment/)
