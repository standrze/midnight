# Midnight product requirements

Status: initial product direction, September 11, 2026. Requirements describe the intended product; they are not claims that every capability already exists.

## Product purpose

Midnight is a local inference and inspection server for one person doing
cybersecurity model work on their own computer. Its primary job is to serve the
selected checkpoint reliably through a documented API, including compatible
exports from external abliteration workflows, while delivering the fastest
useful inference that the person's hardware can provide. Lowlight and other
applications consume its APIs.

**Single-user inference speed wins every feature tradeoff.** If an optional capability slows ordinary inference, it must be redesigned, isolated, disabled by default, or deferred. Team use and aggregate server throughput do not drive the current design.

Speed means producing a correct, usable answer sooner. Dropping requested work, silently reducing context or output length, or degrading model quality does not count as an optimization. Approximate methods and quality-changing configurations must be explicit choices and measured separately.

## Priorities

| Priority | Outcome | Decision rule |
| --- | --- | --- |
| P0 | Reliable API serving for local cybersecurity and externally abliterated checkpoints, with fast single-user inference | Every implementation and release decision is evaluated against this outcome. |
| P1 | Midnight Activation Capture: explicit activation and layer inspection for any API consumer | Capture may cost time while active. Its inactive state must not introduce measurable inference overhead. |
| P2 | Image understanding, including screenshots supplied by headless-browser clients | Add only when it preserves the P0 text path. Defer if this cannot be demonstrated. |

## Intended user and scope

The user runs a model locally, asks questions, continues conversations, and uses tools through applications such as Lowlight. The same user may occasionally inspect a model or use a vision-capable model to interpret a screenshot.

Apple silicon with Metal and supported Linux systems with CUDA are the primary accelerated targets. Performance is evaluated separately for each hardware and model combination. CPU fallback is useful for compatibility but must not constrain accelerated implementations.

One user can still generate overlapping requests from a browser, agent, or second application. Midnight must handle these safely; this does not create a requirement for team scheduling or continuous batching.

## P0 requirements: personal inference

1. Minimize time to first visible text, time to a completed answer, and pauses during streaming. Improve generation speed without sacrificing the first two measurements.
2. Prioritize the active conversation. Reuse compatible conversation state when that reduces latency; avoid rebuilding partial shared caches instead of retaining useful conversation state.
3. Choose caching by measured model and workload behavior. Shared-prefix caching is a fallback where beneficial, not a universal default that overrides faster conversation reuse. A disabled cache must bypass its work, not merely stop retaining results.
4. Keep prompt preparation, tokenization, memory copies, GPU synchronization, and allocation out of the repeated generation path wherever possible.
5. Avoid background model work and optional services that compete for compute, memory bandwidth, or GPU memory during ordinary inference. Expensive diagnostics and profiling are explicit operations.
6. Keep model loading, switching, unloading, and local model removal straightforward. Removal must distinguish deleting a downloaded model from unloading it and must not delete active or unrelated data accidentally. Extra model residency must earn its memory cost through measurements.
7. Preserve OpenAI-compatible Chat Completions and Responses behavior needed by Lowlight, including streaming, cancellation, tool use, and supported structured outputs. Pay feature-specific costs only when the request uses them. Document the supported subset.
8. Maintain a bounded, cancellable request queue. Do not add batching, fairness machinery, or simultaneous generation unless a single-user workload demonstrates a net responsiveness benefit.
9. Keep the runner independently buildable and usable. Client applications, inspection tools, and training packages must not become mandatory runtime dependencies.
10. Changes to cache behavior and other internal optimizations must benefit Lowlight through the existing API when possible. If a new client capability is required, implement and verify its Lowlight integration before treating that feature as complete.

### Planned: sampled speculative decoding and Laguna assistants

Added September 15, 2026. Status: implementation required; existing experimental assistant paths do not satisfy these acceptance criteria.

**Nonzero-temperature speculative sampling**

- Support assistant acceleration for sampled requests, starting with the Gemma assistant and Laguna DFlash paths. Today, nonzero temperature bypasses the assistant; preserve that fallback until each architecture is validated.
- Implement architecture-appropriate proposal probabilities, acceptance/rejection, and corrected sampling after rejection. Removing the temperature-zero guard alone is not an implementation. Preserve the target sampling distribution under supported temperature, `top_p`, and other enabled probability transformations.
- Validate distributional correctness against target-only sampling with tractable reference cases and repeated trials. Do not require identical sampled text from the same seed as proof of correctness; separately test deterministic behavior and document numerical differences.
- Cover rejection/cache rollback, stop tokens, output limits, streaming, cancellation, and consecutive requests. Unsupported tool or structured-output combinations must use an explicit, observable target-only fallback.
- Verify Chat Completions and Responses through Lowlight with nonzero temperature. Report the assistant actually used, proposed/accepted token counts, and fallback reasons with architecture-correct labels; loading an assistant alone is not evidence of use.

**Proper Laguna DFlash / assistant support**

- Establish a supported target/drafter pairing with pinned revisions, compatible tokenizer and embeddings, target hidden-state layer mapping, weight format, and quantization settings. Reuse a compatible pretrained drafter where available; do not assume an arbitrary small model or a Gemma assistant is compatible.
- Validate loading, target feature extraction, draft generation, verification, cache rollback, and model switch/unload behavior. Test greedy decoding first, then the sampled path above. Existing experimental Laguna DFlash wiring is a starting point, not completion.
- Investigate the existing negative Laguna DFlash result recorded in [personal PRD status](personal-prd-status.md): roughly 45 tokens/s versus target-only 141, with no exact paired-answer matches. Establish whether correctness, checkpoint compatibility, or runtime overhead explains the result before claiming effective support.
- If no suitable drafter meets the quality/performance gates, scope external drafter fine-tuning or training as a separate follow-up with a dataset, compute budget, and evaluation plan. Training orchestration remains outside Midnight's runtime.
- Benchmark target-only versus assistant decoding on representative chat, coding, long-output, and multi-turn workloads, separately on Metal and CUDA. Record acceptance, first-text and total latency, decode rate, draft/verification overhead, and peak/resident memory, including assistant quantization tradeoffs. Acceptance rate alone does not establish a speedup.

Completion requires correctness evidence, end-to-end Lowlight validation, and repeatable user-visible benefit on each claimed model/hardware combination. Keep assistants opt-in where gains remain unproven, and preserve a target-only path without assistant execution overhead. Update the API Field Guide alongside any implementation that changes client-visible fields or behavior.

## P1 requirements: opt-in Activation Capture

[Midnight Activation Capture](activation-capture.md) is Midnight's documented integration surface for consumers that need model structure, selected layers, and residual activations. Consumers include midnight-moonshine, ABSlayer, evaluation, and analysis tools. The public operations are `GET /v1/inspector/model` for discovery and `POST /v1/inspector/trace` for bounded capture. Model support, available observation points, and limits must be discoverable rather than assumed universal; arbitrary intermediate tensors are not currently supported.

- Inspection is off by default and enabled explicitly for a bounded operation or dedicated inspection session.
- Callers can select the layers, tensors, and token positions they need. Full-model capture is never an implicit side effect.
- Normal inference must not capture tensors, copy them to the CPU, serialize activations, synchronize the GPU for inspection, or emit per-token inspection events.
- Inspection buffers and handles have explicit lifetimes and memory limits. Completion, failure, and cancellation release temporary state.
- Inspection may be slower while active. The tool must make that state visible, and ordinary inference must not silently overlap with competing inspection work.
- If runtime hooks prevent the fastest kernels or execution strategy when inactive, use a separate execution path, build, or worker rather than compromise normal inference.
- Midnight owns Activation Capture and its consumer-neutral API. External tools own their datasets, training, transformations, analysis, and acceptance decisions. Midnight provides the bounded capture interface and serves compatible exported models.

Acceptance requires both a working inspection integration test and an inactive-versus-absent performance comparison. Merely leaving a flag unset is not evidence of zero cost.

## Model-specific voice and vision work

Voice and image capabilities remain part of the product. The user explicitly loads the model they want to use; that model may require text, speech, or vision components. The requirement is isolation of ordinary text inference, not simultaneous residency of separate models.

- Loading a text model must not load speech or vision weights, preprocess audio/images, start their generation tasks, or allocate their modality-specific caches. No speech or image work belongs in the text token loop.
- Explicitly loading a voice or vision model activates the components that model needs. It may replace the resident text model through the normal drain/unload/load lifecycle. Do not keep every modality resident by default.
- Optional workers, device contexts, and model resources are initialized on actual use. Generic cleanup must not initialize an unused modality's execution machinery merely to tear it down.
- Model changes are explicit. An unsupported input returns a clear capability error; do not silently swap a model, discard an image, or alter context/quantization to make a request fit.
- Unload/cancellation drains producers and releases the selected model's temporary resources before another model takes over. Preserve the user's saved configuration and client transcript; disclose any reload/prefill cost after an explicit switch.
- Optional vision implementation and dependencies remain outside the ordinary text build. Existing voice support must continue working; linked code alone is not evidence of active inference work or of a speed regression. Any dependency/startup cost must be measured separately from per-token performance.
- Shared GPU compute and memory still impose costs when a voice or vision model is actively used. Separate processes do not make concurrent work free. Do not add automatic background or overlapping modality work to ordinary text inference.

## P2 requirements: image input

- Accept explicit image input from clients using the supported OpenAI-compatible multimodal request shape.
- Support screenshots generated by headless-browser tools without embedding browser automation in the runner. Capture belongs to the client or tool.
- Require explicit selection of a compatible vision model. A text-only model continues to reject image input clearly.
- Build the vision runtime independently, with explicit model configuration and its own model state. Installing or making it available must not automatically launch a worker, download/load a model, or warm image caches.
- Keep a selected vision model resident for its intended session when memory allows; do not reload it for every image merely to achieve isolation. Its active resource use and cold-start cost must be visible and bounded.
- Allocate and execute image preprocessing and encoding only for inputs that need them. A text-only prompt explicitly sent to a vision model is a different workload from an ordinary text-only model.
- Bound image dimensions, image count, and memory use. Document detail-versus-latency choices.
- Lowlight must support explicit vision endpoint/model selection and image submission without silently changing its text conversation. Existing text and voice workflows remain usable.
- Verify an end-to-end Lowlight image request and a headless-browser screenshot workflow before calling the supported vision path complete. State platform/model limitations explicitly.

### Modality isolation acceptance

Compare identical text models and workloads before and after optional modality changes, including a build/path without vision. Test disabled/unloaded vision and text inference after an explicit voice/vision-to-text switch. Measure cold start, first-text latency, generation speed, retained/peak memory, and Lowlight conversation behavior. A repeatable ordinary-text regression beyond baseline variability blocks shipping the change.

Test unsupported inputs, insufficient memory, cancellation, and worker/model initialization failure. These must not trigger hidden fallback, processing by the wrong model, or retained work after unload. Active voice/image latency and explicit switching costs are reported separately; a model explicitly selected for another modality is not expected to match the text model's timings.

Image support remains deferred on any platform where this isolation or basic correctness cannot be demonstrated.

## Performance evidence and release gates

The product does not use a single universal tokens-per-second target. Baselines are recorded per machine, model revision, quantization, context size, and runtime configuration.

Every inference-affecting change must be compared with the current baseline using:

| Workload | Primary measurements |
| --- | --- |
| Short one-off prompt | First-text latency and total latency |
| Fresh long prompt | Prefill/first-text latency and peak memory |
| Exact repeated long prompt | Reuse benefit, first-text latency, and retained memory |
| Multi-turn Lowlight conversation | First-text latency per turn and completed-answer latency |
| Long output | Generation rate, total latency, and streaming stalls |
| Cancellation followed by another request | Cancellation cleanup and next-request responsiveness |
| Cold start and model switch | Time until the model can answer and peak memory |

Use identical recorded inputs for controlled comparisons, and separately test natural conversations that carry generated answers forward. Record output lengths and quality checks; temperature zero alone does not establish output equivalence. Exclude and disclose warmups, rotate test order, repeat trials, and report variation as well as medians. Measure Lowlight end to end in addition to server timings.

For optional features, compare the inactive implementation against a build or path with that feature absent. A repeatable regression beyond baseline measurement variability blocks default enablement. Uncertain measurements are inconclusive, not a pass. Microbenchmarks alone cannot establish user benefit.

Before shipping an optimization, record what improved, what regressed, and which workloads it affects. If fresh and repeated prompts trade off, select a model-specific policy supported by evidence instead of hiding the tradeoff in an overall average.

## Explicit non-goals for this phase

- Team accounts, tenant isolation, quotas, billing, and distributed serving.
- Maximizing aggregate throughput at the expense of the individual's response time.
- Continuous batching or speculative concurrency without a demonstrated personal-use benefit.
- Embeddings, vector databases, and retrieval/RAG; these remain shelved.
- Anthropic API compatibility.
- Always-on activation capture, profiling, or heavyweight telemetry.
- Built-in browser automation, training orchestration, or weight-transformation
  algorithms. External abliteration tools own transformations; Midnight owns
  bounded inspection and serving of compatible exports.
- Feature parity with Ollama, vLLM, or other runners as an objective in itself.

## Immediate execution order

1. Verify the conversation-first cache routing change against controlled histories and natural Lowlight conversations on Mac and Linux. Check output behavior as well as latency before deployment.
2. Establish reproducible per-model baselines and use them to find the next largest source of single-user delay.
3. Audit inactive inspection hooks and demonstrate their lack of measurable impact. Isolate any that fail that test.
4. Evaluate image input only after the P0 baseline and isolation requirements are in place.

Any future proposal must answer: **Does this make one person's local inference faster? If not, can it be completely out of the way until explicitly requested?** If neither answer is yes, it is outside the current product direction.
