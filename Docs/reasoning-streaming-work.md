# Reasoning streaming and Gemma 4 26B-A4B — implementation in progress

User target: optional live model-emitted reasoning in Midnight, Lowlight, and
Lowlight Relay, followed by measured Gemma 4 **26B-A4B** optimization on Mac and
RTX 4090. Do not substitute another model or claim wire tests prove end-to-end
completion. Preserve existing uncommitted work in all repositories.

## Current implementation

- Chat Completions accepts optional `include_reasoning`; omission is opt-out.
- Delta and message output support separate `reasoning_content`.
- Pinned `mlx-swift-lm-reasoning-stream.patch` preserves decoder reasoning as
  `Generation.reasoning`; preparation applies it through the existing patch flow.
- Both Midnight generation loops forward these as `LocalModelRunnerEvent.reasoning`.
- HTTP Chat Completions streams/collects reasoning only when explicitly enabled.
- Benchmark consumers ignore reasoning text but retain generation metrics.
- Swift build passed after adding the runtime event. Focused wire tests passed
  before runtime integration; the follow-up run log is
  `/private/tmp/midnight-reasoning-tests.log` (check result before relying on it).

## Required remaining work

- Sliding-window decode reads were evaluated with eight A4B pairs beyond the
  window boundary. Median paired gain was only +0.22%, and output equality did
  not hold. Keep `MIDNIGHT_GEMMA4_WINDOW_SLICING` disabled by default.
- The Gemma text MTP adapter is implemented and CPU-tested. Both text wrappers
  now provide scaled embeddings and opt-in hidden/KV snapshots with absolute
  offsets/source indices. The actual assistant works with either wrapper and
  bounds its own sliding read without mutating target state. Small-model token
  sequences match ordinary greedy decoding at block sizes 2/4 with actual
  proposals. The official A4B QAT assistant's 48 tensors load successfully.
  **Remaining:** LocalModelRunner still accepts only Laguna DFlash drafters;
  wire Gemma loading/compatibility checks, benchmark the actual A4B pair, then
  expose an option only after validated token agreement and speed measurements.

- The long-prompt sweep found and fixed missing sliding-window masking in the
  retained-cache Gemma path. The 1,791-token A4B prompt now produces a full
  256-token answer at all four chunk sizes. Wording is not bitwise identical
  across quantized chunk sizes. The tiny CPU regression fails on the old mask
  for chunks 1/2/4/8 (max logit errors 2.12–3.78) and passes with the fix at
  tolerance 1e-4. No prefill default was changed.
- Cancellation during reasoning now prevents an answer callback from the same
  combined SSE event in both clients. The regression failed before correction;
  all 19 HTTP-session tests in each client pass afterward.

- Mac dense-activation fusion was tested with eight alternating pairs on one
  loaded A4B model. Median paired change was −0.07%; all outputs matched.
  No default optimization was justified. Results and reproduction scripts are
  in `benchmark-results/gemma4-a4b-reasoning/`. Continue with MoE paths and
  prefill measurements; 4090 host access remains unresolved.

- Responses opt-in events and final items are implemented and unit tested;
  continue checking cancellation and tool boundaries.
- Lowlight and Lowlight Relay now send the option to Midnight-owned endpoints;
  request capture and recorded real A4B replay tests pass. Both terminal UIs
  have been exercised with the recorded stream, reasoning on and off.
- Gemma's `enable_thinking` template flag is wired through prompt preparation.
- Verify streaming with real model output, disabled exposure, split markers,
  tool transitions, cancellation, and no answer contamination.
- Add dependency-patch idempotency/decoder tests and document the patch.
- Review useful Gemini CLI ideas for both clients; adapt relevant streaming
  lifecycle and presentation behavior, preserving Swift and local endpoints.
- Locate exact A4B weights and available Mac/4090 hosts, establish matched
  baseline, optimize and record throughput/latency/memory and quality checks.

## Actual A4B HTTP verification

Checkpoint: `mlx-community/gemma-4-26B-A4B-it-qat-4bit`, pinned revision
`0e3cbab38ce568cf6e23543010d08d03b731910c`.

The first live test exposed a real parser bug: Gemma's thought delimiters and
reasoning were reaching `delta.content`. The dependency patch now routes
`<|channel>thought` / `<channel|>` through `ReasoningEventEmitter` before tool
parsing, and reports its reasoning state to the token decoder contract.

All eight live combinations passed: Chat Completions and Responses, streaming
and nonstreaming, reasoning enabled and disabled. Every answer was `323`;
enabled requests had separate reasoning, disabled requests had none, and answer
text had no channel delimiters. Raw responses, summary, and Ruby reproduction
script are in `benchmark-results/gemma4-a4b-reasoning/`. These are correctness
smokes, not representative throughput benchmarks. The standalone Mac baseline
is measured separately. RTX 4090 access remains pending the user's SSH address.

## Sources inspected September 13, 2026

- https://huggingface.co/google/gemma-4-26B-A4B-it
- https://docs.vllm.ai/en/latest/features/reasoning_outputs/
  Current vLLM calls the field `reasoning` (formerly `reasoning_content`). Both
  Lowlight clients already decode these aliases. Gemma 4 requires thinking enabled.
- https://docs.ollama.com/capabilities/thinking
- https://github.com/google-gemini/gemini-cli
  Review actual relevant source before selecting adaptations; README alone is
  insufficient to claim an implementation was imported.

No performance improvement or full feature completion is claimed yet.

## Next client integration decision

Both client model catalogs can decode owned_by=midnight to gate the custom
include_reasoning request extension. Do not send Midnight-only fields to arbitrary
hosted providers. Connect each app’s showThinking option to per-request exposure
and filter received reasoning when disabled. Existing client callbacks always
accept reasoning today; wire and settings edits remain outstanding.

## Client progress

Both Lowlight and Lowlight Relay now decode catalog owned_by and set the
Midnight exposure capability at connection time. Their existing showThinking
preference is passed per generation, the request extension is gated to Midnight,
and callbacks are suppressed when disabled. New transport preference tests and
existing Responses stream suites pass: Lowlight 18 tests; Relay 19 tests.
Logs: /private/tmp/lowlight-reasoning-tests.log and
/private/tmp/lowlight-relay-reasoning-tests.log. Still need HTTP request/callback
integration tests and actual model verification; enabling the Gemma template
flag is not implemented yet. Earlier remaining-work bullets are historical
where superseded by this update.

## Gemma template progress

Gemma preparePrompt and stream now accept thinkingEnabled and pass
enable_thinking through the model template. The prepared-prompt identity checks
the preference; fallback rendering preserves it. HTTP routes derive this from
include_reasoning. Structured-output requests suppress Gemma thinking because
their raw JSON path cannot mix reasoning with grammar output. Eight focused
wire/runtime tests pass (/private/tmp/midnight-thinking-template.log).

Mac capacity: 64 GiB, Mac17,7. Native download dry-run confirmed
mlx-community/gemma-4-26B-A4B-it-qat-4bit at
0e3cbab38ce568cf6e23543010d08d03b731910c: 15.64 GB, 11 files. Download
started via midnight; log /private/tmp/midnight-a4b-download.log. Check live
process handle from thread before deciding whether to restart. User has a
pending asynchronous question for the RTX 4090 SSH address/checkpoint path.

## Live validation starting

A4B download finished successfully. Release product midnight built successfully.
Both clients passed real URLSession request preference tests for both API paths,
with enabled/disabled and known-Midnight/other-provider combinations. Logs:
/private/tmp/lowlight-reasoning-http-tests.log and
/private/tmp/lowlight-relay-reasoning-http-tests.log.

Local Metal server started on 127.0.0.1:21326, served name gemma-4-26b-a4b,
context 4096, max output 1024, prefill 512. Log /private/tmp/midnight-a4b-server.log.
Inspect live process and actual responses before claiming this launch succeeded.

## September 13 continuation — A4B assistant measured, held

The earlier startup notes above are historical. Current evidence is in
`benchmark-results/gemma4-a4b-reasoning/README.md` and its raw reports.

- Optional reasoning has been exercised through both HTTP API styles and both
  Swift clients. Both clients also pass cancellation-during-reasoning tests.
- Gemma retained-cache sliding-window masking was corrected and regression
  tested. Dense fusion and window slicing did not establish a useful paired
  speed gain; both remain off by default.
- The official A4B QAT assistant loads, and its text-target MTP adapter passes
  small-model token-equivalence tests. The runtime now validates checkpoint
  geometry and can use the factory-registered assistant. Only the benchmark
  links/registers MLXVLM; ordinary Midnight retains its prior dependency scope.
- Actual warmed A4B block-2 runs: target median 127.21 tok/s, assistant 124.17,
  accepted 108/145 draft proposals each trial. Greedy outputs diverged at byte
  212. Block 4 was slower in an initial smoke run. No assistant is enabled in
  the installed or ordinary CLI runner.
- A temporary serial verifier still diverged. Its source change was restored
  byte-for-byte; diagnostic artifacts are retained. Root cause is unproven.
- Final focused tests pass: compatibility, MTP state/token equivalence and
  sliding masks (six tests across three suites, with parameterized cases).
  Patch roundtrip passes. No optimization gain is claimed from these tests.

Remaining: diagnose actual A4B speculative divergence, investigate a worthwhile
speed improvement (a smaller quantized assistant is a candidate, not a result),
and measure the exact A4B model on RTX 4090. The requested SSH/checkpoint details
have not been supplied in this task. No release, installation, or GitHub push
was performed during this work.

## Follow-up — explicit text factory and actual batch divergence

Found and fixed a loader confound: MLX automatic factory discovery prefers
MLXVLM when an optional caller links that library. The actual diagnostic
records `MLXVLM.Gemma4` versus explicitly requested `MLXLLM.Gemma4Model`.
LocalModelRunner now uses `LLMModelFactory.shared` explicitly, preserving the
text runner implementation when assistant/vision libraries are present.
RuntimeBenchmark records `model_implementation` in every new report.

The earlier 127.21/124.17 measurements and byte-212 divergence describe the
other implementation, not Midnight's ordinary text path. The earlier serial
reference did not exercise its patched code. These reports remain historical
and are explicitly superseded in the benchmark README.

The corrected text block-2 benchmark gives 128.16 versus 128.97 tok/s across
three warmed pairs, but diverges at byte 539. The 53-token fixed-prefix test
matches the benchmark's ordinary output; 192 MTP-state forwards have exactly
zero logit error. Target-only batches of two/four tokens flip the winner at
step 108, without running a drafter. The correctly routed serial verifier
matches all 256 generated tokens, at lower speed. Thus the assistant state
adapter is not responsible for this observed difference; target batching is.
Widening floating-point parameters to float32 retains batch differences and
is only a diagnostic, not a promoted fix. All temporary dependency edits were
restored, and patch roundtrip verification passes.

The new large-checkpoint test is opt-in through MIDNIGHT_TEST_GEMMA_TARGET and
MIDNIGHT_TEST_GEMMA_DIAGNOSTIC_OUTPUT. Its optional precision experiment uses
MIDNIGHT_TEST_GEMMA_FLOAT32=1. Use debug `swift test` for this repository: the
release test launcher entered the Midnight argument parser before running
Swift Testing tests; no test passed in that failed launch. Runtime performance
reports still come from release executables.

No meaningful assistant speed gain is established, and no assistant/default
precision change is enabled. Further performance work and the exact A4B/4090
measurements remain outstanding.

## Q4 assistant candidate — measured improvement, still experimental

Added assistant-only in-memory affine quantization (4/8 bits, group size 64)
to the experimental runner initializer and benchmark. Target weights are not
modified. The loader rejects already-quantized assistant sources and verifies
actual conversion; reports record the precision and count 22 quantized linear
modules. Normal CLI defaults and installed binaries are unchanged.

Best measured setting is Q4/block 4. Predeclared 512-token confirmation:
- Tutorial: 121.96 → 140.73 tok/s, median paired +15.92%, six positive pairs.
- Prose: 110.72 → 121.61 tok/s, median paired +7.73%, four positive pairs.
- Long prompt (1,791 tokens), 256 output: 112.01 → 122.65 tok/s,
  median paired +10.01%, four positive pairs.
- Swift LRU: large timing drift in both arms; do not promote the apparent
  larger gain. Post-run snapshots do not establish the cause.

Eight predeclared answer checks pass for both baseline and Q4 assistant with
both block sizes 2 and 4, including JSON and 3,318-token retrieval. This is a
small screen, not a general quality guarantee. Free-running outputs still
vary with batched target evaluation. Benchmark reports now explicitly mark
such runs failed_output_parity rather than hiding the nonzero exit.

Reproduction scripts, immutable raw trials, quality fixtures and summary are
under benchmark-results/gemma4-a4b-reasoning; q4-summary.json gives the scope.
Seven focused compatibility/MTP/sliding-window tests pass after the change.
The experimental assistant implementation is still registered only by the
benchmark through MLXVLM. A user-facing fast path must preserve the normal
runner's modality isolation and document greedy-only, non-bitwise behavior.
No RTX 4090 connection/checkpoint has been supplied for this task yet.

## Native A4B assistant and normal runner integration

The earlier benchmark-only limitation is now removed in the workspace build.
The assistant is implemented in MLXLLM using the text decoder and explicitly
registered by LocalModelRunner when requested. Ordinary decoder construction
is unchanged. The release midnight binary has no MLXVLM symbols.

CLI options `--gemma-assistant-model`, `--gemma-assistant-block-size`, and
`--gemma-assistant-quantization-bits` flow through preflight, model switching,
restoration and managed file-use protection. Equivalent model-load API and
model-stack JSON fields are supported. Incorrect pairs, unsupported engines,
and conflicting drafter options fail preflight. Settings do not leak to another
selected target. See Docs/gemma-a4b-assistant.md for the greedy-decoding and
non-bitwise-output limitations.

Sixteen tests in five suites passed, including actual official assistant weight
loading, native/reference forward comparisons in BF16 and Q4 (zero measured
hidden/logit error for both checked offsets), window masks, MTP state and loader
preflight. The actual CLI server passed all eight Chat/Responses × streaming ×
reasoning combinations. Raw responses and binary/patch provenance are in
benchmark-results/gemma4-a4b-reasoning/native-assistant-reasoning. The temporary
test server was stopped; installed binaries and releases were not changed.

## RTX 4090 work paused by the user

The Linux host was located through the existing Linux build task. An isolated
candidate was staged at `/home/sandrzej/release-builds/midnight-gemma-a4b-20260913`.
No GPU inference was started and the existing GPU services were not modified.
The user then requested a temporary pause of 4090 use. The candidate build is
terminal and the target checkpoint transfer was interrupted; assume the staged
target is incomplete. The assistant transfer completed. Do not resume remote
builds, transfers, or GPU tests until the user explicitly resumes 4090 work.

Linux preparation exposed two patch portability issues, now fixed locally:
explicit new-file metadata and final-stack recognition of overlapping tuning
declarations. The generated Swift source is unchanged. The patch round-trip
test now verifies recognition of every patch in the fully patched checkout.

## Metal-only continuation: expert activation screen

The user explicitly allowed continued Gemma/Metal work while keeping all 4090
work paused. A fused GELU-times-up expert activation was tested on the exact
A4B QAT target. Four alternating process pairs, two measured 512-token trials
per process and one excluded warmup gave +0.113% median paired decode change
(−1.978% to +1.968% across pairs). All 16 outputs matched exactly. This is not
a convincing speed gain; the candidate was removed from the active patches
and dependency source. Its patch, test fixture, frozen plan and raw results
are archived under benchmark-results/gemma4-a4b-reasoning/expert-fusion-screen.

The next candidate is combining the experts' separate gate/up projections.
MLXLMCommon already provides FusedGateUpSwitchGLU, while this Gemma text model
currently constructs SwitchGLU and loads separate gate/up tensors. Any change
must preserve quantization scales/layout, weight loading and exact-output
checks before considering a speed claim. No gate/up fusion changes have been
implemented for Gemma yet.

## Metal-only combined expert projection candidate

The next candidate is now implemented behind
`MIDNIGHT_GEMMA4_EXPERT_GATE_UP=1`, default off. It uses
FusedGateUpSwitchGLU and concatenates the packed gate/up weight, scale and bias
rows during loading. It does not requantize or modify checkpoint files.
Actual layer-0 A4B expert comparisons were exact at token batches 1, 4 and 8.

The full target-only screen gave +3.651% median paired decode throughput,
with all four pairs positive and all 16 measured generations identical.
The Q4-assistant combination was inconclusive for speed: +1.365% median,
two regressing pairs and a −1.369%...+7.187% range; all 16 outputs were equal
within that comparison and all trials proposed draft tokens. Eight predeclared
answer checks passed for both modes with combined projections enabled,
including the 3,318-token retrieval case.

Evidence is under benchmark-results/gemma4-a4b-reasoning/expert-gate-up-screen.
This supports broader target-only workload testing, not a default change or
an additive assistant speed claim. The prototype has only been validated with
the exact affine Q4/group-64 checkpoint; generic quantization overrides and
adapter compatibility need explicit validation before user-facing promotion.
The RTX 4090 remains paused; no remote commands were issued in this continuation.

The final filtered regression command did not execute tests: compilation of
Tests/ModelRunnerProtocolTests/TalkieTokenizerTests.swift failed at line 41
(`extra argument 'addGenerationPrompt' in call`). Log:
/private/tmp/midnight-gemma-gate-up-regressions.log. Do not report that regression
run as passing. The separately executed actual-expert diagnostic, release
benchmark, 32 exact-output measured generations across the two comparisons,
eight-case answer screen, and patch round-trip are the completed evidence.

## Second Metal workload and preflight verification

Prose confirmation measured +3.439% median paired target-only throughput with
all four pairs positive and all 16 outputs identical. Together with the
tutorial (+3.651%), the target-only gain repeats on two workloads. The
assistant combination remains inconclusive.

GemmaExpertProjectionCompatibility now rejects unsupported engines, adapters,
non-A4B geometry, and incompatible expert quantization before a model switch or
direct runner initialization. The actual downloaded checkpoint passes this
preflight and again gives exact expert outputs at token batches 1, 4 and 8.

The test-compilation blockers were resolved with explicit tokenizer protocol
arguments, an explicit EOS-token Set conversion, and extracting dictionary
removals before Testing macros. These preserve the intended checks. The new
compatibility test also needed its ModelRunnerProtocol import. A focused run
passed 27 tests in 10 suites; a separate actual-expert/preflight test passed.
Logs: /private/tmp/midnight-gemma-focused-final.log and
/private/tmp/gemma-expert-eligibility-tests.log. Two Talkie Q4 numerical tests
still fail in the broader run; their tolerances/assertions were not changed.
No 4090 work has resumed.

The normal release midnight build succeeded after these checks. With combined
projections enabled, its actual A4B server passed all eight reasoning/streaming
cases. A rejected CPU model-load request returned HTTP 400 while preserving
the active Metal model and modelGeneration. Evidence and binary provenance are
under expert-gate-up-screen/reasoning, rejected-load.json and verification.json.
The temporary server was stopped; no installed binaries or releases changed.

## Current client integration audit

Rebuilt and ran `swift test --filter ResponsesSessionTests` in both local
Lowlight and Lowlight Relay checkouts: each passed 19 tests in one suite.
The inspected tests replay recorded A4B Chat Completions and Responses streams
with reasoning enabled and disabled, assert the answer remains `323`, keep
reasoning out of conversation history, and verify Midnight-only request options.
They also cover cancellation during a reasoning callback before an answer in
the same event. This is fixture-backed client integration evidence, not a new
live GPU performance measurement.

Logs: `/private/tmp/lowlight-reasoning-current-audit.log` and
`/private/tmp/lowlight-relay-reasoning-current-audit.log`.
The RTX 4090 remains explicitly paused; this audit performed no remote work.
