# Paired performance checks

`compare.py` compares two explicitly configured local server executables. It starts only its own loopback servers, writes a separate configuration and log for each process, alternates baseline/candidate order by block, and terminates only those owned process groups. It does not connect to production port 8080, install models, build binaries, or change product settings.

Requires Python 3.9 or later and the Python standard library. macOS and Linux are supported. Runtime and model files must already exist locally. Use direct executable payload paths rather than shell launchers, and declare all non-system runtime dependencies under `identity_paths`.

```sh
python3 Scripts/Performance/compare.py run \
  --manifest /absolute/comparison.manifest.json \
  --out /absolute/new-evidence-directory

python3 Scripts/Performance/compare.py analyze /absolute/new-evidence-directory

python3 -m unittest discover -s Scripts/Performance -p 'test_*.py' -v
```

The output directory must not already exist. Exit status 0 means the declared nonregression or improvement gate passed; 2 means regression, quality regression, or inconclusive. Do not treat an inconclusive run as a pass. Synthetic HTTP tests bind only an owned random loopback port and need local networking permission in a sandbox.

## Manifest

Start with `example.manifest.json`. Replace every absolute path before running. The first variant is the baseline, the second is the candidate; names are labels. Both can point to the same executable with different declared environment settings for a controlled configuration comparison.

The command is an argument array, never a shell expression. Supported placeholders are `{config}`, `{port}`, `{model}`, and `{served_model}`. A separate `{port}` argument and an absolute executable path are required. Configure the listener host as `127.0.0.1`. A variant may override the common `config` object. `identity_paths` should include the loaded Metal library or CUDA/runtime bundle as applicable, plus dependency lockfiles. These are explicitly declared dependencies, not automatic discovery of the full operating-system library closure.

Model files, binaries and declared dependencies are hashed before and after the run. Symlinked files include the target bytes. Effective command, configuration hash and environment are recorded per process. The environment inherits only basic execution paths/locale and performance-related prefixes (`MODEL_RUNNER_`, `MIDNIGHT_BENCH_`, `MLX_`, `CUDA_`, `OMP_`, `OPENBLAS_`, `VECLIB_`), plus explicit variant overrides. Arbitrary shell API-key variables are not inherited. Explicit environment values are part of the evidence: do not put credentials in a manifest.

Every request retains `/v1/runtime` when available, including the resolved model/settings and reported runtime memory counters. An unavailable runtime endpoint is recorded honestly for compatible servers that do not implement it. Resolved model settings must remain stable within each variant and match across arms. For an intentional setting comparison, explicitly allow the corresponding flattened key, for example `"runtime_allowed_differences": ["loadedModel.prefill_step_size"]`. The difference is still reported. Generated IDs, model creation timestamps and dynamic memory counters are excluded from this configuration equality check.

The default gate allows at most a 3% latency regression and requires at least four complete independent process pairs. Eight or more pairs are preferable. The analysis uses a deterministic 95% percentile bootstrap over paired block ratios, with median aggregation. Repeated requests within a process are not independent replicates. A candidate/baseline latency ratio above one is worse; the rate ratio is inverted so positive change always means worse. Early/late drift above the declared threshold makes a result inconclusive. A passing neutral interval must fit below the regression tolerance; an improvement interval must be wholly below zero. Missing requests, unequal outputs, failed quality checks, changed identities and insufficient pairs cannot pass.

Declare cache expectations explicitly when they matter:

```json
"cache_expectations": {
  "fresh_long": "zero",
  "repeat_long": "positive",
  "shared_prefix": "positive"
}
```

Accepted policies are `zero`, `positive`, and `observe`. The default is `zero` for `fresh_long` and `observe` elsewhere. A variant can supply its own complete `cache_expectations` mapping; for a deliberately disabled-cache arm, use `zero`. Cache counts are reported separately from output parity. A disabled or missed cache must not be described as a measured cache hit.

## Workloads and cache state

Each block starts a fresh server process for each arm. Workloads run in this fixed sequence with distinct prompt namespaces. All input text stays identical across arms and blocks: a block number in a prompt would change tokenization or answers and confound a drift check. Warmups are disjoint from the measured long prompt and remain in raw evidence with `stage: warmup`.

| Workload | Measurement and setup |
|---|---|
| Startup | Launch to the expected model ID appearing in `/v1/models`; runtime snapshot follows. |
| Short | Exact marker extraction from a short independent prompt. |
| Fresh long | Exact marker extraction after a fixed repeated instruction document; default expected cached tokens: zero. |
| Repeat long | Exact same long input, immediately following fresh long. |
| Shared prefix | Same long system document, different extraction question. |
| Decode | A fixed request for a long bicycle-maintenance explanation, capped at 256 output tokens. |
| Multiturn | Three marker-recall turns using Responses and `previous_response_id`; actual prior output is preserved. |
| Cancel/next | Close after first visible text, then send a separate marker request until accepted; retain every busy response. |

If repeat/shared is selected without fresh long, a fresh long setup request runs first and is explicitly labeled `cache_setup`. Cold, repeated and shared-prefix results are never pooled. A new process gives a process-cold model load; operating-system file caches remain uncontrolled. This is not a disk-cold or machine-reboot benchmark.

`conversation_api: "chat"` uses full chat history instead of Responses continuation. A custom `request.stop` value is preserved for configurations that exercise the shared fallback path. All requests use the same declared request options, while measured output caps are explicit: marker requests 32 tokens, decode 256, cancellation 256 by default (`cancel_tokens` overrides cancellation). The server's own `--max-tokens` limit must permit these caps. Common `long_repeats` defaults to 70.

The marker checks use exact extraction/recall. They avoid arithmetic and backward-spelling tasks that can already fail in small baseline models. Freeform decode requires visible nonempty output; exact content, reasoning, tool calls, finish status and token-count parity are still required for a controlled timing comparison. A freeform wording change is reported as an exclusion, not automatically a semantic regression. Natural conversation input divergence is retained as an observation and cannot support a causal performance claim.

## Metrics and limits

TTFT begins with the first non-whitespace visible text. Role-only, reasoning-only and tool events do not count as visible text. Total latency ends when the terminal SSE event has been received and the response closed. Hidden server work after that terminal event is not included; cancel-to-next readiness separately verifies that a subsequent inference can complete.

Visible decode rate is an approximation from reported output tokens and the span between first and last visible chunks. It is advisory because HTTP chunks need not map one-to-one to tokens. Reasoning/tool outputs suppress this approximation. EOS and terminal-event delay do not extend the visible decode span. Raw events, content, reasoning, tools, finish fields, token counts, cache counts, input hashes and original requests remain available for inspection.

The startup poll interval defaults to 20 ms and is recorded; set `limits.startup_poll_seconds` to 0.01 for 10 ms. Interpret startup differences below the poll interval as measurement-limited. Process RSS is sampled every 500 ms and is advisory. It excludes child processes and GPU allocations; runtime-reported counters are preserved as additional evidence without pretending they equal process RSS.

The host should be idle, on consistent power/thermal settings, with other model runners stopped by their owners. This harness neither checks nor kills unrelated applications. Paired order reduces ordering bias; it does not make a noisy machine quiet. No statistical interval compensates for mismatched settings or a changed workload.

This initial harness measures the streaming API. A real Lowlight PTY check can be run separately with the existing `benchmark-results/personal-prd-20260911/lowlight-tui-timing.py`; retain its client identity and terminal traces, and replace difficult freeform recall prompts with exact marker recall. API TTFT is not a measurement of the time a user sees pixels on the display. Lowlight-visible timing must not be inferred from this API result.

## Evidence

The output directory contains the immutable input manifest, a copy of the harness source used, before/after file identities, host/harness identity, append-only sample records, integrity result, analysis JSON and a concise Markdown report. Each owned process has its effective launch, configuration, server log and RSS samples. Warmups, setup requests, HTTP errors, partial output and cancellation retry attempts remain in the evidence. Never delete an inconvenient trial and rerun analysis as though it had completed successfully; use a new output directory for a separately declared campaign.
