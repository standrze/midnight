# GPT-OSS optimization checks — 2026-09-06

Device: Apple M5 Max, 40 GPU cores, 64 GB unified memory. Native Swift 6.3.3
release executables, MLX Metal, one inference/benchmark process at a time.
This is a single-device investigation, not a claim of globally optimal inference.

## Decision

Retain the downloaded mixed GPT-OSS-20B checkpoint: **MXFP4/G32 experts,
affine Q4/G64 attention/embedding/head, affine Q8/G64 routers**. Affine Q4R8
conversion has not demonstrated a whole-model speed or quality advantage.
Use explicit low reasoning effort for latency-sensitive requests. Keep
uncompressed KV and the 512-token prefill default. Ordinary text avoids
session retention; declared-tool requests can retain a pending call's session.

The checkpoint is `mlx-community/gpt-oss-20b-MXFP4-Q4`, exact revision
`f356f2747216d7e98fee755df25987459fc19089`. See
[checkpoint-layout.json](checkpoint-layout.json) for header-only verification of
all 775 tensors and 11,178,480,768 payload bytes. This verifies layout and size,
not the complete weight content. Its hypothetical affine-G64 expert payload
would add 597,196,800 bytes. No requantization or quality comparison was performed.

## Runtime evidence

[final-runtime.json](final-runtime.json) verifies the final tool-only retention
policy: one warm-up and four complete 256-token trials, median **138.530 tokens/s**,
median **295.112 ms** to first visible text, and maximum measured MLX active peak
**11.617 GB**. All four outputs match each other and the initial low-effort output.
The [compact summary](final-runtime-summary.json) records the checks. Initial and
final runs were not interleaved AB/BA, so their difference is not an established
speedup.

[low-runtime.json](low-runtime.json) records one warm-up and four 256-token
short-prompt generations at low effort: median **135.654 tokens/s**, median
**305.109 ms** to first visible text. All four continuations have identical
text. This initial build used the general GPT-OSS session path before retention
was narrowed to pending tool calls; final verification is recorded separately.
Peak model-loading allocation was approximately 12.130 GB, with about 11.649 GB
active during measured short generations. MLX active memory is not whole-process
RSS. Generation rate includes reasoning tokens; TTFT waits for visible content.

The original binary answered `Hello` correctly in
[baseline-hello.json](baseline-hello.json), but its longer default-effort test
prompts produced no visible output before termination. Their logs are retained;
they cannot establish a before/after throughput ratio. Low effort is a reasoning
policy tradeoff, not a kernel acceleration claim.

The plain follow-up [cache probe](hot-cache.log) found zero reused tokens.
GPT-OSS drops private analysis when rendering completed ordinary turns, and its
rotating cache can require rebuilding. This led to retaining sessions only for
pending tool calls. A tool continuation's actual reuse remains conditional on
upstream token reconciliation; no general chat or tool-cache speedup is claimed.

## Format primitives

[metal-formats-idle-analysis.json](metal-formats-idle-analysis.json) analyzes
16 balanced AB/BA pairs with 128 queued operations per arm using exact GPT-OSS
20B geometry. Ratios are paired candidate speed relative to affine Q4/G64;
values above one favor the candidate.

| Workload / candidate | Paired ratio | Bootstrap 95% interval | Result |
| --- | ---: | ---: | --- |
| Query / MXFP4 | 0.886 | 0.851–0.978 | Affine faster in this primitive |
| Expert gate/up / MXFP4 | 1.038 | 1.022–1.094 | Small MXFP4 advantage |
| Attention O / affine G128 | 0.989 | 0.980–1.019 | No demonstrated advantage |
| Attention O and expert down / MXFP4 | Withheld | — | Timing drift/order gates failed |

These are random BF16 source weights, independently quantized into each format.
The workloads exclude routing, attention and KV, and are not a quality test or
whole-model speed comparison. The initial `metal-formats.log` overlapped a Swift
compile and is explicitly marked contaminated in its analysis. The subsequent
idle run still has rejected metrics; rejected results were not promoted.

## Prefill comparison

The [campaign](prefill-campaign/summary.json) ran four balanced 512-versus-2048
pairs with the same 4,305 prompt tokens, low effort and a 256-token output target.
Every 512 arm stopped after 85 tokens; every 2048 arm reached 256. Consequently
there are no valid comparable pairs or accepted speed ratios. The separate
[early-stop diagnostic](prefill512-early-diagnostic.json) records the 512 arm's
refusal to produce 300 entries in that format. This is generated-behavior
divergence, not an established cache or attention defect. The 2048 timings also
drifted. Prefill defaults remain unchanged.

## Reproduction and checks

Use [the setup guide](../../Docs/gpt-oss.md) and
[request example](../../Examples/gpt-oss-fast-request.json).
The campaign manifest, commands, source/dependency inventory, executable hashes,
all process outputs and failed runs are retained in `prefill-campaign`.
Standalone executable/Metal hashes are recorded separately. Source hashes and
dirty diffs do not independently prove the binary was built from those sources.
Final build/test logs and final HTTP/native reports accompany this directory.

Final verification passed: **36 Swift tests in seven suites**, **25 campaign
tests**, **five primitive-analyzer tests**, the Metal CLI's 16 argument checks,
and the existing Mistral hot-cache shell regression. The release build includes
the `midnight` server and native benchmark executables. The final
[HTTP smoke](http-smoke-final.json) passed text at both efforts, SSE, a tool call
and its mocked result; actual initial/follow-up requests contain one/three
messages respectively. Single HTTP timings are functional diagnostics, not an
accepted latency comparison. The temporary localhost test server was stopped
after validation.

Tests cover reasoning wire/defaults, template context, cache invalidation and
context bounds, preserved Mistral behavior, tool plumbing, benchmark arguments,
paired statistics and campaign rejection of silently ignored reasoning controls.
HTTP smoke checks use only synthetic requests and a mocked `lookup_code` result;
Midnight does not execute the tool. The initial smoke artifact mutated its saved
first tool request and used a literal ASCII-hyphen assertion; it is diagnostic
only. Final smoke requests are deep-copied and normalize the model's nonbreaking
display hyphen when checking the returned code.
