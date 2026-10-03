# Quantization and runtime experiments — 2026-09-04

## Quality: AWSS improves teacher fidelity, while LS2 has lower reference NLL

Across these 128 fixed code/math references, AWSS reduced full-vocabulary BF16-teacher KL by **48.69% versus ScaleSearch LS2**, but increased reference NLL by **0.014165 nats/token**. Both effects persist in the code and math subsets. This supports AWSS as a fidelity candidate; it does not establish better generated-task accuracy or justify selecting a production default from KL alone.

The [native report](teacher-kl-code-math.json) covers 64 MBPP references (6,153 scored tokens) and 64 GSM8K references (11,925 scored tokens), for **18,078 scored tokens**. No sample was truncated at the 512-token limit. All sample IDs, categories, token fingerprints and token counts match across the teacher and three students. Recomputed token-weighted totals reconcile with the native report.

| Scope | Model | KL(BF16 teacher ∥ student) ↓ | Reference NLL ↓ |
| --- | --- | ---: | ---: |
| All 128 references | Standard Q4 | 0.021750 | 1.414373 |
| All 128 references | ScaleSearch LS2 | 0.022799 | **1.406084** |
| All 128 references | AWSS | **0.011699** | 1.420249 |
| Code: 64 references | Standard Q4 | 0.022465 | **1.602756** |
| Code: 64 references | ScaleSearch LS2 | 0.028122 | 1.603415 |
| Code: 64 references | AWSS | **0.011467** | 1.627801 |
| Math: 64 references | Standard Q4 | 0.021381 | 1.317173 |
| Math: 64 references | ScaleSearch LS2 | 0.020053 | **1.304266** |
| Math: 64 references | AWSS | **0.011819** | 1.313157 |

Teacher reference NLL was 1.415724 overall, 1.619431 on code and 1.310616 on math. A student's lower reference NLL can coexist with higher teacher KL: these metrics ask different questions.

### Paired uncertainty

The [analysis JSON](teacher-kl-code-math-analysis.json) contains all three pairwise comparisons and sample-level win counts. The intervals below use 10,000 paired-sample bootstrap draws with seed `20260904`. Each draw recomputes token-weighted estimates; the overall bootstrap preserves the observed 64/64 category composition.

| AWSS versus LS2 | Teacher-KL reduction, 95% interval | Reference NLL change, 95% interval |
| --- | ---: | ---: |
| Overall | 48.69% [45.22%, 52.47%] | +0.014165 [+0.011045, +0.017377] |
| Code | 59.22% [53.35%, 64.66%] | +0.024387 [+0.018967, +0.030061] |
| Math | 41.06% [38.78%, 43.26%] | +0.008891 [+0.005215, +0.012626] |

Against Standard Q4, AWSS reduced teacher KL by 46.21% overall [44.32%, 48.11%]. LS2's overall teacher-KL difference from Standard crosses zero, while its code-subset KL is 25.18% higher [12.82%, 37.62%] and its math-subset KL is 6.21% lower [2.23%, 10.21%]. LS2's overall reference NLL is lower than Standard by 0.008290 [0.004050, 0.012668]. The apparent ranking therefore depends on the metric and domain.

These are percentile intervals over a fixed convenience subset, not population-accuracy guarantees. They do not capture run-to-run numerical variability, and no multiple-comparison correction is applied. The reference strings contain the complete task/question and reference solution/answer, so the scores include prompt text as well as answer text. **No generated-code pass@1 or generated math accuracy was measured.** Teacher-forced token top-1 accuracy is also not task accuracy.

### Provenance and reproduction

- **Model source:** `mistralai/Ministral-3-14B-Instruct-2512-BF16`, immutable revision `3cea74c1ebaf5ce5f5a2553de470e2ceab825142`. The retained [BF16 source manifest](bf16-source-manifest.json) contains the previously verified source shard hashes. [Model inventory](model-inventory.json) records all four local paths, dimensions, selected-file byte totals and configuration hashes. The native report additionally records each checkpoint's sampled-content fingerprint; those fingerprints are not full-file SHA256 hashes.
- **Corpus:** [Source provenance](corpus-provenance.json) pins MBPP at `f82046ba5aabbbb427dbfd38a254d26bff08b533` and GSM8K at `b0bb162abedc65e1fdd8e93ed090fd7598ee68bc`. `Scripts/prepare-quantization-corpus.py` reproduces MBPP task IDs 11–74 followed by GSM8K test records 0–63. The exact combined payload SHA256 is `5fc84a9794338a138e7284f415018323aa6bfec54227d09db8a95ad562e532f6`. Source corpora are not copied into this result archive.
- **Execution:** The existing release teacher-KL executable was built from `f214f1671bd045726e32099a3ba42b8d4f943766`. It used the subsequently rebuilt **patched** Metal library, rather than an unmodified library from that commit. [Execution provenance](execution-provenance.json) records executable/metallib SHA256 hashes, modification times, MLX dependency revisions and this attribution. These hashes were captured after the completed evaluation; the recorded artifacts' modification times precede report creation.
- **Timing:** Download and CPU build activity overlapped the experiment. Native elapsed-time fields are retained as diagnostics and **are not used for speed comparisons**. The quality result is not a runtime benchmark of the current source tree.

Recreate the corpus separately, then reproduce the archived analysis without inference or downloading a model:

```bash
python3 Scripts/prepare-quantization-corpus.py --output-dir /tmp/quantization-corpus
python3 benchmark-results/quantization-20260904/analyze-teacher-kl-code-math.py
```

For offline corpus reproduction, add `--cache-dir /absolute/cached/corpora --offline`. The analyzer is pinned to the retained native report's SHA256 and needs only that report; it does not require local model files or source corpora. Numeric estimates and bootstrap intervals reproduce exactly; the output timestamp and local report path reflect the rerun.

## Runtime campaign: decode comparisons fail drift checks

All **16 scheduled invocations completed** in the [Ministral runtime campaign](ministral-runtime/summary.json): four counterbalanced pairs each for Standard versus LS2 and Standard versus AWSS. Each invocation used one warmup and one measured 256-token greedy continuation, with the same fixed Swift-scheduler prompt. Exact rendered prompt token fingerprints matched; generated lengths were complete. The models produced different output text, so these are comparisons of their natural generation trajectories, including potentially different expert routing.

Ratios below are medians of paired ratios. Rate ratios are candidate/Standard; TTFT ratios are Standard/candidate, so **greater than one favors the candidate**. Intervals resample four pairs with replacement (2,000 seeded draws). A 10% threshold is applied separately to both arms' first-half/second-half drift and to the AB/BA order effect for each metric.

| Candidate | Decode ratio, 95% interval | Prefill ratio, 95% interval | TTFT speed ratio, 95% interval |
| --- | ---: | ---: | ---: |
| LS2 | 0.984 [0.966, 1.025] — withheld | 0.975 [0.829, 1.076] — withheld | 0.973 [0.836, 1.078] — withheld |
| AWSS | 1.002 [0.957, 1.090] — withheld | 0.875 [0.793, 0.953] — qualified measurement | 0.879 [0.801, 0.954] — qualified measurement |

LS2's decode drift reached 15.9% in Standard and 20.8% in LS2; its prefill/TTFT comparisons also failed drift and order gates. AWSS's decode drift reached 11.2% in the candidate arm, invalidating a speed claim despite its near-one paired median. AWSS's prefill and TTFT comparisons passed the configured gates but indicate **slower prefill and longer TTFT**, not improvement: median prefill rates were 541.0 versus 475.9 tokens/s and median TTFTs were 1,082.2 versus 1,224.9 ms. Ratios of these marginal medians differ from the reported paired-ratio estimates.

“Qualified” means that this measurement passes the configured checks. Four pairs in one session are insufficient for promotion, repeatability or broad performance claims. Thermal autocorrelation and workload specificity remain. These results neither establish a decode winner nor justify changing a production default.

The archive retains the [manifest](ministral-runtime/manifest.json), [actual schedule](ministral-runtime/schedule.json), [provenance](ministral-runtime/provenance.json), [complete invocation records](ministral-runtime/results.json), all 16 native reports and their invocation metadata in `ministral-runtime/runs/`, and the original summary. Redundant successful stdout/stderr logs are omitted. [Archive integrity metadata](runtime-kernel-archive-index.json) records each retained file's SHA256.

The campaign provenance captures executable/Metal-library hashes, dependency revisions, machine details and source-file hashes. **Unrelated calibration/kernel source changes occurred during the run; its source snapshot is not proof of the exact sources compiled into that binary.** Retain that distinction when reproducing the run. No newly implemented kernel was enabled by this checkpoint campaign.

## Metal kernel candidate: exact fixtures, no demonstrated model decode gain

The original [v1 primitive report](primitives/fused-gather-ab.json) failed all eight numerical fixtures, with maximum absolute deviations of 0.046875–0.0625 from stock BF16 output. It contains no valid timing trials. That failure remains in the archive.

Both revised reports, [v2 with AB first-call order](primitives/fused-gather-v2-ab.json) and [v2 with BA first-call order](primitives/fused-gather-v2-ba.json), matched stock **bit-exactly on all eight fixtures**, including exact intermediate projection equality, identical output hashes, and zero maximum/RMS output error. This validates those fixtures only.

The synthetic shape was `E256 top8 K2048 gateUp1024 hidden512 Q4/G64`, BF16, on `applegpu_g17s`, with 16 warmups and queue depth 32. Each report contains nine queued timing pairs and 20 synchronized pairs. Reported speedups below are ratios of marginal medians, not paired confidence estimates.

| Revised primitive run | Queued execution, stock → fused | Queued execution speedup | Queued total speedup | Synchronized stock → fused |
| --- | ---: | ---: | ---: | ---: |
| AB first call | 0.02993 → 0.02400 ms | 1.247× | 1.228× | 0.18365 → 0.18271 ms (1.005×) |
| BA first call | 0.02831 → 0.02020 ms | 1.401× | 1.355× | 0.16392 → 0.16102 ms (1.018×) |

Queued execution is promising; synchronized calls are essentially flat. First-call times include JIT and strong order effects and are excluded from these comparisons. The two short sessions differ materially, and **synthetic primitive speedups are not model or serving speedups**. The raw primitive reports do not independently identify exact executable/kernel-source hashes; the archive index verifies report integrity, not complete build provenance.

The same-loaded native runs are now retained with their independent analyses:
[256-token native report](laguna-kernel-runtime/laguna-gather-silu-ab-256.json),
[256-token analysis](laguna-kernel-runtime/laguna-gather-silu-ab-256-analysis.json),
[512-token native report](laguna-kernel-runtime/laguna-gather-silu-ab-512.json), and
[512-token analysis](laguna-kernel-runtime/laguna-gather-silu-ab-512-analysis.json).
Both compare the existing compiled decode/router path with the opt-in fused
gate/up-SiLU path on one loaded Laguna XS-2.1 Q4R8 checkpoint. They use the same
50-token rendered Swift-scheduler prompt, 512-token prefill chunks, uncompressed
KV, and greedy generation. All measured decoded outputs matched exactly within
each run, including identical rendered-prompt fingerprints and complete lengths.

The 256-token run has four balanced pairs and one warmup per arm. **Every timing
metric fails drift checks**, so it supplies correctness observations rather than
a retained performance estimate. Decode drift was 11.24% in stock and 16.50% in
the candidate. Do not interpret its raw rates as a performance result.

The longer run has eight balanced pairs (four AB, four BA), five warmups per arm,
and 512 generated tokens in each measured trial. Ratios below are paired median
ratios with 2,000-draw seeded percentile intervals; greater than one favors the
candidate. These differ from ratios of marginal medians in the native report.

| 512-token same-loaded metric | Paired ratio, 95% interval | Interpretation |
| --- | ---: | --- |
| Decode throughput | 0.99565 [0.93663, 1.14639] | Passes configured gates; no demonstrated gain |
| Reported prefill throughput | 1.07890 [1.00285, 1.11190] | The candidate is decode-only; this does not establish a prefill-kernel benefit |
| TTFT speed | 1.07310 [0.99507, 1.10787] | Interval includes no improvement |
| Total request speed | Withheld | Candidate drift 10.06% exceeds the 10% threshold |

Stock/candidate marginal decode medians were 115.91/115.15 tokens/s. The decode
candidate drift was 9.61%, close to the cutoff, and paired ratios ranged from
0.92912 to 1.15940. Eight pairs in one session on one prompt do not establish
repeatability. **The fused kernel remains opt-in; its promising primitive result
has not translated into an established model throughput improvement.** Exact
decoded text does not independently prove identical generated token IDs, expert
routes, or every internal activation.

[Invocation provenance](laguna-kernel-runtime/invocation-provenance.json) retains
commands reconstructed from the execution tool history, explicitly distinguished
from contemporaneous launch manifests. The [production build log](laguna-kernel-runtime/production-runtime-build.log)
records a successful 83.50-second release build of the runtime product, without
an added testability flag. The [artifact snapshot](laguna-runtime-artifact-snapshot.json)
was captured after the runs and before the next build. Its runtime/metallib hashes
also match the subsequent prefill campaign's contemporaneous provenance. This
supports attribution to the tested artifacts; it does not make the changing
worktree a clean, exact-source build manifest. The parent reports no other GPU
work, builds or downloads during either native run and paused project mutations
for the 512-token run.

Reproduce the statistical analysis without inference:

```bash
python3 Scripts/analyze-laguna-gather-silu-ab.py \
  benchmark-results/quantization-20260904/laguna-kernel-runtime/laguna-gather-silu-ab-512.json \
  --output /tmp/laguna-gather-silu-ab-512-analysis.json
```

### Instrumented trace verification: functional evidence only

The later [native verification report](laguna-kernel-runtime/laguna-fused-trace-verification.json)
and [execution log](laguna-kernel-runtime/laguna-fused-trace-verification.log)
record `laguna_gather_silu_trace_count: 39`, with identical decoded text and full
32-token output in the stock and candidate arms. This counter increments only
when the candidate successfully builds a fused gate/up-SiLU graph; it confirms
the opt-in path was selected during this evaluated run. It counts graph
constructions, **not individual GPU dispatches or generated tokens**.

There is only one measured AB pair, plus one warmup per arm. The retained
[analysis](laguna-kernel-runtime/laguna-fused-trace-verification-analysis.json)
therefore withholds every timing estimate: minimum-pair, balanced-order and drift
requirements are not met. This is functional verification, not evidence of a
model speedup. The native JSON/log do not independently attest executable or
Metal-library hashes; do not assign the earlier artifact snapshot to this later
instrumented run.

The analyzer now requires a **positive integer** when the trace-count field is
present. Zero, negative, null, Boolean, floating-point and string counts make
comparisons invalid and withhold timing estimates. Historical reports without
the field remain readable and retain their statistical diagnostics with explicit
`not_recorded` trace provenance. In particular, the archived 256/512-token runs
predate this counter: their unchanged original analyses preserve the historical
calculation, and a positive counter from this later run is not retroactive proof
of dispatch in those earlier runs. Thirteen focused CPU tests cover the analyzer,
including counter validation and legacy-report behavior.

## Laguna prefill chunks: 2048 is promising; quality gate pending

The completed [prefill campaign](laguna-prefill/summary.json) contains all 24
scheduled invocations: four balanced pairs each comparing chunks 256, 1024 and
2048 against 512. Each arm starts a new process with one warmup and one measured
32-token greedy continuation. The fixed repeated scheduler-notes prompt renders
to **3,495 tokens**; context length is 8,192 and KV compression is disabled. This
campaign uses the ordinary runtime path, with no fused gate/up opt-in.

| Candidate chunk size | Prefill throughput ratio, 95% interval | TTFT speed ratio, 95% interval | Change in marginal median peak active memory |
| --- | ---: | ---: | ---: |
| 256 | 0.76157 [0.74820, 0.78039] | 0.76812 [0.75323, 0.78109] | −203.7 MB |
| 1024 | 1.08075 [0.99988, 1.14254] | Withheld: baseline drift | +292.0 MB |
| 2048 | 1.07022 [1.03255, 1.13577] | 1.07430 [1.03658, 1.13032] | +412.0 MB |

For 2048, the paired prefill-throughput estimate is **+7.02% [3.25%, 13.58%]**
and the TTFT speed estimate is **+7.43% [3.66%, 13.03%]**. Its marginal median
peak MLX active memory increases from 19.819 to 20.231 GB (approximately 393 MiB).
Chunk 256 is about 24% slower in prefill throughput. Chunk 1024's prefill interval
includes one; it does not establish an improvement. Decode is not a promotion
signal here: the 2048 comparison fails its order-effect gate, 256 fails candidate
drift, and 1024's accepted interval includes one.

**Each alternative chunk size produces different greedy output from 512.** The
prompt fingerprints and generation lengths match, but this is not an equal-output
performance comparison. The chunk-aware reference NLL check below does not establish a quality
improvement; broader generated-task evidence remains absent. No default prefill size is changed. The
confidence intervals resample four pairs within one session and do not capture
all thermal autocorrelation, prompt variation or run-to-run effects. Peak active
memory is an MLX metric, not total system memory or energy use.

The archive includes [manifest](laguna-prefill/manifest.json),
[actual schedule](laguna-prefill/schedule.json),
[provenance](laguna-prefill/provenance.json),
[all invocation records](laguna-prefill/results.json), the original summary, and
all 24 native reports plus their invocation metadata under `laguna-prefill/runs/`.
Successful per-run stdout/stderr logs are omitted. The runtime SHA256 is
`379179d3343679277cf1a8bf9f172d4ba5c30420178a08f3f59f8645858faf73`; the Metal-library
SHA256 is `903daf038bc9e65c6b77ccb3dc023df6435cf50d4d2dc78ed950a711f68be48c`.
The campaign captures Apple M5 Max, 64 GiB memory, macOS 26.6.2 and Swift 6.3.3,
with patched MLX dependencies. Source snapshots taken around a run do not attest
that every recorded source file was compiled into the identified binary.

### Confirmation campaign: prefill and TTFT gains were not retained

The [confirmation summary](laguna-prefill-confirm/summary.json) contains all
16 completed invocations: four balanced 512-versus-2048 pairs for each of two
scheduler-note prompts. The actual rendered lengths are **3,495 and 13,986 tokens**
(the manifest labels are `scheduler-3600-confirm` and `scheduler-14000`). Each
process uses one warmup, a 32-token greedy continuation, a 32,768-token context
limit and uncompressed KV. The independent schedule seed is `20260905`.

| Confirmation prompt | Prefill comparison | TTFT comparison |
| --- | --- | --- |
| 3,495 tokens | Withheld: stock/candidate drift 26.07%/28.08%; order effect 10.32% | Withheld: stock/candidate drift 34.48%/37.82% |
| 13,986 tokens | Withheld: stock drift 11.15% | Withheld: stock/candidate drift 13.88%/11.15% |

Thus **neither confirmation workload retains a prefill or TTFT improvement**
under the predeclared 10% gates. The initial 2048 result remains a qualified
single-session observation; it has not become a repeatable basis for a default
change. Diagnostic ratios in the raw summary must not be substituted for the
withheld accepted estimates.

The longer prompt's decode ratio passes its gates at 1.04808
[1.02235, 1.08561], but both prompt comparisons produce different greedy output
between chunk sizes. This natural-generation decode observation does not show
that changing prefill chunks makes the decode kernel more efficient, nor does
it establish an equal-quality serving improvement. The short-prompt decode
comparison fails drift as well. The reference NLL comparison below has an interval spanning zero, and the default remains 512.

The archive retains the [manifest](laguna-prefill-confirm/manifest.json),
[schedule](laguna-prefill-confirm/schedule.json),
[complete invocation records](laguna-prefill-confirm/results.json),
[provenance](laguna-prefill-confirm/provenance.json), original summary, and all
16 native reports with their invocation metadata under
`laguna-prefill-confirm/runs/`. Successful stdout/stderr logs are omitted.
The recorded executable and Metal-library hashes match the initial prefill
campaign and the retained artifact snapshot. This confirmation ran on the same
machine and date; four pairs per workload remain a small, temporally correlated
sample, despite the independent schedule.

### Reproducible long-context likelihood corpus

[Corpus provenance](corpora/wikitext2-long-8.provenance.json) pins an existing
64-record WikiText-2 test JSONL source with SHA256
`be40b2e92a40298a2e7c29b1b52844f4c57b096a27250ddcaf4733cc5dcba0e7`.
[The preparation script](prepare-long-context-corpus.py) concatenates every
consecutive group of eight texts with exactly two newlines, preserving source
order. It emits eight records using deterministic IDs, UTF-8 JSONL serialization,
and a final newline. The required output SHA256 is
`c7b87e6366cd2cd02fec1d24f75a9506e6aaaca95c3e4d091325b17d5ea76180`;
reconstruction was verified byte-for-byte against the prepared corpus. Neither
the source nor the derived corpus content is copied into this evidence archive.

```bash
python3 benchmark-results/quantization-20260904/prepare-long-context-corpus.py \
  --source /absolute/path/to/wikitext-2-raw-eval.jsonl \
  --output /tmp/wikitext2-long-8.jsonl
```

The script refuses a mismatched source hash and requires a new output path.
This corpus supports chunked-prefill reference likelihood checks; it does not
measure retrieval ability or generated-task accuracy. Its archive provenance
pins the local test payload, rather than independently establishing broader
training-data exclusion or benchmark representativeness.


### Long-context reference NLL: no demonstrated quality improvement

The completed [512-chunk report](laguna-long-nll/laguna-long-nll-512.json) and
[2048-chunk report](laguna-long-nll/laguna-long-nll-2048.json) score the same
Laguna Q4R8 checkpoint on all eight long-prose records above. All sample IDs,
corpus/token fingerprints and token counts match. **32,528 next-token targets**
were scored, with no truncation at the 8,192-token sample limit. Each sample
starts with a fresh native cache, and the scorer includes targets crossing
chunk boundaries.

| Chunk size | Token-weighted reference NLL ↓ | Perplexity ↓ | MLX peak memory |
| --- | ---: | ---: | ---: |
| 512 | 2.803886536 | 16.50868 | 19.804 GB |
| 2048 | 2.799426490 | 16.43522 | 21.234 GB |

The [paired analysis](laguna-long-nll/analysis.json) estimates an
**NLL change of −0.004460046 nats/token**, with a **95% interval of
[−0.010149118, +0.000573441]** for 2048 minus 512. The percentile bootstrap
resamples eight paired records with replacement (10,000 draws, seed `20260904`)
and recomputes token-weighted NLL for each draw. Five records improve and three
worsen. The interval spans zero: these data do not establish improved NLL or
formal noninferiority. A noninferiority margin was not specified.

This is one run per setting on eight convenience records from a single prose
source. The interval does not capture correlated material, numerical/run
variation or population generalization. **Reference NLL is not generated-task
accuracy or proof of identical greedy output.** Combined with output divergence
and the failed prefill/TTFT confirmation gates, this result does not justify
changing the default from 512.

The larger chunk's MLX peak is 1.430 GB higher in this scoring workload. MLX peak
memory is distinct from process maximum RSS and total system footprint. CPU/build
work overlapped these quality runs; elapsed times are retained in the native
reports and short logs as diagnostics and are not used for a speed claim.

The [post-run artifact snapshot](laguna-long-nll/post-run-artifact-snapshot.json)
records quality executable and Metal-library SHA256 hashes and modification
times before subsequent builds. The executable includes the awaited Laguna
registration fix and was built in release mode with testability enabled for the
coordinated regression build. The snapshot is post-run evidence; the native JSON
does not independently prove exact binary/source attribution or hash all model
weights. Model metadata and sampled-weight hashes are retained in the
[earlier runtime provenance](laguna-prefill/provenance.json).

The [standalone analysis script](laguna-long-nll/analyze.py) pins both native
report SHA256 hashes, validates paired identities and recomputes totals before
producing the interval. Reproduce the analysis without models or inference:

```bash
python3 benchmark-results/quantization-20260904/laguna-long-nll/analyze.py
```

## Laguna activation collection: process-memory retention addressed

The [calibration report](laguna-activation-collection/laguna-calibration-v2.json)
and [development report](laguna-activation-collection/laguna-dev-v2.json) both
completed all 40 native BF16 layers, collecting 320 dense and 78 routed
projection statistics. Each run verified the same 14 indexed source shards:
`indexed-safetensors-full-content-fnv1a64-v1`, digest
`fnv1a64:d76936c8b3e3e53b`. This is full-content reproducibility checking, not
cryptographic publisher authentication.

| Run | Tokens / segments | Collector report time | Whole-process wall time | MLX peak allocation | Maximum RSS | Peak process footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Calibration v2, before lifetime fix | 65,536 / 168 | 97.99 s | 169.53 s | 3.952 GB | 28.442 GB | 70.151 GB |
| Development v2, after lifetime fix | 15,997 / 47 | 34.64 s | 104.04 s | 3.952 GB | 2.511 GB | 4.332 GB |

Values use decimal GB. The [measurement summary](laguna-activation-collection/measurement-summary.json)
retains exact bytes and timing values. MLX peak allocation is **not whole-process
memory**; it did not expose the original high Foundation-memory retention.
Maximum RSS and peak footprint come from `/usr/bin/time -l` and have different
accounting from each other and from MLX. The collector's own elapsed timer starts
**after full-source hashing and corpus preparation** and stops before final
statistics/report serialization. Use 169.53 and 104.04 seconds for total observed
command wall time; the smaller report fields are not end-to-end duration.

The fix drains autoreleased Foundation file/data temporaries per tensor and
around bounded segment/layer work. MLX copies each tensor's data synchronously,
so this changes temporary ownership without changing model arithmetic. The
development run shows bounded whole-process memory after the fix, consistent
with resolving that retention. **The corpora, token counts and executable builds
differ**, so these rows do not establish an exact percentage memory reduction,
a matched speedup, or a worst-case memory guarantee. Both still read the complete
source for identity and execute the same complete layer sequence. A matching
corpus rerun would be required for a controlled reduction claim.

At the minimum 32 selected positions, calibration covers 19,684 of 19,968
expert/projection entries; development covers 17,190. Their joint eligible set
contains 17,190 entries. Counts refer to two routed projections per expert,
not that many unique experts. Refinement must retain template weights wherever
either corpus lacks coverage. Collecting moments alone establishes neither a
better checkpoint nor a full-model quality improvement.

The [calibration log](laguna-activation-collection/laguna-calibration-v2.log),
[development log](laguna-activation-collection/laguna-dev-v2.log), and respective
[calibration build snapshot](laguna-activation-collection/laguna-calibration-v2-build-provenance.json)
and [development build snapshot](laguna-activation-collection/laguna-dev-v2-build-provenance.json)
are retained unchanged. Both executables were built in release mode with
testability enabled for correctness/memory work; these durations are not a
production performance comparison. The snapshots retain different executable
hashes and the same Metal-library hash. They identify selected artifacts and
source files, not an immutable, clean reconstruction of the complete worktree.
The approximately 101 MiB calibration/development safetensors payloads are
intentionally omitted; their original output paths remain in the reports.

### Template identity depends on quantization device

The bounded [CPU investigation](laguna-template-identity/identity-investigation.json)
and [GPU investigation](laguna-template-identity/identity-investigation-gpu.json)
used Python MLX 0.31.2 on three sampled BF16 tensors: one shared-expert down
projection, one routed-expert down projection and one Q8 router. GPU conversion
matched the public Q4R8 packed words, scales and biases exactly on all three.
CPU conversion differed in 1,730, 1,826 and 1,629 packed words respectively,
and in a small number of scales. Sampled unquantized norms matched on both.
These packed-word counts are not numerical error magnitudes or quality scores.

The exact template guard must use the intended conversion device, and conversion
provenance now records `quantization_device`; a device mismatch is not grounds
to loosen equality checks. This investigation does not prove whole-checkpoint
identity or behavior across every MLX version. Retained historical
[CPU script](laguna-template-identity/investigate-identity.py) and
[GPU script](laguna-template-identity/investigate-identity-gpu.py) preserve their
original local model/output paths and are provided for inspection, not automatic
execution. The conversion's actual complete preflight remains authoritative.


## Reproducible Laguna standard/LS2/AWSS reference comparison

The [192-record corpus provenance](corpora/laguna-heldout-192.provenance.json)
pins WikiText-2 test64 followed by MBPP64 and GSM8K64. The
[reproduction script](prepare-laguna-heldout-corpus.py) verifies both source
hashes, preserves all records and unique IDs, and emits compact UTF-8 JSONL with
one LF per record. Its output was verified byte-for-byte against the prepared
payload, SHA256 `9490178eb34b57526a8245632272824bab040dd3c8556735233b2dc30b69a5b5`.
Source and derived corpus contents are not copied into this archive.

```bash
python3 benchmark-results/quantization-20260904/prepare-laguna-heldout-corpus.py \
  --wikitext /absolute/wikitext-2-raw-eval.jsonl \
  --code-math /absolute/code-math-128.jsonl \
  --output /tmp/laguna-heldout-192.jsonl
```

`Scripts/analyze-quality-nll.py` compares completed standard, LS2 and AWSS
native quality reports scored with the same `--max-tokens-per-sample 2048
--prefill-step-size 512` settings. It verifies matched per-sample identities,
lengths, categories and scoring settings, reconciles totals, and produces all
three pairwise overall/category NLL comparisons. Its 10,000-draw paired
bootstrap preserves the observed category sample counts and recomputes
token-weighted estimates. Report and analyzer hashes are retained. The completed
quantization results and domain-specific tradeoffs appear below.

```bash
python3 Scripts/analyze-quality-nll.py \
  --report standard=/absolute/standard-native.json \
  --report ls2=/absolute/ls2-native.json \
  --report awss=/absolute/awss-native.json \
  --output /absolute/laguna-heldout-analysis.json
```

This analysis measures reference likelihood, not generated-task accuracy,
BF16-teacher KL, or speed. Category stratification does not establish population
representativeness or training-data exclusion, and intervals do not capture
within-source correlations or numerical/run variation.


## Laguna LS2: full conversion and reference evaluation complete

The completed [conversion report](laguna-ls2-conversion/q4r8-scale-search.json)
records **438 Q4 modules rescored**, with all **39 Q8 router modules** and the
standard Q4 embedding preserved. The GPU conversion used affine Q4/G64,
expert batches of 16, two joint affine-refinement iterations and one
bias-refinement iteration. It wrote four shards and released all **30,513 source
tensors**, leaving zero source tensors retained in the rescorer's map at completion.
These counts establish completion of the intended transformation; they do not
establish improved reference likelihood or generated-task quality.

The [original log](laguna-ls2-conversion/laguna-ls2-conversion.log) reports
**17,742,233,636 bytes of peak MLX allocation**, **5,444,878,336 bytes of maximum
process RSS**, and **18,127,537,232 bytes of peak process footprint**. These are
separate memory measures with different accounting and scopes. Total observed
command wall time was **736.06 seconds**. CPU build work overlapped this run,
and the executable was a release build with testability enabled, so this is a
conversion timing diagnostic, not a quiet speed benchmark or runtime speedup.

The [measurement summary](laguna-ls2-conversion/measurement-summary.json) retains
exact counters, SHA256 hashes and byte sizes for the output config, safetensors
index and conversion report. The index contains 1,634 tensor keys across four
shard filenames and declares 18,821,963,264 bytes of tensor payload; this is not
a hash of the model weights or an exact on-disk checkpoint size. No weight
payload was copied or read for archival hashing.

The [build snapshot](laguna-ls2-conversion/laguna-conversion-build-provenance.json)
pins the recorded rescorer executable, Metal library and selected source files.
The [invocation provenance](laguna-ls2-conversion/invocation-provenance.json)
retains the exact command reconstructed from the recorded tool history after
completion, explicitly distinguished from a contemporaneous invocation
manifest. The original log records passing template identity preflight; the
checked tensor scope is defined by that rescorer implementation and should not
be expanded into a full-checkpoint byte-identity claim.

The output remains an experimental candidate. The completed paired likelihood
comparison below finds higher overall NLL than Standard, with lower code NLL
and higher math/prose NLL. Conversion alone changes neither the runner default
nor the recommended model.


## Laguna AWSS: full conversion complete, reference evaluation shows domain tradeoffs

The completed AWSS checkpoint refines the **LS2 template** using BF16 source
weights and the calibration/development moments archived above. It processed
438 Q4 modules, preserved 39 Q8 routers and the standard Q4 embedding, wrote
four shards, and released all 30,513 source tensors. Source identity was checked
against the same full-content fingerprint recorded by both collectors.

The [objective summary](laguna-awss-conversion/diagnostic-summary.json) covers
25,785 routed expert/projection entries and 321 dense matrices. **4,167 of
29,952 routed expert/projection entries retain their LS2 template** under the
coverage/usable-moment guards; these represent 1,389 layer/expert pairs across
the separate gate, up and down projections. Retained entries have no fitting
diagnostic and are excluded from the objective aggregates.

Among the 26,106 diagnostic entries, the median reduction in local weighted MSE
is **3.70% on calibration moments** and **3.72% on development moments**. Raw
weight MSE worsens in 4,497 entries. The element-weighted aggregates in the
summary use a different weighting from these per-entry medians and counts.
These locally normalized weight-error objectives are not full-model activation
error, reference NLL, or generated-task accuracy. Development data participate
in candidate selection and therefore are not an independent quality test.

Of 450,735,104 evaluated groups, 286,493,061 changed from LS2. The
`validation_rejected_group_count` totals 134,110,704 groups with **at least one**
calibration-improving candidate rejected by the development guard across the
search factors. Those groups can accept another candidate, so this count can
overlap changed groups and must not be read as the final retained-group count.

The [original log](laguna-awss-conversion/laguna-awss-conversion.log) and
[measurement summary](laguna-awss-conversion/measurement-summary.json) record
**21,370,269,732 bytes of peak MLX allocation**, **5,976,866,816 bytes of maximum
process RSS**, **21,549,401,432 bytes of peak process footprint**, and **733.74
seconds** of command wall time. Memory measures have different accounting. The
[shared conversion build snapshot](laguna-awss-conversion/laguna-conversion-build-provenance.json)
identifies the same release executable with testability enabled used for LS2;
its earlier CPU-overlap annotation is not an independent AWSS concurrency
manifest. Durations remain diagnostics, not a controlled conversion-speed or
serving-speed comparison. The [invocation record](laguna-awss-conversion/invocation-provenance.json)
explicitly identifies its reconstruction from recorded tool history.

The full [conversion report](laguna-awss-conversion/q4r8-scale-search.json.gz)
is preserved as deterministic gzip with a zero modification timestamp and no
embedded filename. The original 20,846,801-byte JSON has SHA256
`2949f49bd3d5a9ce4b527ce5c5a4c2302e0a932615e357748d154fa2d7c693e1`;
the compressed archive is 2,862,667 bytes. Both hashes/sizes are recorded in the
measurement summary. No uncompressed duplicate, model weight payload, or
activation-statistics safetensors is copied. Output config/index hashes and
sizes are also retained.

Recompute the compact objective summary directly from the compressed evidence,
without models or inference:

```bash
python3 benchmark-results/quantization-20260904/laguna-awss-conversion/summarize-diagnostics.py
```

The script validates compressed and original report hashes before computing
counts and aggregates. To inspect the complete JSON, use `gzip -dc` on the
archived report. The completed likelihood comparison below supports AWSS on
aggregate and code references while identifying a math regression against
Standard. Fitting diagnostics alone justify no default or model recommendation
change.


## Laguna held-out NLL: AWSS leads overall, Standard remains best on math

The completed [Standard](laguna-heldout-192/laguna-heldout-standard.json),
[LS2](laguna-heldout-192/laguna-heldout-ls2.json) and
[AWSS](laguna-heldout-192/laguna-heldout-awss.json) reports score the same
**192 references and 50,096 next-token targets**, with zero truncation. All
sample IDs, categories, corpus/token fingerprints, evaluated lengths and scoring
settings match. The shared settings were `--max-tokens-per-sample 2048` and
`--prefill-step-size 512`, using a fresh native cache per sample.

**AWSS has lower aggregate reference NLL than both Standard and LS2, but worse
math-reference NLL than Standard.** It is a stronger same-format candidate for
these aggregate/code references, not a universal quality winner. No generated
code pass@1, generated math accuracy or BF16-teacher KL was measured by this
campaign, and no runner or quantizer default changes follow from it.

| Reference scope | Targets | Standard NLL ↓ | LS2 NLL ↓ | AWSS NLL ↓ |
| --- | ---: | ---: | ---: | ---: |
| All 192 records | 50,096 | 2.435966 | 2.452241 | **2.426263** |
| Code: 64 records | 5,689 | 1.962727 | 1.929743 | **1.910631** |
| Math: 64 records | 11,935 | **1.558559** | 1.585023 | 1.572413 |
| WikiText-2 prose: 64 records | 32,472 | 2.841364 | 2.862524 | **2.830431** |

The aggregate is token-weighted: prose accounts for roughly 65% of scored
tokens, despite equal record counts across categories. Domain weighting therefore
matters when selecting a quantizer.

### Paired uncertainty and domain regressions

The [analysis JSON](laguna-heldout-192/laguna-heldout-analysis.json) contains all
three pairwise comparisons. The following deltas are candidate minus baseline
NLL in nats/token; negative values favor the candidate. Intervals use 10,000
paired record bootstrap draws with seed `20260904`; overall draws preserve each
category's observed record count and recompute token-weighted estimates.

| Comparison | Scope | NLL delta | Paired 95% interval |
| --- | --- | ---: | ---: |
| AWSS minus Standard | Overall | −0.009702 | [−0.017679, −0.001615] |
| AWSS minus Standard | Code | −0.052096 | [−0.073848, −0.029682] |
| AWSS minus Standard | Math | **+0.013854** | **[+0.001742, +0.026317]** |
| AWSS minus Standard | Prose | −0.010933 | [−0.021680, −0.000023] |
| AWSS minus LS2 | Overall | −0.025977 | [−0.032151, −0.019791] |
| AWSS minus LS2 | Code | −0.019112 | [−0.039133, +0.000534] |
| AWSS minus LS2 | Math | −0.012610 | [−0.021929, −0.003429] |
| AWSS minus LS2 | Prose | −0.032093 | [−0.040217, −0.024445] |

AWSS's code point estimate improves on LS2, but that interval crosses zero.
Its prose interval against Standard only narrowly excludes zero. Multiple
comparisons are not corrected, and source correlations, numerical/run variation
and corpus representativeness are not captured. The math regression against
Standard must remain visible alongside the aggregate gain.

LS2 versus Standard increases overall NLL by +0.016275
[+0.007886, +0.024519]. Its code references improve by −0.032984
[−0.057347, −0.008389], while math and prose worsen. Improved weight reconstruction
alone consequently did not predict the full-model likelihood ranking.

### Reproduction and execution scope

[Independent verification](laguna-heldout-192/verification-summary.json)
recomputed all sample totals, all 12 overall/category paired point estimates and
all bootstrap intervals, matching retained estimates within 1e-12. The
[archived analyzer](laguna-heldout-192/analyze-quality-nll.py) matches the script
hash recorded in the original analysis. Reproduce it from the native reports:

```bash
python3 benchmark-results/quantization-20260904/laguna-heldout-192/analyze-quality-nll.py \
  --report standard=benchmark-results/quantization-20260904/laguna-heldout-192/laguna-heldout-standard.json \
  --report ls2=benchmark-results/quantization-20260904/laguna-heldout-192/laguna-heldout-ls2.json \
  --report awss=benchmark-results/quantization-20260904/laguna-heldout-192/laguna-heldout-awss.json \
  --output /tmp/laguna-heldout-analysis-reproduced.json
```

The [recorded build provenance](laguna-heldout-192/laguna-heldout-build-provenance.json)
identifies the release quality executable, testability flag, Metal library and
selected source hashes. The [invocation record](laguna-heldout-192/invocation-provenance.json)
reconstructs exact arguments from the coordinator's command template and native
settings after completion; it is not a contemporaneous launch manifest. The
native reports identify each model path. The [complete checkpoint inventory](laguna-heldout-192/laguna-model-full-hashes.json)
records SHA256 of every byte of all four safetensors shards per model, plus
configuration, tokenizer and other metadata hashes. It was captured after the
quality runs using `model_info(full_hash=True)`; it is not a sampled-weight hash
or a contemporaneous launch manifest. Original `/usr/bin/time -l` logs are
retained alongside the reports, but **elapsed times are not used for speed
claims**. MLX allocation peaks, process RSS and footprint use distinct accounting.


## Final beta.2 validation

The [validation summary](validation/summary.json) retains the final release build,
**166 Swift Testing tests plus 2 XCTest tests**, **49 Python tests**, and **29
shell test scripts**, all passed. Swift and shell logs are archived; the Python
result is identified by the original agent tool output because no filesystem
log was saved. The production build used `swift build -c release --jobs 8` and
reported version `0.2.0-beta.2`.

The resulting production runtime completed 128-token generation checks on both
LS2 and AWSS. These are functional checks with one measured trial per model;
different generated trajectories and the small run count support no speed
comparison. The summary records production executable and Metal-library hashes,
separately from the testable release executable used for quality scoring. The
512-token prefill default and disabled fused gather/SiLU default remain intact.
