# Reproducible benchmark campaigns

`Scripts/benchmark-campaign.py` compares local checkpoints or runtime configurations using the existing native runtime and quality executables. It never builds, downloads, quantizes, or serves a model. It runs one process at a time and preserves every process's command, report, stdout, stderr, exit status, and elapsed wall time, including failed runs.

The campaign measures native generation without HTTP overhead. It is not a serving throughput/load test. Runtime arms generate natural greedy continuations; different quantizations can produce different output tokens and MoE expert routes. Even a valid speed ratio describes those end-to-end trajectories, not identical computation or equal model quality. Run held-out quality evaluation alongside runtime measurements.

## Prepare and run

Build both tools in release mode before benchmarking:

```bash
MODEL_RUNNER_BUILD_CONFIGURATION=release MODEL_RUNNER_BUILD_PRODUCT=model-runner-runtime-bench ./build.sh
MODEL_RUNNER_BUILD_CONFIGURATION=release MODEL_RUNNER_BUILD_PRODUCT=model-runner-quality-bench ./build.sh
```

Create an explicit manifest for the existing Ministral standard, ScaleSearch LS2, and AWSS checkpoints:

```bash
python3 Scripts/benchmark-campaign.py init \
  --binary-dir "$(pwd)/.build/release" \
  --models-root /Users/stephen/Documents/ChatGPT/model-runner-mlx-archive-20260904/tmp/models \
  --output /tmp/ministral-campaign.json

python3 Scripts/benchmark-campaign.py run /tmp/ministral-campaign.json \
  --mode all --dry-run --output /tmp/ministral-campaign-plan

python3 Scripts/benchmark-campaign.py run /tmp/ministral-campaign.json \
  --mode runtime --output /tmp/ministral-runtime-results

python3 Scripts/benchmark-campaign.py run /tmp/ministral-campaign.json \
  --mode quality --output /tmp/ministral-quality-results
```

Every output directory must be new. Review `commands.json` and `schedule.json` from the dry run before execution. `--mode all` runs both workloads. `--timeout 1800` limits each subprocess; failures are retained and subsequent arms still run. Interrupting the campaign terminates the current process group and retains completed records. There is no resume/retry facility in this version; a retry is a new campaign.

The generated example uses one fixed 256-token generation prompt and the authored general smoke corpus. That corpus is a wiring/sanity check, not sufficient evidence of broad quantization quality. Add representative, disjoint held-out code, prose, reasoning, and long-context corpora before selecting a production quantizer. Quality runs use the native teacher-forced NLL tool, not the separate BF16-teacher KL tool.

## Manifest and configuration sweeps

The manifest has `version: 1`, an integer `seed`, even `pairs` (default four), a `baseline` label, explicit absolute local `models` paths, and absolute `binaries.runtime` / `binaries.quality` paths. Executables must resolve inside a `release` directory. That guards against accidental debug selection; it does not independently prove compilation flags.

`runtime.prompts` contains `{ "label": "name", "text": "fixed prompt" }` entries. `quality.corpora` contains `{ "label": "name", "path": "/absolute/corpus.jsonl" }` entries. Labels must be unique within their list and contain only letters, digits, periods, underscores, and hyphens.

Shared `native_args` are passed as argument arrays, never shell text. Supported runtime arguments are `--engine`, `--tokens`, `--warmups`, `--context-length`, `--prefill-step-size`, `--kv-compression`, and `--allow-early-stop`. Quality supports `--cpu`, `--max-tokens-per-sample`, and `--prefill-step-size`. The campaign supplies positional paths, `--prompt`, and `--trials 1`; native A/B/cache modes are deliberately excluded because their reports contain multiple trial types. The order seed is not passed to the model: the native runtime has no sampling-seed option.

Each model can additionally supply `runtime_native_args` and `quality_native_args`. These override shared settings, except output length and quality sample token limits must stay fixed. Multiple labels may point to the same checkpoint. For a prefill sweep, use the same `path` in these model entries:

```json
{
  "baseline": "prefill512",
  "models": [
    {"label": "prefill512", "path": "/absolute/model", "runtime_native_args": ["--prefill-step-size", "512"]},
    {"label": "prefill128", "path": "/absolute/model", "runtime_native_args": ["--prefill-step-size", "128"]},
    {"label": "prefill256", "path": "/absolute/model", "runtime_native_args": ["--prefill-step-size", "256"]},
    {"label": "prefill1024", "path": "/absolute/model", "runtime_native_args": ["--prefill-step-size", "1024"]}
  ]
}
```

This is a fragment to merge into a generated manifest. Supply a fixed prompt long enough to exercise the prefill sizes, and compare TTFT and prefill metrics as well as decode. Context/KV policy differences are recorded and permitted; differing prompt token identities or output lengths invalidate comparisons.

For an existing opt-in runtime knob, list its exact name in top-level `environment_allowlist`, then assign string values in top-level `environment` or each model's `environment`. Only explicit `MODEL_RUNNER_*` and `MLX_*` names are allowed. Model values override shared values. An allowlisted variable omitted from both maps is unset in that arm, so an ambient opt-in cannot leak into the control. Other inherited runtime environment settings are recorded in provenance; the harness does not provide an isolated OS environment.

## Pairing, evidence, and acceptance

Every candidate is compared directly with the baseline for each prompt/corpus. A seeded schedule alternates AB/BA order across repetitions, giving equal counts of each order; the comparison-case order is also shuffled. The seed, complete schedule, native arguments, and per-arm environment are retained. Each runtime process performs the requested warmups and one measured generation. Loading is excluded from native decode/TTFT metrics; process wall time includes loading. Filesystem caches are not flushed.

`summary.json` reports medians and **paired** ratios, not a ratio assembled from unrelated aggregate samples. Decode/prefill rate ratios are candidate/baseline; TTFT speed ratios are baseline/candidate. Ratios above one favor the candidate. Each metric has separate baseline drift, candidate drift, AB/BA order effect, paired range, median absolute deviation, and a deterministic 2,000-resample paired-bootstrap 95% interval.

An accepted runtime ratio requires all scheduled pairs to be complete and comparable, at least four pairs, verified exact prompt-token fingerprints, and both arm drift and order effect within `maximum_drift_fraction` (default 10%). Drift compares the first and second halves of that arm's repetitions. Missing token fingerprints retain diagnostic measurements but cannot produce an accepted ratio. Short generations fail even when native `--allow-early-stop` is requested. Quality comparisons require identical corpus and token fingerprints, scored token counts, sample counts, special-token policy, and device.

“Accepted” means a measurement passed these gates. It is not proof of improvement. An interval spanning one does not support a speedup claim; small samples, thermal autocorrelation, and multiple tested configurations further limit conclusions. Use more pairs and confirmation on another session/device before promoting a small gain. A campaign can finish with exit code zero while a metric's accepted ratio is null because drift or sample-count gates failed; inspect `acceptance_reasons`, not only the process status. Failed/missing/incomparable arms produce exit code one.

## Provenance and limitations

`provenance.json` records executable and adjacent metallib SHA256 hashes, model configuration/tokenizer metadata hashes, model weight inventories, corpus hashes, compiler/platform/hardware information, relevant environment, and start/end power/memory/thermal state where available. It records main/dependency Git revisions and dirty diffs, including the embedded MLX/MLX-C repositories. Source file hashes include untracked new sources and patches, which Git diffs alone omit.

Weight hashing defaults to file size, modification time, and a clearly labeled first/last 1 MiB sample hash to avoid reading tens of gigabytes before every experiment. This is **not** a full content identity. Pass `--full-weight-hash` for durable checkpoint SHA256 provenance. Keep models, sources, binaries, and system workload unchanged during a campaign. The harness records provenance but cannot ensure that binaries were built from the currently hashed source tree. Native reports currently provide active MLX peak memory, not complete process RSS or energy consumption.

Run the CPU-only harness tests with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s Tests/Python -p 'test_benchmark_campaign.py' -v
```

## Pinned code/math reference corpus

`Scripts/prepare-quantization-corpus.py --output-dir /tmp/quantization-corpus` downloads exact pinned revisions of MBPP and GSM8K, verifies source SHA256 hashes, selects MBPP tasks 11–74 and the first 64 GSM8K test records, and reproduces the experiment's exact 128-record UTF-8 payload. It also verifies each derived corpus and the combined SHA256 (`5fc84a9794338a138e7284f415018323aa6bfec54227d09db8a95ad562e532f6`). Add `--cache-dir /absolute/existing/corpora --offline` to reproduce from cached `mbpp-source.jsonl` and `gsm8k-source.jsonl` without network access. Matching outputs are reused without rewriting; corrupt caches and differing output files cause an error before output writes. `provenance.json` records the pinned repositories, revisions and hashes.

These records support **reference likelihood and teacher-distribution fidelity**. NLL scores the supplied full question/task plus reference answer/solution; full-vocabulary `KL(teacher || student)` measures deviation from the BF16 teacher at those same token positions. Neither is generated-code pass@1 or generated math accuracy, and teacher-forced next-token top-1 accuracy is not task accuracy. A quantizer can improve KL while worsening reference NLL. Generated-task evaluation needs a separate prompt-only generation protocol and answer/code scoring; do not promote a quantizer from these reference metrics alone.


## Analyze the same-loaded gather/SILU kernel experiment

The native runtime executable accepts `--laguna-gather-silu-ab`, alternating the stock and opt-in fused primitive within one loaded Laguna model. Use an even `--trials` count of at least four pairs and retain the raw JSON plus executable/Metal-library hashes from the run. Analyze the completed report separately:

```bash
python3 Scripts/analyze-laguna-gather-silu-ab.py /absolute/native-kernel-ab.json \
  --output /absolute/kernel-ab-analysis.json
```

This CPU-only analyzer excludes warmups and checks consecutive paired sequences, exactly one stock/fused arm per pair, matching prompt-token fingerprints, complete and equal generation lengths, equal prefill/cache lengths, and exact output text across all measured trials. Missing identities, mismatched outputs, incomplete arms or invalid measurements withhold all accepted ratios. Each of decode, prefill, TTFT and total latency then has its own two-arm drift, AB/BA order-effect and paired-bootstrap calculation; a stable decode rate cannot hide drifting TTFT. At least four pairs with equal AB/BA counts are required. The default drift/order threshold is 10% (`--maximum-drift-fraction`); the bootstrap seed defaults to `20260904` (`--seed`). Ratios greater than one favor the candidate.

An accepted ratio is an eligible measurement, not proof of improvement or approval to change defaults. The report preserves diagnostic ratios when timing gates fail and records the input, analyzer and statistics-helper SHA256 hashes. Native JSON alone does not attest the exact binary/metallib build. Exact decoded text plus token count does not independently verify generated token IDs or every internal activation. Exit code 0 means structurally comparable evidence (timing gates may still fail), 1 means incomparable evidence with an analysis written, and 2 means a malformed input or I/O failure. Existing differing output files and the native input are never overwritten.


## Check prefill chunking against held-out reference NLL

The quality executable accepts optional `--prefill-step-size`. Zero or omission preserves its existing all-at-once `cache:nil` scoring path and the 2,048-token sample limit. Positive sizes 1–8,192 use a fresh native `model.newCache(parameters: nil)` for each sample and permit up to 32,768 encoded tokens per sample. Defaults remain 512 maximum sample tokens and zero chunk size. Every chunk evaluates FP32 cross-entropy, including the target crossing its boundary, and realizes its updated KV cache before advancing. The maximum logits tensor is limited to the current chunk; full-attention KV storage still grows with sample length. All evaluated token IDs and sample/combined fingerprints remain unchanged by the choice of chunk size.

For a 512-versus-2048 quality gate, use the same checkpoint and held-out corpus in both arms, shared quality arguments `['--max-tokens-per-sample', '8192', '--prefill-step-size', '512']`, and candidate `quality_native_args: ['--prefill-step-size', '2048']` (use JSON double quotes in a manifest). The campaign permits this policy difference, requires the report to confirm the requested chunk setting, records settings in the comparison, and continues requiring matching corpus/token identities and scored lengths. The old binary's missing field is treated as zero only when no chunk flag was requested; it cannot silently satisfy an explicit chunking request.

Reference NLL is a teacher-forced likelihood check, not generated-task accuracy. Equal NLL within sampling uncertainty would not prove that chunking preserves greedy output or user-visible task behavior; retain generated-task evaluation and representative workloads as separate gates before choosing a default.


### Chunking regression and numerical interpretation

The scoring regression uses a deterministic tiny Laguna model and an independent CPU log-softmax target oracle. It checks every target across chunk sizes 0, 1, 3, 8, 16 and 32, including two wraps of an eight-token sliding window and fresh-cache sample repetition. The aggregate CPU tolerance is 0.0001 nats across 20 targets. A separate native-device test compares compiled attention/MoE combinations with their eager counterpart at each chunk size; it does not assume all GPU chunk sizes have identical logits.

During diagnosis on this Apple GPU, seed-101 FP32 fixture NLL totals were approximately 69.773346 for one-token chunks and 69.768387 for multi-token/full-pass execution. All four compiled-attention/MoE on/off combinations produced the same respective totals. CPU one-token, multi-token and full-pass results agreed within 0.000008 nats, and cached per-token logits matched uncached-prefix logits within 0.0000006 through both wraps. These observations exclude target loss, generic cache-wrap handling and the two project fusion toggles as explanations for that fixture; they do not identify or establish an upstream kernel bug. The original GPU cross-chunk equality assertion was replaced with the stricter CPU correctness oracle and per-size GPU fusion-parity checks, rather than loosening its tolerance. This is another reason to measure held-out NLL when changing prefill policy.


## Compare quantizations on the 192-record held-out corpus

`benchmark-results/quantization-20260904/prepare-laguna-heldout-corpus.py` combines the pinned WikiText-2 test64 source with the reproducible MBPP64/GSM8K64 references. It verifies both input SHA256 hashes, preserves every record and unique ID in WikiText→code→math order, and writes compact UTF-8 JSONL with an LF after every record. The required 192-record output hash is `9490178eb34b57526a8245632272824bab040dd3c8556735233b2dc30b69a5b5`. Matching output can be reused; differing files are never overwritten. Source/derived corpus contents are not retained in the result archive.

After scoring the same corpus with the same `--max-tokens-per-sample 2048 --prefill-step-size 512` settings on each checkpoint, compare completed native reports:

```bash
python3 Scripts/analyze-quality-nll.py \
  --report standard=/absolute/standard-native.json \
  --report ls2=/absolute/ls2-native.json \
  --report awss=/absolute/awss-native.json \
  --output /absolute/paired-quality-analysis.json
```

The analyzer accepts two to eight labeled reports and computes all pairwise contrasts, overall and per category. It requires matching corpus/token fingerprints, ordered sample IDs, categories, original/evaluated/scored lengths, truncation decisions and scoring settings. Totals must reconcile with individual samples. Differing model paths are recorded because these are quantization comparisons; separate checkpoint/binary provenance is still required.

The default 10,000 paired bootstrap draws (`--draws`) use seed `20260904` (`--seed`). Each overall draw resamples records inside each category while preserving its observed sample count, then recomputes token-weighted NLL. Negative candidate-minus-baseline NLL favors that candidate on these references. Category intervals are also paired. These intervals do not account for source correlations, run/numerical variation or multiple comparisons; they do not measure generated-task accuracy, teacher KL or runtime speed. A zero-crossing interval does not establish equivalence or noninferiority. JSON includes raw input and analyzer SHA256 hashes, and an existing differing output cannot be overwritten.
