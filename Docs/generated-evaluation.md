# Generated-output evaluation

Use this workflow to compare one greedy answer per task under recorded runtime
settings. It complements reference NLL and teacher KL with generated math
answers, Python programs and synthetic retrieval. It does not establish a
universal model ranking or a runtime speed improvement.

## Prepare the public tasks and separate scoring key

The offline builder reads the verified `mbpp-source.jsonl` and
`gsm8k-source.jsonl` cache used by `model-runner-prepare-corpus reference`. It downloads
nothing and refuses changed source bytes or differing existing outputs.

```sh
python3 Scripts/prepare-generated-eval-corpus.py \
  --cache-dir /absolute/source-cache \
  --output-dir /absolute/generated-corpus \
  --math-count 64 --code-count 64 \
  --retrieval-token-estimates 4096 16384 32768 --seed 20260904
```

This explicit full selection contains **137 fixed tasks** in source order:
64 MBPP task IDs 11–74, 64 GSM8K test records 0–63, then nine synthetic retrieval
tasks. Builder defaults are a smaller pilot of 16 code and 16 math tasks plus
nine retrieval tasks; always retain the chosen counts in provenance.

`tasks.jsonl` contains `{id, category, prompt}` and optional public metadata.
**Pass only this file to generation.** `answers.jsonl` contains numeric/exact
answers or code setup/tests and is created with mode `0600`. The builder never
copies reference function bodies into prompts. MBPP prompts include the task
text and required function signatures inferred with AST parsing; they exclude
assertions, test inputs and expected outputs. GSM8K prompts contain the question
and ask for a final `#### <number>` line, without reference reasoning or answers.
Retrieval prompts necessarily contain the supporting key/value fact in their
context, while the scoring key remains separate.

The immutable input identities are:

| Source | Revision | Source payload SHA256 |
| --- | --- | --- |
| `google-research/google-research`, `mbpp/mbpp.jsonl` | `f82046ba5aabbbb427dbfd38a254d26bff08b533` | `ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f` |
| `openai/grade-school-math`, `grade_school_math/data/test.jsonl` | `b0bb162abedc65e1fdd8e93ed090fd7598ee68bc` | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` |

For the September 4 full suite, the frozen files are in
`/private/tmp/midnight-priorities-20260904/generated-corpus/`:

- `tasks.jsonl`: 308,405 bytes, SHA256 `f0bc4abb263adf68d3de793c6e5128417b5969b8314d5f6aaa9962b2eb8af114`.
- `answers.jsonl`: 19,710 bytes, SHA256 `2ffe2f3de3957a259244842ace1d10ccac0d1a3d0b61bb9e2525227ee143ff0f`.
- `provenance.json` records selections, seed, source identities and builder SHA256 `28d44452da5ef0ac0021228dd4e5174eaf62d18070ae0cc2bb941fd75ded3a6d`.

A new builder revision may change provenance even if task bytes remain the
same. Preserve the original files and use a new output directory when needed.

## Render actual prompts, then generate

Build once before the evaluation interval:

```sh
swift build -c release --product model-runner-generation-bench --jobs 8
bash Tests/Shell/ModelGenerationBenchmarkTests.sh
```

The shell tests use `--validate-only` and never load a model. That mode checks
public-schema validity, unique IDs and UTF-8, and records corpus/prompt hashes; it cannot establish
token lengths or request admission. Optional metadata is ignored by native
prompt construction. Extra top-level fields such as `answer` or `tests` fail
validation.

Use `--prepare-only` to load the checkpoint once and record actual, fully
rendered prompt-token counts before generating. The following context ceiling
is an example for checkpoints whose metadata permits it:

```sh
.build/release/model-runner-generation-bench \
  /absolute/checkpoint /absolute/generated-corpus/tasks.jsonl \
  /absolute/results/prepared.json \
  --engine metal --tokens 512 --context-length 65536 \
  --prefill-step-size 512 --kv-compression none --prepare-only
```

Retrieval labels **4K/16K/32K are estimates**, based on record count, not certified
token lengths. Targets occur near 10%, 50% and 90% of record positions; these are
not guaranteed tokenizer-coordinate positions. Report actual
`prompt_token_count` for each checkpoint and task. Native preparation uses the
checkpoint's chat template and includes its generation prefix; raw character
counts and tokenizer estimates cannot replace that measurement.

Generate with a new report path by removing `--prepare-only`:

```sh
.build/release/model-runner-generation-bench \
  /absolute/checkpoint /absolute/generated-corpus/tasks.jsonl \
  /absolute/results/standard-native.json \
  --engine metal --tokens 512 --context-length 65536 \
  --prefill-step-size 512 --kv-compression none
```

Each invocation loads one model once and processes every record sequentially.
Each prompt is exactly one user turn with no added system text. Generation uses
`temperature=0`, `top_p=1`, a fresh independent request cache, no speculative
model, and no prompt-cache reuse. Laguna uses the ordinary production compiled
MoE/attention paths and Metal block-tail/router defaults. The experimental
fused gate/up-SiLU kernel is explicitly disabled. No retries or best-of-N
selection are performed. Greedy decoding does not promise bit-identical output
across different kernels, devices or floating-point execution settings.

`--tokens` accepts 1–32,768 and is an output ceiling, not a required output
length. `--prefill-step-size` accepts 1–8,192. The context limit must fit model
metadata, and the complete rendered prompt plus reserved output budget must
pass context and memory admission. **Prompts are never silently shortened.**
Oversized requests stop with an explicit error instead of evaluating a prefix.
EOS stops are valid. `stop_reason=length` and `output_limit_reached=true` expose
output-budget exhaustion; such responses stay in the scoring denominator.

## Reports and interrupted runs

The JSON contains the exact prepared-token FNV fingerprint, prompt SHA256,
actual prompt/generated token counts, stop reason, generated text, cache/prefill
accounting and diagnostic timing for each record. The top level records corpus
SHA256/FNV identities, invocation, model config/index hashes, model-load count,
requested/resolved settings and selected environment values, including
`MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER`. These metadata hashes do
not hash model weights; retain an external full checkpoint/artifact manifest
for an exact-build comparison. MLX memory is distinct from process RSS or
footprint. Time-to-first-content measures the first emitted nonempty text
chunk, which need not coincide with the first generated token.

The output path must be new. After exclusive initial creation, each snapshot
is written by atomic replacement. Completed records survive an interruption;
the most recent report names any active sample and reports completion counts.
A caught request error retains that sample's partial text, marks it `error`,
sets top-level `status=failed`/`failure_status`, and exits nonzero. Abrupt process
termination can leave a `running` report with earlier completed records. There
is no automatic resume or replacement of existing reports. Use a new path and
keep incomplete evidence identifiable.

Only a fully finished generation run has `status=completed`. Validation and
preparation reports have `validated` and `prepared` status and are not generated
accuracy results. The scorer rejects inconsistent counts or falsely claimed
completion, retains missing/failed records as unscored, and withholds full-suite
accuracy whenever any record is unscored. On macOS, CryptoKit supplies corpus
and prompt SHA256. Builds without CryptoKit explicitly omit SHA256; the current
Python scorer requires these hashes and will reject those reports rather than
accept an FNV substitution.

## Score with the private key

```sh
python3 Scripts/evaluate-generated.py \
  --tasks /absolute/generated-corpus/tasks.jsonl \
  --answers /absolute/generated-corpus/answers.jsonl \
  --report standard=/absolute/results/standard-native.json \
  --report awss=/absolute/results/awss-native.json \
  --output /absolute/results/generated-scores.json \
  --sandbox-image python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea \
  --code-timeout 15 --bootstrap-draws 10000 --seed 20260904
```

The image must already be installed locally and its exact Linux digest must
pass inspection; the scorer never pulls it. Generated code is never executed
by host Python. Each code task runs in an unprivileged Docker container with
no network, read-only root/input, dropped capabilities and bounded CPU, memory,
process count, output and time. Only a temporary grader directory is mounted.
There is no host-execution fallback. Without a configured sandbox, executable
code responses remain explicitly unscored; syntax/format errors can still be
identified without executing code.

The container is an isolation boundary, **not a tamper-proof adversarial
grader**. Candidate code runs alongside its test harness and can inspect grader
inputs. The completion marker detects simple early exits; it does not establish
integrity against arbitrary score-spoofing programs. Use a trusted disposable
Docker environment and retain sandbox provenance. Code success means passing
only the supplied MBPP tests; it is not a HumanEval result or comprehensive
functional correctness.

Before trusting code scores, verify that the pinned reference programs pass
the same private tests in the same sandbox:

```sh
python3 Scripts/validate-generated-code-references.py \
  --tasks /absolute/generated-corpus/tasks.jsonl \
  --answers /absolute/generated-corpus/answers.jsonl \
  --cache-dir /absolute/source-cache \
  --sandbox-image python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea \
  --output /absolute/results/reference-grader-check.json
```

This validates grader/source compatibility; it does not feed reference solutions
to the model. Preserve every reference failure and the frozen denominator.

Math scoring requires one final `####` number and compares its exact numeric
value, accepting valid decimal/fraction forms without units. Retrieval requires
the exact value after surrounding-whitespace removal. Code accepts plain
Python or one complete Python/plain fence; ambiguous blocks and surrounding
prose fail rather than being guessed away.

For retrieval-format investigation, add `--retrieval-diagnostics` to the scorer.
This records whether the expected value appears, the count of emitted values
matching the fixed value pattern, whether all such values equal the expected
value (`null` when none appear), and whether the response contains a tag.
These are **post-generation diagnostics, not retrieval accuracy or a second
success rule**. For example, the correct value repeated twice with `</think>`
still fails strict exact match. The flag does not change pass/fail decisions,
accuracy denominators or paired intervals.

Paired comparisons require matching corpus/prompt identities, prompt lengths
and every recorded execution setting. For a deliberate KV experiment, explicitly
add `--allow-setting-difference kv_compression`; only that setting may differ,
and both values remain recorded. This labels a runtime-setting experiment,
not a matched-checkpoint quantization comparison. Other settings, including
prefill size and environment, still must match.

## Interpreting the fixed suite

The scorer reports category accuracy and record-weighted overall accuracy,
candidate-only/baseline-only successes, and paired bootstrap intervals. The
10,000-draw overall bootstrap preserves the observed code/math/retrieval counts.
Missing or unscored records prevent qualified paired comparisons; reported
subset accuracy is diagnostic only. Timing is excluded from accuracy estimates.

These 137 convenience tasks include public benchmarks with unknown training
contamination and prior exposure in reference-likelihood comparisons. They are
not a blind, untouched holdout. Avoid fitting or selecting further candidates
on this suite and then calling its score independent validation. Three public
MBPP tests per task provide limited coverage; strict answer formatting also
affects success. Nine synthetic key lookups do not measure broad long-context
reasoning. Bootstrap intervals do not account for source correlations, repeated
candidate selection, multiple comparisons, or numerical/run variability.
Review category regressions, length stops and actual retrieval lengths before
changing a default; aggregate improvement alone is insufficient.
