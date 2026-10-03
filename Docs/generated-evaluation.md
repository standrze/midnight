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

## Coding and cybersecurity quantization evaluation

For the Gemma quantization campaign, select on coding and cybersecurity tasks;
math has zero selection weight. Keep calibration, candidate screening and final
holdout examples separate. The historical 137-task suite below remains a
convenience suite, not an untouched holdout for this campaign.

The scorer also accepts public tasks with `category: "cybersecurity"`. Their
private answer has `kind: "security-json"`, `verdict` (`safe` or `vulnerable`),
`cwes` (distinct canonical labels such as `CWE-89`), `evidence_lines` (distinct
one-based source-line numbers), and a positive `line_count`. Define the threat
scope and source-line numbering in the public prompt. A safe label requires
empty CWE and evidence arrays; a vulnerable label requires both arrays to be
nonempty. Keep the private labels out of generation inputs.

An optional private `evidence_anchor_lines` list requires at least one cited
vulnerable-operation line, in addition to relevant data-flow evidence. Anchors
must be distinct members of the accepted evidence set, nonempty for a vulnerable
key and empty for a safe key. Existing keys without anchors keep their original
scoring rules. State the operation-citation requirement in the public prompt.

Responses must be exactly an object with `verdict`, `cwes` and `evidence_lines`,
either plain JSON or one complete JSON fence. Duplicate keys, extra fields,
nonstandard numbers and ambiguous fences fail. A task passes only when its
verdict and CWE set match and its cited lines are nonempty and all belong to
the labeled evidence set (or empty for a safe task). This workflow never
executes security-review source or generated review responses.

Security category summaries include classification precision, recall,
false-positive rate, invalid-response count and classification accuracy.
Invalid responses count against recall on vulnerable tasks and against
classification accuracy; a missing prediction is never counted as a true
negative. Complete task success additionally requires CWE and evidence
correctness. Missing generation records withhold qualified suite accuracy;
partial diagnostics retain an explicit completeness flag.

These labels assess only the declared threat scope. They do not establish
broad security expertise, remediation quality or exploit validity. Use both
vulnerable examples and plausible safe controls, inspect false positives and
false negatives, and verify executable coding/patch references separately in
the pinned sandbox. Small pilot gains cannot justify a default change.

### Authored development pilot

`Scripts/prepare-cyber-coding-pilot.py --output-dir /new/pilot` creates an
18-task coding/cybersecurity screen: six executable secure-coding tasks and
six paired vulnerable/safe review families. Preparation downloads nothing,
records file and builder hashes, refuses existing output directories, and
keeps `answers.jsonl` and `references.jsonl` at mode `0600`. IDs and public
metadata do not disclose the review verdicts. Only `tasks.jsonl` goes to the
native generator.

Before model scoring, verify every reference implementation passes and every
intentionally faulty implementation fails the same private tests:

```sh
python3 Scripts/validate-cyber-coding-pilot.py \
  --corpus-dir /absolute/pilot \
  --sandbox-image python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea \
  --output /new/reference-validation.json
```

The same builder accepts `--code-cases-json /private/blueprints.json` for a
separate authored development remediation slice. Blueprints contain family,
prompt, reference, fault, tests, optional setup and optional `extra_faults`.
They are private authoring inputs; only the resulting `tasks.jsonl` may reach
generation. The prepared references retain additional faulty controls, and the
validator requires every reference to pass and every control to fail the tests
in the pinned sandbox. A code-only slice does not establish the complete equal
coding/cybersecurity selection score.

The Gemma campaign added six such remediation tasks on 1 October 2026. Three
recipes completed 18 generations under identical prompt token IDs/settings.
The frozen supplied-test results were 1/6 for installed A4B, 1/6 for its Q8-router
candidate and 2/6 for installed 31B. These are authored development findings;
syntax/format failures and the router's 2048-token tool-validation length stop
remain failures. All references pass and all 18 predeclared faulty controls fail.
No output, private key or prompt was repaired after generation.

All recipes emit the same extra quote in an incident-summary dictionary. Local
source inspection found tokenizer cleanup enabled by a generic default when the
Gemma configuration omits the flag. A Foundation probe with valid vocabulary IDs
reproduces doubled quotes through the inspected cleanup/common-prefix equation.
It does not instantiate the actual native tokenizer or capture the generations'
raw token IDs; the precise cause still needs that direct evidence. Treat these
raw scores as a rendering/runtime-inclusive development result, not an isolated
quantizer ranking. Native tokenizer validation and any corrected-runtime reruns
are separate from the retained original scores. See the campaign's
`security-remediation-findings-v2.json` for identities and scope.

These authored examples have not received independent label review and cannot
serve as a final holdout. Passing one faulty control does not establish complete
test coverage. Freeze equal coding/cybersecurity category weights (0.5 each);
the scorer's existing overall accuracy is record weighted and differs when
category counts differ. Report category results separately. Six pairs provide
six review families, not twelve independent source samples. Use an output
budget sufficient for complete implementations and retain all length stops.

### Publisher-labeled OWASP development proposals

The October Gemma campaign pins `OWASP-Benchmark/BenchmarkJava` at
`8b67a88d73b2594570fc21150705283de884620b`. The selected development slice has
64 cases: eight vulnerable and eight safe for each of CWE-22, CWE-78, CWE-89
and CWE-330, sampled with seed 20261001 before generation. Publisher verdicts
and CWE labels come from its hashed `expectedresults-1.2.csv`; source files and
GPL-2.0 notices remain archived. Six files inspected during source evaluation
are excluded from the selection.

`Scripts/prepare-owasp-review-proposals.py` works offline from the frozen source
manifest and selection. It verifies every source/label/license hash, consistently
renames the case IDs, preserves line numbering and logic, and separates public
prompts from private source mappings and answer proposals. The declared CWE
scope is public, so this slice assesses within-scope review rather than
unconstrained taxonomy discovery. Public benchmark training exposure is unknown;
it is a development set, not an untouched final holdout.

The publisher does not supply this workflow's evidence-line labels. Mechanical
sink proposals and prompt/helper assumptions therefore require review before
scoring. `proposed-answers.jsonl` intentionally uses the unsupported kind
`security-json-proposal`; the scorer rejects it. Native `--validate-only`
acceptance establishes public-schema validity only. Do not interpret preparation,
source acquisition or schema validation as generated model accuracy.

`Scripts/prepare-reviewed-owasp-development.py` freezes a separate reviewed
48-case development slice after primary source inspection of path traversal,
command injection and SQL injection. It verifies pinned helper/resource hashes,
requires evidence and operation anchors for every vulnerable case, and shuffles
case IDs before generation. The helper context describes actual pinned behavior,
including input-preserving helpers and platform-specific command execution.
All 16 randomness cases are excluded before generation: publisher negatives
include low-entropy floating-point remember-me tokens, which do not support the
broader insufficient-unpredictability scope. This is a label-quality exclusion,
not a decision based on generated results. Public comments/names can still hint
at labels; evidence was reviewed by the primary evaluator, not an independent
reviewer. This slice remains development evidence and cannot promote a recipe.

The distinction is token entropy versus generator choice: Java documents
[`nextFloat`](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Random.html#nextFloat())
as drawing from at most 2^24 values, while
[`SecureRandom`](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/security/SecureRandom.html)
provides a cryptographically strong generator. Choosing that generator does not
establish sufficient entropy after conversion to a floating-point token. This
is our scope assessment, not a relabeling of the publisher's benchmark.

The clarified v3 prompts explicitly require canonical CWE strings. A subsequent
citation audit found legitimate input-source lines omitted from the original
private evidence sets. `owasp-evidence-corrections-v1` records a source inspection
of all 24 vulnerable cases and expands only relevant input/guard context; every
verdict, CWE, operation anchor and safe-case evidence set is unchanged. An
unreachable constant-switch branch remains excluded. All models are rescored
against the same corrected keys, while original results remain preserved.
This review occurred after output inspection: corrected development scores are
not untouched holdout results, and changes caused by key corrections are not
model-accuracy gains. Use the corrected key path explicitly for subsequent
comparisons on the same v3 prompts and retain its SHA-256 in scoring provenance.

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

## Evaluate a saved adapter

Use the exact base checkpoint used for training, then add the native MLX
adapter directory. It must contain `adapter_config.json` and
`adapters.safetensors`:

```sh
.build/release/model-runner-generation-bench \
  /absolute/base-checkpoint /absolute/generated-corpus/tasks.jsonl \
  /absolute/results/adapter-native.json \
  --adapter /absolute/saved-adapter --engine metal --tokens 512 \
  --context-length 8192 --prefill-step-size 512 --kv-compression none
```

Omitting `--adapter-scale` uses the saved scale. An explicit finite,
nonnegative value replaces that scale; it does not multiply it. A scale
override requires `--adapter`. Scale zero is not guaranteed to reproduce a
model without adapter layers: the existing FP32 adapter branch can change
BF16 arithmetic even when the adapter update is zero.

The optional `adapter` report object records the absolute standardized path,
configured/requested/effective scale, config and weights byte lengths and
SHA256, and loading/verification state. Files are rechecked after successful
loading and after the run when SHA256 is available; changes fail the report.
On generation failure, the last completed verification state remains visible.
Runs without `--adapter` retain the previous report shape.

`--validate-only` checks adapter files and the native configuration schema
without loading tensor arrays or establishing compatibility with the base.
Failures are retained as `adapter_validation` with zero model loads. Tensor
or model incompatibility remains a loading failure. Loading an adapter or
passing a small generation check does not establish that training improved
held-out task quality; compare complete reports on the same public tasks.

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

Qualified comparisons also include a conservative weighted Hoeffding bound on
paired accuracy differences, grouping tasks by declared category/family where
available. It preserves record weights and never turns zero discordance into
an interval of `[0,0]`. This finite-sample bound assumes independent sampled
units; it cannot establish independence or representative sampling for authored
or convenience tasks. Its `promotion_gate` is explicitly false. Keep the
empirical bootstrap as a diagnostic, and do not infer a tight noninferiority
margin from a small or saturated suite.

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
## Native Gemma quote-boundary diagnostic

The opt-in `GemmaQuoteBoundaryDiagnosticTests` replays valid token IDs through
the installed tokenizer, `#adaptHuggingFaceTokenizer`, and
`NaiveStreamingDetokenizer`. It loads only local tokenizer assets. Set
`MIDNIGHT_GEMMA_TOKENIZER_PROBE_MODEL` to the installed checkpoint directory and
`MIDNIGHT_GEMMA_TOKENIZER_PROBE_OUTPUT` to a new report path, then run
`swift test --configuration release --filter GemmaQuoteBoundaryDiagnosticTests`.
Without the model variable, the diagnostic is disabled. It does not transfer
weights or alter checkpoint metadata.

For both installed Gemma 4 A4B and 31B tokenizers, the October 1 native replay
confirmed a streaming/batch mismatch when cleanup is absent or explicitly true:
batch decoding returns `{'outcome','source'}`, while streaming returns
`{'outcome', ''source'}`. Explicitly false returns `{'outcome', ' source'}` in
both paths. The space inside that key belongs to the replayed token sequence;
equality here does not prove that the generated program meets its contract.

This establishes a native decoder failure mode independently of quantization.
It does not establish which tokens produced the shared quote artifact in the
remediation benchmark, because that report did not capture generated token IDs.
Keep the original scores and raw responses. Capture generated tokens next,
compare their batch and streamed decoding, and identify any corrected runtime
separately before rerunning the frozen tasks. The native reports and logs live
under `benchmark-results/gemma-quantization-20261001/gemma-native-quote-boundary-replay-*`;
the earlier benchmark executable and Metal library are preserved under
`frozen-runtime-before-decoder-probe-v1/` with verified hashes.

## Generated-token evidence for the Gemma decoder defect

The opt-in `GemmaGeneratedTokenDiagnosticTests` uses the native library's
`generateTaskRecordingTokens` with the normal text/tool handler, greedy
sampling, the existing Gemma pad-token suppression, prefill 512, and no
assistant or compressed KV cache. It rejects mismatched rendered prompt tokens
before generation and records raw tokens, streamed content/reasoning, native
batch decoding, and naive streamed decoding. Set these environment variables:

- `MIDNIGHT_GEMMA_GENERATED_PROBE_MODEL`: local checkpoint directory.
- `MIDNIGHT_GEMMA_GENERATED_PROBE_CORPUS`: frozen public tasks JSONL.
- `MIDNIGHT_GEMMA_GENERATED_PROBE_PREVIOUS`: frozen native generation report.
- `MIDNIGHT_GEMMA_GENERATED_PROBE_OUTPUT`: new capture JSON path.
- `MIDNIGHT_GEMMA_GENERATED_PROBE_TASK`: task ID, default `cyber-remediation-code-03`.

Run `swift test --configuration release --filter GemmaGeneratedTokenDiagnosticTests`.
The diagnostic is disabled without the model variable. Model memory follows the
default native guard; these captures used 54,975,581,389 bytes (about 51.2 GiB),
matching the earlier benchmark, and a 256 MiB allocator cache. This is not the
48 GiB process-footprint acceptance gate for controlled serving measurements.

All three task-03 captures (installed A4B, Q8-router A4B, installed 31B) exactly
reproduced their original streamed answers and prompt fingerprints. Each
streamed Python body has invalid doubled quotes. Batch decoding those same
recorded tokens produces a syntactically valid body. The raw capture also
includes the terminal `<turn|>` outside the code fence.

The retained `tools/analyze_gemma_token_capture.py` calls the existing scorer
and its pinned Docker sandbox to test each literal batch-decoded fence body;
it does not repair characters or execute generated code on the host. All three
bodies still fail the supplied security-contract tests. This confirms a
runtime-caused syntax failure while retaining evidence of other correctness
failures. The fence-body replay is a post-generation decoder diagnostic, not
strict response accuracy or a new model-generation result. Frozen scores stay
unchanged, and a corrected runtime still needs separately identified full-task
runs. The capture reports, sandbox result, and full checkpoint recheck are in
`benchmark-results/gemma-quantization-20261001/gemma-generated-token-*`.

## Scoped Gemma cleanup default correction

`Patches/swift-transformers-gemma-cleanup-default.patch` changes the absent
cleanup-flag default to false for `GemmaTokenizer` and `GemmaTokenizerFast`.
The pinned dependency helper is wired into `prepare-dependencies.sh`, checks
forward/reverse applicability atomically, and refuses unrecognized revisions
or conflicting edits. `Tests/Shell/GemmaCleanupDefaultPatchTests.sh` covers
clean and currently overlaid source, idempotence, reversal and refusals.

The correction follows the family defaults documented by Hugging Face at
<https://huggingface.co/docs/transformers/v4.52.2/en/model_doc/gemma>. Explicit
true/false flags remain authoritative. Other families retain their prior default.
Explicit cleanup true can still rewrite earlier streamed text; this scoped
correction does not make arbitrary cleanup-enabled decoding append-safe.

Eleven native tests passed across Gemma defaults, installed Gemma replay,
Talkie and exact EOS compatibility. The pre-correction tokenizer source and
replay test remain in `decoder-default-before-correction-v1/`. The new
generation executable is frozen under `corrected-runtime-gemma-cleanup-v1/`;
its Metal library matches the earlier runtime. All quantization comparisons
must identify this runtime change separately. Code-body batch replay from old
tokens is not equivalent to fresh generation with cleanup disabled.

The corrected-runtime remediation slice is complete: installed A4B 2/6,
Q8-router A4B 3/6, and installed 31B 3/6. Earlier-runtime results remain 1/6,
1/6 and 2/6. All six prompts match their earlier token fingerprints, full
checkpoint files match prior identities, and each wrapper verifies unchanged
checkpoint/runtime/tasks before and after generation. No corrected answer
fails syntax parsing. Task 04 SQL construction now passes for all three;
incident-summary contract tests still fail. These are development findings,
not precision promotion or a final accuracy estimate. See
`security-remediation-cleanup-findings-v1.json` for the complete evidence.
