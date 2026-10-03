# Gemma Metal performance work

This campaign prioritizes Gemma 3 and Gemma 4, including Gemma 3 270M. All weight
and KV quantization experiments must use **at least 4 bits**. The installed Gemma 4
checkpoints remain unchanged. Do not enable an experimental kernel by default
until numerical checks and counterbalanced full-model measurements pass. The
user accepts a modest speed cost for a demonstrated quality improvement;
evaluate quality, speed and memory together. Changed output fails an exact
preservation claim, but does not by itself establish worse quality. A noisy
NLL improvement whose confidence interval crosses zero is not demonstrated
quality improvement.

Evidence is retained in `benchmark-results/gemma-performance-20260929/`. The
preceding investigation and vLLM-Metal comparisons are in
`benchmark-results/metal-opportunities-20260929/README.md`.

## Implementation and validation lanes

| Lane | Change | Validation and remaining work |
|---|---|---|
| Q4 matrix-vector kernels | Opt-in full-tile plus tail kernels for affine Q4/G64 FP16/BF16; useful widths include 640, 704, 2816 and 5376. No weight padding or requantization. | The original 16-value arithmetic changed fixed-prefix predictions and amplified one-BF16-ULP differences. Stage 4's eight-value ordering preserves all 132,306 synthetic and 1,182,720 real-A4B-weight fixture outputs exactly, plus the captured 16-token 270M projection trace. Single A4B and 31B whole-model fixed-prefix A/B checks each match all 24 record losses and 2540 recorded winners at prefill step one; 31B NLL is exactly 8.965165077419732 in both arms. These capped reference checks do not establish generated-task quality or full-vocabulary equality. Fresh ABBA 256-token screens preserve exact output: A4B 134.28 → 146.35 tok/s (+8.99%); 270M 561.33 → 557.91 (−0.61%). The longer four-pair A4B recheck preserves output but fails the 10% drift/order gates: diagnostic paired +8.10%, interval −8.67% to +9.97%, baseline/candidate drift 21.91%/27.84%. The exploratory two-pair 31B run also preserves output but fails drift/order gates. Neither provides an accepted repeatable speed claim. Historical failed results remain distinguishable. |
| Attention | Opt-in two-pass fused D512 decode for one query, batch one, FP16/BF16, GQA up to 8 and no array mask. | D512 Metal numerical/stride/sink/fallback tests and both D256 pruning arms pass. A4B full-model screens show no material speed gain; D512 changes generated text. D512 multi-query prefill remains separate work. |
| Prefix reuse | Gemma 4 uses exact prepared token IDs for independent prompt checkpoints; checkpoints retain the existing configurable memory/entry limits. | CPU branch/cancellation checks pass. Four alternating marker continuations match exactly: A4B TTFT −58.8%, 31B −72.4%, 512/656 tokens reused. Generated KV and paged copy-on-write storage remain separate work. |
| Cache memory | Admission accounts for Gemma 4's retained history and distinct global head geometry. | An opt-in chronological bounded cache and matched admission policy are implemented. Stage 3 reuses an owning concatenation when it fits the window; initial/cropped tails still copy. Numerical, ownership and disk round-trip tests pass. A4B's marker recheck preserves exact output and saves 408.47 MB (2.63%); TTFT is 1096.62 → 1105.61 ms. Four generated tokens cannot establish sustained decode speed. A separate 3837-position probe changes 117 winners after the 1024-token window; reference NLL decreases by 0.010622, but the change's 95% interval crosses zero, proving neither quality improvement nor equivalence. The earlier 31B screen saved 1.68 GB; its copy-removal recheck remains outstanding. Default storage remains retained history. |
| Small models | Gemma 3 recognizes explicit attention layouts and the alternate sliding-window-pattern key; tiny fixtures no longer assume a sixth layer exists. | CPU Q4/cache tests and a real pinned Gemma 3 270M Q4 checkpoint cover this lane. The opt-in compiled post-attention tail preserves a 2550-position fixed-prefix pilot and records one compiled trace per layer. Its four-pair decode interval spans −5.24% to +1.96%; no speed gain is established. Diagnostic TTFT and memory increase. See [compiled-tail evidence](Gemma3CompiledTailExperiment.md). Fine-tuning throughput and adapter quality require separate training measurements. |
| Quantized checkpoints | Preserve 4-bit minimum; investigate selective G128, better fitting, metadata dtype alignment and grid-preserving official QAT import. | Four 270M Q4 candidates convert from one verified BF16 source with exact unchanged tensors/norms and tied-head checks. Selective G128 saves 1.106 MB. Full-192 and nonpilot-168 reference screens favor ordinary selective G128's pooled NLL, but it worsens code while improving prose. Searched G64 improves code/math while worsening prose, and leads under an explicitly secondary equal-category weighting. Teacher KL, task accuracy and speed remain distinct outcomes. See the [quality tradeoffs](../benchmark-results/gemma-performance-20260929/gemma270-four-arm-quality/README.md) and [quantization plan](GemmaQuantizationPlan.md). QAT import remains unimplemented. |
| Speculative verification | Reuse expert weights across candidate positions and choose draft block lengths by accepted tokens per second. | Six grouped-expert Metal numerical cases pass, including more than four repeated assignments. A4B assistant screens remain slower than their target-only controls and all fail exact target-output parity; grouped verification also changes assistant output and accepted/proposed counts. Empty-mask omission matches baseline assistant output in this screen, without a repeatable speed gain. Stage 3's adaptive arms bypassed the policy. Stage 4 repairs stateless-processor eligibility, with all eight iterator tests passing under both flags; its fresh adaptive screen reaches 125.77 versus target-only 130.32 tok/s (−3.50%) and still changes output. Task-quality improvement has not been established. |
| Serving and scheduling | Measure submission gaps, concurrent request throughput and chunked prefill before choosing scheduling changes. | Bounded, opt-in command-buffer telemetry passes CPU publication checks and an actual A4B off/on ABBA screen: identical output, 135.081 → 135.169 tok/s (+0.0653%) and TTFT 69.994 → 70.079 ms (+0.1226%). This short screen shows little measured timing change, not a general overhead bound. Both enabled traces reach the 65536-sample cap; queue-local gaps have no workload-phase labels and are not total GPU utilization. Continuous batching remains separate work. Existing thread-pinning experiments did not establish a repeatable gain. |

The [stage 3 integrated test log](../benchmark-results/gemma-performance-20260929/root-tests-stage3-pointer-repair.log)
records 519 passing Swift Testing tests and nine passing XCTest tests, with two
CUDA tests skipped. This includes a Metal compatibility check; it is not an
entirely CPU-only run. The initial test-only raw-pointer lifetime crash and
successful repair are both retained. The [campaign evidence](../benchmark-results/gemma-performance-20260929/README.md)
links the grouped-expert, real-checkpoint, amplification, bounded-cache and
assistant reports. These checks have not promoted an experimental default.
The [stage 4 full run](../benchmark-results/gemma-performance-20260929/root-tests-stage4.log)
passes 522 Swift Testing tests and nine XCTest tests, with two CUDA tests skipped.
The later [repaired stage 6 full suite](../benchmark-results/gemma-performance-20260929/root-tests-stage6-q4-output-repair.log)
passes 540 Swift Testing tests and nine XCTest tests, again with two CUDA tests
skipped. Separately enabled experimental/real-model fixtures remain gated in
this invocation. Its four EOS vocabulary-identity regressions pass without changing
the existing unknown-token termination policy. The Q4_0 primitive passes five
synthetic cases and an official 32-by-5376 row slice with original FP16 scales;
maximum BF16-output absolute error versus the CPU FP64 dot product is
0.000030512914. The original JIT output-cast failures remain in the earlier
stage 6 log. These checks do not establish a complete QAT model import, quality
gain or throughput improvement, and no experimental default is promoted.

The [short synthetic retrieval development screen](../benchmark-results/gemma-performance-20260929/gemma270-short-retrieval-stage8/README.md)
completes eight BF16 responses and scores 0/8 under its frozen case-sensitive
criterion. Each response is the correct requested word with initial capitalization;
the lowercase answer keys impose a casing requirement absent from the prompt.
This is a format-criterion mismatch, not evidence of failed word retrieval. The
separate posthoc casefold diagnostic is 8/8, while the primary score and failed
eligibility gate remain unchanged. All 24 held-out tasks stay unexecuted; this
screen supplies no quantizer quality comparison.

A separate [two-step native LoRA check](../benchmark-results/gemma-performance-20260929/training-smoke/README.md)
passes on the real 270M mixed Q4/Q8-query/key checkpoint: all 493 frozen base
tensor hashes and 127 linear geometries stay exact, the 14-tensor adapter updates,
and saved/reloaded logits match exactly. The existing FP32 adapter branch has a
precision effect even before training: enabling a zero-valued adapter changes
BF16 logits to FP32, with maximum difference 0.49017334 on the fixed toy input.
The separate debug Studio worker also completes two Adam steps on tiny synthetic
text and saves the expected 14 finite FP32 tensors, with all seven B tensors
nonzero and recorded source/runtime/data identities unchanged. The saved worker
adapter now [passes direct runtime reload and generation](../benchmark-results/gemma-performance-20260929/generation-adapter-cli-studio/README.md)
in two fresh serial processes at saved scale 16: the same 20-token prompt
produces exactly 16 tokens with identical text, fingerprints, counts and length
stop. Adapter hashes are checked after load and run; all model/input/runtime
hashes remain unchanged. The native verifier's separately trained SGD adapter
loads successfully but [fails the same public generation prompt](../benchmark-results/gemma-performance-20260929/generation-adapter-cli/README.md)
without producing text; that failed fixture remains preserved. These are
functionality checks, with no quality, inference-speed or training-throughput
conclusion, and no claim that enabling an adapter preserves base precision.

## Prompt preparation — stage 8

The tokenizer now omits redundant capture groups when all added tokens are
nonempty literals without whitespace stripping. Empty tokens and stripping
retain the original path. Gemma 3 270M's 6,415 added-token alternatives exposed
an expensive capture-group scan: the isolated split fell from 112.42 ms to
0.10 ms with identical segments.

The [whole-request comparison](../benchmark-results/gemma-performance-20260929/stage8-tokenizer-request/analysis-repair.md)
measured **123.34 → 6.78 ms to first visible text (−94.50%)** on the same
35-token prompt and searched Q4/G64 checkpoint. Four alternating process pairs
passed the preregistered drift/order gates; all 24 warmup and 40 measured outputs
matched in prompt identity, content, reasoning, token counts and stop reason.
The paired latency ratio was 18.23× (95% pair-bootstrap interval 17.79–18.80×).
The 32-token decode measurements do not establish a sustained-throughput gain.
Installed Gemma 4 tokenizers have only 24 added tokens, and Laguna has 70;
this latency result cannot be transferred to those models.

The [stage 8 full suite](../benchmark-results/gemma-performance-20260929/root-tests-stage8-tokenizer.log)
passes 551 Swift Testing tests and nine XCTest tests, with two CUDA skips.
It includes the empty-token regression found in stage 7, plus mixed empty and
nonempty tokens, stripping, overlapping literals and Unicode cases. Stage 7's
failed tests and unexecuted benchmark plan remain preserved. The completed
stage 8 benchmark's original analysis failed on Python tuple/list equality after
JSON reload; a separate analysis-only repair verified unchanged raw reports and
normalized those containers without changing metric gates or measurements.

## Selective precision results

The [270M query/key candidate](GemmaQuantizationPlan.md#gemma-3-270m-querykey-and-tied-head-candidates)
retains a Q4/G64 body and uses Q8 only for query/key projections, adding 4.9%
to tensor storage. On 192 references, NLL falls from 5.000518 to 4.929368
(difference -0.071149, paired 95% interval [-0.082698, -0.059942]). The accepted
eight-pair resident timing comparison measured 565.61 versus 562.77 tok/s;
its paired change was -0.64%, with an interval from -2.14% to +0.10%.
This is a demonstrated reference-likelihood benefit with no clear throughput
change in that screen, not established answer or trained-adapter accuracy.
The Q8 tied-head alternative is less attractive: 55.6% more storage and worse
pooled reference NLL despite better teacher KL. More bits alone do not select
the better recipe.

The separate [A4B router candidate](GemmaQuantizationPlan.md#recovering-a4b-routers-from-the-original-source)
uses Q8/G64 for all 30 routers, built from recovered original BF16 weights after
exactly reproducing the installed Q4 triplets. Its 1,249 other tensor payloads
remain unchanged. Extra storage is only 5.407 MB (0.0381%). A matched 192-record,
50,669-token reference screen gives NLL 6.755030 versus 6.692167, but the paired
difference interval [-0.140592, +0.019437] spans zero. All three category means
favor Q8; code's uncorrected interval narrowly excludes zero. Overall quality
improvement and equivalence are both unproven. The separate 22-task generated
screen passes 12/22 for Q4 versus 11/22 for Q8. Both retrieve all six answers
correctly; ten versus eleven math responses hit the 256-token cap without the
required final answer. A separately frozen 1,024-token follow-up on all the same
tasks gives 21/22 for both arms, with no truncations; the same `#### 60%` answer
fails the strict numeric parser in both. Neither screen demonstrates higher
answer accuracy for Q8 routers, and the later result does not replace the
earlier cost-budget result or establish equivalence.
The [eight-process timing run](../benchmark-results/gemma-performance-20260929/gemma4-a4b-router-q8-throughput/summary.json)
preserves full identities and exact output within each arm, but fails all timing
acceptance gates. Decode drift is 27.08%/31.24% and the order effect is 15.31%,
above the predefined 10% limits. The diagnostic +1.08% paired change is not an
accepted speed estimate. The installed checkpoint remains unchanged.

The [read-only residency audit](../benchmark-results/gemma-performance-20260929/memory-residency-audit/README.md)
identifies a separate runtime policy difference worth testing. Midnight already
requests a measured wired-memory budget, including its 256 MiB cache allowance
and headroom. The eight timing logs show roughly 16.51 GB budgets and measured
trial active peaks roughly 2.12 GB lower. Local vLLM-Metal retains the full
recommended working-set budget for the worker lifetime; Midnight ends its
ticket after each request, removing allocation membership until the next one.
This is a source-backed hypothesis about residency lifetime, not an identified
cause of throughput drift. Post-run swap/compressor observations also do not
establish paging or thermal causation. A same-budget lifetime experiment must
hold checkpoint, cache allowance and kernel settings fixed before any promotion.

## Measurement policy

Run GPU workloads serially, retain every trial, record the executable and Metal
library identities, and alternate baseline/candidate process order. Keep exact
weights, tokenizer, prompt IDs, context, output limits and quantization fixed.
For kernel comparisons set `MODEL_RUNNER_PREFIX_CACHE_ENTRIES=0` so prefix reuse
does not change the workload. Cache experiments must instead report actual reused
token counts and time to first visible text.

`model-runner-runtime-bench --prompt-cache` now alternates cold/cached continuation
order and clears retention before each probe. Reports retain reasoning as well
as visible content. The cache parity result also requires matching prompt token
fingerprints, output token counts and stop reasons. Changed outputs fail an
exact-preservation claim. A candidate with changed outputs may still offer a
useful quality/speed/memory tradeoff, but needs representative quality evidence
and explicit evaluation of the cost; a speed result alone cannot supply that.

A passing tiny numerical fixture is necessary but insufficient. Record numerical
error, exact-output differences and application quality separately from speed.
Do not extrapolate a short single-chat result to concurrent serving or training.
