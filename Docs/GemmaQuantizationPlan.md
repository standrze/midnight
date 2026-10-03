# Gemma quantization experiments at four bits and above

The current checkpoints already use BF16 activations and BF16 affine scale/bias
metadata. Selective G128 is a modest bandwidth experiment; preserving trained
QAT grids requires a separate importer/format experiment. Neither is a measured
speed or quality improvement by their byte estimates alone. All proposed weight formats retain at least four
bits. Keep Gemma's Q/K normalization parameters and normalization semantics.

## Read-only checkpoint plan

`../wick/Scripts/plan-gemma-quantization.py` reads only safetensors headers. It checks
packed affine shapes, declared bit widths, scale/bias shapes and dtypes, and
reports grouping choices. It rejects below-four-bit declarations, contradictory
quantization aliases, and foreign compressed-tensors metadata. Header hashes do
not identify weight payloads, and this check does not establish finite values,
correct dequantization, native loading or model quality.

```sh
python3 ../wick/Scripts/plan-gemma-quantization.py \
  artifacts/gemma3-270m-it-4bit --output /tmp/gemma3-270m-plan.json
python3 ../wick/Scripts/plan-gemma-quantization.py \
  "$HOME/.midnight/models/gemma-4-31B-it-midnight" --output /tmp/gemma4-31b-plan.json
python3 ../wick/Scripts/plan-gemma-quantization.py \
  "$HOME/.midnight/models/gemma-4-26B-A4B-it-midnight" --output /tmp/gemma4-a4b-plan.json
python3 ../wick/Tests/Python/test_gemma_quantization_plan.py
```

The live verifier's status wording now reflects Wick's implemented opt-in G128
conversion. The prior verifier and tests are preserved by their original SHA256
under `artifacts/gemma-quantization-verifier-revisions/`; historical plans and
reports retain their original identities. This wording revision changes no
geometry or byte calculation.

Output files must be new. Actual plans from September 29 are retained in
`benchmark-results/gemma-performance-20260929/*-quantization-plan.json`.
The calculations count all nonexpert tensors once and weight expert tensors by
top-k/expert-count. They omit KV/activation traffic, caching and launch cost.

| Installed checkpoint | Current tensor bytes | Estimated active bytes/token | G128 saving retaining the G64 tail path | Ideal active-byte speedup ceiling |
| --- | ---: | ---: | ---: | ---: |
| Gemma 3 270M | 150,885,632 | 150,885,632 | 1,105,920 | 1.0074x |
| Gemma 4 31B | 17,269,170,808 | 17,269,170,808 | 313,098,240 | 1.0185x |
| Gemma 4 A4B | 14,194,649,148 | 2,151,180,348 | 12,615,680 | 1.0059x |

The conservative plan changes only `down_proj` and attention `o_proj` for 31B
and 270M; A4B changes only attention `o_proj`. It preserves embeddings/tied heads
and routers. A4B's expert/shared down widths 704/2112 are not divisible by 128.

Changing every compatible nonprotected matrix to G128 would save 3,133,440 bytes
for 270M, 915,210,240 for 31B, and 521,635,840 for A4B (75,581,440 active bytes).
The corresponding ideal ceilings are 1.0212x, 1.0560x and 1.0364x. However, the
new opt-in Q4 tail kernel admits **G64 only**: changing K640/2816/5376 to G128
loses that dispatch. Fewer bytes can therefore make a projection slower. Compare
G64-tail, G128-generic and an eventual G128-tail kernel separately.

## Implemented source-based Gemma conversion

Wick now supports explicit `--gemma-group-policy` and repeatable
`--g128-module` selections for native Gemma 3/4. Its
[generic CLI](../../wick/Sources/ModelQuantizer/main.swift) accepts both ordinary
and searched Q4 with mixed G64/G128 modules, plus the existing explicit Q8/skip
policies. Default conversion behavior is unchanged. The
[policy implementation](../../wick/Sources/QuantizerSupport/GemmaGroupSizePolicy.swift)
rejects prequantized sources, incompatible matrix widths, protected
embedding/head/router selections, conflicting overrides, and below-four-bit
output metadata. Actual weight/scale/bias headers and both config aliases are
verified before the output transaction commits. Editing config alone never
converts weights.

The backend's existing ScaleSearch primitive supported G128; its generic
validation, Linear/SwitchLinear replacement and metadata callers previously
fixed G64. The applied follow-up passes each selected group's geometry through
those callers. Native model sanitization retains source projection conventions,
explicit attention layout, sliding-window metadata, and normalization behavior.
Gemma 3's omitted head, or a stored head whose full floating payload exactly
matches its embedding, is preserved once on disk. Explicitly untied heads remain
independent; a declared tie with different payloads fails. This prevents the
ordinary embedding and separately searched head from breaking an original tie.

The reviewed [patch bundle](../artifacts/facet-gemma-g128-20260929/README.md)
records the exact source snapshot from when Wick was named Facet, plus its patch
and dependency replay. That Facet snapshot was
built and all **10 selected CPU tests actually executed and passed**: eight
policy checks and two MLX conversion tests, including ordinary/searched mixed
geometries, exact unchanged tensors/norms, tied-head native reload and gathered
expert G128 parity. The
[test log](../benchmark-results/gemma-performance-20260929/facet-gemma-g128-tests.log)
contains each test case; this is not a zero-test success. Source/geometry tests
are separate from whole-model quality and throughput measurements.

The unquantized small source is now available at
`artifacts/gemma3-270m-it-bf16`, pinned to
`mlx-community/gemma-3-270m-it-bf16` revision
`c806ef3a4ed971bd75aaee3346e0fef808512f03`. The
[fixture provenance](../benchmark-results/gemma-performance-20260929/gemma3-bf16-fixture-provenance.json)
contains full hashes for every downloaded file. It contains 871,740,672 tensor
bytes, including an explicit BF16 head that is byte-identical to the embedding;
[the direct payload comparison](../benchmark-results/gemma-performance-20260929/gemma3-bf16-tied-head.json)
records the matching hashes. The source configuration and weights remain intact.
The original 31B/A4B BF16 paths recorded in installed provenance are still absent.

Run the policy on both controls and candidates from that **same unquantized
source**. For 270M and 31B, select `down_proj` and attention `o_proj`; for A4B,
start with attention `o_proj` only because its down widths cannot use G128.
The explicit four-arm commands are:

```sh
# GEMMA_BF16 is the verified unquantized source; CANDIDATE_ROOT is a new directory.
../wick/.build/release/wick "$GEMMA_BF16" "$CANDIDATE_ROOT/ordinary-g64" \
  --gemma-group-policy --standard-q4 --bounded-memory
../wick/.build/release/wick "$GEMMA_BF16" "$CANDIDATE_ROOT/searched-g64" \
  --gemma-group-policy --bounded-memory
../wick/.build/release/wick "$GEMMA_BF16" "$CANDIDATE_ROOT/ordinary-selective-g128" \
  --gemma-group-policy --standard-q4 --bounded-memory \
  --g128-module '*.mlp.down_proj' --g128-module '*.self_attn.o_proj'
../wick/.build/release/wick "$GEMMA_BF16" "$CANDIDATE_ROOT/searched-selective-g128" \
  --gemma-group-policy --bounded-memory \
  --g128-module '*.mlp.down_proj' --g128-module '*.self_attn.o_proj'
```

Add `--dry-run` to inspect each resolved policy without conversion. The
[270M four-arm runner](../benchmark-results/gemma-performance-20260929/convert_gemma270m_arms.py)
first verifies full source hashes, snapshots the converter and matching Metal
library, then performs CPU dry runs. Its separate `--run` operation serializes
conversion into new destinations and verifies module geometry, source norms,
unchanged matrices and final file identities. No installed Q4 checkpoint is
requantized. Conversion completion alone is not a promotion or quality result.

All four 270M arms have now converted successfully on Metal into
[`artifacts/gemma3-270m-four-arm-20260929-r2/models`](../artifacts/gemma3-270m-four-arm-20260929-r2/models).
The [conversion summary](../artifacts/gemma3-270m-four-arm-20260929-r2/conversion-summary.json)
and [plan with full identities](../artifacts/gemma3-270m-four-arm-20260929-r2/plan.json)
record the results. Both G64 arms contain 150,885,632 tensor bytes; both selective
G128 arms contain 149,779,712 bytes, the predicted 1,105,920-byte reduction.
Exactly 36 of 127 packed modules change groups. Original floating tensors/norms
remain exact, no duplicate tied head is stored, and all unselected tensors are
bit-identical within the ordinary and searched G64/G128 pairs. Source and
converter hashes remained unchanged.

The four conversion commands took 0.625, 3.862, 0.630 and 2.501 seconds in the
listed order. These are conversion times, **not inference-speed results**.
Different quantizers may legitimately produce different text. Evaluate
quality, inference speed and memory separately; the user's preference allows a
modest slowdown when quality improves.

For A4B, inspect `--q8-module '*.router.proj' --dry-run`, then apply the identical
Q8 router policy to both candidate arms if testing router protection. The
installed A4B checkpoint has Q4 routers; Gemma's opt-in group policy does not
automatically change their precision. Router Q8 is a separate quality
intervention with a small byte increase, not a speed optimization. Existing
activation statistics/rescoring and the bounded GPTQ probe in Wick remain
Laguna/Mistral paths; activation-calibrated Gemma fitting is not implemented.

## Higher-precision quality candidates

### Recovering A4B routers from the original source

The installed A4B conversion manifest names Google revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`. Its original BF16 router tensors
have now been recovered with bounded HTTP ranges, without downloading the
complete 51.6 GB source checkpoint. The
[recovery manifest](../artifacts/gemma4-a4b-router-recovery-20260929/acquisition-1/recovery.json)
records all 30 `[128, 2816]` matrices and their companion scaling vectors.
The source configuration agrees with the installed configuration after removing
quantization metadata. All companion vectors match exactly; total transfers
including the earlier metadata inspection were 22,292,550 bytes.

The [native reproduction check](../artifacts/gemma4-a4b-router-recovery-20260929/proof-1.json)
passes **all 30 routers**: applying the recorded searched Q4/G64 conversion to
the recovered BF16 matrices reproduces every installed packed code, BF16 scale
and BF16 bias byte. The original manifest did not retain source payload or
converter-build hashes, so this finite reconstruction check is recorded
explicitly. Full remote-shard hashes remain unverified by these partial reads.

Only after that check passed, ordinary Q8/G64 router triplets were produced
directly from the recovered BF16 tensors and verified after saving/reloading.
They contain 11,489,280 payload bytes, an increase of 5,406,720 bytes over the
installed routers. The [standalone candidate](../artifacts/gemma4-a4b-router-q8-20260929/model)
has now been assembled and [validated](../artifacts/gemma4-a4b-router-q8-20260929/validation.json).
Exactly 90 router arrays changed; the other 1,249 tensor payloads and copied
sidecars remain exact. Both configuration aliases contain the same 30 Q8/G64
module overrides. The candidate holds 14,200,055,868 tensor bytes, just 0.0381%
more than the installed model. Full installed shard hashes agree before and
after assembly. Installed model files remain intact.

The [matched reference screen](../benchmark-results/gemma-performance-20260929/gemma4-a4b-router-q8-quality/analysis.json)
completed both arms on the same preserved stage 8 runtime, with every model and
input identity unchanged. It scored 192 references and 50,669 tokens at maximum
512 tokens per sample and prefill step 128. Pooled NLL fell from 6.755030 to
6.692167, a candidate-minus-baseline difference of -0.062863; the paired 95%
record-bootstrap interval is **[-0.140592, +0.019437]**. This is inconclusive
overall, not an established quality improvement or equivalence.

| Reference category | NLL difference | Paired 95% interval |
| --- | ---: | --- |
| Code | -0.165334 | [-0.319016, -0.001093] |
| Math | -0.040010 | [-0.112336, +0.029032] |
| Prose | -0.050279 | [-0.167131, +0.071251] |

All category means favor Q8 routers, but only code's uncorrected interval
excludes zero, narrowly. No multiple-comparison correction or independent blind
holdout is claimed. Reference likelihood does not measure generated-answer
accuracy; native scoring times do not measure generation throughput.

The subsequent [22-task answer screen](../benchmark-results/gemma-performance-20260929/gemma4-a4b-router-q8-generation/scores.json)
completed both arms with identical prompt fingerprints and scoring settings.
Q4 routers passed 12/22 tasks; Q8 routers passed 11/22. Both passed all six
retrieval questions. Math scores were 6/16 versus 5/16; all ten and eleven
respective failures reached the fixed 256-token output cap without the required
final-number marker. There were no refusal cues. The paired overall accuracy
difference is -4.55 percentage points, interval [-13.64, 0.00]; the only
discordant task favors Q4. This capped convenience screen does not establish
higher answer accuracy, broader inferiority, or equivalence. The retrieval
ceiling and math truncation limit its usefulness for ranking quantizers. Its
corpus and token budget were not changed after observing the results.
The router precision default is unchanged.

A separate [1,024-token follow-up](../benchmark-results/gemma-performance-20260929/gemma4-a4b-router-q8-generation-1024/scores.json)
reran **all 22 unchanged tasks for both arms**, with a protocol frozen after
observing the earlier truncations and before collecting these new outputs.
Both score 21/22: 15/16 math and 6/6 retrieval, with no output-limit stops.
Their sole strict-scoring failure is identical: `gsm8k-test-14` ends in
`#### 60%`, which the frozen numeric parser rejects. This is a format failure,
not evidence of an incorrect percentage calculation. Baseline and candidate
generate 5,467 and 5,489 total tokens, respectively. These matched outcomes and
the degenerate [0, 0] paired-task interval do not establish general equivalence.
The follow-up supplies no higher-accuracy evidence for Q8 routers and does not
replace the separate 256-token cost-budget result.

The [four-pair natural-generation timing comparison](../benchmark-results/gemma-performance-20260929/gemma4-a4b-router-q8-throughput/summary.json)
completed all eight fresh processes, each with three warmups and three measured
512-token outputs. Full identities and within-arm output repeatability pass;
cross-quantizer output differs, as allowed. All timing metrics fail the
preregistered consistency gates. Decode has 27.08% baseline drift, 31.24%
candidate drift and a 15.31% order effect, against a 10% threshold. Its
diagnostic paired change is +1.08%, interval [-9.98%, +12.79%], but there is
**no accepted throughput estimate**. These measurements establish neither a
speed gain nor a slowdown for router Q8, and no adverse pair was discarded.

### Gemma 3 270M query/key and tied-head candidates

Two additional 270M recipes preserve the matched searched Q4/G64 body except
for explicit Q8 overrides, using the same pinned original BF16 source:

| Candidate | Q8 modules | Tensor bytes | Increase versus searched Q4/G64 |
| --- | --- | ---: | ---: |
| Tied embedding/output head | One stored embedding, reused by native head | 234,771,712 | 83,886,080 (+55.6%) |
| Query/key projections | All 18 layers' `q_proj` and `k_proj` | 158,258,432 | 7,372,800 (+4.9%) |

These header-derived estimates double selected packed-code bytes while retaining
G64 and the BF16 scale/bias dimensions. They are quality experiments; no measured
sensitivity or quality improvement follows from choosing these modules. The head
candidate has a substantial memory/bandwidth cost even on this small model.

The [applied Q8 tied-head fix](../artifacts/facet-gemma-q8-tied-head-20260929/README.md)
adds an `lm_head` Q8/G64 metadata alias in **both** quantization dictionaries
when the stored Q8 embedding is tied. Native Gemma 3 otherwise recreates its head
from Q8 tensors but inherits the default Q4 metadata. The fix stores the embedding
triplet once, rejects conflicting geometry, records the alias in provenance and
leaves Q4/untied configuration bytes unchanged. All **13 selected CPU tests**
passed, including native Q8 conversion/reload, matching head/embedding projection
values, exact unchanged body tensors/norms and finite forward logits.

The [two-arm runner](../benchmark-results/gemma-performance-20260929/convert_gemma270m_q8_arms.py)
verifies source/control payload identities, snapshots the tested converter, and
separates CPU dry runs from serial GPU conversion. Its
[conversion plan](../artifacts/gemma3-270m-q8-arms-20260929/plan.json)
records the exact commands and required geometry/payload checks. The former
[blocked head-only plan](../artifacts/gemma3-270m-q8-head-plan-20260929/plan.json)
is retained as historical evidence and must not be run with its older converter.
Both candidates have now converted successfully. The
[conversion summary](../artifacts/gemma3-270m-q8-arms-20260929/conversion-summary.json)
confirms the exact tensor-byte estimates above, one stored tied head/embedding,
correct per-module bits/groups, original norms and all unselected payloads
identical to the searched Q4/G64 control. Source/converter hashes stayed fixed.
Conversion command times were 3.823 seconds for the Q8 head and 2.086 seconds
for Q8 query/key; these are not inference-speed measurements.

The subsequent [Q8 quality screen](../benchmark-results/gemma-performance-20260929/gemma270-q8-quality/README.md)
used preserved stage3 binaries and matching flags, with full source/checkpoint,
corpus and runtime identity checks. Native NLL covered 192 records and 50,801
scored tokens, maximum 512 tokens per record and prefill chunks of 128. The
searched Q4/G64 control NLL was 5.000518.

| Candidate versus searched Q4/G64 | Full192 NLL delta | Paired 95% interval | Prose delta | Code delta | Math delta |
| --- | ---: | --- | ---: | ---: | ---: |
| Q8 tied embedding/head | +0.023816 | [+0.005179, +0.042606] | -0.041925 | +0.178728 | +0.114391 |
| Q8 query/key projections | -0.071149 | [-0.082698, -0.059942] | -0.088809 | -0.112699 | -0.000436 |

Negative NLL deltas favor the candidate on these references. Q8 query/key's
math interval spans zero; no clear math change or equivalence is established.
Its prose and code intervals favor the candidate. Excluding the 24 inspected
pilot IDs leaves 168 records and the same pattern: Q8 query/key pooled delta
-0.072372 [-0.085195, -0.059754], versus Q8 head +0.027065
[+0.007061, +0.047500]. This post-pilot subset shares the original sources and
is not an independent blind holdout.

The full-vocabulary 24-record teacher pilot (2,550 scored positions, maximum
128 tokens, reduction chunks of 16) reduced KL from 0.973853 for the control to
0.815202 for Q8 head and 0.858193 for Q8 query/key. Teacher/control reruns matched
prior aggregate NLL/KL exactly. The head's closer teacher distribution still
comes with worse code/math and pooled reference NLL, plus 55.6% more tensor
storage. Q8 query/key is the stronger measured quality candidate here at +4.9%
bytes. These are 270M reference-likelihood/fidelity results, not evidence about
31B/A4B quality, generated-task accuracy, or inference speed.

The subsequent [stage6 resident comparison](../benchmark-results/gemma-performance-20260929/paired-resident/stage6-searched-q8-qk/README.md)
passed its A/A control and eight-pair drift/order gates. With both 270M models
resident, searched Q4/G64 measured 565.61 tok/s versus 562.77 for Q8 query/key;
the paired change was -0.64% (95% pair-bootstrap interval [-2.14%, +0.10%]),
showing no clear decode difference. On the 35-token prompt, paired prefill was
+1.16% and first-content latency -0.33%; these small differences do not establish
broad prefill improvements. All 512-token trials were retained, each arm's text
was exactly repeatable, and different quantizer text was allowed. The roughly
315.5 MiB process peaks include both resident models and do not show per-model
memory savings; Q8 query/key still costs +7.03 MiB/+4.9% tensor storage. Combined
with its earlier reference-NLL gain, this is a promising 270M quality/storage/speed
tradeoff, not a demonstrated generated-task gain or larger-Gemma result. The quality
and timing screens used separate preserved stage3/stage6 builds. No default
checkpoint or conversion policy is changed by these screens.

For an explicit 270M quality/storage tradeoff, the measured query/key recipe is
available at
`artifacts/gemma3-270m-q8-arms-20260929/models/searched-g64-q8-query-key`.
To reproduce it from the same pinned BF16 source into a new destination:

```sh
../wick/.build/release/wick artifacts/gemma3-270m-it-bf16 NEW_DESTINATION \
  --gemma-group-policy --bounded-memory \
  --q8-module '*.self_attn.q_proj' --q8-module '*.self_attn.k_proj'
```

This keeps the remaining quantized projections at Q4/G64. It also passes the
native LoRA save/reload check and the existing Studio worker's two-step Adam
smoke test. Reference NLL is the demonstrated quality benefit; generated-task
accuracy and trained-adapter quality still need representative task evaluation.

## Preserve trained QAT grids instead of assuming interchangeability

Google publishes QAT unquantized and GGUF Q4_0 Gemma 4 31B/A4B variants. Its model
card lists compressed-tensors W4A16 for the dense family including 31B, without
listing A4B. It also requires a corresponding QAT assistant when using a QAT
target. This is a checkpoint-basis change relative to the current locally
ScaleSearch-converted original checkpoints. A matching architecture alone is
insufficient evidence of assistant compatibility.
[Google's QAT model card](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized/blob/main/README.md)

The official 31B compressed-tensors config declares symmetric integer Q4,
group size 32, no activation quantization, and `pack-quantized` storage. It leaves
the head and vision linears outside that quantization group. Those config keys
and packed tensor names/layout are not native MLX affine storage. The config's
`scale_dtype: null` does not establish the stored scale tensor dtype; inspect
headers during an eventual import.
[Official W4A16 configuration](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct/blob/main/config.json)

For Q4_0, each group of 32 uses one half-precision scale and packed codes with an
implicit offset of eight. A prospective native Q4_0 kernel can retain that
single-scale storage: 4.5 bits/weight, equal to current affine Q4/G64 with two
16-bit metadata values. Repacking it into ordinary affine Q4/G32 with explicit
scale **and** bias instead costs 5 bits/weight, 11.1% more for affected tensors.
The algebraic mapping is `affine_scale = d`, `affine_bias = -8*d`; preserve scale
bits and verify all decoded values against the source format before claiming a
grid-preserving import. A G64/G128 refit merges trained G32 grids and is a new
quantizer experiment. The pinned runtime has no native Q4_0 quantization mode.
[Q4_0 reference implementation](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c)

A bounded [test-only prototype](../artifacts/gemma-q4zero-prototype-20260929/README.md)
now preserves this exact source grid. Its Foundation row reader/repacker accepts
only confirmed rank-2 GGUF Q4_0, G32, original F16 scales and the implicit offset
of eight. It rejects incompatible metadata, nonfinite scales, invalid geometry,
overflow, truncation and over-budget reads. The pinned MLX C++ GGUF reader already
knows Q4_0 but expands it to ordinary affine scale/bias storage; Swift's current
array-loading boundary does not expose GGUF. This experiment does not modify
either loader or add a global MLX quantization mode.

The official fixture is pinned to Google 31B QAT GGUF revision
`59dde24573e7e61570dba08b18a2e1fe246955ed`. The first metadata read stopped at its
4 MiB cap; a separately authorized continuation reused those verified ranges
with a 32 MiB cumulative cap. Acquisition stopped after **15,825,408 total bytes**
of metadata/ranges, including a **96,768-byte** row slice from
`blk.0.attn_q.weight`: rows `1..<33` of GGUF dimensions `[5376,8192]`, yielding a
native `[32,5376]` matrix. Every request required HTTP 206, exact Content-Range
and exact body length. No full checkpoint was downloaded; the published full-file
hash is recorded as advertised, not verified by partial reads. The
[range provenance](../artifacts/gemma-q4zero-prototype-20260929/fixture-continuation-2/provenance.json)
records the source and every downloaded range hash.

The [official CPU validation](../artifacts/gemma-q4zero-prototype-20260929/official-native-grid/native-grid.json)
proves all **172,032 decoded FP32 weight bit patterns** and a complete source-byte
round trip match exactly. All 5,376 F16 scales survive unchanged, including 1,729
negative scales. Test-only native payloads use U32 `[32,672]` (86,016 bytes) and
F16 `[32,168]` (10,752 bytes), with no stored bias. Six Foundation checks also
passed, including every finite F16 bit pattern and all 16 codes, signed zero and
subnormals; eight bounded-reader/parser checks and Swift formatting passed.

Three [frozen test files](../artifacts/gemma-q4zero-prototype-20260929/swift-source-freeze.json)
are integrated under `Tests/ModelRunnerProtocolTests`. Their opt-in MLXFast
fixture uses BF16 activations/output, untouched F16 scales and local FP32
accumulation without whole-tensor promotion. **Metal fixture validation passed**
on all five synthetic shapes and the official `[32,5376]` slice. The official
maximum BF16 output error against a CPU FP64 dot product was `3.0512914e-5`;
the original F16 scale bits remained unchanged. The first GPU run exposed an
explicit-output-cast compilation error; the original frozen source and failed
log are preserved, and the [one-line cast revision](../artifacts/gemma-q4zero-prototype-20260929/revisions/explicit-bf16-output/manifest.json)
records the tested fix. The [repaired test log](../benchmark-results/gemma-performance-20260929/root-tests-stage6-q4-output-repair.log)
contains the fixture outputs. This is an anonymous row-slice import and reference QMV fixture, not
a complete Gemma importer or a tuned kernel. It does not implement projection
axis/name mapping, unquantized head/norm loading, tied storage, MoE gathering or
prefill. Full-model quality, task accuracy, memory and speed remain unmeasured;
preserving QAT grid values alone establishes none of those outcomes.

## Dtype alignment and quality screens

All 127 packed modules in 270M, 411 in 31B and 326 in A4B have BF16/BF16 metadata;
the planner reports no mismatches. This is already the desired alignment.
Pinned MLX `ops.cpp` promotes affine activations and scale/bias tensors to one
common dtype in dense and gathered multiplication. Mixing BF16 activations with
FP16 metadata promotes to FP32; it also misses the new half/BF16 tail gate.
An importer must not silently fix this by casting learned metadata after fitting:
that changes the stored grid. Either fit/validate BF16 metadata or implement an
explicit mixed-metadata kernel without whole-tensor promotion. The latter is
especially relevant to Q4_0's FP16 scale storage. See the existing measured
[dtype analysis](quantization-challengers-20260904.md#autoround-feasibility).

Use the same tokenizer, chat template, norm tensors, source revision and
per-module policy when comparing quantizers. Record actual packed-grid weighted
projection error on disjoint calibration/development inputs, not only weight
MSE. For A4B, record routed expert coverage and router top-k agreement; sparse
experts without sufficient observations retain the baseline. Select a recipe
on development data, then report held-out reference NLL by prose/code/math,
full-vocabulary teacher KL/top-1 agreement, generated task outcomes and long
generation repetition separately. An improved fitting objective is not evidence
of improved task accuracy.

The existing Midnight tools provide the following screens after candidates and
corpora exist. A held-out likelihood corpus contains `id`, `category`, `text`;
public generation inputs contain `id`, `category`, `prompt` and no answers/tests.
The first command fits all three model sizes; full BF16 teacher runs additionally
need a measured memory plan, especially for 31B on a 64-GiB machine. `--cpu`
does not remove that unified-memory requirement.

The teacher-KL tool requires an unquantized BF16 teacher by default. If that
teacher cannot fit, `--allow-quantized-reference` explicitly permits a quantized
reference whose config still declares BF16. Such reports set
`reference_kind: quantized_reference_provisional` and
`represents_bf16_teacher: false`. They measure deviation from that quantized
reference, not from BF16; retain its exact precision policy and checkpoint
identity. This option does not establish that a full Q8 reference fits, or make
reference fidelity a substitute for generated coding/security correctness.

```sh
.build/release/model-runner-quality-bench "$GEMMA_CANDIDATE" "$HELDOUT_JSONL" \
  "$NEW_REPORT_DIR/reference-nll.json" --max-tokens-per-sample 2048 --prefill-step-size 512
.build/release/model-runner-teacher-kl-bench "$GEMMA_BF16" "$HELDOUT_JSONL" \
  "$NEW_REPORT_DIR/teacher-kl.json" --student "$GEMMA_CONTROL" --student "$GEMMA_CANDIDATE" \
  --max-tokens-per-sample 512 --position-chunk-size 16
.build/release/model-runner-generation-bench "$GEMMA_CANDIDATE" "$PUBLIC_PROMPTS_JSONL" \
  "$NEW_REPORT_DIR/generations.json" --engine metal --tokens 512 \
  --context-length 8192 --prefill-step-size 512 --kv-compression none
```

Keep fit/dev/test records disjoint and retain per-record differences rather than
only an aggregate. Compare throughput in alternating fresh processes and on a
matched fixed token prefix, then natural generation separately. Quantizer changes
can alter EOS, routing and future work. No candidate is selected by these plans.
