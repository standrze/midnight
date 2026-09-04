# MLX quantization and kernel research — 2026-09-04

**Recommendation:** retain affine Q4/G64 with Q8 routers as Laguna's production baseline. The completed Laguna reference evaluation favors AWSS overall and on code, while Standard remains best on math; keep both as workload-dependent candidates. LS2's lower weight reconstruction error did not yield better aggregate reference NLL. Prioritize a stronger same-format quantizer and a separately measured Metal kernel program. A custom format becomes worthwhile only when it improves the measured quality–memory–latency tradeoff enough to justify new execution paths.

Research inspected Midnight at `f214f167`, its MLX core at `1f8e74e3f12f31365464a6867c6579f0e9b29d85`, and a fresh official MLX main clone at `b6368984b8e02a3fb3ee7986846c0fb85e1fccf7` (September 3). Upstream main identifies itself as 0.32.3 development; the release page's latest published version was 0.32.2. Also inspected oMLX source at `e467261edc786efd33b1e9023d5c4a827f8aa1c1`. These identifiers describe the initial source survey. Subsequent implementation and measured experiments are summarized below and retained in the September 4 archive.

**Three independent decisions explain the choices.** Q4R8 is this project's mixed-precision policy: ordinary affine Q4 weights, Q8 routers, and calibrated optional Q8 promotions. ScaleSearch decides how to create the codes, scales and biases offline. MLX's kernels decide how those arrays execute. LS2 and AWSS keep the existing packed representation, so a faster affine kernel would benefit both as well as standard affine checkpoints. Better weight fitting alone does not reduce the work in a matrix multiplication. Generated token and expert trajectories can nevertheless change observed end-to-end speed.

The project's AffineScaleSearch adapts the scale-search idea to integer affine Q4. It is not the NVFP4 algorithm or attention implementation in the [ScaleSearch paper](https://arxiv.org/abs/2605.12464), and the paper's reported gains cannot be assigned to this implementation. The center, bias and joint-fit patches are components of LS2, not three additional runtime formats. See [the implemented algorithm](scaleplan-q4r8.md).

**The initial BF16-teacher quality evidence.** Ministral 3 14B was evaluated on 30,118 held-out WikiText-2 tokens, with the same BF16 teacher and token IDs:

| Quantizer | Perplexity ↓ | Teacher KL ↓ | Teacher top-1 agreement ↑ |
|---|---:|---:|---:|
| Standard affine Q4/G64 | 8.523476 | 0.033313 | 90.27% |
| ScaleSearch LS2 | 8.473043 | 0.032580 | 90.52% |
| Activation-weighted ScaleSearch (AWSS) | 8.377132 | 0.017672 | 92.90% |

AWSS cuts KL 45.76% relative to LS2 on that corpus, with unchanged 7,597,762,560 indexed tensor bytes. Its raw weight MSE actually worsens 0.86% while teacher agreement improves: raw reconstruction error alone is not a sufficient model-quality objective. However, the separate 698-token authored smoke corpus worsens from NLL 2.488547 to 2.514955. The runtime campaign's drift overwhelms its apparent small timing difference. These results justify further evaluation, not an across-the-board AWSS promotion. [Retained measurements](../benchmark-results/ministral3-awss-20260830/README.md)

Laguna now has routed-expert calibration, development coverage guards, complete LS2/AWSS conversions, and a matched 192-reference NLL evaluation. AWSS has the lowest aggregate NLL, while Standard has the lowest math NLL; broad BF16-teacher KL and generated-task validation remain missing. The collector measures actual selected-expert inputs and post-SwiGLU down-projection inputs, and under-covered experts retain LS2 weights. Q8 routers and compatible gate/up layouts are preserved. [Laguna evidence](laguna-metal-q4r8.md), [current AWSS objective](../Sources/MistralActivationScaleSearchCore/ActivationWeightedScaleSearch.swift)

**External candidates worth pursuing, in order.**

| Candidate | Value to this project | Runtime consequence |
|---|---|---|
| AutoRound native MLX export | A readily available calibrated challenger that optimizes rounding; compare at Q4/G64 and the same router policy | Experimental exporter supports native MLX and mixed layer settings. Validate Swift loading/fusion; embedding quantization is unsupported. |
| GPTQ with static groups | Uses activation correlations and compensates errors between weight columns; complements AWSS's diagonal objective | Can target existing affine storage. Avoid dynamic group maps or permutations requiring extra inference work. |
| oMLX oQe | Close practical comparator for importance-weighted affine fitting, with expert coverage accounting and broader bit/group choices | Source deliberately emits normal MLX affine arrays. Equalize actual bytes and layer policy; an “oQ4” label is not necessarily identical to Q4R8. |
| MLX LM DWQ | Fine-tunes nonquantized parameters, including scales and biases, against a teacher | Can retain the packed layout; needs training resources and held-out quality checks. |
| AWQ / HQQ | AWQ rescales important channels; HQQ provides a calibration-free optimization baseline | Some AWQ transforms fold into adjacent weights; others add operations. HQQ rowwise groups can map algebraically to affine, with stored-dtype rounding checked. |
| MXFP4 / NVFP4 | Different grids and metadata budgets already supported in MLX | Existing kernels make them testable now; current local primitive results do not establish an advantage. |
| Rotations, QuIP#, AQLM | Potentially better quality at lower bit rates | Full rotation/codebook schemes can require online transforms, new loaders and efficient Metal kernels. Higher-effort research after native affine comparisons. |

Primary sources: [AutoRound export support](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md), [GPTQ](https://github.com/IST-DASLab/gptq), [oQe implementation](https://github.com/jundot/omlx/blob/e467261edc786efd33b1e9023d5c4a827f8aa1c1/omlx/oq.py#L5499), [MLX LM learned quantization](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md), [AWQ](https://github.com/mit-han-lab/llm-awq), [HQQ](https://github.com/dropbox/hqq), [MLX formats](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html), [SpinQuant](https://github.com/facebookresearch/SpinQuant), [QuIP#](https://github.com/Cornell-RelaxML/quip-sharp), [AQLM](https://github.com/Vahe1994/AQLM).

The oQ documentation mixes older GPTQ/oQ+ descriptions with its newer source. The inspected `oq.py` implements imatrix-weighted clipping in `_weighted_affine_quantize`, and supports per-expert importance/coverage. Benchmark that exact implementation rather than assuming every “enhanced” name denotes the same algorithm.

**Compare effective storage, not format names.** For quantized tensors with 16-bit affine scales and biases, Q4/G64 is 4.5 bits/weight; Q4/G128 is 4.25 and Q4/G32 is 5.0. MXFP4/G32 is 4.25; NVFP4/G16 is 4.5 before additional global metadata. Thus G128 saves about 5.6% against affine G64 for affected tensor bytes, while using coarser groups. Include routers, unquantized tensors, metadata and padding in complete-model totals. For MoE decode also measure active bytes/token: improving checkpoint size can have little effect if it shrinks experts seldom selected at runtime.

**Yes, customizing MLX can make a difference.** Target a measured format, device and operation shape. The most useful tracks are:

1. **Batch-one MoE decode:** selected-expert gather-QMV, fused gate/up, activation and weighted expert reduction, launch overhead and memory access. Preserve the existing fused paths. Gate/up is already combined in one matrix multiplication; the additional experiment is fusing its SiLU/product epilogue, or the down-projection expert reduction, to avoid intermediate traffic. Dense `qmv_wide` does not automatically apply to gathered MoE calls. A local R2/R4/R8 QMV experiment already found stock R4 best: scoped R2 was 0.68% slower and R8 regressed. Repeating tile changes without a new profile is low priority. [Experiment](../benchmark-results/q4r8-qmv-specialization-20260830/README.md)
2. **Prefill and larger batches:** affine QMM/gather-QMM and attention on M5 Neural Accelerators. Current MLX already dispatches eligible QMM shapes to NAX; merely enabling an “M5 mode” is not an unexplored optimization. The inspected affine NAX kernel explicitly dequantizes tiles into threadgroup memory. A prototype that removes that staging is a specific hypothesis. [Inspected kernel](https://github.com/ml-explore/mlx/blob/b6368984b8e02a3fb3ee7986846c0fb85e1fccf7/mlx/backend/metal/kernels/quantized_nax.h#L485)
3. **Small verification batches:** profile QMV/QMM/split-K crossover at the model's real shapes. Upstream issue #4198 documents a dispatch discontinuity on M5; it is a lead to reproduce, not a guaranteed speedup for Laguna. Correctness and accepted tokens/second remain the deciding metrics for speculative decoding. [Upstream report](https://github.com/ml-explore/mlx/issues/4198)

Apple's WWDC26 guidance introduces additional TensorOps quantized types and scale planes in macOS 27, plus cooperative-tensor inputs that can avoid threadgroup staging. Integer 4/8 support also exists in an update to macOS 26. This motivates two prototypes: preserve affine scale/bias semantics with custom dequantization, and compare a native MX format. Native support does not prove that affine's arbitrary floating-point scale plus bias maps directly to the new scale plane. The current machine runs macOS 26.6.2, so gate newer APIs by SDK/OS availability. [Apple TensorOps guidance](https://developer.apple.com/videos/play/wwdc2026/330/)

The same favorite checkpoint can use different kernels for decode and prefill; it need not use one universal tile. Start with a narrow MLX extension or opt-in patch, retain the upstream fallback, and carry an internal prepacked view only if its load time and extra resident memory are included. [MLX custom kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)

Expected benefit follows the measured bottleneck. If an operation accounts for half the request time, making it twice as fast improves overall speed 33%; if it accounts for 10%, the same kernel improvement yields only 5.3%. These are illustrative Amdahl calculations, not forecasts. Faster arithmetic cannot eliminate the bandwidth needed to read the model and KV state.

**A concrete upstream correctness item preceded large-batch experiments.** The initial pin lacked [d73eb752 — sorted gather-QMM NAX row overflow fix](https://github.com/ml-explore/mlx/commit/d73eb752ef2e6288fd95b032c0bff0a15a4a9e93). The older kernel narrows remaining rows to signed 16-bit before clamping. Here rows mean flattened token×expert assignments, not the context limit. Laguna's default 512-token/top-8 chunk has at most 4,096 assignments; an unaligned 4,097-token/top-8 chunk has 32,776 and can reach the problematic range. The fix has since been backported with source/JIT consistency checks and a regression test; the dependency commit remains pinned.

**Recommended experiment sequence and promotion gates.**

1. **Completed for this campaign:** backport the sorted-gather NAX correctness fix and retain checkpoint, tokenizer, converter, binary, Metal-library, patch and source provenance. Complete SHA256 hashes cover all four weight shards of each Laguna candidate. Repeat these captures for every future release candidate.
2. **Initial comparisons complete; challengers pending:** Standard, LS2 and AWSS now have Ministral teacher/reference evidence and a matched Laguna reference campaign. Add AutoRound, static-group GPTQ and oQe at matching G64 geometry and router policy. Keep calibration, development and test sets separate, and expand beyond prose/code/math reference likelihood to generated tools, retrieval and task scores.
3. **Laguna AWSS implemented and evaluated:** routed calibration, development guards, coverage fallbacks and full checkpoint conversion are complete. AWSS leads the current aggregate/code reference scores; Standard remains best on math. Next define domain weights and generated-task tolerances. Replace the illustrative ScalePlan ledger with measured quality, bytes and latency; its example costs remain fabricated and are not a deployable optimization plan.
4. **First kernel and prefill experiments complete; no speed promotion:** the fused gathered gate/up-SiLU prototype passes numerical checks but shows no reliable same-loaded model decode gain. An initial 2048-token prefill result did not survive the confirmation campaign's drift checks, and generated outputs changed. Keep the fused path off and prefill at 512 by default. Next profile actual dense/gather shapes, layer time, occupancy, memory traffic and launch gaps before choosing another prototype, using identical token prefixes as well as natural generation.
5. **Pending format comparisons:** test G128/selective group sizes and MXFP4/NVFP4 at matched quality and actual bytes. Reserve rotations/codebooks or activation quantization for a demonstrated gap that simpler methods cannot close.
6. **Validation complete; performance promotion gate unmet:** beta.2 passes the release build, 166 Swift Testing plus 2 XCTest tests, 49 Python tests, 29 shell scripts and functional generation on both new Laguna checkpoints. These checks establish the exercised behavior, not a speed win. Future promotion requires repeated end-to-end decode and prefill/TTFT gains across 128/4K/16K/32K contexts, peak memory, energy/token and sustained thermals. Report p50/p95 and all failures. A proposed threshold is a repeatable ≥5% target-workload improvement with predefined quality and other-workload tolerances; it remains a project decision, not a measured result.

Weight quantization and the runner's new KV compression are independent. Evaluate their combination separately: their errors and latency costs can interact. A quality improvement at unchanged weight size, a capacity improvement at unchanged quality, and a faster kernel are all useful results, but they answer different questions.

## Follow-up experiments

Implementation and measurements following this source review are retained in
[the September 4 experiment archive](../benchmark-results/quantization-20260904/README.md).
The new 128-sample code/math reference evaluation confirms that teacher fidelity
and reference-answer likelihood can rank quantizers differently. AWSS cuts
teacher KL relative to LS2 by 48.69%, but increases reference NLL by 0.014165.
Neither quantity is a generated coding or math task score. LS2 and AWSS therefore
remain separate candidates instead of selecting AWSS solely from KL.

The sorted-gather NAX fix has been backported with source/JIT consistency checks.
A fused gathered affine-Q4 gate/up-SiLU prototype matched stock BF16 outputs
exactly on eight synthetic fixtures in two runs. Queued primitive timing improved,
while synchronized timing changed little. The subsequent same-loaded model
comparison found no reliable decode gain; fused gather/SiLU remains opt-in. The [campaign harness](benchmarking.md) retains failures, exact
input fingerprints, paired uncertainty, and separate drift checks for decode,
prefill, and time to first token. [Laguna calibration](laguna-activation-calibration.md)
now has an explicit layerwise path and expert-coverage fallbacks.

Layerwise BF16 calibration has now completed real 40-layer runs. The first
65,536-token run reported 3.952 GB in MLX allocation accounting while the process
peaked at 28.442 GB RSS and 70.151 GB footprint. Bounded Foundation autorelease
lifetimes were then added; a 15,997-token development run reported 3.952 GB MLX,
2.511 GB RSS and 4.332 GB footprint. Different corpora prevent a matched reduction
claim, but the measurements demonstrate why MLX peak alone is insufficient.
The full command durations were 169.53 and 104.04 seconds; the collector's 97.99
and 34.64-second timers exclude full-source hashing and final report writing.
[Reports, logs, build snapshots and coverage limits](../benchmark-results/quantization-20260904/README.md#laguna-activation-collection-process-memory-retention-addressed)
are retained. These are calibration capability/memory results, not evidence of
an improved quantized checkpoint or production speed.


## Completed Laguna reference-quality result

The matched Standard/LS2/AWSS campaign scored 192 references and 50,096 targets
with identical per-sample token identities, no truncation, a 2,048-token sample
limit and 512-token scoring chunks. Aggregate reference NLL was **2.435966 for
Standard, 2.452241 for LS2 and 2.426263 for AWSS**. AWSS minus Standard was
−0.009702 nats/token, with a paired category-stratified 95% interval of
[−0.017679, −0.001615]. Against LS2 the aggregate change was −0.025977
[−0.032151, −0.019791].

The domain tradeoff matters: **AWSS worsens math-reference NLL versus Standard
by +0.013854 [+0.001742, +0.026317]**. AWSS has the lowest code point estimate,
but its code comparison with LS2 crosses zero. Standard is best on math; AWSS
is best on the observed aggregate, code and prose point estimates. Prose supplies
about 65% of scored tokens. There is no universal winner, and these are
teacher-forced reference scores rather than generated-task accuracy or teacher KL.

[Raw reports, independent verification, category tables and reproducible analysis](../benchmark-results/quantization-20260904/README.md#laguna-held-out-nll-awss-leads-overall-standard-remains-best-on-math)
retain the regressions and uncertainty. Intervals resample 64 records inside
each category for 10,000 seeded draws; they do not account for source correlations,
run/numerical variation or multiple comparisons. Native elapsed times support
no speed claim. The next quantizer decision should use explicit workload/domain
weights and generated-task tolerances. Keep the existing defaults while adding
those tests and controlled runtime comparisons; stronger affine fitting has now
shown a measurable reference-likelihood benefit, not a blanket quality or
performance promotion.
