# Disaggregated quantization for Midnight/MLX — 2 October 2026

My recommended order is: improve execution of existing affine Q4 weights first,
test shared-weight prefill/decode format specialization next, then consider a
separately adapted prefiller. SSD streaming is a later experiment. This order
isolates kernel speed from changes in the model's numerical behavior.

## Current research and implementation

[Disaggregated Quantization: Specializing LLM Prefill and Decode](https://arxiv.org/html/2609.26333v1)
was posted on 22 September 2026. It distinguishes shared-weight format
specialization, separately trained phase weights, and SSD-offloaded prefill.
Shared-weight specialization removes activation quantization during decode.
Separate prefill weights are trained to produce useful KV representations for
the decoder. Reported performance is on DGX Spark, with an offloaded llama.cpp
experiment reaching 1.78× TTFT speedup at 8K context. This is not an Apple/MLX
measurement. The paper explicitly leaves multi-turn and agentic cache-policy
robustness untested: retained assistant KV entries and a prefill rebuild can
represent the same history differently.

The [official implementation](https://github.com/IST-DASLab/disaggregated-quantization)
already provides training/export, vLLM serving, CUDA/Triton LUT kernels and
NVFP4 offloading experiments. DQ itself is therefore an implemented technique,
not an untried idea. Those artifacts do not establish a working native MLX port.
No DQ model artifacts were downloaded or substituted during this investigation.

## MLX experiments worth doing

These are proposed local experiments, not claims of published speed or accuracy.

| Experiment | Prefill | Decode | What it isolates |
| --- | --- | --- | --- |
| A: execution control | Existing affine Q4, direct cooperative dequantization, A16 | Existing affine Q4 matrix-vector path, A16 | Removes weight staging while retaining the grid |
| B0: native weight format | Compatible MXFP4 TensorOps with A16, if supported | Same MXFP4 weights with A16 | Whether native weight operands suffice without activation error |
| B1: shared-weight format DQ | Same weight grid, native low-precision activations | Same weights with A16 | Incremental activation-compute speed versus quality cost |
| C: separate prefiller | Adapted hardware-native prefill weights | Frozen compact decode weights | Whether adapted prompt KV improves the speed/quality frontier |

`A16` means FP16 or BF16 activations, matching the checkpoint. A kernel-only
choice in A is not yet the paper's format-disaggregated quantization.
[The affine probe](tensorops-quantized-matmul-20261002.md) is implemented and tested;
B0/B1/C are not implemented here. B0 needs the macOS 27 native-scale capability
check, including mixed operand support. B1 must measure activation packing,
scales, extra launches and temporary buffers in its total cost. A fake-quantize
then dequantize path can study error but cannot prove native low-bit speed.

For an affine checkpoint, casting its weights to FP4 changes the decoded grid.
Do not present that as a zero-error execution improvement. MXFP4 and NVFP4 also
have different scale representations and grouping. Preserve format identity and
evaluate any conversion explicitly. The paper's NVFP4 results cannot establish
that MXFP4 will retain the same quality on this model.

The likely practical opportunity is long dense prefill with high-precision decode
activations. Keep MLX's specialized matrix-vector path for short decode unless a
direct comparison says otherwise. MoE expert gathers, speculative verification,
short prompts and cache-heavy requests need separate measurements; their effective
matrix dimensions and reuse differ. A prefill row count is a performance input,
not a reliable semantic phase label.

## Runtime implementation boundaries

Carry explicit execution phase and format identity into graph construction or
compiled function selection. Include every variant in the compiled-cache key.
MLX evaluates lazily; toggling a global flag between graph creation and evaluation
can apply the wrong phase to pending work. A one-token prompt tail, a generated
token and a multi-token speculative verification block need deliberate routing.

For separate weights, require matching tokenizer/template, architecture, RoPE,
positions, head layout and cache dtype. Matching tensor shapes alone does not
establish semantic compatibility. Pin both artifact revisions and their format
metadata. A KV cache belongs to the entire prefill/decode pair and its execution
policy; changes require invalidation or a separately keyed cache namespace.
The current `SharedPromptCache` assumes one loaded model and exact token keys,
so adding a swappable prefiller requires revisiting that assumption.

Model preparation, calibration and quantization belong in Afterglow. New QADD
distillation work belongs in midnight-moonshine. Midnight owns execution,
Activation Capture and runtime/quality evaluation of those artifacts. This
investigation adds no quantizer, training implementation or public API setting.

## Evaluation gate, especially for agentic use

Use the same checkpoint pair, prompt tokens, sampling settings and output budget
for all arms. Start with existing local eligible checkpoints and record immutable
identity. Compare uniform weight-only, uniform low-precision compute, prefill-only
low-precision compute and decode-only low-precision compute before claiming a DQ
benefit. Separate kernel-only results from format-conversion quality results.

Measure prompt lengths 128, 512, 2K, 8K and 16K with short and long output budgets.
Record prefill rate, TTFT, decode rate, total latency, resident/peak memory and
temporary packing costs. Include retrieval/long-context tasks and reasoning/code
tasks. Fixed-prefix logits, KL, NLL and winner margins diagnose numerical changes;
generated-answer grading is also required. Agree on a quality budget before
selecting the fastest arm.

For multi-turn histories, compare these cache routes explicitly:

1. Continue using retained KV entries from generated assistant tokens.
2. Rebuild the identical full token history through the prefill path.
3. Restore a saved prefix and prefill only the suffix.
4. Repeat after cache eviction, trimming and a cancelled request.

Record which phase produced each cache segment, exact reused/prefilled token
counts, fixed-prefix next-token differences and task outcomes. For experiment A,
require parity consistent with the kernel gate. For B/C, quantify differences
under a declared quality budget rather than requiring identical generations or
silently treating cache-route changes as equivalent.

`model-runner-runtime-bench --prompt-cache` and `--hot-cache-ab` already contain
controlled cache-route comparisons; they need a DQ-aware variant, not an assumption
that current parity tests establish paired-model compatibility. Reuse
`model-runner-quality-bench`, `model-runner-teacher-kl-bench` and
`model-runner-generation-bench` for complementary quality evidence. The generation
benchmark explicitly treats its timings as diagnostics, not controlled speed tests.

## SSD streaming gate

Measure local SSD throughput and transformer-block compute time first. A useful
double-buffering estimate is initial load plus the sum of
`max(block_load_time, block_compute_time)`, plus restoring decode weights and
non-overlapped work. Compare that measured total to current weight-only prefill;
an isolated native FP4 GEMM win is insufficient.

On unified memory, two resident checkpoints still consume RAM. Count KV, scratch,
allocator retention and file-backed pages in the actual peak. Borrowing decode
storage requires proving no lazy graph or request retains access to overwritten
buffers, and cancellation must restore a usable decoder or invalidate/reload it.
Begin with resident small paired weights to validate semantics. Add block streaming
only after long-prompt compute can amortize measured I/O and restore costs.
