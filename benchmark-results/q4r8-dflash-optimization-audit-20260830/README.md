# Q4R8, ScaleSearch, DFlash, and backend portability audit — 2026-08-30

## Decision summary

- The accepted Metal production path is already faster than the historical
  exact Ollama Q4R8 control: `185.291435 tok/s` versus `179.566223 tok/s`, a
  `+3.188357%` lead. The winning stack is direct KV slice update, the compiled
  single-token Laguna block tail, and the fused 256-expert/top-8 router.
- ScaleSearch improves quantization quality, not inference speed. It emits the
  same affine Q4/group-64 storage and calls the same runtime kernels as standard
  Q4. The remaining ScaleSearch optimization opportunity is offline conversion
  time.
- Do not replace Q4R8 with all-Q4 for DFlash by default. At Laguna's exact
  `2048 -> 256` router shape, Q4 and Q8 were tied within sub-percent measurement
  noise. The 39 Q8 target routers account for only about 0.64% of active weight
  traffic, while changing them can alter expert selection and draft acceptance.
- DFlash is block-diffusion speculative decoding, not FlashAttention. Its current
  problem is a combination of target block-versus-sequential numerical framing,
  a target/drafter precision mismatch, and a multirow verifier that cannot use
  today's single-token compiled/fused Laguna production path.
- ScaleSearch calibration and DFlash scheduling are portable to CUDA. The
  accepted fused-router win is Metal-only, and the DGX Spark needs a separate
  CUDA/SM121 measurement and likely an NVFP4-specific execution profile.

## Current consolidated Metal position

The retained production evidence is under
[`laguna-fused-router-topk-20260830`](../laguna-fused-router-topk-20260830/README.md).

| Runtime | Median decode | Relative to Ollama |
| --- | ---: | ---: |
| Current default-on Swift/Metal Q4R8 ScaleSearch | 185.291435 tok/s | +3.188357% |
| Historical exact Ollama Q4R8 ScaleSearch | 179.566223 tok/s | control |

This comparison is the retained 79-prompt-token, 128-generated-token benchmark
reference. It is not a claim for every prompt length, temperature, or thermal
state.

## New router-precision measurement

The benchmark now has a dedicated `--router-precision-ab` mode. It uses one
BF16 input row and the exact Laguna router geometry, builds Q4/G64 and Q8/G64
from the same source weights, warms both, and alternates measurement order.

```sh
swift build --configuration release \
  --product model-runner-metal-quant-bench

.build/arm64-apple-macosx/release/model-runner-metal-quant-bench \
  --router-precision-ab \
  --warmup 32 \
  --queue-depth 512 \
  --queue-rounds 21
```

Host: 40-GPU-core M5 Max. Each value is milliseconds per operation. A speedup
above one favors Q4.

| Run | Q4 total | Q8 total | Total speedup | Q4 execution | Q8 execution | Execution speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.002980 | 0.002994 | 1.004807x | 0.002742 | 0.002767 | 1.009113x |
| 2 | 0.002749 | 0.002752 | 1.001095x | 0.002523 | 0.002535 | 1.004806x |
| 3 | 0.002885 | 0.002853 | 0.988858x | 0.002662 | 0.002631 | 0.988383x |
| Geometric ratio | — | — | **0.998230x** | — | — | **1.000727x** |

The total and execution ratios disagree by less than 0.3 percentage point and
both round to a tie. This rejects a meaningful router-latency argument for
all-Q4. It does not replace a full-model quality and DFlash-acceptance A/B.

## New ScaleSearch microbenchmark confirmation

The fixed synthetic `6144 x 2048` BF16 weight uses ordinary affine Q4/G64 as
the per-group fallback, then nine scale factors, one least-squares bias update,
and two joint affine fits.

| Run | Standard MSE | ScaleSearch MSE | Conversion overhead | QMV ratio | Stored bytes match |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 3.314103e-6 | 2.812670e-6 | 215.425x | 1.007x | yes |
| 2 | 3.314103e-6 | 2.812670e-6 | 207.290x | 0.981x | yes |
| 3 | 3.314103e-6 | 2.812670e-6 | 228.993x | 1.018x | yes |

The reconstruction MSE improves by 15.13%, but the QMV ratio crosses one and
the stored byte count is identical (`7,077,888` each). Full-checkpoint evidence
is less extreme on conversion cost but reaches the same conclusion: the Laguna
48-tensor audit improved weighted MSE by 15.0543%, took about 21.4 times the
standard conversion time, and did not change the runtime format.

The safest conversion acceleration is to evaluate all nine factors together
inside a bounded group chunk, retain each group's best refinement, and pack
only the winner. This can replace nine graph-materialization boundaries with a
chunk memory bound. Promotion must require bit-for-bit equality of packed
weights, scales, and biases against the current LS2 implementation. Source-shard
grouping/LRU and bounded tensor batches are separate I/O and memory improvements.

## What is actually wrong with DFlash

1. **The current exact-output gate compares different numerical frames.**
   Sequential target decoding executes one-row quantized matmuls. DFlash target
   verification executes a block-width forward. Small reduction-order changes
   can change a close greedy argmax or MoE route even when both paths are honest.
   Conventional Q4R8 and ScaleSearch Q4R8 both diverge in the local evidence, so
   ScaleSearch is not the root cause.
2. **External no-drafter evidence isolates the same phenomenon.** The MLXFast
   DFlash correctness study observed 14 divergence events, all 14 in a
   target-only block-versus-sequential baseline, and zero DFlash-versus-block
   mismatches. That study used NVFP4 rather than this Q4R8 artifact, so it
   validates the class of failure rather than proving this implementation's
   cache parity.
3. **The official INT4 drafter is paired to a different target.** Poolside's
   official INT4 target is not all-Q4: it uses symmetric INT4/G128 expert weights
   in layers 1–30, INT8/G128 expert weights in layers 31–39, and excludes
   attention, routers, shared experts, layer 0, and the LM head. Its official
   drafter is explicitly precision-paired. It accepted 0/1,785 proposals against
   the custom affine Q4R8 target in the retained CUDA experiment.
4. **The verifier misses today's production fast path.** DFlash requests five
   intermediate hidden-state taps and verifies multiple rows. The compiled
   block tail requires no captures and sequence length one; the fused router is
   also single-row only. DFlash therefore cannot inherit the production path
   that produced 185.29 tok/s.
5. **Block width changes kernel routing.** On Metal, Laguna's head counts imply
   that block sizes 3 and 4 retain fused vector SDPA in all target and draft
   layers. Block 5 falls off that path in 30 of 40 target layers and all five
   draft layers. Upstream DFlash also recommends block size no larger than five
   for quantized MLX target/draft models because small-M quantized matmul scales
   poorly.
6. **The throughput evidence is not stable enough for default-on use.** Metal
   has ranged from `-4.26%` to `+8.09%` in short probes. CUDA block 4 produced a
   promising `+7.36%` median but two 31–33 tok/s outliers, making its mean slower.

The next correctness harness should compare DFlash with a trusted target-only
**block-frame** oracle on the same emitted prefix. It should record per-row
target logits, final hidden states, router indices, and committed cache offsets.
Sequential greedy output remains a useful quality diagnostic, but it should not
be the sole correctness gate for a block verifier.

## Recommended implementation order

1. Add the real-checkpoint block-frame parity harness. Require DFlash-versus-
   block logits/hidden/cache parity before optimizing it further.
2. Specialize `K=3` and `K=4`. On Metal, keep both warm and measure accepted
   output tokens per second, target verify time, draft time, and acceptance.
3. Build a capture-aware compiled verifier tail plus a batched 256-expert/top-8
   router for three or four rows. This attacks the gap between the 185.29 tok/s
   production target and the current DFlash verifier.
4. Keep the quantized BF16-target drafter as the current pairing. If maximum
   acceptance is required, train or calibrate a drafter directly against the
   custom Q4R8 target rather than relabeling Poolside's official INT4 drafter.
5. Only then test an all-Q4 drafter/target as an experimental control. Judge it
   by end-to-end accepted output tokens per second and quality, not isolated Q4
   matmul latency.
6. Optimize ScaleSearch conversion only after the runtime work above; it cannot
   close a tokens-per-second gap.

## Mistral and Mixtral portability

- ScaleSearch is already validated on dense Ministral 3: the retained audit
  reduced representative reconstruction MSE by 15.0436% and slightly improved
  the fixed teacher-forced quality smoke corpus. Fixed-token runtime differed by
  only 0.21%, which is noise-sized.
- The current practical Ministral 3 comparison measured native MLX ScaleSearch
  at 67.995 tok/s versus Ollama Q4_K_M at 50.287 tok/s (`+35.21%`). That test
  matched the user prompt but not exact prompt token IDs, so it is a practical
  package/runtime comparison, not proof that ScaleSearch made inference faster.
- Direct KV updates, prompt-prefix reuse, and incremental ByteLevel decoding are
  architecture-general when the cache/tokenizer contract matches.
- Laguna's compiled tail and custom 256-expert/top-8 router are architecture-
  and shape-specific. Dense Mistral does not use them. Mixtral needs its own
  representative full-checkpoint benchmark before claiming a similar gain.

## DGX Spark applicability

The repository has a Linux/CUDA `dgx-spark` build profile for ARM64, CUDA 13,
and `sm_121`, but it remains an experimental, not yet on-device-validated path.

What carries over without a kernel port:

- checkpoint values and ScaleSearch's quality effect;
- Q4R8 policy and ordinary MLX affine storage;
- direct KV slice updates and most high-level scheduling/caching logic;
- DFlash's block drafting, verification, and acceptance accounting.

What does not carry over automatically:

- the custom Metal fused router;
- Metal shader/QMV specializations and Metal SDPA block-size thresholds;
- the Metal-only automatic compiled-tail/fused-router selection;
- the 185.29 tok/s result itself.

The pinned MLX CUDA source has different routing. Its custom vector SDPA accepts
query length below four; supported larger block forwards use cuDNN. Affine Q4
and Q8 with fewer than eight rows use CUDA QMV, but the explicit small-M kernel
optimized for compute capability above 9 excludes affine mode and serves
floating formats such as NVFP4. Therefore the Spark plan should benchmark:

1. affine Q4R8 target-only and DFlash at `K=3` and `K=4`;
2. a matched NVFP4 Laguna candidate using the Blackwell-oriented path;
3. a CUDA fused/batched Laguna router and capture-aware verifier graph;
4. the same OpenAI request against MLX-Swift, vLLM, and TensorRT-LLM where the
   Laguna architecture is supported.

NVIDIA's current Spark guidance favors NVFP4 models with vLLM or TensorRT-LLM,
which is consistent with GB10's FP4 hardware. It is not evidence that a current
Laguna/DFlash integration is available or faster; that remains an on-device A/B.

## Primary external references

- [DFlash paper](https://arxiv.org/abs/2602.06036)
- [DFlash MLX recommendation for quantized block sizes](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/README.md#L72-L85)
- [MLX v0.32.2 Metal SDPA routing](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/scaled_dot_product_attention.cpp#L633-L766)
- [MLX v0.32.2 CUDA SDPA routing](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/cuda/scaled_dot_product_attention.cu#L661-L682)
- [MLX v0.32.2 CUDA affine QMV/QMM dispatch](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/cuda/quantized/quantized.cpp#L101-L130)
- [MLXFast DFlash correctness contract and measurements](https://github.com/Layr-Labs/mlxfast-challenge-dev/blob/main/docs/dflash-track-correctness-contract.md#why-the-retired-mtp-contract-cannot-be-reused)
- [Poolside INT4-paired DFlash model card](https://huggingface.co/poolside/Laguna-XS-2.1-DFlash-INT4/blob/main/README.md)
- [Poolside official INT4 target configuration](https://huggingface.co/poolside/Laguna-XS-2.1-INT4/blob/main/config.json)
- [NVIDIA DGX Spark hardware](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [NVIDIA DGX Spark vLLM and NVFP4 guide](https://build.nvidia.com/spark/vllm/agent-ready-models)
