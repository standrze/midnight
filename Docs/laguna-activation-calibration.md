# Laguna expert activation calibration

The activation collector accepts an unquantized Laguna target and records its
actual routed inputs. It does not estimate routed-expert importance by sharing
one dense activation vector across all experts.

```sh
.build/release/model-runner-mistral-activation-stats \
  /absolute/path/Laguna-BF16 \
  /absolute/path/calibration.jsonl \
  /absolute/path/laguna-calibration.safetensors \
  --segment-tokens 512 --maximum-total-tokens 65536 \
  --minimum-expert-positions 32
```

The executable retains its existing name for compatibility. Mistral collection
keeps the existing `mistral_activation_stats_v1` schema. Laguna writes the
separate `laguna_expert_activation_stats_v1` schema.

For each routed `language_model.model.layers.N.mlp.switch_mlp` projection:

- `gate_up_proj.input_second_moment` contains `[experts, input_channels]`
  conditional moments of the hidden state selected for each expert.
- `down_proj.input_second_moment` contains the actual expert-specific
  post-SwiGLU moments, also `[experts, input_channels]`.
- Each projection has an `.expert_position_count` Int32 tensor of shape
  `[experts]`. The companion JSON reports these counts and the eligibility mask.
- An unobserved expert has zero moments and zero count. No synthetic moments
  are inserted. Experts below the configured count retain their template
  quantization during expert rescoring.

Dense projections, shared experts and routers also retain their individual
`.input_second_moment` vectors. Router statistics do not authorize reducing
router precision; the Q4R8 router policy still applies. The recorded routed
objective is conditional on selection and does not apply routing-score weights.
It is a diagonal local projection-error proxy, not a full-model quality score.

The observer follows the same sorted and unsorted expert dispatch paths as the
native fused switch layer and observes the actual tensor consumed by each
projection. Compiled block tails are bypassed while an observer is installed so
collection occurs on every forward call. Removing the observer restores the
normal serving path.

`MistralActivationWeightedScaleSearch.rescoreExperts` accepts stacked BF16
weights, the original affine-Q4 packed arrays, per-expert moments and separate
calibration/dev counts. It searches only sufficiently covered experts. Dev
coverage and the existing stored-dtype weighted-error veto both apply. Experts
with a zero-mass input group retain their template bytes as well. Callers must
use disjoint fingerprinted calibration and dev corpora and preserve their
template's gate/up layout and Q8 routers when assembling a checkpoint.

For a 64 GB Mac, use `--layerwise`. This mode reads exact tensor ranges from
safetensors, keeps one native transformer block resident, and stores each
independent segment's BF16 hidden state in a private temporary spool. It
preserves tokenizer inputs, sample/segment identifiers and their fingerprints.
The previous activation stage is deleted only after the next stage completes;
the owned spool is removed on successful completion or a handled error.

```sh
.build/release/model-runner-mistral-activation-stats \
  /absolute/path/Laguna-BF16 /absolute/path/calibration.jsonl \
  /absolute/path/laguna-calibration.safetensors \
  --layerwise --segment-tokens 512 --maximum-total-tokens 65536 \
  --spool-directory /absolute/path/scratch

.build/release/model-runner-laguna-q4r8-rescore \
  /absolute/path/Laguna-BF16 /absolute/path/Laguna-Q4R8-LS2 \
  /absolute/path/Laguna-Q4R8-AWSS \
  --activation-stats /absolute/path/laguna-calibration.safetensors \
  --validation-stats /absolute/path/laguna-dev.safetensors \
  --expert-batch 16
```

The unified quantizer accepts the same activation options with `--template`.
Without those options, its existing LS2 conversion is unchanged. Activation
refinement supports standard-Q4R8 and source-verified LS2 templates; using LS2
makes the existing LS2 grid the strict fallback. Q8 routers and the standard
embedding are copied unchanged. Both the public split expert gate/up layout
and previously fused templates are accepted, and the output keeps the input
layout, tensor names and shard index. Split expert gate/up projections each use
the collector's same per-expert gate/up input moments; dense/shared split
projections likewise use their corresponding fused input vector. No additional
full-checkpoint repacking step is needed. Exact source path/config/index provenance,
projection geometry, expert coverage, and preserved-router identity are checked
before output. Calibration/dev must have distinct corpus and token fingerprints
and no identical recorded token samples or segments. This detects exact overlap;
it does not claim to detect paraphrases or every overlapping substring.

At 65,536 tokens and hidden width 2,048, the two BF16 activation stages require
about 512 MiB of disk payload, plus small safetensors headers. The typical
Laguna sparse block has about 1.6 GB of BF16 expert weights; peak allocation
also includes fusion copies, activations, collected moments and MLX workspace.
Those are sizing estimates, not measured peak-memory guarantees. The collector
prints its spool estimate, checks available disk, and records measured MLX peak
memory. The source checkpoint itself still needs its full on-disk space.
Full-model collection remains available for machines with sufficient memory.

The [September 4 measured runs](../benchmark-results/quantization-20260904/README.md#laguna-activation-collection-process-memory-retention-addressed)
show why allocator and process memory must be reported separately. Before the
Foundation lifetime fix, 65,536 calibration tokens across 168 segments reported
3.952 GB MLX peak, but `/usr/bin/time -l` recorded 28.442 GB maximum RSS and
70.151 GB peak footprint. The collector timer was 97.99 seconds; total command
wall time was 169.53 seconds because complete source hashing occurs before the
collector timer and final output writing occurs after it.

Per-tensor and per-segment/layer autorelease pools now release Foundation
file/data temporaries promptly; MLX copies tensor data synchronously before a
read's pool drains. A subsequent development run with 15,997 tokens across 47
segments reported 3.952 GB MLX peak, 2.511 GB maximum RSS and 4.332 GB peak
footprint, with 34.64 seconds on the collector timer and 104.04 seconds total.
Both runs verified the same 14 source shards and traversed all 40 layers, but
they used different corpora and token counts. These measurements support the
lifetime fix; they are not a controlled percentage-reduction or speedup claim.
Memory values here use decimal GB. Retain both process and allocator metrics
when qualifying a larger workload or another machine.

New Laguna statistics additionally bind the complete indexed source weight
files using `indexed-safetensors-full-content-fnv1a64-v1`. The streaming digest
uses at most 8 MiB of file data at a time and includes sorted shard names, file
sizes, and every byte. Conversion verifies this identity once and reuses that
verified value for calibration/dev checks. Changed payload bytes are rejected
even when config, index, path and file size match. The method and digest are
recorded in both collection and conversion provenance. FNV-1a is a
noncryptographic reproducibility check; it does not authenticate a publisher or
protect against deliberately constructed collisions. Older statistics without
this content identity must be recollected.

Layerwise loading requires rank-3 BF16 routed expert weights in addition to
BF16 dense Linear weights. Source readers support ordinary Hugging Face
snapshot symlinks into shared blob directories, while rejecting index paths
that lexically escape the selected checkpoint directory.

The rescorer releases BF16 source arrays after their last output module has been evaluated. A use-count plan preserves sources shared by fused and split gate/up outputs; shape-only preflight never evaluates all source weights. The JSON provenance records `peak_mlx_memory_bytes`, `preflight_peak_mlx_memory_bytes`, `source_tensors_released`, and `source_tensors_remaining`. MLX memory is allocator accounting rather than total process RSS; the real conversion benchmark must still verify the host's peak RSS. Source retention is bounded by active projection consumers, while the existing template shard and pending cross-shard replacements also contribute to peak memory.

CPU and GPU affine quantizers can produce different exact packed codes and
scales. Template identity checks therefore use the selected conversion device,
require exact equality, and record `quantization_device`. A bounded Python MLX
0.31.2 investigation matched public Q4R8 packed weights/scales/biases exactly on
GPU for three sampled shared/routed/router tensors; CPU conversion differed on
those samples. This is not a full-checkpoint identity result or proof about all
MLX versions. The rescorer's actual device-specific preflight remains required.
[Sample reports and historical scripts](../benchmark-results/quantization-20260904/README.md#template-identity-depends-on-quantization-device)
are retained for the measured scope.
