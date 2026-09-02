# Runtime and API model research

Last updated: 2026-08-29

## Decision

The runtime priorities for this project are:

1. Mistral text models and Voxtral speech models.
2. Poolside Laguna, especially the 4-bit MLX checkpoint and its DFlash draft model.
3. OpenAI GPT-OSS.
4. Qwen compatibility when it comes from upstream MLX support, without a custom Qwen TTS runtime.

Production inference stays inside the Swift process. Python is acceptable for an
offline checkpoint conversion, but it is not part of the serving path.

## API comparison

Model architecture and HTTP protocol are separate decisions. A native Laguna,
Mistral, GPT-OSS, or Qwen text model can all sit behind the same local chat
contract.

| Surface | Text endpoint and stream | Speech endpoint and stream | Important differences |
| --- | --- | --- | --- |
| OpenAI-compatible runner contract | `POST /v1/chat/completions`; JSON or data-only SSE ending in `data: [DONE]` | `POST /v1/audio/speech`; binary chunks or OpenAI speech SSE | This is the stable client-facing contract. Supporting it does not imply an OpenAI model architecture. |
| Mistral API | Also `POST /v1/chat/completions`; data-only SSE ending in `[DONE]` | `POST /v1/audio/speech`; JSON `audio_data` or event-stream output | Mistral adds fields such as `guardrails`, `safe_prompt`, `prompt_mode`, `random_seed`, `prompt_cache_key`, and speech fields including `voice_id`, `ref_audio`, and `stream`. The runner implements its documented common subset and its separate speech dialect. |
| Alibaba Model Studio / Qwen text | Offers OpenAI-compatible Chat Completions and Responses endpoints as well as native DashScope interfaces | Qwen realtime TTS is a stateful WebSocket session | Qwen-specific text options may be passed as nonstandard fields. Realtime TTS uses events such as `session.update`, `input_text_buffer.append`, `input_text_buffer.commit`, and base64 `response.audio.delta`; that is not the OpenAI Chat Completions wire format. |

Qwen does not have a unique architectural advantage merely because Alibaba's
hosted runtime streams audio. Incremental audio is a capability of the model,
decoder, server, and transport together. Mistral speech and this runner's
OpenAI speech surface can stream too. The meaningful Qwen difference is its
official realtime protocol: a full-duplex WebSocket event session rather than
the runner's HTTP/SSE speech contracts.

## Why Laguna belongs in the Swift runtime

Poolside describes Laguna XS 2.1 as a 33B-total, 3B-active MoE model with 40
layers: 10 global-attention layers, 30 sliding-window layers, 256 routed
experts plus one shared expert, a 512-token sliding window, and a 262,144-token
context window. The MLX 4-bit community checkpoint is about 18.8 GB.

Python is one way to serve Laguna through Transformers, vLLM, or SGLang; it is
not a model requirement. The runner now implements Laguna directly in
`ModelRunnerCore` with:

- mixed full and sliding-window attention;
- layer-specific query-head counts, RoPE/YaRN settings, Q/K RMS normalization,
  and per-head output gates;
- a dense first MLP followed by 39 top-k routed MoE layers;
- correction-bias routing with the unbiased scores retained for expert weights;
- a shared expert and hybrid KV caches;
- tied or independent output embeddings;
- mixed per-module quantization matching the MLX checkpoint.

The architecture is registered through the upstream MLX Swift model registry,
so the project does not need a private `mlx-swift-lm` fork.

## Implemented performance work

### MLX 0.32.2 Metal backend

The macOS build now pins official MLX Swift update revision `72f3c3a`, MLX
0.32.2 core revision `1f8e74e`, and MLX-C revision `c74db53`. The core is newer
than the `c793734` revision observed in Ollama 0.33.1 and includes the audited
M5 QMV, NAX quantized-MoE, GQA-8 decode, and wide-GEMV work. Linux remains on
its separate exact CUDA-compatible revision.

Dependency preparation verifies all three Darwin revisions. The Metal build
compiles the authoritative static source set from that same MLX checkout,
including the 0.32.2 `dot` and `fence` kernels, rather than relying on the
incomplete generated-kernel copy.

A balanced old → new → new → old full-model sequence preserved the
Q4R8 runtime's observed 149–152 tok/s performance class before sustained device
heat collapsed the final old-core block. That sequence does not support a
precise backend-only speedup claim. It does establish that the upgrade did not
regress the deployment checkpoint. On the new backend, the same-loaded-model
Laguna fusion A/B remained exact and improved median decode by 5.80% under the
thermally stressed run.

### Fused expert gate/up projection

Laguna's checkpoint stores quantized expert gate and up projections separately.
The native model combines their independently quantized output rows and uses
`FusedGateUpSwitchGLU`, reducing two selected-expert quantized matrix operations
to one. An ordinary checkpoint is fused lazily at load time.

For lower GPU residency, the offline utility below creates a standalone compact
checkpoint. It rewrites the four weight shards, replaces the six gate/up
tensors in every sparse layer with three fused tensors, validates every output
shard, and writes an optimization manifest. Tensor bytes are replaced in place;
the output does not retain the obsolete projections.

```bash
python3 Scripts/pack-laguna-gate-up.py \
  /absolute/path/Laguna-XS-2.1-4bit \
  /absolute/path/Laguna-XS-2.1-4bit-fused-gate-up-compact
```

The utility requires PyTorch and Safetensors only while packaging. The output
runs through native Swift/MLX and requires no Python environment.

### Persistent MLX streams

The prior request path created a new MLX stream for every generation. On the
Linux CUDA backend, each completed request left a worker spinning at 100% CPU.
Decode therefore deteriorated as requests accumulated. Model loading,
generation, adapter application, and Voxtral synthesis now reuse the device's
persistent default stream; these paths are actor-serialized already.

A regression test prevents dynamic GPU stream creation from returning to the
serialized text and Voxtral paths.

### Metal token scheduling

The token iterator now keeps a one-token Metal pipeline on macOS: it queues the
next token asynchronously and synchronizes when the previously queued token is
read. Linux/CUDA retains a strict token-plus-cache evaluation boundary because
asynchronous cache mutation there produced non-finite logits in long runs.
Allocator cache clearing was removed from the hot decode loop.

On the local 40-GPU-core M5 Max with the Laguna Q4 checkpoint, a roughly
990-token prompt and 100 generated tokens improved from a 115.99 tok/s median
to 124.70 tok/s, a 7.5% gain. A 512-token stability run completed at
120.99 tok/s. An exact same-machine `mlx-vlm` control measured about
135.99 tok/s, leaving a smaller runtime integration gap to investigate.

### Compiled Laguna MoE graph

Ollama's native Laguna source exposed a remaining model-graph difference: it
compiled the sigmoid top-8 router and fused routed reduction, scale, shared
expert, and residual. The Swift Laguna model now uses the same graph boundaries.

On the local M5 Max, one loaded target alternated unfused and fused 512-token
generations. Median decode improved from 139.46 to 151.17 tok/s, an 8.40% gain,
while median prompt processing improved from 1,006.50 to 1,032.94 tok/s
(2.63%); all generated outputs matched exactly. Repeated fragment probes
measured about 1.5x faster router execution and 1.42x faster final MoE
reduction. The optimized path is the default; `--laguna-fusion-ab` preserves
the full-model regression test.

### MLX Q4R8 path

The Metal target is affine Q4 group-64 by default with Q8 only for
calibration-selected modules. The stock checkpoint already follows the most
important rule: its 39 MoE router projections are Q8, while the other
quantized modules are Q4. Local Laguna-shaped selected-expert measurements
showed Q8 moving the hot gate/up and down operations from about 0.029/0.016 ms
to 0.042/0.025 ms. Interleaved identical-tensor runs on the former MLX 0.31.1
core did not establish a stable winner among affine Q4, MXFP4, and NVFP4. The
rerun on MLX 0.32.2 remains mixed: NVFP4 reached 0.939x affine-Q4 throughput
for dense QMV, 0.999x for gathered expert gate/up, and 1.018x for gathered
expert down. Q4R8 therefore remains the deployment choice until a
quality-matched full-model format comparison establishes a better alternative.

The project now has a native Swift/MLX Q4R8 converter. It can read Poolside's
per-expert BF16 checkpoint, stack and fuse the routed experts, force routers to
Q8, and apply an exact Q8 module allowlist while retaining ordinary MLX packed
weights and Metal kernels. Gate and up are one fused quantization unit.

That converter has produced the 18,821,963,264-byte
`Laguna-XS-2.1-MLX-Q4R8-v1` artifact on the project server: 1,517 tensors
in four shards, strict Q4 group-64 plus exactly 39 Q8 routers. In a matched
five-trial RTX 4090 comparison after a 256-token warm-up, it reached a 126.93
tok/s median decode rate versus 127.45 tok/s for the prior compact Q4 control.
The 0.4% difference is noise-level and shows no new packing cost.

The artifact is the accuracy-first baseline, not the end of accuracy
calibration. No dense or expert module is promoted to Q8 until BF16-teacher
KL/NLL and downstream task evaluation show enough recovered quality to justify
its active-byte cost.

See [Laguna Q4R8 for MLX and Metal](laguna-metal-q4r8.md) for the measured
bandwidth budget, accuracy-selection procedure, converter commands, and why a
new physical bit layout is not the first optimization step.

## RTX 4090 measurements

Environment: GeForce RTX 4090 with 24,564 MiB VRAM, Ubuntu 24.04, Swift 6.3,
MLX CUDA, `sm_89`, and `mlx-community/Laguna-XS-2.1-4bit`. Each measured trial
used the same 33-token rendered prompt, greedy decoding, and 256 generated
tokens. A 64-token warmup preceded five sequential trials.

| Variant | Median decode | Median end-to-end | GPU used after trials |
| --- | ---: | ---: | ---: |
| Original checkpoint, lazy fused weights | 136.34 tok/s | 129.75 tok/s | 18,606 MiB |
| Compact prepacked checkpoint | 136.08 tok/s | 132.40 tok/s | 17,662 MiB |

The small throughput difference between packaging forms is within run-to-run
noise. The compact form's repeatable gain is 944 MiB less GPU residency, about
5.1%, while retaining the same generated text. It is the preferred deployment
artifact when the extra standalone checkpoint copy is acceptable.

The stream lifecycle fix is the larger operational result. Before it, identical
sequential requests declined from 121.45 to 43.23 tok/s as spinning workers
accumulated. With persistent streams, the post-warmup trials remain between
135.57 and 139.08 tok/s, and the process task count no longer grows per request.
The final SSE correctness probe returned exactly `native Laguna works` and a
terminal `[DONE]` event.

The Linux runtime also performs explicit terminal cleanup for mlx-swift's
thread-local and global cross-thread command encoders. The Q4R8 runner now
handles SIGTERM after inference and exits with status 0; without global cleanup,
CUDA teardown aborted from a late stream synchronization in a C++ static
destructor.

These are single-sequence local measurements, not claims about multi-user
throughput, long-context prefill, or model quality.

## Native Laguna DFlash

Poolside's Laguna DFlash speculator is a 5-layer Llama-style draft model that
can propose up to 15 tokens per verification step. Poolside reports
workload-dependent speedups from 1.67x on GSM8K v2 to 2.64x on HumanEval for its
BF16 target measurements. Those figures are useful targets, not a promise for
this 4-bit MLX/CUDA runtime.

`mlx-swift-lm` contains generic speculative-decoding, MTP, and draft-model
infrastructure. This project now:

1. ports `DFlashLagunaForCausalLM` to Swift;
2. loads the draft checkpoint beside the target without a Python sidecar;
3. exposes block sizes for sweeps such as 4, 8, and 16;
4. reports proposed and accepted draft tokens with end-to-end throughput;
5. keeps non-greedy requests on the target-only path.

DFlash is implemented natively. Its separate BF16 checkpoint can also be
converted by `model-runner-quantize` with an experimental DFlash-specific
Q4R8 ScaleSearch profile; target-paired acceptance and throughput remain the
decision criteria for keeping that quantized drafter.

## GPT-OSS and Mistral implications

OpenAI documents GPT-OSS 20B as a 21B-total, 3.6B-active open-weight text model
with streaming, function calling, and structured-output support. The pinned
upstream MLX Swift dependency already has a GPT-OSS architecture, so this
project should benchmark and tune it rather than create a second model
implementation. The local client contract remains `/v1/chat/completions`.

Mistral and Voxtral remain first-class. The persistent-stream correction is
shared by text generation and native Voxtral synthesis, and the public server
keeps both OpenAI and Mistral speech dialects without a provider-mode switch.

The Mistral runtime policy is family-wide rather than checkpoint-name-specific.
Classic Mistral, Mistral 3/Ministral, and Mixtral now share the zero-copy hot
conversation path, while cache-disabled requests preserve one-shot generation.
Classic Mistral still uses the upstream Swift `LlamaModel`, augmented by a
separate pinned patch that honors `layer_types` and `sliding_window`; Mistral 3
keeps its existing `Mistral3TextModel` hybrid cache. Laguna's independent
compiled and branch-snapshot paths are unchanged.

The remaining Mistral graph work is model-specific rather than API-specific.
Classic Mistral and Mistral 3 both issue separate gate/up and Q/K/V projections.
Mistral-7B-shaped Metal probes were repeated on MLX 0.32.2. Compiled-only
SwiGLU reached 0.983x current throughput, packed-row gate/up reached 0.722x,
and packed QKV reached 0.955x end-to-end. All outputs matched, but none is
enabled because none cleared the performance gate. The next gate is a real
checkpoint benchmark. The pinned Swift backend now backports Mixtral's
single-row Metal router top-k/gather specialization, guarded to GPU evaluation
mode with the original expression retained for CPU, training, and multi-token
prefill. Its outputs are tested bit-for-bit, including zero and
negative logits, while further routed-reduction work must preserve the
selected-logit softmax semantics. No full Mistral checkpoint is currently
installed on this MacBook, so no Mistral throughput claim has been made yet.

See [MacBook MLX performance](macbook-mlx-performance.md) for the exact core
revision audit, current Ollama controls, DFlash result, and acceptance gates.

## Primary sources

- [Poolside Laguna XS 2.1 model card](https://huggingface.co/poolside/Laguna-XS-2.1)
- [MLX 4-bit Laguna checkpoint](https://huggingface.co/mlx-community/Laguna-XS-2.1-4bit)
- [Poolside Laguna DFlash model card](https://huggingface.co/poolside/Laguna-XS-2.1-DFlash)
- [MLX quantization modes and packing](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
- [Poolside Laguna NVFP4 model card](https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4)
- [Mistral Chat API](https://docs.mistral.ai/api/endpoint/chat)
- [Mistral Speech API](https://docs.mistral.ai/api/endpoint/audio/speech)
- [Alibaba Qwen OpenAI-compatible Chat API](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions)
- [Alibaba Qwen realtime TTS WebSocket API](https://www.alibabacloud.com/help/en/model-studio/interactive-process-of-qwen-tts-realtime-synthesis)
- [OpenAI GPT-OSS 20B model reference](https://developers.openai.com/api/docs/models/gpt-oss-20b)
- [Ollama MLX runner](https://github.com/ollama/ollama/blob/v0.33.1/x/mlxrunner/runner.go)
- [Ollama Laguna MLX model](https://github.com/ollama/ollama/blob/v0.33.1/x/models/laguna/laguna.go)
- [MLX Swift 0.32 update](https://github.com/ml-explore/mlx-swift/pull/450)
- [MLX 0.32.2 release](https://github.com/ml-explore/mlx/releases/tag/v0.32.2)
- [MLX-LM heterogeneous Mistral attention](https://github.com/ml-explore/mlx-lm/commit/dcb4b9ba6db5)
- [MLX Swift LM fused Mixtral router](https://github.com/ml-explore/mlx-swift-lm/commit/db767efca373bcc215e2c340e97751c28f570491)
- [Authoritative Ministral 8B hybrid configuration](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410/blob/main/config.json)
- [MLX M5 Max NVFP4 QMV optimization](https://github.com/ml-explore/mlx/commit/5a1e44c3bb991dab753cee394b0b1d889e2eb9a7)
