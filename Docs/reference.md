# Midnight: complete reference

Midnight Runner is a standalone Swift server that loads an MLX `.safetensors`
model in-process and exposes OpenAI-compatible model, chat, and speech contracts
plus local Mistral speech and voice contracts. It uses Metal on macOS, CUDA on
Linux, or MLX's CPU backend.

> **Beta:** Midnight is a prerelease. CLI, API, and model-runtime behavior may
> change before the first stable release.

The separate `model-chat-swifttui` client connects to the runner over HTTP.

## Model-family priorities

The runner optimizes for Mistral/Voxtral, Poolside Laguna, and GPT-OSS. Model
execution stays in the Swift process; production inference does not launch a
Python model server.

| Family | Native runtime | HTTP surface | Project status |
| --- | --- | --- | --- |
| Mistral / Voxtral | Swift + MLX | OpenAI-compatible chat; OpenAI and Mistral speech/voice dialects | Primary |
| [Poolside Laguna](https://huggingface.co/poolside/Laguna-XS-2.1) | Swift + MLX hybrid-attention/MoE implementation | OpenAI-compatible chat | Primary |
| [GPT-OSS](https://developers.openai.com/api/docs/models/gpt-oss-20b) | Upstream MLX Swift architecture | OpenAI-compatible chat | Supported |
| Qwen | Upstream MLX Swift architectures where available | OpenAI-compatible chat | Compatibility only; no project-specific TTS path |

"OpenAI-compatible" describes the runner's HTTP contract, not the model's
internal architecture. Laguna, Mistral text models, and GPT-OSS therefore use
the same `/v1/chat/completions` client path. Vendor-specific fields are supported
only when documented below. Audio streaming is also an endpoint/runtime
capability rather than a reason to couple the server to Qwen TTS.

See [runtime and API model research](runtime-model-research.md) for the
OpenAI/Mistral/Qwen protocol comparison, the native Laguna architecture,
measured Metal and RTX 4090 results, and the DFlash roadmap. The detailed
[Laguna Metal Q4R8 design](laguna-metal-q4r8.md) contains the native
converter, bandwidth budget, and accuracy-selection rules.
The [MacBook MLX performance audit](macbook-mlx-performance.md) explains
why Ollama's MLX Laguna path could be faster, records the synchronized MLX
0.32.2 Metal-backend upgrade and the implemented compiled-MoE gain, and ranks
the remaining Laguna and Mistral work.
The follow-up [Ollama optimization-gap audit](ollama-optimization-gap.md)
includes the permanent-pthread implementation and A/B. It remains opt-in with
`MODEL_RUNNER_PINNED_MLX=1` because it did not deliver a repeatable speed gain.
The experimental [ScalePlan-Q4R8 implementation](scaleplan-q4r8.md)
adds same-format affine-Q4 scale search and validation-gated per-layer planning
without changing the deployment kernels.
The [Laguna DFlash guide](laguna-dflash.md) documents the native
five-layer block drafter, checkpoint pairing, and benchmark procedure.

### Standalone Swift ScaleSearch quantizer

`model-runner-quantize` is the architecture-aware Swift quantization program;
the shell files in `Scripts/` are optional build/launch conveniences, not the
quantizer implementation. Build it with the MLX Metal library on macOS:

```bash
MODEL_RUNNER_BUILD_CONFIGURATION=release \
MODEL_RUNNER_BUILD_PRODUCT=model-runner-quantize \
./build.sh
```

Inspect a local Mixtral, Mistral, Llama, GPT-OSS, Qwen, Poolside Laguna
DFlash drafter, or other registered MLX Swift text model without writing
output, then perform the conversion directly:

```bash
.build/release/model-runner-quantize \
  /absolute/path/unquantized-model \
  /absolute/path/model-q4r8-scalesearch \
  --dry-run

.build/release/model-runner-quantize \
  /absolute/path/unquantized-model \
  /absolute/path/model-q4r8-scalesearch
```

Create a matched ordinary-affine-Q4 control from the same source by adding
`--standard-q4` and choosing a separate destination:

```bash
.build/release/model-runner-quantize \
  /absolute/path/unquantized-model \
  /absolute/path/model-q4-standard-control \
  --standard-q4
```

The default is ordinary MLX affine Q4 group-64 storage with ScaleSearch LS2
used to derive `Linear` and `SwitchLinear` arrays. Embeddings and custom
quantizable modules remain standard MLX Q4 because they do not accept the
searched arrays. Architecture profiles force known MoE routers to standard
affine Q8: Mixtral's `block_sparse_moe.gate`, Laguna's routed gate, GPT-OSS's
router, and supported Qwen MoE gates. Repeated `--q8-module` and
`--skip-module` path globs add explicit policy; unmatched or conflicting
patterns fail before output mutation.

The control uses the same affine Q4 group size, architecture sanitizer,
per-layer Q8 overrides, and sharding path; only Q4 calibration differs. This
keeps storage geometry and inference kernels matched for benchmarking. The
default candidate writes `scale-search-quantization.json`; the control writes
`standard-q4-quantization.json` and truthfully records every Q4 module as
standard rather than searched. `--standard-q4` is intentionally unavailable
with Laguna `--template`, so the proven template layout remains unchanged.

### Matched model-quality benchmark

`model-runner-quality-bench` measures teacher-forced next-token NLL and
perplexity with the tokenizer and model loaded from one local checkpoint. It
loads only one model per process and evaluates MLX cross-entropy from FP32 logits,
while copying only the scalar loss to the CPU. Build it once, then run the BF16,
ordinary-Q4, and ScaleSearch-Q4 artifacts separately against the same authored
corpus:

```bash
MODEL_RUNNER_BUILD_CONFIGURATION=release \
MODEL_RUNNER_BUILD_PRODUCT=model-runner-quality-bench \
./build.sh

.build/release/model-runner-quality-bench \
  /absolute/path/Ministral-3-14B-BF16 \
  Benchmarks/Quality/general-smoke.jsonl \
  benchmark-results/ministral-bf16-quality.json

.build/release/model-runner-quality-bench \
  /absolute/path/Ministral-3-14B-Q4-standard \
  Benchmarks/Quality/general-smoke.jsonl \
  benchmark-results/ministral-q4-standard-quality.json

.build/release/model-runner-quality-bench \
  /absolute/path/Ministral-3-14B-Q4-ScaleSearch \
  Benchmarks/Quality/general-smoke.jsonl \
  benchmark-results/ministral-q4-scalesearch-quality.json
```

The three reports must have identical `corpus_fingerprint` and
`token_id_fingerprint` values before comparing `token_weighted_nll` or
`perplexity`; lower is better. The included corpus is a bounded comparative
smoke gate, not a substitute for a broad external evaluation. Weight MSE alone
is not a quality gate because it cannot capture activation propagation or
next-token behavior.

DFlash is a separate checkpoint from the Laguna target. The official
`Laguna-XS-2.1-DFlash-INT4` artifact is a BF16 drafter trained to match the
INT4 target, so quantize it with a second invocation:

```bash
.build/release/model-runner-quantize \
  /absolute/path/Laguna-XS-2.1-DFlash-INT4 \
  /absolute/path/Laguna-XS-2.1-DFlash-INT4-MLX-Q4R8-ScaleSearch
```

The `DFlashLagunaForCausalLM` architecture selects the native drafter
registry automatically. The default experimental profile keeps the shared
target-context projection (`fc`) and each per-head attention gate (`g_proj`)
in affine Q8, applies searched affine Q4 to the larger Q/KV/O and MLP
projections, and uses DFlash's sanitizer to convert the source fused QKV and
separate gate/up tensors into the exact runtime module layout. The resulting
checkpoint is loadable by `--dflash-model`; no Python converter or sidecar is
involved.

Every newly converted DFlash artifact is conservatively labeled unbenchmarked
at creation time. A smaller drafter can be faster but reduce draft acceptance,
so retain the BF16 source and compare acceptance plus end-to-end tokens per
second before keeping a Q4R8 artifact. The specific full-checkpoint artifact
below was benchmarked after conversion; that evidence does not automatically
validate another checkpoint or target pairing.

The first full RTX 4090 validation is now recorded in
[`benchmark-results/dflash-quantizer-20260829`](../benchmark-results/dflash-quantizer-20260829/README.md).
The BF16-target drafter shrank from 882 MiB to 259 MiB and retained nearly all
observed acceptance. Block size 4 reached a 140.43 tok/s median versus 130.81
target-only, but intermittent 31–33 tok/s DFlash trials made its mean slower.
Keep the artifact for investigation; do not enable it by default yet.

The first post-scheduling-fix M5 Max probe is recorded in
[`benchmark-results/serving-optimizations-20260829`](../benchmark-results/serving-optimizations-20260829/README.md).
DFlash block 3 measured 157.36 tok/s versus 145.58 target-only (+8.09%) over
three 128-token trials per mode, but exact output still diverged at UTF-8 byte
109. A block-2 probe diverged at the same byte, so DFlash remains opt-in despite
the positive median.

This is not an arbitrary-format converter. It accepts an unquantized
safetensors text model whose `model_type` can be instantiated by the pinned
MLX Swift LLM registry, or a supported architecture from the drafter registry.
It rejects already-quantized input, PyTorch
`.bin`, GGUF, unknown/custom architectures that have not been registered, and
VLM-only model types. The generic path uses each registered model's own weight
sanitizer, including Mixtral's per-expert-to-`SwitchLinear` stacking.

For Laguna, retain the measured bounded-memory and fused-layout path by giving
the same binary a standard Q4R8 template:

```bash
.build/release/model-runner-quantize \
  /absolute/path/Laguna-XS-2.1-BF16 \
  /absolute/path/Laguna-XS-2.1-Q4R8-ScaleSearch-LS2 \
  --template /absolute/path/Laguna-XS-2.1-Q4R8-standard \
  --expert-batch 16
```

That profile preserves the template's 39 Q8 routers and standard-Q4 embedding,
fuses Laguna gate/up projections in the established order, validates source
identity, and streams the existing shard layout. A two-layer real-safetensors
Mixtral regression test verifies searched expert arrays, Q8 router overrides,
and the emitted mixed-precision configuration. A separate DFlash regression
starts from the official fused tensor conventions, converts to searched Q4/Q8,
then reloads the result through the native drafter registry.

### Native Laguna deployment

Laguna runs directly in Swift/MLX; no Python server or sidecar is involved. An
ordinary MLX checkpoint works as-is and its sparse expert gate/up rows are
fused lazily during loading:

```bash
MODEL_RUNNER_MLX_MEMORY_LIMIT_GIB=20 \
./run-cuda.sh rtx-4090 \
  --model /absolute/path/Laguna-XS-2.1-4bit \
  --name laguna-xs-2.1-4bit \
  --host 127.0.0.1 \
  --port 18081
```

For a compact deployment artifact with lower GPU residency, prepack the 39
sparse MoE layers offline, then serve the destination directory normally:

```bash
python3 Scripts/pack-laguna-gate-up.py \
  /absolute/path/Laguna-XS-2.1-4bit \
  /absolute/path/Laguna-XS-2.1-4bit-fused-gate-up-compact
```

The packer needs PyTorch and Safetensors, rewrites and verifies each weight
shard atomically, and refuses an existing destination. Python is not needed to
load or serve its output. On the project's RTX 4090, five 256-token trials had
a 136.08 tok/s median decode rate and 132.40 tok/s median end-to-end rate. The
compact artifact used 17,662 MiB of GPU memory after the trials, 944 MiB less
than the same fused runtime loading the original checkpoint.

### Native Laguna Q4R8 conversion

The project can create an MLX Q4R8 checkpoint directly in Swift: affine Q4
group-64 weights with every MoE router forced to affine Q8 group-64. An optional
policy file can promote calibration-selected modules. Routed gate/up rows are
fused before quantization and must share one precision.

Inspect module paths without loading or writing the BF16 weights:

```bash
Scripts/quantize-laguna-q4r8.sh \
  /absolute/path/Laguna-XS-2.1-bf16 \
  /absolute/path/Laguna-XS-2.1-mlx-q4r8 \
  --dry-run
```

Create a calibrated candidate with `--policy /absolute/path/policy.json`.
Pass `--q4-scale-search` to build an explicitly experimental same-format Q4
candidate; standard Q4 remains the fallback for every group and all routers
remain standard Q8. A measured ScalePlan bundle can instead select standard
Q4, searched Q4, or Q8 per indivisible module with `--scale-plan`.
Run `Scripts/audit-laguna-q4-scale-search.sh` first to measure the search on a
representative set of real BF16 Laguna tensors without writing a checkpoint.
The input must be an unquantized safetensors checkpoint, and conversion needs
memory headroom beyond the roughly 62 GB BF16 source. Pass `--cpu` on a
discrete-GPU conversion host when the source exceeds VRAM; this changes only
where conversion is evaluated, not the output format or serving engine.
Serving the resulting checkpoint remains entirely in the native Swift/MLX
runtime.

The converter has produced
`/home/example/models/Laguna-XS-2.1-MLX-Q4R8-v1` on the project server:
18,821,963,264 tensor bytes, four shards, Q4 group-64 plus exactly 39 Q8 routers.
A matched five-trial RTX 4090 run measured 126.93 tok/s median decode versus
127.45 tok/s for the prior compact Q4 control, so the new native conversion has
no measurable packing penalty. Extra Q8 modules remain opt-in until a
BF16-teacher calibration demonstrates an accuracy gain.

The completed same-format LS2 candidate is
`/home/example/models/Laguna-XS-2.1-MLX-Q4R8-ScaleSearch-LS2`. Its conversion
searches nine nearby Q4 scales, one least-squares bias update, and two joint
slope/intercept refinements while retaining ordinary MLX affine-Q4 arrays and
the 39 standard Q8 routers. A 48-tensor/411-million-value BF16 audit reduced
weighted reconstruction MSE by 15.0543%. Direct safetensors verification found
all 399 searched Q4 modules changed and all 320 preserved tensors exactly
identical. Balanced ten-trial CUDA medians were 126.815 tok/s standard versus
125.650 tok/s searched (-0.919%); this is an accuracy candidate, not a speed
optimization. Use `Scripts/rescore-laguna-q4r8.sh` to build from a standard
layout, `Scripts/verify-laguna-q4r8.sh` for exact integrity checks, and
`Scripts/benchmark-cuda-laguna-ab.sh` for matched CUDA A/Bs. The direct
`Scripts/benchmark-runtime-model.sh` harness records native runner metrics on
Metal or CUDA without HTTP overhead. A standalone run of the archived LS2
checkpoint on the local M5 Max measured 150.02 tok/s median across five
256-token trials (149.90-150.23 tok/s); the exact standard-checkpoint Metal A/B
was deferred, so this is not a relative speed claim. See
[ScalePlan-Q4R8](scaleplan-q4r8.md) for methods and evidence boundaries.

The Metal token scheduling work also improved stock Laguna Q4 decode on the
local M5 Max from a 115.99 tok/s median to 124.70 tok/s for the measured
roughly 990-token prompt and 100-token generation.

### Laguna DFlash speculative decoding

Greedy Laguna generation can use Poolside's DFlash checkpoint directly. The
drafter is loaded once, reuses the target embedding and output head, proposes
up to 15 tokens in one five-layer pass, and lets the 40-layer target verify the
block in one staged pass:

```bash
./run.sh \
  --model /absolute/path/Laguna-XS-2.1 \
  --dflash-model /absolute/path/Laguna-XS-2.1-DFlash \
  --dflash-block-size 16
```

Send `"temperature": 0` to select DFlash. Requests with a nonzero temperature
continue through ordinary target generation because the current verifier does
not yet implement probability-ratio rejection sampling; the runner does not
silently change sampling behavior. For an affine Q4R8 target, benchmark both
the BF16-trained and INT4-trained Poolside drafters—the latter is the more
plausible match, but acceptance rate and end-to-end throughput decide.

The runtime benchmark accepts the same options and records proposed and
accepted draft-token counts. Greedy verification batches all verifier argmaxes
into one realization per round, and target-only fallback retains the ordinary
Metal one-token pipeline. `--dflash-ab` also reports the first differing UTF-8
byte when target-only and DFlash output do not match:

```bash
swift run -c release model-runner-runtime-bench \
  /absolute/path/Laguna-XS-2.1 \
  /absolute/path/dflash-benchmark.json \
  --dflash-model /absolute/path/Laguna-XS-2.1-DFlash \
  --dflash-block-size 16
```

## Run

The repository, build product, and command are named `midnight`; the Swift
package and executable target use the conventional module spelling `Midnight`:

```bash
swift build --product midnight
swift run midnight --list-models
```

Load an MLX model and listen on a chosen local port:

```bash
./run.sh --model /absolute/path/to/mlx-model --host 127.0.0.1 --port 8080
```

When model loading finishes, connect an OpenAI-compatible client to
`http://127.0.0.1:8080/v1`. Change `--port` to any available port; it defaults
to `8080`.

Models are discovered by name under `~/.runner/models` (override with
`MODEL_RUNNER_MODELS_DIR`). A bundle such as
`~/.runner/models/gemma-4-e2b-it-cyber` may contain `base-model` and `adapter`
directories; `--model gemma-4-e2b-it-cyber` then loads both automatically and
uses the bundle folder name as the served model name. Absolute and relative
paths remain supported.

List every valid model directory without loading a model:

```bash
./run.sh --list-models
```

The OpenAI-compatible `GET /v1/models` endpoint returns only the model loaded by
the current process, because it is the only model clients can select for an
inference request. `GET /v1/models/{model}` retrieves that model's descriptor.

Chat completions support streaming and non-streaming requests, `developer`
messages, `max_tokens` or `max_completion_tokens`, `temperature`, `top_p`, and
up to four `stop` strings. Invalid parameters use the standard OpenAI error
shape with `message`, `type`, `param`, and `code` fields.

OpenAI chat and model responses contain only OpenAI-compatible fields. Local
performance details, including prompt and generated tokens per second, are
available through `--verbose` terminal logging and are not inserted into
response JSON.

Add `--verbose` to log request flow without printing prompt or tool contents:

```bash
./run.sh --model gemma-4-e2b-it-cyber --port 8080 --verbose
```

Verbose logs include a request ID, client address, HTTP method and path, body
size, generation settings, output counts, finish reason, elapsed time, and
errors. Prompt text and tool arguments remain redacted.

The shared `../model-stack.local.json` already contains the local Gemma path, served name, host, port, and token limit. Start it without arguments:

```bash
./run.sh
```

Settings are discovered automatically. `MODEL_STACK_CONFIG` or `--config PATH` can select another file, and individual command-line options override file values. `mlxRunner.maximumTokens` (or `--max-tokens`) is both the default when an HTTP request omits `max_tokens` and a hard per-request ceiling. Requests with a non-positive value or a value above that ceiling are rejected before streaming starts.

### Resource guard

The runner resolves, applies, and reads back conservative MLX allocator limits immediately before the Hugging Face model-container load. The process refuses to load the model if an override is malformed or outside its backend's bounds, or if MLX does not report the requested limits after they are applied.

| Backend | Default memory limit | Absolute memory limit | Default / maximum cache |
| --- | ---: | ---: | ---: |
| CUDA | 18 GiB | 20 GiB | 128 MiB |
| Metal | 24 GiB, clamped down when needed | the smaller of 32 GiB or half physical RAM | 256 MiB |
| CPU | 18 GiB, clamped down when needed | the smaller of 20 GiB or half physical RAM | 128 MiB |

These strict plain-decimal environment overrides are available:

```bash
# Examples that lower the CUDA defaults
MODEL_RUNNER_MLX_MEMORY_LIMIT_GIB=16 \
MODEL_RUNNER_MLX_CACHE_LIMIT_MIB=64 \
./run-cuda.sh rtx-4090
```

- `MODEL_RUNNER_MLX_MEMORY_LIMIT_GIB` must be greater than zero and no larger than the backend's absolute limit.
- `MODEL_RUNNER_MLX_CACHE_LIMIT_MIB` may be zero (which disables buffer recycling) and may not exceed either the backend cache maximum or the selected memory limit.
- Whitespace, signs, exponents, `nan`, `inf`, empty values, and values outside the permitted range are rejected before model loading.

On Metal, the runner also performs one isolated 513-token prefill after model
load, measures MLX's real peak active memory, and derives a wired-residency
ticket capped by both the allocator limit and Metal's recommended working set.
Completed requests can only raise this budget based on their observed peak.
Set `MODEL_RUNNER_MLX_WIRED_MEMORY=0` to disable this policy, or
`MODEL_RUNNER_MLX_WIRED_TUNE_TOKENS=128...4096` to change the calibration
length. A failed measurement is non-fatal and leaves generation unwired.

Native Laguna conversations retain a hot linear session and a bounded LRU of
immutable completed-message checkpoints for interleaved conversations and
branches. Defaults are four checkpoints and 2,048 MiB; set
`MODEL_RUNNER_PREFIX_CACHE_ENTRIES=0...64` and
`MODEL_RUNNER_PREFIX_CACHE_MIB=0...16384` to tune or disable it. This is exact
completed-transcript matching, not a promise of bit-identical floating-point
generation and not token-radix reuse across unrelated prompts. A short
140-token sibling-branch probe reused 111 tokens and, after one full warm-up,
reduced median TTFT from 180.95 ms cold to 86.16 ms cached over three trials.
The first unwarmed restore was slower than cold and output was not bit-identical,
so measure representative long prefixes and apply the required reproducibility
gate before treating the cache as a deployment win.

Mistral, Mistral 3/Ministral, and Mixtral chat checkpoints use the same exact
transcript reconciliation for the common append-only conversation path. Their
hot session is zero-copy; unlike Laguna, they do not create immutable branch
snapshots by default because copying a production KV cache after every turn can
erase the prefill win and sharply raise peak memory. Setting
`enablePromptCache` to false keeps the historical one-shot path. Custom stop
strings also stay one-shot.

Measure that hot append-only path against a forced-cold replay on one loaded
checkpoint with:

```bash
Scripts/benchmark-runtime-model.sh \
  /absolute/path/Mistral-or-Mixtral-checkpoint \
  /absolute/path/mistral-hot-cache-ab.json \
  --engine metal --tokens 512 --warmups 1 --trials 5 \
  --mistral-hot-cache-ab
```

Each measured continuation gets its own identical one-turn seed because a
forced-cold request intentionally clears the runner's hot session. The probe
alternates the cold-seed/cold and cached-seed/cached pair order, and fails if
the two seeds differ, the hot path does not reuse the seed's full prompt plus
generated token count, the cold path reuses any tokens, or prefill is not
reduced. Cached and cold numerical schedules can make greedy text diverge, so
that difference is nonfatal: the JSON comparison records
`outputs_match_exactly` and, when different, the earliest
`first_output_divergence_utf8_offset`. The comparison is stored separately as
`mistral_hot_cache_ab_comparison`;
Laguna's `--prompt-cache` sibling-branch benchmark and
`prompt_cache_comparison` remain unchanged. This A/B mode rejects
`--allow-early-stop` so an unreported consumed EOS token cannot invalidate its
exact cache-ledger assertion.

The pinned Swift backend additionally honors classic Mistral checkpoint
`layer_types` and `sliding_window` metadata. Explicit mixed full/sliding layouts
are preserved layer by layer; a non-null window without `layer_types` is treated
as the original all-sliding Mistral layout, while a null window remains full
attention. The existing Mistral 3 hybrid implementation and Laguna model are
unchanged. Older converted checkpoints whose `config.json` discarded the
source model's hybrid metadata must be reconverted or have that metadata
restored; the runner deliberately does not guess architecture from a folder or
repository name.

Mixtral also uses a pinned inference-only Metal specialization for the
single-token router top-k and selected-logit gather. It is enabled only for an
evaluation-mode model on the GPU; CPU execution, training, and multi-token
prefill retain the original MLX expression. Dependency tests
cover tied logits, zero-versus-negative ordering, exact selected values, and the
CPU fallback. This removes router dispatches but does not imply an end-to-end
tokens-per-second gain until a full Mixtral checkpoint is benchmarked.

This controls memory tracked by the MLX allocator; it is not an operating-system limit on the entire process. Use a Linux cgroup or systemd memory limit as an additional boundary when a hard whole-process host-RAM limit is required.

The `mlxRunner.engine` setting and `--engine` accept:

- `auto`: Metal in a macOS build, CUDA in a normal Linux build, or CPU in a CPU-only Linux build
- `metal`: require a macOS Metal build
- `cuda`: require a Linux CUDA build
- `cpu`: run inference on MLX's CPU device

To override the configured model explicitly:

```bash
./run.sh \
  --model ../models/gemma-4-e2b-it-4bit \
  --port 8080
```

By default, the endpoint model name is the model folder's name. Override it with `--name NAME`.

The server exposes:

- `GET /v1/models`
- `POST /v1/chat/completions` with `stream: true`, including OpenAI-compatible
  function-tool declarations, assistant `tool_calls`, and `tool` result
  messages

### Speech and voice API contracts

The same server also recognizes these routes directly under `/v1`; there is no
provider or server-mode switch:

- `POST /v1/audio/speech`
- `GET` and `POST /v1/audio/voices`
- `GET`, `PATCH`, and `DELETE /v1/audio/voices/{voice_id}`
- `GET /v1/audio/voices/{voice_id}/sample`

`POST /v1/audio/speech` accepts both official request dialects without changing
the existing OpenAI chat API. A request containing OpenAI's required `voice`
field uses the OpenAI speech contract. A request containing Mistral's
`voice_id`, `ref_audio`, `metadata`, `prompt_cache_key`, or `stream` fields uses
the Mistral contract. Mixing the two sets of fields is rejected.

Mistral non-streaming speech responses are JSON with base64 `audio_data`.
Streaming uses named `speech.audio.delta` and `speech.audio.done` server-sent
events, including the event `type` inside each JSON data object, and ends at
EOF without an OpenAI `[DONE]` marker. OpenAI's default `stream_format: audio`
uses a binary chunked response. Its `stream_format: sse` uses data-only
`speech.audio.delta` objects with base64 `audio` and a final
`speech.audio.done` object with `input_tokens`, `output_tokens`, and
`total_tokens`; it also ends at EOF rather than `[DONE]`. PCM encoding is kept
dialect-specific: Mistral uses 24-kHz Float32 little-endian samples, while
OpenAI uses 24-kHz signed Int16 little-endian samples.

The voice wire format, pagination, FastAPI-style validation errors, and
read-only preset discovery are implemented. Presets expose stable UUID
resource IDs while retaining their checkpoint names as `slug`; either value is
accepted when selecting a preset. Mistral voice creation is distinguished by
its JSON content type. OpenAI's multipart `POST /v1/audio/voices` is recognized
separately and receives an OpenAI-shaped unsupported-feature error rather than
being decoded as Mistral JSON. The open Voxtral checkpoint does not contain the
encoder weights needed for `ref_audio` or custom voice creation, and it does
not contain the original preset WAV samples. Those operations therefore return
explicit errors; preset voices cannot be patched or deleted.

The contained native Swift/MLX Voxtral generator loads the language backbone,
flow-matching acoustic transformer, codebook feedback embeddings, and codec
decoder directly from the checkpoint. It does not invoke Python or a hosted
API. Start it with:

```bash
./run.sh \
  --model voxtral-4b-tts \
  --host 127.0.0.1 \
  --port 8080 \
  --max-tokens 512 \
  --verbose
```

For Voxtral, `--max-tokens` is the hard maximum number of generated 80-ms audio
frames; generation normally stops earlier when the model emits end-of-audio.
This first native slice supports 24-kHz mono WAV and PCM. OpenAI defaults
`response_format` to `mp3`, so select `wav` or `pcm` explicitly:

```bash
curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "voxtral-4b-tts",
    "input": "Hello from native Swift.",
    "voice": "casual_male",
    "response_format": "wav"
  }' \
  --output speech.wav
```

Use `GET /v1/audio/voices` to list the checkpoint's preset voices. Speed is
currently fixed at `1.0`; instructions, reference-audio cloning, custom voice
creation, and compressed output formats remain unavailable for this checkpoint
path.

For the in-process MLX backend, tools are passed to both the model's chat
template and MLX Swift LM's tool-call parser. Accepted calls stream in
`choices[0].delta.tool_calls`, and the terminal chunk uses
`finish_reason: "tool_calls"`. The runner proposes calls but never executes
them: the client (for example, the neighboring SwiftTUI app's MCP client) must
invoke the selected tool, append the assistant message with its `tool_calls`,
append a `role: "tool"` result with the matching `tool_call_id`, and send the
expanded conversation back for the model's final answer.

The request shape is the normal OpenAI function-tool form:

```json
{
  "model": "gemma-4-e2b-it-4bit",
  "stream": true,
  "messages": [{"role": "user", "content": "What is the weather in Boston?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Read current weather for a city",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }]
}
```

Then start any compatible client, including the neighboring SwiftTUI app:

```bash
cd ../model-chat-swifttui
./run.sh --endpoint http://127.0.0.1:8080/v1 --model gemma-4-e2b-it-4bit
```

On macOS, the first build compiles MLX Metal shaders. If the Metal compiler is unavailable, install it with:

```bash
xcodebuild -downloadComponent MetalToolchain
```

You can require Metal explicitly:

```bash
./run.sh --engine metal
```

## CUDA on Linux

On Linux, `./run.sh` builds the same runner with MLX's CUDA backend. Building directly on the machine with the default `native` profile is preferred:

First create the local settings file and replace its placeholder model path/name with the absolute path and served name on the GPU host:

```bash
cp ../model-stack.example.json ../model-stack.local.json
```

You can instead skip the settings file and append `--model /absolute/path/to/your/mlx-model` (and, if needed, `--name your-model-name`) to any run command below.

```bash
./run-cuda.sh
```

Named profiles are provided for both target machines:

```bash
# NVIDIA DGX Spark / GB10 / compute capability 12.1 / CUDA 13
./run-cuda.sh dgx-spark

# NVIDIA GeForce RTX 4090 / compute capability 8.9
./run-cuda.sh rtx-4090
```

The corresponding build-only commands are `./build-cuda.sh`, `./build-cuda.sh dgx-spark`, and `./build-cuda.sh rtx-4090`. The profiles set `CUDA_ARCH` to `native`, `sm_121`, and `sm_89` respectively. The build script checks that nvcc supports a selected named architecture. Named-profile cache identity includes the resolved nvcc path and version, the selected architecture, the resolved `/usr/local/cuda` toolkit path, and the resolved roots, include paths, versions, release-evidence fingerprints, and representative-header fingerprints for both pinned header dependencies. Because MLX's Swift build plugin does not reliably fingerprint the GPU resolved by `CUDA_ARCH=native`, the native profile starts from a clean Swift package build each time.

The pinned Linux MLX-Swift revision makes ordinary streams thread-affine, while
SwiftNIO handlers and Swift tasks may resume on another operating-system
thread. An exact-revision-gated experimental `new_thread_unsafe_stream` bridge
is retained only as an explicit diagnostic opt-in. Linux defaults to the clean
patched checkout (`mlx-cross-thread-stream-overlay=on`), fails closed on
source drift, and removes the exact overlay if a prior opt-in left it applied.
Darwin uses the pinned official MLX-Swift 0.32 update revision, synchronized
with MLX 0.32.2 and MLX-C; Linux retains its separately verified CUDA revision.
The conditional package manifest selects the pinned CUDA compatibility revision
on Linux, so Metal and CUDA now build from this same canonical source tree.

Dependency preparation also applies the Gemma 4 non-rotating-cache patch from
the CUDA host. It keeps full K/V storage while leaving Gemma 4's sliding-window
attention mask intact, avoiding corrupted generation when the former rotating
cache wrapped at 512 tokens.

The verified RTX 4090 runner was built with that overlay off. A bounded
Qwen2.5 0.5B 4-bit test now loads, reaches health, returns the exact model
identity, and completes a streamed request. The earlier pre-health overlay
experiment remains useful historical evidence, but it is not the configuration
published by the current stable runner.

Linux builds produce the optimized `release` executable and default to two
compile jobs to avoid large parallel compiler memory spikes. Override the job
count only when the host has enough headroom:

```bash
SWIFT_BUILD_JOBS=1 ./build-cuda.sh rtx-4090
```

`SWIFT_BUILD_JOBS` must be a positive integer.

The resulting executable is under the selected SwiftPM scratch tree at
`release/midnight`. With the default scratch tree, the build verifies that
SwiftPM's standard compatibility path resolves to that same regular executable:

```text
/path/to/midnight/.build/release/midnight
```

To keep a Linux build completely separate from the package's default `.build`
tree, select a dedicated SwiftPM scratch directory:

```bash
MODEL_RUNNER_SCRATCH_PATH="$HOME/.cache/midnight/rtx-4090" \
./build-cuda.sh rtx-4090
```

The same setting must be present when using `./run-cuda.sh`; the wrapper carries
it consistently through dependency resolution/patching, build and clean
commands, the backend profile marker, and executable lookup. Relative paths are
resolved below the package root. The package root, home directory, `/tmp`, and
`/var/tmp` are rejected as scratch roots; use a dedicated child directory.
When `MODEL_RUNNER_SCRATCH_PATH` is unset, all existing `.build` behavior is
unchanged.

For an isolated named `rtx-4090` build, the post-build publisher verifies the
custom scratch tree's one-line release profile marker, requires the produced
binary to be a regular executable inside that scratch tree, hashes it, and then
atomically installs this stable copy:

```text
bin/midnight-rtx4090
```

It writes `bin/midnight-rtx4090.manifest` with the full build profile,
source scratch/binary paths, destination paths, size, and SHA-256. The expected
`.build/release/midnight` path is an absolute symlink to that verified
stable copy, so it is never an unexplained duplicate in the default SwiftPM
cache. `run-cuda.sh rtx-4090` verifies the manifest, profile provenance, hash,
and symlink before executing it. The publisher refuses a missing or
non-executable source, a non-release or non-`sm_89` profile, a missing/mismatched
profile marker, a changed prior hash, and any existing unmanaged file or link at
the compatibility path. Conversely, a default-scratch build requires its
SwiftPM product to be a regular executable and verifies that
`.build/release/midnight` resolves to that exact product, so the managed
custom publication cannot be mistaken for a default-cache build.

The stable publication is intentionally specific to the named `rtx-4090`
profile. Other custom-scratch profiles continue to execute directly from their
selected scratch tree, so a DGX Spark or CPU artifact cannot silently acquire a
4090 filename.

Requirements:

- Linux and Swift 6.3
- an NVIDIA driver and CUDA toolkit supported by MLX
- CUDA installed at `/usr/local/cuda`
- Clang 18, 19, or 20 supported by that CUDA toolkit (`clang-18` is available
  on Ubuntu 24.04 and is the pinned default)
- cuDNN, cuBLAS, BLAS, LAPACK, OpenBLAS, and gfortran development libraries
- NVIDIA `cudnn-frontend` v1.16.0 (Ubuntu's `libcudnn-frontend-dev` 0.9.2 is too old for this pinned MLX checkout)
- NVIDIA CUTLASS v4.3.5, including its CuTe headers
- CUDA 13 for the DGX Spark `sm_121` profile

By default, the CUDA build looks for the pinned user-local headers at:

```text
~/.local/opt/cudnn-frontend-v1.16.0/include/cudnn_frontend.h
~/.local/opt/cutlass-v4.3.5/include/cute/numeric/numeric_types.hpp
```

If either dependency is installed elsewhere, point the build at its root or its include directory:

```bash
CUDNN_FRONTEND_ROOT=/absolute/path/to/cudnn-frontend \
CUTLASS_ROOT=/absolute/path/to/cutlass \
./build-cuda.sh rtx-4090
```

`CUDNN_FRONTEND_INCLUDE_DIR` and `CUTLASS_INCLUDE_DIR` remain available when the headers are not under `<root>/include`; when a root and include override are both provided, the resolved include must belong to that same root. The selected directories are prepended to `CPATH` and passed explicitly to the MLX CUDA build plugin through `MLX_CUDA_INCLUDE_PATHS`, so incompatible system-packaged headers cannot take precedence in either compilation stage.

Those environment include lists govern the build plugin, but pinned MLX's
NVRTC runtime compiler does not read either `CPATH` or
`MLX_CUDA_INCLUDE_PATHS`. It discovers auxiliary headers relative to the
executable. An RTX publication therefore copies the already-validated
CUTLASS 4.3.5 `cutlass/` and `cute/` trees into this project's managed
`include/` directory before publishing the binary. The publisher does not
retain links, rejects links or special files in either tree, records SHA-256 for
both complete trees and the five release-defining headers, and will neither
repair nor overwrite an unmanaged, partial, or drifted prior tree.

An existing stable runner can receive only this runtime-header companion,
without rebuilding or modifying its binary and release manifest:

```bash
bash Scripts/cuda-runtime-environment.sh --publish-only
```

The same helper can launch an arbitrary command after resolving the exact CUDA
toolkit, libraries, Clang host compiler, cudnn-frontend, and CUTLASS inputs:

```bash
bash Scripts/cuda-runtime-environment.sh -- /absolute/path/to/command arg
```

The CUDA build fails before compilation unless it can prove exact `cudnn-frontend` v1.16.0 from public version-header macros, an exact `VERSION` file, matching CMake release metadata, or an exact Git tag. A version-looking directory name is not sufficient. CUTLASS must provide its full CUTLASS/CuTe header set and declare exactly 4.3.5 in `cutlass/version.h`. Resolved roots/include paths and content fingerprints for both dependencies are included in the build-cache profile identity, so replacing headers in place invalidates the prior Swift build cache.

The Linux wrapper selects `/usr/bin/clang++-18` as the CUDA host C++ compiler.
This avoids the Clang 21 bundled with Swift 6.3, which CUDA 13.0 rejects. It
also intentionally rejects GCC: `encuda` asks NVCC to emit host C++, and GCC 13
leaves glibc 2.39's native `_FloatN` C++ types in that generated source. SwiftPM
then compiles the generated source with Swift Clang 21, which cannot parse those
types. Clang 18-20 selects glibc's compatible typedef path, so the generated
source remains valid in the second stage. Override the default only with an
absolute Clang 18-20 executable path; `MLX_CUDA_HOST_CXX` takes precedence over
the conventional `CUDAHOSTCXX` alias:

```bash
MLX_CUDA_HOST_CXX=/usr/bin/clang++-18 ./build-cuda.sh rtx-4090
```

Empty, relative, missing, directory, non-executable, non-Clang, Clang 17-or-old,
and Clang 21-or-new values fail before SwiftPM starts compiling. The selected
compiler is resolved and exported as `MLX_CUDA_HOST_CXX`. The project pins the
official upstream MLX-Swift 0.32 compatibility PR and applies a reproducible
Linux CUDA overlay that passes that same path to both `encuda compile` and
`encuda link`; the plugin's original bundled
`clang++` lookup remains the fallback only when neither environment variable is
set. The resolved compiler path, family, major version, and `--version`
fingerprint are part of the backend build profile, so a compiler change
invalidates the old build cache.

The x86-64 `rtx-4090` profile builds on the target NVIDIA host. Rebuild it after
the executable rename to publish `bin/midnight-rtx4090` with a fresh manifest
and SHA-256; `.build/release/midnight` will be the verified symlink to that
stable copy.
The managed runtime companion contains complete, SHA-256-verified
CUTLASS 4.3.5 `cutlass/` and `cute/` trees at the exact runner-relative path
used by MLX's NVRTC compiler.

The capped Qwen CUDA smoke run `20260825T134830Z-509715-20351` passed: load to
readiness took 1.035 seconds, `/v1/models` and streamed chat were
HTTP 200, and the chat completed in 19.896 seconds. The transient unit peaked
at 4,629,516,288 bytes of host memory; the combined GPU-used timeline peaked at
13,313 MiB while the separately protected GPU service remained present. No
fatal/runtime-compiler/OOM signature was found, the unit and listener were
gone afterward, and the stable runner and release-manifest hashes were
unchanged.

The RTX 4090/Qwen path has therefore crossed the MLX-CUDA runtime smoke gate.
The DGX Spark `sm_121` profile remains experimental and has not received the
same build or runtime validation; the Qwen result also does not establish that
every MLX architecture is supported.

The pinned MLX Swift release exposes CUDA on Linux, and its upstream CUDA CI is useful evidence that the backend can build. That CI currently exercises an x86-64 Linux CUDA CMake build, however; it does not validate this runner's SwiftPM dependency/plugin path or executable. DGX Spark adds a separate ARM64 host, CUDA 13, and `sm_121` combination that upstream CUDA CI does not cover. The runner performs no model conversion: it reads the same MLX-compatible config, tokenizer, and `.safetensors` folder on either backend, subject to the model operations currently supported by MLX CUDA.

For a CPU-only Linux binary that does not link against CUDA:

```bash
SPM_CUDA=0 ./run.sh --engine cpu
```

Selecting `--engine cpu` inside a CUDA build executes on CPU, but that executable still links CUDA libraries. Use the CPU-only build command above for a machine without CUDA.

## Bounded CUDA smoke test

After building on the Linux GPU host, run exactly one local-model request through
the fail-closed smoke harness. The model folder and served name must be supplied
explicitly; the harness never discovers them from the shared settings file:

```bash
./Scripts/smoke-cuda-model.sh \
  --model "$HOME/models/Qwen2.5-0.5B-Instruct-4bit" \
  --name qwen-smoke
```

The default `qwen` profile limits MLX to 8 GiB, its cache to 64 MiB, the whole
systemd user unit to 16 GiB of host RAM, swap to 1 GiB, and the only generation
to 96 tokens. The larger profile is an explicit opt-in:

```bash
./Scripts/smoke-cuda-model.sh \
  --profile gemma \
  --model "$HOME/models/gemma-4-e2b-it-bf16" \
  --name gemma-4-e2b-smoke
```

| Profile | MLX memory | MLX cache | Host MemoryHigh / MemoryMax | Runtime | CPU quota | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen` | 8 GiB | 64 MiB | 12 / 16 GiB | 600 s | 200% | 96 |
| `gemma` | 18 GiB | 128 MiB | 24 / 32 GiB | 1500 s | 400% | 96 |

Every resource value may be lowered, but hard ceilings cannot be raised:

- `MODEL_RUNNER_SMOKE_MLX_MEMORY_GIB` (maximum 20)
- `MODEL_RUNNER_SMOKE_MLX_CACHE_MIB` (maximum 128)
- `MODEL_RUNNER_SMOKE_HOST_MEMORY_HIGH_GIB` and
  `MODEL_RUNNER_SMOKE_HOST_MEMORY_MAX_GIB` (maximum 32)
- `MODEL_RUNNER_SMOKE_SWAP_MAX_GIB` (maximum 2)
- `MODEL_RUNNER_SMOKE_RUNTIME_SECONDS` (maximum 1800)
- `MODEL_RUNNER_SMOKE_HEALTH_TIMEOUT_SECONDS` (maximum 900)
- `MODEL_RUNNER_SMOKE_HTTP_TIMEOUT_SECONDS` (maximum 600)
- `MODEL_RUNNER_SMOKE_MAX_TOKENS` (maximum 96)
- `MODEL_RUNNER_SMOKE_PORT` (1024 through 65535)
- `MODEL_RUNNER_SMOKE_CPU_QUOTA_PERCENT` (maximum 400)

By default, the harness refuses to start if any `midnight` process, NVIDIA
compute process, second visible GPU, or listener on the selected port already
exists. A fail-closed opt-in permits exactly one explicitly protected compute
PID to coexist:

```bash
PROTECTED_PID=12345
PROTECTED_CMDLINE_SHA256="$(sha256sum "/proc/$PROTECTED_PID/cmdline" | awk '{print $1}')"
MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID="$PROTECTED_PID" \
MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_CMDLINE_SHA256="$PROTECTED_CMDLINE_SHA256" \
./Scripts/smoke-cuda-model.sh \
  --model "$HOME/models/Qwen2.5-0.5B-Instruct-4bit" \
  --name qwen-smoke
```

Both authorization variables must be present or both absent. Each preflight
revalidates the positive PID, the exact SHA-256 of its raw `/proc/PID/cmdline`,
and that every NVIDIA compute row belongs to that PID; any other row fails the
run. With or without this opt-in, free VRAM must be at least the configured MLX
memory cap plus a 2 GiB reserve. The harness checks again immediately before
launch. The transient service always binds to
`127.0.0.1`, runs under a uniquely generated systemd user unit with
`MemoryHigh`, `MemoryMax`, `MemorySwapMax`, `RuntimeMaxSec`, and a capped
`CPUQuota`, and receives the same MLX allocator limits. Before launching the
model, the harness re-runs the exact CUDA/dependency/host-compiler resolvers
and resolves the selected runner through every symlink. It proceeds only when
`dirname(realpath(runner))/../include` is exactly this project's managed
`include/`; the verified compatibility link to `bin/midnight-rtx4090`
passes, while a SwiftPM scratch binary or arbitrary external `--runner` fails.
Only after that check does it atomically publish a missing CUTLASS/CuTe
companion and verify every recorded SHA-256. It never publishes into an
arbitrary runner parent. The harness then passes the resolved CUDA bin/library,
`CUDA_HOME`, compiler, and build-context variables explicitly into the
transient unit; these variables keep toolchain selection deterministic, while
the runner-relative copies are what satisfy MLX runtime JIT header discovery.
It waits for `/v1/models`,
then verifies the exact line-buffered `MLX resource guard: memory=... cache=...`
startup record in the unit journal before it verifies the exact served name
through `/v1/models`, and sends one benign streaming `/v1/chat/completions`
request. Cleanup stops only the unit name created by that invocation and, after
verifying its parent PID and that it is not the protected PID, terminates only
its own one-second sampler PID. It never signals the protected process and never
uses process-wide `pkill` or `killall`.

Results are written below `.smoke-results/<run-id>/`: load-to-health timing,
per-request curl timings, request/SSE response, the verified MLX guard line,
systemd journal, cgroup process memory, and a one-second host-memory/GPU-VRAM
timeline spanning model load through chat. Override the parent with
`MODEL_RUNNER_SMOKE_OUTPUT_ROOT`. `--runner` remains available for an explicit
binary or link whose real path is inside this project's `bin/` directory; it
does not authorize a different runtime-header root. When the verified RTX 4090
publication manifest is present, the harness validates and uses the exact
`.build/release/midnight` compatibility link without needing the custom
scratch variable to be repeated.

## Use a remote GPU from the Mac client

The safe default binds the runner only to `127.0.0.1`. Keep that setting on the DGX Spark or 4090 host and create an SSH tunnel from the Mac:

```bash
ssh -L 8080:127.0.0.1:8080 USER@GPU-HOST
```

The existing Mac chat setting can then stay at `http://127.0.0.1:8080/v1`. Binding the runner to `0.0.0.0` is also possible with `--host 0.0.0.0`, but this minimal server does not yet authenticate inbound requests; expose it only on a trusted, firewalled network.
