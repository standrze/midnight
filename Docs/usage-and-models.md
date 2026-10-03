# Usage and model details

## Chat with the model

For a local visual layer explorer, use [Midnight Studio](../../midnight-studio/README.md),
a Hummingbird web app that shows loaded modules and tensor shapes, compares
model structures, and captures bounded per-token residual activations on
supported architectures.

[Midnight Activation Capture](activation-capture.md) lets any consumer request
selected layers, token positions, and bounded raw residual vectors. Consumers
include midnight-moonshine, ABSlayer, evaluation, and analysis tools. Discover
support and limits with `GET /v1/inspector/model`, then request an explicit capture
with `POST /v1/inspector/trace`. These are Midnight-specific API operations.
Activation Capture installs temporary observers only for an explicit capture and
restores the original model before ordinary generation resumes. Current captures
are bounded JSON operations with at most 256 rendered prompt tokens and 64
generated tokens; supported models and observation points must be discovered
before capture.

The examples assume `MIDNIGHT_API_KEY` is unset on the server. If a key is
configured, add `-H "Authorization: Bearer $MIDNIGHT_API_KEY"` to each request.

```bash
curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ministral-local",
    "messages": [
      {"role": "user", "content": "Explain affine 4-bit quantization."}
    ],
    "temperature": 0,
    "max_tokens": 256
  }'
```

Set `"stream": true` for data-only server-sent events ending in
`data: [DONE]`. Midnight proposes function calls but never executes tools on
the host; tool execution remains the client's responsibility.

Use OpenAI's `response_format` with `json_object` or `json_schema` for constrained
JSON generation. See [structured output](structured-output.md) for an example,
supported schemas, streaming, and token-limit behavior.

OpenAI SDK clients can also use `client.responses.create()` at `/v1/responses`,
including `text.format`, semantic streaming events, function calls, and
`previous_response_id` continuation. See the [Responses API guide](responses-api.md)
for examples, local history retention, and supported features.

## Generate speech

Serve a Voxtral TTS checkpoint (`model_type=voxtral_tts`) in its own Midnight
process:

```bash
./run.sh \
  --model /absolute/path/to/voxtral-tts \
  --name voxtral-local \
  --port 8081 \
  --max-tokens 512
```

List the checkpoint's voices, then request WAV output:

```bash
curl --silent http://127.0.0.1:8081/v1/audio/voices

curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8081/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "voxtral-local",
    "input": "Hello from Midnight.",
    "voice": "casual_male",
    "response_format": "wav"
  }' \
  --output speech.wav
```

The endpoint accepts the implemented OpenAI and Mistral speech dialects.
Voxtral uses preset voices and uncompressed 24 kHz WAV/PCM. Qwen3-TTS
CustomVoice also uses preset speakers, while Qwen3-TTS Base and VibeVoice accept
request-time reference audio through the Mistral dialect. Qwen3-TTS Base also
requires the reference transcript in `ref_text`. Speech speed remains fixed;
compressed output and persistent custom-voice creation are unavailable.

## Model focus

| Family | Runtime path | Status |
| --- | --- | --- |
| Mistral / Ministral / Mixtral / Codestral / Devstral text | Native Swift + MLX | Primary; native Mistral-family prompt caching |
| Voxtral TTS | Native Swift + MLX speech stack | Primary; preset voices |
| Qwen3-TTS | Native Swift + MLX speech stack on Metal | Base reference conditioning; CustomVoice preset speakers |
| VibeVoice 1.5B / 7B | Optional local Python/PyTorch worker | Speech and request-time reference conditioning; Metal or CPU |
| Poolside Laguna | Native hybrid-attention/MoE implementation | Primary |
| GPT-OSS | Upstream MLX Swift architecture | Supported |
| Other MLX Swift LM families | Upstream architecture when compatible | Best effort |

“OpenAI-compatible” describes the local HTTP contract, not the model
architecture. The server loads one checkpoint per process and serializes
generation through that model.

Codestral and the text-only Devstral releases use the native Mistral-family
path when their `config.json` declares a Mistral, Mistral 3, or Ministral 3
text architecture. This includes MLX conversions: routing is based on
checkpoint metadata, not the repository or folder name. Multimodal Devstral
releases need a compatible image runtime for image inputs; use their text-only
checkpoint when serving code/chat requests through Midnight today.

## Quantization

The architecture-aware Swift quantizer is now [Wick](../../wick/README.md),
an independent project with its own model adapters, build scripts, and dependency
patches. It no longer requires Midnight. From this directory, build it with:

```bash
WICK_BUILD_CONFIGURATION=release \
WICK_BUILD_PRODUCT=wick \
../wick/build.sh
```

Create a ScaleSearch affine-Q4 checkpoint without changing the standard MLX
Q4 storage layout or generation kernels:

```bash
../wick/.build/release/wick \
  /absolute/path/to/bf16-model \
  /absolute/path/to/q4-scalesearch-model
```

Exports materialize Hugging Face snapshot metadata as regular files, so moving
the model or removing the source cache does not break its configuration or
tokenizer. The converter writes both quantization config fields with the actual
per-module overrides and fails if it cannot read or update the output config.

Wick retains the legacy `model-runner-quantize` executable for existing scripts.
Conversions stage their output before replacing an existing checkpoint.

Wick also includes an experimental activation-weighted second pass for
dense Mistral matrices. AWSS chooses affine grids using observed input-channel
second moments and can veto calibration gains that fail on a separate dev set.
It retains the same Q4 format, group size, tensor payload geometry, and runtime
kernels.
See the [complete command reference](reference.md#standalone-swift-scalesearch-quantizer)
and the [AWSS experiment](../benchmark-results/ministral3-awss-20260830/README.md).

## Measured results

The repository keeps raw JSON alongside each report. These are bounded
same-machine measurements, not universal performance claims.

| Experiment | Observed result | Important limitation |
| --- | --- | --- |
| [Ministral 3 14B AWSS](../benchmark-results/ministral3-awss-20260830/README.md) | 45.76% lower exact BF16-teacher KL than ScaleSearch LS2 on 30,118 held-out tokens, with identical Q4 tensor payload geometry | A separate 698-token authored smoke set regressed; AWSS remains experimental |
| [Native MLX vs Ollama Q4_K_M](../benchmark-results/ministral3-ollama-20260830/README.md) | 67.995 vs 50.287 median decode tok/s on one 64 GB M5 Max workload (+35.21%) | This measures the whole package/runtime path, not a ScaleSearch-specific speedup |

The benchmark archive also records Laguna Q4R8, compiled-MoE, DFlash, Metal,
and RTX 4090 work. Start with [runtime and model research](runtime-model-research.md)
or browse [`benchmark-results`](../benchmark-results).

## API surface

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/v1/models` | Preflight-valid installed models; `loaded` marks the resident selection |
| `GET` | `/v1/models/{model}` | Descriptor for an installed, available model |
| `GET` | `/v1/inspector/model` | Activation Capture model structure, support, and limits (Midnight-specific) |
| `POST` | `/v1/inspector/trace` | Bounded Activation Capture at selected layers and token positions (Midnight-specific) |
| `POST` | `/v1/chat/completions` | Chat and client-executed tool calls |
| `POST` | `/v1/responses` | OpenAI Responses with text, structured output, tools, and streaming |
| `GET` / `DELETE` | `/v1/responses/{response_id}` | Retrieve or delete a locally stored response |
| `GET` | `/v1/responses/{response_id}/input_items` | List and paginate the response's input history |
| `POST` | `/v1/audio/speech` | OpenAI/Mistral speech requests |
| `GET` | `/v1/audio/voices` | Preset voice discovery |
| `GET` | `/v1/audio/voices/{voice_id}` | Preset voice metadata |

Additional voice-management routes are recognized but currently return
explicit unsupported or read-only errors. The
[complete reference](reference.md#speech-and-voice-api-contracts)
documents request dialects, streaming, formats, limits, configuration, LoRA,
resource guards, and smoke tests.

## Linux and CUDA

Linux builds use the same package with a pinned MLX CUDA integration. Build on
the target GPU when possible:

```bash
./run-cuda.sh native \
  --model /absolute/path/to/mlx-model \
  --name local-model \
  --port 8080
```

The `rtx-4090` profile is validated. A `dgx-spark` profile is available but
remains experimental. CUDA deployment has stricter toolchain and header
requirements; read the [Linux/CUDA reference](reference.md#cuda-on-linux)
before building. A CPU-only Linux build is available with `SPM_CUDA=0`.
