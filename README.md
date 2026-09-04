# Midnight

**Native Swift/MLX inference, tuned for Mistral, Voxtral, and Poolside Laguna.**

[![Swift 6.3](https://img.shields.io/badge/Swift-6.3-F05138?logo=swift&logoColor=white)](https://www.swift.org/)
[![Apple silicon](https://img.shields.io/badge/Apple_silicon-Metal-111111?logo=apple)](https://github.com/ml-explore/mlx-swift)
[![Linux](https://img.shields.io/badge/Linux-CUDA-FCC624?logo=linux&logoColor=111111)](Docs/reference.md#cuda-on-linux)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

Midnight is a standalone local model server for MLX `safetensors` checkpoints.
It loads the model directly into a Swift process and serves an OpenAI-compatible
subset for chat and speech over Metal on macOS, CUDA on Linux, or MLX's CPU
backend. There is no Python inference server or hosted dependency in the
serving path.

> **Beta:** Midnight is prerelease software. CLI, API, and runtime behavior may
> change before the first stable release.

## Memory and context controls

Version **0.2.0-beta.1** adds request memory admission, device-aware Metal
budgets, shared conversation-cache accounting, configurable chunked prefill,
and opt-in KV compression. The Mac launcher now uses release by default.
See [memory and long-context operation](Docs/memory-and-context.md) for settings
and experimental-feature limits.

## What is included

- Native Mistral-family text inference with append-only conversation-prefix
  caching and hybrid-attention support.
- Native Voxtral text-to-speech with 24 kHz WAV/PCM output and checkpoint
  preset voices.
- Native Poolside Laguna hybrid-attention/MoE execution, including the existing
  fused expert path, compiled graph optimizations, mixed Q4R8 checkpoints, and
  opt-in greedy DFlash speculative decoding.
- Model discovery, streaming and non-streaming chat, function-tool calls,
  speech, and voice discovery through a focused OpenAI-compatible HTTP surface.
- Swift quantization and evaluation tools for ordinary affine Q4, ScaleSearch,
  activation-weighted ScaleSearch (AWSS), teacher-KL, NLL, and runtime tests.
- One source tree for Apple-silicon Metal and Linux CUDA deployments.

## Quick start on Apple silicon

Requirements: macOS 15 or newer, Swift 6.3, and the Xcode command-line tools.
The first build resolves pinned dependencies and compiles the MLX Metal library.
If the Metal compiler is missing, the script prints the one-time installation
command.

```bash
git clone https://github.com/standrze/midnight.git
cd midnight

./run.sh \
  --model /absolute/path/to/mlx-model \
  --name ministral-local \
  --host 127.0.0.1 \
  --port 8080
```

The model directory must contain an MLX-compatible `config.json`, tokenizer,
and `.safetensors` weights. Midnight also discovers named models under
`~/.runner/models`:

```bash
./run.sh --list-models
```

## Chat with the model

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

## Generate speech with Voxtral

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
Current open-checkpoint support is limited to preset voices and uncompressed
24 kHz WAV/PCM; reference-audio cloning, custom voice creation, speed changes,
and compressed output are not yet available.

## Model focus

| Family | Runtime path | Status |
| --- | --- | --- |
| Mistral / Ministral / Mixtral | Native Swift + MLX | Primary |
| Voxtral TTS | Native Swift + MLX speech stack | Primary; preset voices |
| Poolside Laguna | Native hybrid-attention/MoE implementation | Primary |
| GPT-OSS | Upstream MLX Swift architecture | Supported |
| Other MLX Swift LM families | Upstream architecture when compatible | Best effort |

“OpenAI-compatible” describes the local HTTP contract, not the model
architecture. The server loads one checkpoint per process and serializes
generation through that model.

## Quantization

Build the architecture-aware Swift quantizer:

```bash
MODEL_RUNNER_BUILD_CONFIGURATION=release \
MODEL_RUNNER_BUILD_PRODUCT=model-runner-quantize \
./build.sh
```

Create a ScaleSearch affine-Q4 checkpoint without changing the standard MLX
Q4 storage layout or generation kernels:

```bash
.build/release/model-runner-quantize \
  /absolute/path/to/bf16-model \
  /absolute/path/to/q4-scalesearch-model
```

Midnight also includes an experimental activation-weighted second pass for
dense Mistral matrices. AWSS chooses affine grids using observed input-channel
second moments and can veto calibration gains that fail on a separate dev set.
It retains the same Q4 format, group size, tensor payload geometry, and runtime
kernels.
See the [complete command reference](Docs/reference.md#standalone-swift-scalesearch-quantizer)
and the [AWSS experiment](benchmark-results/ministral3-awss-20260830/README.md).

## Measured results

The repository keeps raw JSON alongside each report. These are bounded
same-machine measurements, not universal performance claims.

| Experiment | Observed result | Important limitation |
| --- | --- | --- |
| [Ministral 3 14B AWSS](benchmark-results/ministral3-awss-20260830/README.md) | 45.76% lower exact BF16-teacher KL than ScaleSearch LS2 on 30,118 held-out tokens, with identical Q4 tensor payload geometry | A separate 698-token authored smoke set regressed; AWSS remains experimental |
| [Native MLX vs Ollama Q4_K_M](benchmark-results/ministral3-ollama-20260830/README.md) | 67.995 vs 50.287 median decode tok/s on one 64 GB M5 Max workload (+35.21%) | This measures the whole package/runtime path, not a ScaleSearch-specific speedup |

The benchmark archive also records Laguna Q4R8, compiled-MoE, DFlash, Metal,
and RTX 4090 work. Start with [runtime and model research](Docs/runtime-model-research.md)
or browse [`benchmark-results`](benchmark-results).

## API surface

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/v1/models` | Loaded model descriptor |
| `GET` | `/v1/models/{model}` | Descriptor for the loaded model |
| `POST` | `/v1/chat/completions` | Chat and client-executed tool calls |
| `POST` | `/v1/audio/speech` | OpenAI/Mistral speech requests |
| `GET` | `/v1/audio/voices` | Preset voice discovery |
| `GET` | `/v1/audio/voices/{voice_id}` | Preset voice metadata |

Additional voice-management routes are recognized but currently return
explicit unsupported or read-only errors. The
[complete reference](Docs/reference.md#speech-and-voice-api-contracts)
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
requirements; read the [Linux/CUDA reference](Docs/reference.md#cuda-on-linux)
before building. A CPU-only Linux build is available with `SPM_CUDA=0`.

## Repository map

| Path | Contents |
| --- | --- |
| [`Sources/ModelRunner`](Sources/ModelRunner) | CLI and HTTP server |
| [`Sources/ModelRunnerCore`](Sources/ModelRunnerCore) | Model loading, generation, Laguna, Voxtral, caches |
| [`Sources/ModelQuantizer`](Sources/ModelQuantizer) | Architecture-aware ScaleSearch quantizer |
| [`Sources/MistralActivationScaleSearchCore`](Sources/MistralActivationScaleSearchCore) | Activation-weighted affine-Q4 search |
| [`Benchmarks`](Benchmarks) | Authored corpora and benchmark programs |
| [`benchmark-results`](benchmark-results) | Reproducible reports and raw measurements |
| [`Docs`](Docs) | Designs, research, deployment notes, and full reference |

## Project status

Midnight is optimized for local, single-model serving and research. The server
does not provide authentication or TLS. Keep it on `127.0.0.1` unless you place
it behind your own trusted access-control boundary, and treat model output as
untrusted. Validate quantized checkpoints against your actual workloads. Model
weights are not included here and remain subject to their own licenses.

## License

Midnight is licensed under the [Apache License 2.0](LICENSE). Dependency and
patch attribution is collected in [third-party notices](THIRD_PARTY_NOTICES.md).
