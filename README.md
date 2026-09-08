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

The long-term target is a project with no Python dependency, including
quantization helpers, evaluation, diagnostics, and tests. See the
[Python removal roadmap](Docs/python-removal-roadmap.md) for the migration scope.

Pruning, distillation, training datasets, and experiment checkpoints belong in
the sibling **Training** project (`../training`). Midnight loads and serves the
exported models. Training reuses the `ModelRunnerCore` and `ModelRunnerProtocol`
libraries; Midnight has no dependency on the Training package. See the local
[Training README](../training/README.md) for builds, experiments, and exports.

Studio, the menu bar app, Chat, and Quantization have their own sibling projects:

| Project | Responsibility |
| --- | --- |
| [Midnight Studio](../midnight-studio/README.md) | Web model controls, preparation, inspection, and its native adapter/shard worker |
| [Midnight menu bar](../midnight-menubar/README.md) | Native macOS launcher for the installed Runner |
| [Midnight Chat](../model-chat-mlx/README.md) | Terminal conversations and client-side session management |
| [Midnight Quantization](../midnight-quantization/README.md) | Checkpoint conversion, calibration, packing, and quantization tests |

Runner builds independently of these applications. See the
[project layout](Docs/project-layout.md) for dependencies and launch commands.

> **Beta:** Midnight is prerelease software. CLI, API, and runtime behavior may
> change before the first stable release.

## Install the current prerelease

```sh
curl -fsSL https://midnightrun.sh/install.sh | bash
```

The installer selects the newest published release, including prereleases,
verifies its checksum, and asks whether to add Midnight to PATH. The prebuilt
runner supports Apple silicon on macOS 26+ and is ad-hoc signed, not notarized.
Linux/CUDA remains available through the source build below.

Version **0.2.0-beta.4** separates the runner from the surrounding tools and adds
Hugging Face downloads, an editable download catalog, native corpus preparation,
and the current model/audio compatibility work. Use `midnight --list` for local
models or `midnight download` for downloadable presets. Exact checkpoint
compatibility and memory requirements still vary; see the model notes below.

## Quantization and reproducible benchmarks

Version **0.2.0-beta.3** adds generated code/math and long-context retrieval
checks, isolated code scoring, and bounded standard/ScaleSearch Q4/G128 conversion.
See the [generated evaluation workflow](Docs/generated-evaluation.md),
[runtime profiling](Docs/runtime-profiling.md),
[measured results](benchmark-results/next-priorities-20260904/README.md), and
[quantization research](Docs/mlx-quantization-research-20260904.md).
Quantization quality varies by model and domain; runtime defaults change only
when controlled measurements support them.

## Memory and context controls

Version **0.2.0-beta.1** adds request memory admission, device-aware Metal
budgets, shared conversation-cache accounting, configurable chunked prefill,
and opt-in KV compression. The Mac launcher now uses release by default.
See [memory and long-context operation](Docs/memory-and-context.md) for settings
and experimental-feature limits.

## What is included

Liquid LFM2/LFM2.5 text architectures are available through the native MLX
loaders, from dense 1.2B Base/Instruct models to 8B-A1B and 24B-A2B MoE models.
See [Liquid model compatibility](Docs/liquid-models.md) for formats, usage,
configuration checks, and the remaining checkpoint-level validation limits.

- Native Mistral-family text inference with append-only conversation-prefix
  caching and hybrid-attention support.
- Native GPT-OSS text inference with explicit reasoning-effort controls and
  mixed MXFP4/affine checkpoints. See the [GPT-OSS setup and measurements](Docs/gpt-oss.md).
- Native Voxtral text-to-speech with 24 kHz WAV/PCM output and checkpoint
  preset voices.
- Native Poolside Laguna hybrid-attention/MoE execution, including the existing
  fused expert path, compiled graph optimizations, mixed Q4R8 checkpoints, and
  opt-in greedy DFlash speculative decoding.
- Model discovery, streaming and non-streaming chat, function-tool calls,
  speech, and voice discovery through a focused OpenAI-compatible HTTP surface.
- Native evaluation tools for teacher-KL, NLL, generated-task quality, and runtime
  tests. Checkpoint conversion lives in [Midnight Quantization](../midnight-quantization/README.md).
- One source tree for Apple-silicon Metal and Linux CUDA deployments.

## Quick start on Apple silicon

The per-user installation lives in `~/.midnight`: `models/` holds checkpoints,
`logs/` holds saved logs, and `bin/` contains the commands available on PATH.
Build and install the current release with `./install.sh`, or install an already
built runner with `./install.sh --binary .build/release/midnight`. The installer
preserves models and logs and stages the executable with its runtime resources
under `apps/.runner-versions/` before updating `bin/midnight`. Previous versions
remain available to running processes.

Add `export PATH="$HOME/.midnight/bin:$PATH"` to your shell configuration once.
Then use `midnight --list` or `midnight --model MODEL_NAME` from any directory.

Browse publisher checkpoints with `midnight download` or `midnight download --list`, and download with
`midnight download liquid-1.2b`. Use `midnight auth login` for a local Hugging Face
read token, and `--dry-run` to check a download before transferring weights.
The default download limit is 30 GB. See [model downloads](Docs/model-downloads.md)
for preset compatibility notes and custom `owner/model` repositories.
Runtime output stays on stdout/stderr; save it when needed with
`midnight --model MODEL_NAME > ~/.midnight/logs/runner.log 2>&1`.
The separate Lowlight chat application uses `lowlight`; `midnight` is the model runner command.

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
and `.safetensors` weights. Midnight discovers named models under
`~/.midnight/models` and automatically loads user-level settings from
`~/.midnight/model-stack.local.json` when no explicit or project-local config
is supplied:

```bash
./run.sh --list-models
```

Text models can store their own context and output limits in a `midnight.json`
alongside `config.json` (at the root for adapter bundles). It is loaded automatically;
CLI flags override it. See [memory and context configuration](Docs/memory-and-context.md)
and [example model policy](Examples/midnight.json).

Chatterbox Turbo and Multilingual speech models can use the same OpenAI speech
endpoint, with model-specific controls in a model-local `chatterbox.json`.
See [Chatterbox setup, configuration, and compatibility](Docs/chatterbox-integration.md).
The native Chatterbox backend currently requires macOS Metal; compressed audio
formats and speed changes use an optional FFmpeg executable.

## Chat with the model

For a local visual layer explorer, use [Midnight Studio](../midnight-studio/README.md),
a Hummingbird web app that shows loaded modules and tensor shapes, compares
model structures, and captures bounded per-token residual activations on
supported architectures.

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

The architecture-aware Swift quantizer belongs to the sibling
[Midnight Quantization project](../midnight-quantization/README.md). From this
directory, build it with:

```bash
MODEL_RUNNER_BUILD_CONFIGURATION=release \
MODEL_RUNNER_BUILD_PRODUCT=model-runner-quantize \
../midnight-quantization/build.sh
```

Create a ScaleSearch affine-Q4 checkpoint without changing the standard MLX
Q4 storage layout or generation kernels:

```bash
../midnight-quantization/.build/release/model-runner-quantize \
  /absolute/path/to/bf16-model \
  /absolute/path/to/q4-scalesearch-model
```

Midnight Quantization also includes an experimental activation-weighted second pass for
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
| [`Sources/CorpusPreparation`](Sources/CorpusPreparation) | Native text and pinned reference dataset preparation; [usage](Docs/dataset-preparation.md) |
| [`Sources/ModelRunnerProtocol`](Sources/ModelRunnerProtocol) | Shared settings, model descriptors, and API contracts |
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
