# Midnight

**Native Swift/MLX inference, tuned for Mistral, Voxtral, and Poolside Laguna.**

[![Swift 6.4](https://img.shields.io/badge/Swift-6.4-F05138?logo=swift&logoColor=white)](https://www.swift.org/)
[![Apple silicon](https://img.shields.io/badge/Apple_silicon-Metal-111111?logo=apple)](https://github.com/ml-explore/mlx-swift)
[![Linux](https://img.shields.io/badge/Linux-CUDA-FCC624?logo=linux&logoColor=111111)](Docs/reference.md#cuda-on-linux)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

Midnight is a standalone local model server for MLX `safetensors` checkpoints.
It serves an OpenAI-compatible subset for chat and speech over Metal on macOS,
CUDA on Linux, or MLX's CPU backend. Text, Voxtral, and Qwen3-TTS inference run
in the Swift process. VibeVoice is an optional local worker that currently uses
Python and PyTorch; no serving path requires a hosted dependency.

Midnight is the inspection and serving layer for local model work, including
checkpoints transformed by external tools. [Midnight Activation Capture](Docs/activation-capture.md)
provides bounded residual activations through a documented API for any consumer,
including midnight-moonshine, ABSlayer, evaluation, and analysis tools. Model
support and capture limits are discoverable through the API. Weight transformation,
abliteration, and training remain separate workflows so the ordinary inference
path stays small and auditable.

The long-term target is a project with no Python dependency, including
quantization helpers, evaluation, diagnostics, and tests. See the
[Python removal roadmap](Docs/python-removal-roadmap.md) for the migration scope.

New domain-specific distillation development belongs in the separate sibling
project **midnight-moonshine**. Existing pruning, distillation, training datasets,
and experiment checkpoints remain in **Training** (`../training`). Midnight
provides Activation Capture and serves compatible exported models. Training
reuses the `ModelRunnerCore` and `ModelRunnerProtocol` libraries; Midnight has no
dependency on the Training package. See the local [Training README](../training/README.md)
for its preserved builds, experiments, and exports.

Studio, the menu bar app, Chat, and Quantization have their own sibling projects:

| Project | Responsibility |
| --- | --- |
| [Midnight Studio](../midnight-studio/README.md) | loom/weft terminal model controls and inspection; retained browser interface and native worker |
| [Midnight menu bar](../midnight-menubar/README.md) | Native macOS launcher for the installed Runner |
| Midnight Chat | Terminal conversations and client-side session management |
| [Wick](../wick/README.md) | Checkpoint conversion, calibration, packing, and quantization tests |
| [Talkie GQA training](../training/Experiments/TalkieGQARecovery/README.md) | Standalone checkpoint conversion and K/V recovery experiment |
| [Main website](../midnight-site/README.md) / [Midnight Nite](../midnight-nite/README.md) | Independent website code and artwork |

Runner builds independently of these applications. See the
[project layout](Docs/project-layout.md) for dependencies and launch commands.

> **Beta:** Midnight is prerelease software. CLI, API, and runtime behavior may
> change before the first stable release.

## Getting started

See [getting started](Docs/getting-started.md) for installation, command-line help,
the Apple silicon quick start, model downloads, and runtime options.

## Usage and model details

See [usage and model details](Docs/usage-and-models.md) for chat and speech examples,
model support, quantization, measured results, API routes, and Linux/CUDA setup.

## Repository map

| Path | Contents |
| --- | --- |
| [`Sources/ModelRunner`](Sources/ModelRunner) | CLI and HTTP server |
| [`Sources/ModelRunnerCore`](Sources/ModelRunnerCore) | Model loading, generation, Laguna, Voxtral, caches |
| [`Sources/CorpusPreparation`](Sources/CorpusPreparation) | Native text and pinned reference dataset preparation; [usage](Docs/dataset-preparation.md) |
| [`Sources/ModelRunnerProtocol`](Sources/ModelRunnerProtocol) | Shared settings, model descriptors, and API contracts |
| [`Benchmarks`](Benchmarks) | Authored benchmark corpora; benchmark executables live under `Sources/` |
| [`benchmark-results`](benchmark-results) | Local reports and raw measurements are ignored; only test-required evaluation source is published |
| [`Docs`](Docs) | Designs, research, deployment notes, and full reference |

## Project status

Midnight is optimized for local, single-model serving and research. The server
starts without an API key by default. Set `MIDNIGHT_API_KEY` to require its
HTTP Bearer token on every API route. It does not provide TLS, so keep it on
`127.0.0.1` or place it behind a trusted TLS boundary. Treat model output as untrusted. Validate quantized
checkpoints against your actual workloads. Model weights are not included here
and remain subject to their own licenses.

The [30 September 2026 readiness audit](Docs/readiness-audit-2026-09-30.md)
records verified serving behavior, model support levels, cybersecurity and
abliteration boundaries, fixes made during the audit, and the remaining
production work.

## License

Midnight is licensed under the [Apache License 2.0](LICENSE). Dependency and
patch attribution is collected in [third-party notices](THIRD_PARTY_NOTICES.md).

See [concurrent clients and shared prompt caching](Docs/concurrent-requests-and-prompt-caching.md) for queue limits, cache behavior and Lowlight integration.

Native Muse and Laguna tool parsing in Lowlight is documented in [Native tool protocols](Docs/native-tool-protocols.md). Existing OpenAI-compatible requests remain supported.
