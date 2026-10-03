# Getting started with Midnight

## Command-line help

`midnight --help` groups server options into Listener & Configuration, Model &
Adapters, Generation & Cache, and Speculative Decoding. Existing invocations,
short options, defaults, and subcommands remain supported. Use `midnight auth --help`, `midnight api-key --help`, `midnight download --help`, `midnight remove --help`, or `midnight unload --help` for command-specific help. `--list` and `--list-models` retain
their command-specific meaning; root listing does not load configuration or
start a listener. Explicit CLI settings retain precedence over configuration.

## Terminal console

A source-built `midnight` opens a small loom/weft console when both input and
output are terminals and `TERM` is not `dumb`. Launch `midnight` to start idle
and choose an installed model, or pass your usual `--model` and generation options.
The dashboard refreshes four times per second with model lifecycle, live tokens
per second, output tokens against the request limit, elapsed generation time,
time to first token, active/queued model requests and uptime. It also shows MLX
active, cached and peak memory, plus active memory against the configured MLX
budget. Installed models and recent logs remain below the dashboard. Use Up/Down (or j/k) to select, Enter (or l) to
load, a to toggle available/unavailable, u to unload, r to refresh the installed
list, and q, Escape or Ctrl-C to quit.
Loading a selection uses that model's checkpoint settings; launch-only overrides
are retained in its catalog entry after the initial model finishes loading.

The token bar measures the output limit; early stops leave it partially filled.
Live throughput uses emitted token IDs after the first token, excluding prefill;
completed requests retain their authoritative final generation statistics until
the next text generation. First-token timing starts when generation acquires the
model execution slot, excluding queue wait and earlier HTTP prompt preparation.
Text metrics include reasoning and tool tokens. Speech and the optional vision
backend show model request activity without text throughput. Memory covers MLX
allocator buffers, excluding other process allocations; peak is process-wide.
Small terminals prioritize dashboard values and clip the model list and logs.

`midnight --idle --no-ui` keeps ordinary stdout/stderr output. Pipes, redirected
output and noninteractive launchers select this mode automatically. Plain mode
still requires `--model`, a configured model, or `--idle`.

Availability is separate from whether a model is loaded: available models are
listed by the API and can load on demand; unavailable models remain installed
but are hidden from discovery and reject new requests. Marking the resident
model unavailable drains existing work and unloads it. Making it available again
does not load it. The console keeps all installed models visible with +/− markers.

Availability is saved by canonical checkpoint/bundle path in
`~/.midnight/config/model-availability.json`, so aliases share the setting and it
survives restarts. `MIDNIGHT_MODEL_AVAILABILITY_FILE` can select another file.
Each listener reads these settings at launch; other running listeners retain
their own policy until restarted. A malformed file prevents startup and a failed
write leaves the previous policy unchanged.

API-key authentication is optional in both modes. Console controls share the
server's existing lifecycle;
unloading releases memory and retains downloaded files. The console does not
provide chat, downloads or file deletion. Source builds require Swift 6.4 for
loom and weft; published binaries may predate the console.

## Install the current prerelease

```sh
curl -fsSL https://midnightrun.sh/install.sh | bash
```

The installer selects the newest published release, including prereleases,
verifies its checksum, and asks whether to add Midnight to PATH. The prebuilt
runner supports Apple silicon on macOS 26+ and is ad-hoc signed, not notarized.
A separate Linux x86-64 archive targets CUDA 13 and RTX 4090 (sm_89) on Ubuntu 24.04; other Linux/GPU configurations can use the source build below.

Version **0.2.0-beta.5** adds OpenAI Responses, structured output controls, safe queued requests, shared prompt caching and live model switching. Lowlight can use the Responses API and display cached input tokens. Embeddings and document retrieval are not included.

Version **0.2.0-beta.4** separates the runner from the surrounding tools and adds
Hugging Face downloads, an editable download catalog, native corpus preparation,
and the current model/audio compatibility work. Use `midnight --list` for local
models or `midnight download` for downloadable presets. Exact checkpoint
compatibility and memory requirements still vary; see the model notes below.

## Quantization and reproducible benchmarks

Version **0.2.0-beta.3** adds generated code/math and long-context retrieval
checks, isolated code scoring, and bounded standard/ScaleSearch Q4/G128 conversion.
See the [generated evaluation workflow](generated-evaluation.md),
[runtime profiling](runtime-profiling.md),
[measured results](../benchmark-results/next-priorities-20260904/README.md), and
[quantization research](mlx-quantization-research-20260904.md).
Quantization quality varies by model and domain; runtime defaults change only
when controlled measurements support them.

Use the [performance regression checks](performance-regression-checks.md)
to compare an installed runner with a candidate using identical models and
requests, preserved outputs, and alternating process pairs.

## Memory and context controls

Version **0.2.0-beta.1** adds request memory admission, device-aware Metal
budgets, shared conversation-cache accounting, configurable chunked prefill,
and opt-in KV compression. The Mac launcher now uses release by default.
See [memory and long-context operation](memory-and-context.md) for settings
and experimental-feature limits.

## What is included

Liquid LFM2/LFM2.5 text architectures are available through the native MLX
loaders, from dense 1.2B Base/Instruct models to 8B-A1B and 24B-A2B MoE models.
See [Liquid model compatibility](liquid-models.md) for formats, usage,
configuration checks, and the remaining checkpoint-level validation limits.

- Native Mistral-family text inference with append-only conversation-prefix
  caching and hybrid-attention support.
- Native GPT-OSS text inference with explicit reasoning-effort controls and
  mixed MXFP4/affine checkpoints. See the [GPT-OSS setup and measurements](gpt-oss.md).
- Native Talkie historical text inference, including our own MLX Q4/Q8
  conversions. See [Talkie setup and validation](talkie.md).
- Native Voxtral text-to-speech with 24 kHz WAV/PCM output and checkpoint
  preset voices.
- Native Poolside Laguna hybrid-attention/MoE execution, including the existing
  fused expert path, compiled graph optimizations, mixed Q4R8 checkpoints, and
  opt-in DFlash speculative decoding.
- Model discovery, streaming and non-streaming chat, function-tool calls,
  speech, and voice discovery through a focused OpenAI-compatible HTTP surface.
- Muse Glimmer ATEM calls are translated into OpenAI tool calls, including
  streaming and required/named selection. See [Muse tool calling](muse-glimmer.md).
- Native evaluation tools for teacher-KL, NLL, generated-task quality, and runtime
  tests. Checkpoint conversion lives in [Wick](../../wick/README.md).
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
Start the server without configuring an API key:

```sh
midnight --model MODEL_NAME
```

When `MIDNIGHT_API_KEY` is unset, every HTTP route accepts requests without an
Authorization header. The listener defaults to `127.0.0.1`. OpenAI SDKs that
require a nonempty `api_key` argument can use a placeholder such as `unused`
when the server has no configured key.

To enable authentication, generate a private key and set it before startup:

```sh
export MIDNIGHT_API_KEY="$(midnight api-key generate)"
midnight --model MODEL_NAME
```

With a configured key, every route, including discovery and runtime controls,
requires `Authorization: Bearer $MIDNIGHT_API_KEY`. Missing or incorrect client
keys return HTTP 401. Invalid server keys still prevent startup: configured keys
must contain 32–256 ASCII letters, digits, hyphens, periods, underscores, or
tildes. An empty environment value is invalid; use `unset MIDNIGHT_API_KEY` to
disable authentication, then restart the server. This choice applies to the
whole listener, including Activation Capture and recording routes.

`midnight api-key generate` prints a random 256-bit key without storing or
activating it. `midnight --list` and Hugging Face login do not depend on this
server key. Give the configured key to clients through their environment or
secret settings.

Browse publisher checkpoints with `midnight download` or `midnight download --list`, and download with
`midnight download liquid-1.2b`. Use `midnight auth login` for a local Hugging Face
read token, and `--dry-run` to check a download before transferring weights.
The default download limit is 30 GB. See [model downloads](model-downloads.md)
for preset compatibility notes and custom `owner/model` repositories.
Switch models without restarting the server using the local load/unload API,
or start with `midnight --idle` and select a model later. See
[live model switching](live-model-switching.md) for commands and lifecycle behavior.
Runtime output stays on stdout/stderr; save it when needed with
`midnight --model MODEL_NAME > ~/.midnight/logs/runner.log 2>&1`.
The separate Lowlight chat application uses `lowlight`; `midnight` is the model runner command.

Requirements: macOS 15 or newer, Swift 6.4, and the Xcode command-line tools.
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
CLI flags override it. See [memory and context configuration](memory-and-context.md)
and [example model policy](../Examples/midnight.json).

Loading a text model does not load speech weights or run audio preprocessing
or synthesis. Explicitly loading a speech model selects its backend and can
replace the resident text model; unloading drains its work and releases its
model state.

Image input uses the independently built
[optional vision runtime](../Optional/Vision/README.md). Configure its executable
with `MIDNIGHT_VISION_WORKER`, then explicitly select a compatible FastVLM model
through ordinary model loading. Midnight drains the previous model, starts the
worker, and stops it when unloading or returning to text. Text selections start
no vision worker and perform no image processing. Lowlight and Lowlight-browser
provide managed `/vision` controls that preserve the previous text selection
and conversation. This vision backend currently requires Apple silicon.
