# Midnight project layout

Studio, the menu bar app, and quantization were extracted into independent
sibling Git repositories on 8 September 2026. Chat and research Training were
already separate projects.

Wick, then named Facet, became a self-contained quantization package on
19 September. The
30 September ownership cleanup removed Midnight's remaining quantization
launchers and moved its Gemma planner to Wick, Talkie GQA training to Training,
and the alternative website to its own `midnight-nite` folder.
The [ownership manifest](project-separation-20260930.json) records the moves,
source hashes, preserved data, and verification.

```text
ChatGPT/
  midnight/                # model serving, Activation Capture, HTTP API, runtime evaluation
  midnight-moonshine/      # native domain distillation and GPT-OSS expert reduction
  midnight-studio/         # loom/weft terminal workbench; retained web interface and Workers
  midnight-menubar/        # native macOS runner launcher
  afterglow/               # native training, calibration, quantization, evaluation, exports
  wick/                    # preserved quantization source project
  midnight-site/           # main website
  midnight-nite/           # alternative website, including its own Git history
  midnight-diagnostics/   # external reference tools and archived standalone prototypes
  vibevoice/              # unused upstream reference checkout, separate from serving
  model-chat-mlx/          # terminal chat client; installed command: midnight
  training/                # research training, distillation, experiments, exports
    Experiments/TalkieGQARecovery/  # independent package for Talkie conversion/recovery
```

## Ownership and dependencies

| Project | Owns | Uses |
| --- | --- | --- |
| Midnight Runner | Model loading/generation, text/audio APIs, public Activation Capture, memory/context controls, runtime/quality evaluation and corpus preparation | Pinned Swift/MLX dependencies; no sibling project dependency |
| [midnight-moonshine](../../midnight-moonshine/README.md) | Domain-specific distillation, teacher-answer review, GPT-OSS expert reduction, student training, evaluation and exports | Midnight's public native Activation Capture and runtime libraries; independent sibling project with loom/weft console and isolated workers |
| [Studio](../../midnight-studio/README.md) | Terminal model controls, profiles, JSON commands and inspection; retained web preparation UI and native workers | Runner over HTTP or as an owned process; native worker links Runner libraries; quantization jobs launch the separate toolkit |
| [Menu bar](../../midnight-menubar/README.md) | Native status/launch interface | Installed `~/.midnight/bin/midnight-runner`; no MLX package dependency |
| [Afterglow](../../afterglow/README.md) | New model development, decision training, calibration, quantization, evaluation and exports | Own pinned dependencies and model support; no sibling package dependency |
| [Wick](../../wick/README.md) | Preserved quantization implementations, tests and existing research runs | Own pinned dependencies, patches and build scripts |
| [Main website](../../midnight-site/README.md) | Midnight's main public website | Independent website project |
| [Alternative website](../../midnight-nite/README.md) | Midnight Nite artwork and static website | Independent deployment configuration; relocation does not deploy it |
| Chat | Terminal conversations, sessions, attachments, client profiles and skills | OpenAI-compatible HTTP server |
| [Training](../../training/README.md) | Distillation, pruning research, datasets, experiments, and model exports | Runner's shared model/runtime libraries |

Midnight owns its reviewed MLX dependency patches and the shared platform build
recipe. Studio's workers and the Talkie GQA training package use that recipe with
`MODEL_RUNNER_BUILD_PACKAGE_ROOT` pointing to their own Swift package. Each
project owns its own dependency checkouts and build outputs; the runtime source
is a one-way local package dependency. The Studio web package and menu bar
package have no MLX dependency.

Wick owns its complete build and dependency preparation. The older
`midnight-quantization` checkout is retained separately as the earlier dependent
extraction; current Midnight documentation and tools use Wick.

New quantization algorithms, calibration and their tests live in Afterglow.
Existing Wick sources and research runs are preserved. Runtime execution
of quantized weights, model compatibility, quality evaluation, and historical
benchmark reports stay in Runner. Small interactive adapter jobs belong to
Studio; larger training experiments remain in Training.

### Activation Capture ownership

[Midnight Activation Capture](activation-capture.md) belongs to the serving
runtime and is available to any authorized consumer through the documented HTTP
API. Midnight owns capability discovery, model/generation selection, bounded
tensor capture, passive recording of external chat activations, token coordinates,
and observer cleanup. Studio owns its local capture library, JSON artifacts,
recording log, and replay interface. ABSlayer,
midnight-moonshine, and other analysis or evaluation clients own their datasets,
objectives, transformations, and saved artifacts. A consumer does not need an
ABSlayer dependency or job bridge to use this feature.

New distillation development belongs in the independent `midnight-moonshine`
project. Existing Training experiments remain preserved. Model preparation and
quantization remain separate from both distillation and Midnight's serving API.

The [Talkie GQA recovery package](../../training/Experiments/TalkieGQARecovery/README.md)
owns the conversion and adapter-application scripts, their fixture tests, and
the `model-runner-talkie-gqa-recovery` command. It has its own manifest and build
directory. Midnight retains the Talkie loader and inference tests for those
exports. Read-only runtime/KL benchmarks, context probes, and evaluation corpus
preparation remain Midnight responsibilities.

## Build and launch

Run each command from the project named below. See its README for requirements
and optional configuration.

| Project | Command |
| --- | --- |
| Runner | `./build.sh`, then `./run.sh --model MODEL` or `./install.sh` |
| Studio | `./build.sh`, then `./run.sh --midnight-url http://127.0.0.1:8080` (terminal; `./run.sh web` for browser) |
| Studio native workers | `Workers/build.sh` |
| Menu bar | `./script/build_and_run.sh --build`; open `dist/Midnight.app` to launch |
| Wick | `./build.sh`; default product is `wick` |
| Talkie GQA training | `Experiments/TalkieGQARecovery/build.sh` from Training |
| Chat | `./install.sh`, then `midnight` |
| Training | `./build.sh`; see its experiment-specific documentation before running training |

Studio's primary command is `midnight-studio`; `midnight-inspector` remains a
compatibility command. Runner does not launch a web interface. The installed
Runner command is `midnight`, installed in `~/.midnight/bin`.

The menu bar checkout still defaults to `~/.midnight/bin/midnight-runner`.
Set its runner path to `~/.midnight/bin/midnight` when using the current installer;
the launcher default needs to be updated in that sibling project. The historical
`model-chat-mlx` checkout is absent on this host, so its README is not linked here.

Run new model-development and quantization tools from `../afterglow/`.
Preserved Wick launchers remain in `../wick/Scripts/` for existing work. Midnight's five
forwarding scripts have been removed. The Gemma geometry planner and its tests
also live in Wick. No quantizer, training executable, or website is part of the
Midnight package or its active source tree.

The unused upstream `Optional/VibeVoice` checkout now lives at `../vibevoice`.
Midnight's supported VibeVoice worker still uses its separate
`Optional/vibevoice-1.5b-env` environment. The upstream clone's ignored `.venv`
was preserved but contains paths from its former location; recreate that unused
environment before running the reference checkout.

The standalone CUDA decoder prototype moved to
`../midnight-diagnostics/experiments/CUDAReplay`. It is not part of Midnight's
MLX runtime and retains its historical failed numerical gate. The MLX CUDA
kernel and replay integration checks remain here because they exercise the
serving backend. Historical benchmark reports and model files were not moved.

## Preserved data and verification

Model weights and research artifacts were not moved. Studio continues to discover
`~/.midnight/models` and the sibling Runner's existing `tmp/models`. Its future
downloads use `~/.midnight/models/studio`.

The existing Studio profile and empty managed-process metadata were copied
byte-for-byte to `~/.midnight/studio`; the original `.local` directory remains
in the Studio checkout. There were no preparation jobs or downloaded Studio
checkpoints to migrate. Menu bar preferences, bundle identifier, and log paths
are unchanged. No installed application was replaced or launched.

The [extraction manifest](project-separation-20260908.json) records the original
and extracted hashes for moved native sources, tests, examples, and helpers.
Native source files were verified byte-identical before the original paths
were removed. Working-tree changes predating this separation remain intact.

Verification after the original 8 September relocation:

- Runner builds; 21 focused runtime/catalog/configuration/quality tests pass.
- Studio: 37 Swift web tests, 14 frontend tests, and 2 native fixture tests pass.
- Quantization: all nine executables build; 34 Swift tests, 14 Python tests,
  16 CLI validation cases, and static source/patch checks pass.
- Menu bar builds from its final directory; bundle signature and plist validate.
- Shared build-helper tests cover macOS and Linux CPU package/scratch separation
  with mocked tools. Native CUDA execution was not exercised during this move.
- The existing Training compatibility build could not proceed: its manifest
  already references the absent `Experiments/GPTOSSPrune/Sources/GPTOSSPruneCLI`
  target. That manifest predates this separation and was not changed here.

## Afterglow decision-model development (2 October 2026)

Afterglow (`../afterglow`) is the new standalone Swift/MLX model-development
project. It combines copied Wick quantization components with native decision
training, calibration, evaluation and exports. Existing Wick and Training
projects and their runs are preserved during validation. New Nimble training
and preparation belong in Afterglow; Midnight owns the decision serving API.
Afterglow has no sibling package dependencies. The versioned decision artifact
contract connects it to Midnight.
