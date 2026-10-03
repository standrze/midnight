# Midnight project layout

Studio, the menu bar app, and quantization were extracted into independent
sibling Git repositories on 8 September 2026. Chat and research Training were
already separate projects.

```text
ChatGPT/
  midnight/                # inference, HTTP API, compatibility, runtime evaluation
  midnight-studio/         # web workbench and its native Workers package
  midnight-menubar/        # native macOS runner launcher
  midnight-quantization/   # conversion, calibration, packing, quantization tests
  model-chat-mlx/          # terminal chat client; installed command: midnight
  training/               # research training, distillation, experiments, exports
```

## Ownership and dependencies

| Project | Owns | Uses |
| --- | --- | --- |
| Midnight Runner | Model loading/generation, text/audio APIs, inspection API, memory/context controls, runtime/quality evaluation and corpus preparation | Pinned Swift/MLX dependencies; no sibling project dependency |
| [Studio](../../midnight-studio/README.md) | Model-control web UI, downloads, job history, inspection, small adapter-training and shard-preparation workers | Runner over HTTP or as an owned process; native worker links Runner libraries; quantization jobs launch the separate toolkit |
| [Menu bar](../../midnight-menubar/README.md) | Native status/launch interface | Installed `~/.midnight/bin/midnight-runner`; no MLX package dependency |
| [Quantization](../../midnight-quantization/README.md) | Nine quantization/calibration/verification commands, quantization libraries, benchmark and tests | Runner's `ModelRunnerCore`, `ModelRunnerProtocol`, and `ModelQualityCore` libraries |
| [Chat](../../model-chat-mlx/README.md) | Terminal conversations, sessions, attachments, client profiles and skills | OpenAI-compatible HTTP server |
| [Training](../../training/README.md) | Distillation, pruning research, datasets, experiments, and model exports | Runner's shared model/runtime libraries |

Midnight owns the reviewed MLX dependency patches and the shared platform build
recipe. Quantization and Studio's workers use that recipe with
`MODEL_RUNNER_BUILD_PACKAGE_ROOT` pointing to their own Swift package. Each
project owns its own dependency checkouts and build outputs; the runtime source
is a one-way local package dependency. The Studio web package and menu bar
package have no MLX dependency.

Quantization algorithms and their tests live in Quantization. Runtime execution
of quantized weights, model compatibility, quality evaluation, and historical
benchmark reports stay in Runner. Small interactive adapter jobs belong to
Studio; larger training experiments remain in Training.

## Build and launch

Run each command from the project named below. See its README for requirements
and optional configuration.

| Project | Command |
| --- | --- |
| Runner | `./build.sh`, then `./run.sh --model MODEL` or `./install.sh` |
| Studio | `./build.sh`, then `./run.sh --midnight-url http://127.0.0.1:8080` |
| Studio native workers | `Workers/build.sh` |
| Menu bar | `./script/build_and_run.sh --build`; open `dist/Midnight.app` to launch |
| Quantization | `./build.sh`; default product is `model-runner-quantize` |
| Chat | `./install.sh`, then `midnight` |
| Training | `./build.sh`; see its experiment-specific documentation before running training |

Studio's primary command is `midnight-studio`; `midnight-inspector` remains a
compatibility command. Runner does not launch a web interface. The installed
Runner command is `midnight`, installed in `~/.midnight/bin`.

The five existing quantization shell launchers in Runner's `Scripts/` are
compatibility forwarders into `../midnight-quantization/Scripts/`. Set
`MIDNIGHT_QUANTIZATION_ROOT` if that checkout is elsewhere. Direct quantizer
builds, binaries, Python helpers, and tests now belong to Quantization.

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

Verification after relocation:

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
