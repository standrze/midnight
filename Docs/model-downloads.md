# Command-line Hugging Face downloads

```sh
midnight auth login
midnight download
midnight download liquid-1.2b --dry-run
midnight download liquid-1.2b
midnight --model liquid-1.2b
```

`midnight --list` lists models already installed under `~/.midnight/models`;
`midnight --list-models` remains an alias. `midnight download` with no model,
or `midnight download --list`, lists the downloadable catalog.

The editable catalog lives in `~/.midnight/config/downloads.json`. Midnight
creates it with the default presets the first time it is needed. Subsequent
runs and upgrades preserve your edits. Both listing and downloading by preset
name read this file. It is a JSON array; each entry has `name`, `repo`, and `note`:

```json
[
  {"name": "my-model", "repo": "your-account/your-mlx-model", "note": "My optimized release"}
]
```

Add entries to the existing array to retain the other presets. Names must be
unique and use letters, digits, dots, hyphens, or underscores, without a leading
dot or hyphen. An invalid catalog produces an error rather than being replaced.

Login prompts for a read token without echoing it, checks it with Hugging Face,
and stores it in the standard shared Hugging Face token file with mode 0600.
The default is `~/.cache/huggingface/token`; `HF_TOKEN_PATH` or `HF_HOME` can
change it. `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` environment variables take
precedence. Never put a token in a command argument or share it in chat.
`midnight auth status` reports availability without revealing it.
`midnight auth logout` removes the shared token file, affecting other tools that
use it; it does not unset environment variables or revoke the token remotely.
Authentication raises Hub request quotas and enables authorized gated/private
access. It does not guarantee faster transfer bandwidth. Accept gated model
licenses on the publisher's Hugging Face page before downloading.

Downloads default to a 30 decimal GB limit, checked using current file metadata
before weights are transferred. `--max-gb` explicitly overrides it. `--revision`
accepts a branch, tag, or commit; each operation resolves and records an immutable
commit. Downloads select root-level safetensors, JSON, tokenizer data, templates,
and license/readme files, excluding duplicate nested formats and executable code.
A repository with only nested weights or GGUF needs a different layout/conversion.

Files download through the pinned native Swift Hugging Face client, with up to
eight concurrent files and its reusable cache. A hidden staging directory is
published under `~/.midnight/models` only after all selected file sizes are
verified. Existing model folders are never overwritten. Allow disk space for
both the Hub cache and the destination (potentially twice the reported size).
Download completion does not certify inference compatibility or sufficient RAM.

## Publisher presets

Repository metadata checked September 8, 2026. Sizes below are approximate
weight sizes; the CLI checks the exact selected total for the chosen revision.

| CLI name | Publisher repository | Weights | Notes |
| --- | --- | --- | --- |
| `liquid-1.2b` | [LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit) | 4-bit | Native LFM2 loader; exact checkpoint inference not tested here |
| `liquid-8b` | [LiquidAI/LFM2.5-8B-A1B-MLX-4bit](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-MLX-4bit) | 4.8 GB | Native LFM2 MoE loader |
| `liquid-24b` | [LiquidAI/LFM2-24B-A2B-MLX-4bit](https://huggingface.co/LiquidAI/LFM2-24B-A2B-MLX-4bit) | 13.4 GB | LFM2, not LFM2.5; all experts need memory |
| `laguna-xs-2.1` | [poolside/Laguna-XS-2.1-NVFP4-mlx](https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4-mlx) | 21.6 GB | Publisher MLX NVFP4; exact quantization not runtime-validated |
| `gpt-oss-20b` | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | 13.8 GB | Official MXFP4; raw tensors expand in the pinned MLX loader; see [GPT-OSS notes](gpt-oss.md) |
| `gemma-e2b` | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | 10.2 GB | Full precision, not a 4-bit MLX conversion |
| `gemma-e4b` | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | 16.0 GB | Full precision |
| `gemma-12b` | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) | 23.9 GB | Full precision; unified architecture not runtime-validated |

[Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B)
has about 59.6 GB of official weights and is omitted from the presets. A suitable
smaller MLX conversion and runtime support must be established before recommending it.

## Your future optimized releases

No upload or publishing configuration is required now. Any future compatible
repository works with the same command:

```sh
midnight download your-account/your-mlx-model --dry-run
midnight download your-account/your-mlx-model
midnight --model your-account--your-mlx-model
```

The destination name uses `owner--model` for arbitrary repositories. Adding a
short preset later is optional. Keep the native weights and config at repository
root, and include all required tokenizer files. The downloader does not convert
weights or execute repository code.

## Install or update the runner

```sh
curl -fsSL https://midnightrun.sh/install.sh | bash
```

The bootstrap resolves the newest published GitHub release, including
prereleases, and verifies the release archive against its SHA256SUMS file before
installing it. Running it again updates the runner while preserving your models
and editable download catalog. In an interactive terminal it asks whether to add
Midnight to PATH; answering yes updates `.zshrc` or Bash's startup file, without
sudo. No terminal means no shell-configuration change. Open a new terminal after
accepting. You can inspect the script before running it.

The prebuilt beta.4 artifact supports Apple silicon on macOS 26+. It is ad-hoc
signed, not Developer ID signed or notarized. Linux/CUDA remains a source build.
