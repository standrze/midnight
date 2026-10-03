# Command-line Hugging Face downloads

```sh
midnight auth login
midnight download
midnight download laguna-xs-2.1 --dry-run
midnight download laguna-xs-2.1
midnight --model laguna-xs-2.1
```

`midnight --list` lists models already installed under `~/.midnight/models`;
`midnight --list-models` remains an alias. `midnight download` with no model,
or `midnight download --list`, lists the complete selected scope, including unavailable models and their concrete sourcing gaps.

The editable catalog lives in `~/.midnight/config/downloads.json`. Midnight
creates it with the default presets the first time it is needed. Subsequent
runs and upgrades preserve your edits. Listing and preset selection always include the current built-in scope first. Saved entries cannot hide or override those models; additional custom entries are preserved and remain subject to source validation. It is a JSON array; each entry has `name`, `repo`, and `note`:

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

## Mistral-family scope

Current download scope includes Devstral Small 2 and Voxtral, not older
Devstral releases or other Mistral subfamilies. Their runtime implementations
remain available for existing compatible local checkpoints. Publisher-owned
approximately four-bit native MLX downloads for these selected models have not
been verified yet, so they are unavailable in the curated download catalog.

## Current curated downloads

See [the selected scope and tiers](model-support-scope.md). Downloads currently
accept only registered publisher-owned approximately four-bit sources. The
config is checked before weights transfer; community repos, full-precision
substitutions and deferred Muse Glimmer are refused. Some selected families
remain unavailable pending a suitable publisher-owned native MLX release.

| Preset | Publisher repository | Companion |
| --- | --- | --- |
| `laguna-xs-2.1` | `poolside/Laguna-XS-2.1-NVFP4-mlx` | `poolside/Laguna-XS-2.1-DFlash-NVFP4` |
| `gpt-oss-20b` | `openai/gpt-oss-20b` | No eligible OpenAI assistant verified |
| `laguna-s-2.1` | `poolside/Laguna-S-2.1-NVFP4-mlx` | No automatic bundle selected |
| `gpt-oss-120b` | `openai/gpt-oss-120b` | No automatic bundle selected |

Use a preset or the exact repository ID. Existing user-edited catalogs are
preserved, but entries still need an approved repository. Larger models need
an explicit `--max-gb` override. Official GPT-OSS MXFP4 weights can expand in
MLX memory; stored precision does not promise a four-bit resident footprint.

Laguna XS downloads include the NVFP4-target BF16 assistant by default. Metadata
for both repositories is resolved to immutable commits before weight transfer.
The combined selected size must fit `--max-gb`; `--dry-run` reports both commits
and total size without fetching weights. `--target-only` omits the designated
companion. Targets and companions stage together, and the target directory is
published only after both transfers and size checks succeed. Cancellation or
failure removes staging and retains reusable Hub cache files. No existing
installation is replaced. Companions live under the target's `assistant/` and
have their own download provenance; deleting the target removes the bundle.

Downloading does not enable Laguna DFlash. Its existing explicit load options
still apply, for example `--dflash-model ~/.midnight/models/laguna-xs-2.1/assistant`.
Exact NVFP4 pairing quality and performance still need runtime evaluation.

## loom model management

Press `d` to open the publisher download catalog. Up/Down selects a model;
Enter downloads it with its designated companion. Unavailable entries explain
the source or format gap. Progress and errors appear in the console; `c`
cancels an active transfer, `t` reports token presence without revealing it,
and Escape returns to installed models. Use `midnight auth login` in another
terminal or supply the shared token before launching the console; the UI does
not collect credentials. Console downloads use the default 30 GB combined
limit; use the CLI with `--max-gb` for larger installations.

In installed models, `x` requests deletion and `y` confirms the named target;
`n` or Escape cancels. Only managed downloads are removable. Loaded/in-use
models must be unloaded first, and existing file leases protect models used by
other cooperating Midnight processes. Existing `a` availability, Enter load,
`u` unload, and `r` refresh controls remain available.

## Your future optimized releases

No upload or publishing configuration is required now. Our own approved releases can later be added to the source catalog and use
the same native Swift transfer path. They are not enabled during this phase:

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

## Removing downloaded models

Unloading releases RAM; removing deletes the checkpoint's files. These commands use a running local Midnight listener. Updated Midnight processes protect their loaded model, adapter, draft model, and speech dependencies across listeners:

```sh
midnight remove --list
midnight unload
midnight remove laguna-xs-2.1
```

Use `--endpoint http://127.0.0.1:PORT` when the listener uses a different port. Unload is asynchronous: allow outstanding requests to finish before retrying removal. An unused download can be removed while another model remains loaded. Removal detaches the directory before deleting files, and file cleanup runs outside the inference lifecycle actor.

Lowlight supports the same workflow on its local Midnight connection:

```text
/models downloads
/models unload
/models remove laguna-xs-2.1
```

`/model NAME` always selects an endpoint model, including models named `unload`, `downloads`, or `remove`. File management uses the separate `/models` namespace and only runs when entered explicitly.

Removal accepts the folder names returned by the downloads list, not served aliases, repository IDs, or arbitrary paths. It requires the `midnight-download.json` provenance record written by `midnight download`. Manually installed folders, symbolic links, and active model dependencies are refused. Hugging Face's separate download cache and connection settings are retained. Each updated Midnight process takes shared file leases before opening a managed checkpoint or dependency, and retains them until requests and backend producers have drained. Removal requires an exclusive lease, so another updated listener using the same download prevents deletion. This adds no per-token work and does not scan the model catalog during loading. Persistent hidden `.leases` files coordinate processes and must not be deleted manually.

These are advisory leases: older Midnight binaries and other applications do not participate. Upgrade/restart every Midnight listener sharing the download folder before relying on cross-process protection, and stop any other application using a download before removing it. A listener using an external folder does not block unrelated managed downloads.

The local management API is a Midnight extension, separate from OpenAI's model catalog:

- `GET /v1/runtime/downloads`: returns `{ "object": "list", "data": [{ "id": "laguna-xs-2.1", "repository": "owner/model", "revision": "<commit>", "inUse": false }] }`.
- `POST /v1/runtime/remove` with `{ "model": "laguna-xs-2.1" }`: returns `{ "id": "laguna-xs-2.1", "object": "model", "deleted": true }` after deletion completes.
- `POST /v1/runtime/unload` with `{}`: unloads the active model and keeps downloaded files.

All three require a loopback native client and `Content-Type: application/json`; cross-origin browser requests are refused. Active models and load/unload transitions return HTTP 409. Missing downloads return 404; unsafe names and unmanaged folders return 400. There is no force-delete or offline fallback.
