# Curated model registry scope

Owner-selected scope recorded 1 October 2026. This is the intended registry
allowlist, not a claim that every checkpoint or modality has passed Midnight
runtime validation. No runtime support or download defaults are changed by this
document. The online registry and its JSON belong in `../midnight-site`;
quantization recipes and release preparation belong in `../wick`.

## Current priorities

The owner considers the selected models runnable and has finalized this scope
for now. Prioritize Midnight/MLX optimization and evaluation as follows:

1. **First tier:** Gemma 4 26B-A4B / 31B.
2. **Second tier:** GPT-OSS 20B and Laguna XS 2.1.
3. **Third tier:** `Qwen/Qwen3.8-27B`, the selected approximately 28B checkpoint.
4. **Support coverage:** all other selected checkpoints below; no additional
   families or checkpoints are planned for now.

Focus intensive quantization, runtime performance, quality and compatible
assistant-pairing work in tier order: first-tier models, GPT-OSS 20B and Laguna
XS 2.1, then the selected Qwen checkpoint. Keep the other selected models in
the registry without treating them as active optimization priorities. The
owner's statement that they run is not a replacement for recorded
release-specific validation evidence.

## Tier 1 and 2 assistant downloads

Downloading an approved tier 1 or tier 2 release must also download its
designated compatible assistant by default, when one is available. The model
and assistant form one registry download selection; the user should not have
to find or request the companion separately. The CLI and loom console share a
native Swift download service that bundles designated companions into the
target installation before publishing it.

### Current source policy

For now, use Hugging Face repositories owned by the original model maker only:
`google`, `meta-models`, `poolside`, `Qwen`, `openai` and `mistralai` for their
respective selected models. Verify publisher ownership and current metadata at
download time. Target approximately 4-bit quantized weights that the actual
Midnight/MLX path supports; a four-bit label alone does not establish format
compatibility. The preference is for target weights; matching assistants may
remain at their publisher-supported BF16 or other precision.

This supersedes the earlier third-party GPT-OSS assistant candidate. Do not
download community MLX conversions or third-party drafters under this policy,
even when they derive from official weights. Do not silently fall back to BF16,
FP8, GGUF, compressed-tensors or other formats when no suitable approximately
4-bit native checkpoint is available. Report the specific source/format gap
before transferring an alternative. An approved model and its designated
publisher-owned assistant are covered by the same download request.

These policies are linked from `AGENTS.md`. The native Swift CLI and loom
console enforce the approved source catalog and approximately four-bit
configuration preflight. Existing user-edited presets are retained, but a preset
must resolve to an approved source before its weights are downloaded.

Direct Hugging Face downloads use the pinned Swift Hub client and shared token
support: `midnight auth login`, `midnight auth status`, `HF_TOKEN`,
`HUGGING_FACE_HUB_TOKEN`, `HF_TOKEN_PATH` and `HF_HOME`. Login prompts without
echoing, verifies the read token and stores it with mode 0600. Never ask for
tokens in chat or put them in arguments or logs.

Our own quantized models and Hugging Face listings are a future phase. Preserve
source revisions and target/assistant pairing provenance for those releases;
no upload or substitution of our own releases is authorized for this phase.

| Tier | Target | Published companion source | Current Midnight integration |
| --- | --- | --- | --- |
| 1 | Gemma 4 26B-A4B IT | `google/gemma-4-26B-A4B-it-assistant` (MTP) | Automatic discovery/download at load time is implemented for supported configurations |
| 1 | Gemma 4 31B IT | `google/gemma-4-31B-it-assistant` (MTP) | Automatic discovery/download at load time is implemented for supported configurations |
| 2 | GPT-OSS 20B | No OpenAI-owned assistant verified | Third-party `z-lab/gpt-oss-20b-DFlash` is excluded by the publisher-only policy; report the missing eligible companion |
| 2 | Laguna XS 2.1 | `poolside/Laguna-XS-2.1-DFlash`, `poolside/Laguna-XS-2.1-DFlash-INT4` or `poolside/Laguna-XS-2.1-DFlash-NVFP4`, matched to the target representation | Native DFlash loading is implemented; CLI/console downloads bundle the NVFP4-target assistant; runtime enablement stays explicit; exact NVFP4 quality and performance remain to be evaluated |

These are published source candidates, not universal pairings for every base,
QAT or custom quantization variant. Each released bundle must designate the
exact assistant artifact and immutable revision validated with that target.
For Laguna, select the assistant matched to the target representation; do not
choose the INT4-trained assistant just because our target is four-bit. Existing
custom Q4R8 evaluation favored the BF16-target assistant; the official INT4
pair used the INT4-trained assistant (see [Laguna DFlash](laguna-dflash.md)).

Bundled downloads must report both artifacts, revisions and combined download
size before transferring weights, retain reusable Hub cache files, and
clearly report assistant failures or unavailable pairings. A target-only
download must not be reported as a completed bundle. `--target-only` explicitly omits the designated companion. Downloading an assistant does not automatically certify
runtime support or enable speculative decoding; preserve the runtime's
configuration and validation controls.

For GPT-OSS 20B, an official companion is unverified. The third-party DFlash
candidate is excluded and must not be bundled. The source and integration gaps
remain part of the tier 2 work. No model weights were downloaded while
implementing these controls.

### Publisher source and format findings (1 October 2026)

- Poolside's `Laguna-XS-2.1-NVFP4-mlx` config declares native MLX NVFP4,
  four-bit group-16 quantization. `Laguna-XS-2.1-DFlash-NVFP4` is a BF16
  assistant trained for the NVFP4 target, not a four-bit assistant. Validate
  the exact MLX target pairing before marking it ready.
- OpenAI's `gpt-oss-20b` and `gpt-oss-120b` publish MXFP4 MoE weights.
  The pinned Midnight MLX loader can expand those tensors in memory; the
  stored precision is not a four-bit resident-memory guarantee.
- Meta publishes approximately four-bit Muse weights in
  `meta-models/Muse-Glimmer-30B-GGUF`. That release is GGUF, not directly
  usable through Midnight's native MLX safetensors download/load path.
- Google publishes Gemma 4 QAT GGUF variants and a 31B W4A16
  compressed-tensors release. Neither should be presented as a verified native
  MLX release solely because it is official and four-bit. A repository named
  `unquantized` is not a packed four-bit download.
- A publisher-owned approximately four-bit native MLX checkpoint was not
  verified for the selected Qwen, Devstral Small 2 or Voxtral models during
  this source check. Keep those source selections pending and recheck when
  requested; community availability does not satisfy publisher-only sourcing.

## Deferred models

Muse Glimmer is removed from the active download/support scope for now. No
maker-owned approximately four-bit native MLX release has been verified; the
official GGUF release does not meet this runtime requirement. Existing runtime
code and installed files remain intact. Revisit it when a qualifying publisher
release or our own approved release becomes available.

## Selected families

| Family | Intended scope | Open decisions |
| --- | --- | --- |
| Gemma 4 | All official variants through 31B: E2B, E4B, 12B, 26B-A4B, 31B; track base, instruction and QAT releases separately where published | Validate each release and supported modality independently |
| Gemma 3 | All core sizes: 270M, 1B, 4B, 12B, 27B; base, instruction and published QAT variants | Gemma 3n was not explicitly selected and is outside the finalized scope |
| Laguna | XS 2.1 and S 2.1 only, including compatible assistants | Both 2.0 releases are excluded; validate each exact target/assistant pairing |
| Qwen 3.8 | Only `Qwen/Qwen3.8-27B`, the approximately 28B checkpoint selected by the owner | No other Qwen sizes or generations; track our quantized variants of this checkpoint |
| GPT-OSS | 20B and 120B | Validate each Midnight/MLX release independently; both are explicitly selected regardless of the earlier Qwen/Mistral size discussion |
| Voxtral | Retain support and registry coverage | Inventory current supported checkpoints and distinguish ASR, audio understanding, realtime and TTS |
| Mistral / Devstral | Devstral Small 2 24B only, alongside the separate Voxtral entry | No older Devstral releases or other Mistral subfamilies in the curated scope |

Chatterbox, Talkie and all unlisted families are excluded from the curated
support and quantization program for now. Within the Mistral publisher's models,
only Devstral Small 2 and Voxtral are selected. Older Devstral releases,
Ministral, Mistral 7B, NeMo, Pixtral, Codestral, Mistral Small, Magistral,
Mixtral, Mistral Medium and Mistral Large are outside this scope. Existing
files, historical results and loader implementations are not removed by this
scope record.

Laguna XS 2.0 and S 2.0 are excluded. Retain XS 2.1 and S 2.1; the
earlier approximate 40B ceiling discussion does not apply to the explicitly
selected Laguna S 2.1.

## Size decisions

The owner selected only the approximately 28B Qwen checkpoint, published as
`Qwen/Qwen3.8-27B`. Use that exact repository identity in the registry;
the marketed 27B name and approximate 28B inventory count refer to this same
checkpoint. This single-checkpoint selection supersedes the earlier approximate
40B family ceiling. Flash-Next, 2.4T-A95B and all other Qwen checkpoints are
excluded.

The owner narrowed the Mistral selection after reviewing size choices:
`mistralai/Devstral-Small-2-24B-Instruct-2512` and the Voxtral models only.
The earlier broad Mistral candidate list is superseded. Devstral Small 2 fits
the approximate 40B ceiling; larger Devstral 2 is not selected. No Devstral
release older than version 2 belongs in the registry.

## Registry requirements

Each approved checkpoint should record publisher repository and immutable
revision, our MLX release repository and revision when available, quantization
profile, measured download size, validation status and evidence, supported
modalities, license/access restrictions and last source-check date.

Assistant records must identify their method (DFlash, MTP or another method),
repository and immutable revision, stored precision, exact compatible target
release and measured pairing results. Do not infer compatibility from a shared
family name or bit width. Download availability and runtime enablement are
separate decisions. Track upstream changes without silently replacing a
validated release.

## Publisher sources checked

- [Gemma model inventory](https://ai.google.dev/gemma/docs/get_started)
- [Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3)
- [Gemma 3n overview](https://ai.google.dev/gemma/docs/gemma-3n)
- [Laguna XS 2.1](https://huggingface.co/poolside/Laguna-XS-2.1)
- [Laguna S 2.1 collection](https://huggingface.co/collections/poolside/laguna-s-21)
- [Muse Glimmer assistant](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant)
- [Gemma 4 26B-A4B assistant](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant)
- [Gemma 4 31B assistant](https://huggingface.co/google/gemma-4-31B-it-assistant)
- [GPT-OSS 20B DFlash candidate](https://huggingface.co/z-lab/gpt-oss-20b-DFlash)
- [Laguna XS 2.1 DFlash](https://huggingface.co/poolside/Laguna-XS-2.1-DFlash)
- [Laguna XS 2.1 INT4-target DFlash](https://huggingface.co/poolside/Laguna-XS-2.1-DFlash-INT4)
- [Laguna XS 2.1 NVFP4 MLX configuration](https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4-mlx/blob/main/config.json)
- [Laguna XS 2.1 NVFP4-target DFlash](https://huggingface.co/poolside/Laguna-XS-2.1-DFlash-NVFP4)
- [Official Muse GGUF release](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF)
- [Official Gemma 4 31B W4A16 configuration](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct/blob/main/config.json)
- [Official GPT-OSS weights](https://huggingface.co/openai/gpt-oss-20b)
- [Qwen publisher inventory](https://huggingface.co/Qwen/models)
- [Qwen 3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Mistral publisher inventory](https://huggingface.co/mistralai/models)
- [Devstral Small 2](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512)
- [Voxtral collection](https://huggingface.co/collections/mistralai/voxtral)

## Added decision-model scope: Bespoke Nimble (2 October 2026)

The owner approved native Nimble inference and training with preparation in
Afterglow. This scoped addition permits the publisher-owned Qwen3.5-9B
full-precision base and Bespoke LoRA adapter as source artifacts. It does not
change the four-bit download policy for other model families.

`midnight download nimble-9b` resolves immutable revisions and transfers both
components through the shared native Swift Hugging Face service. The recorded
base is `Qwen/Qwen3.5-9B` at `c202236235762e1c871ad0ccb60c8ee5ba337b9a`;
the adapter is `bespokelabs/Bespoke-Nimble-9B` at
`bd792f44ec8e265be861bfcdf4e05967ffe0e858`. Combined selected size is 19.52 GB.
The LoRA component is not a speculative assistant. Downloads remain marked
as requiring Afterglow preparation and do not claim verified runtime quality.

Afterglow verifies adapter hashes, converts the PEFT layout, retains the exact
prompt/token contract and release-specific temperature, and exports standard
MLX artifacts. Local four-bit preparation requires separate quality and
calibration checks; no derived-weight publishing is authorized.
