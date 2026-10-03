# Editable model cards

## Available voices

Speech cards include a runtime-derived `voices` array. You can read it from
`GET /v1/models` or `GET /v1/models/{id}` even before loading the model.
Voxtral and Qwen choices come from checkpoint configuration; VibeVoice choices
come from the installed worker's presets and available local reference files.
Once loaded, the actual synthesizer catalog is authoritative. User-written
`voices` in a card file cannot override the backend's choices.

```json
{"id":"female","slug":"female","name":"Female","languages":["en"],"gender":"female","requires_reference_audio":false}
```

Use `id` or `slug` as the OpenAI speech `voice` or Mistral-style `voice_id`.
Voxtral IDs are stable UUIDs and slugs are readable names. `clone` requires
Mistral-style `ref_audio`; Qwen Base also requires `ref_text`. It is a request-time
selector, not a stored voice. Missing `voices` means unknown/not applicable;
an empty array means the catalog is known but has no choices. Optional gender
is omitted when unknown. Languages describe Midnight's current voice routing.
`GET /v1/audio/voices` still provides the loaded model's fuller voice-resource
metadata; it does not load a model or list voices across all installed models.

```sh
curl -s -H "Authorization: Bearer $MIDNIGHT_API_KEY" http://127.0.0.1:8096/v1/models | jq '.data[] | {id, voices: .model_card.voices}'
```

## Editable display metadata

Each model folder can contain `model-card.json`:

```json
{
  "name": "Talkie 1930 (ScaleSearch)",
  "description": "Historical English model; local ScaleSearch conversion."
}
```

`name` is the common display name for clients. It does not change the served
model ID, folder name, or the `model` value used in chat requests and responses.
`description` is optional. The file supplies presentation metadata; it does not
override context limits, model architecture, quantization, or runtime facts.
Midnight adds vision and audio capabilities to the card it serves, based on the
selected backend.

## Automatic assistants

For Gemma 4 and Muse Glimmer on Metal with uncompressed KV caches and no LoRA
adapter, Midnight automatically pairs compatible assistants at model load time.
Explicit load/CLI/model-stack assistant paths take precedence. Otherwise, it
checks an explicit `assistant_model` in `model-card.json`, a downloaded Hugging
Face `README.md` naming an assistant, embedded `drafter/` or `assistant/`
directories, then `~/.midnight/drafters`. Set `MODEL_RUNNER_ASSISTANTS_DIR` to
change that last directory.

```json
{
  "name": "Gemma 4 26B A4B",
  "assistant_model": "google/gemma-4-26B-A4B-it-assistant"
}
```

`assistant_model` is a Midnight-specific field and is also returned in served
cards. It accepts a repository ID, an absolute local path, or a path relative to
the card directory. Repository IDs resolve to `owner--name` under the assistant
directory or the normal model directory. Missing repository checkpoints download
automatically during loading, using the Hugging Face client's existing credentials.
Private/gated repositories still require access. Downloads resolve `main` to a
fixed commit, enforce the existing 30 GB limit and root-level safetensors format,
validate configuration before fetching weights, and publish a staged download
only after validation. Download or preflight failure retains the resident model;
a later weight-loading failure follows the existing unload/load behavior.
Listing models never downloads weights. An unavailable target model is not downloaded.

README discovery recognizes quoted assistant repository assignments such as
`ASSISTANT_MODEL_ID = "owner/model"` and Hugging Face links whose repository name
contains `assistant` or `dflash`. It reads the downloaded card, does not execute
examples, and does not infer arbitrary prose or browse upstream cards. Multiple
different named repositories require an explicit `assistant_model` selection.
Configuration compatibility is always checked; incomplete installed candidates
are skipped, while an invalid explicitly named assistant causes preflight failure.
When several installed assistants are compatible, Gemma source provenance/model
identity distinguishes original and QAT variants; unresolved ambiguity runs the
target alone. Assistant discovery for other architectures, including Laguna,
is not supported yet; their existing explicit options remain available.

Set `autoAssistant: false` in `POST /v1/runtime/load`, `mlxRunner.autoAssistant:
false` in model-stack settings, or use `--no-auto-assistant` at startup to disable
automatic pairing and downloads. Explicit assistant paths still work. The chosen
path appears in the runtime's `loadRequest.gemmaAssistantModel` or
`loadRequest.dflashModel` and is printed on load. Structured output and forced
tool selection use target-only decoding. Automatic selection retains the source
precision unless assistant quantization is explicitly configured.

No new endpoint is added. Existing `GET /v1/models`, `GET /v1/models/{id}` and
managed runtime model descriptors include `model_card`. Existing
`GET /v1/inspector/model` includes `modelCard`, following the inspector's existing
camel-case field convention. For example, a model lookup contains:

```json
{
  "id": "talkie-1930-13b-it-midnight-q4r8-scalesearch-attn8",
  "object": "model",
  "model_card": {
    "name": "Talkie 1930 (ScaleSearch)",
    "capabilities": {
      "vision": false,
      "audio_input": false,
      "audio_output": false
    }
  }
}
```

`capabilities` is a Midnight-specific extension on the existing model routes.
The same object appears inside the inspector's `modelCard`; its audio fields
remain `audio_input` and `audio_output`. These booleans describe operations
available through the selected Midnight backend, not every capability of the
underlying model architecture:

| Selected backend | `vision` | `audio_input` | `audio_output` |
| --- | --- | --- | --- |
| Text | `false` | `false` | `false` |
| Managed vision | `true` | `false` | `false` |
| Voxtral TTS speech | `false` | `false` | `true` |
| Qwen3-TTS Base | `false` | `true` | `true` |
| Qwen3-TTS CustomVoice | `false` | `false` | `true` |
| VibeVoice 1.5B / 7B | `false` | `true` | `true` |

Vision means image input through non-streaming Chat Completions. It requires a
compatible FastVLM checkpoint and the separately built vision worker on
macOS/Metal. A text backend stays `vision:false` even when its architecture or
configuration includes a vision component.

Audio output means speech synthesis through `/v1/audio/speech`. Audio input
means request-time voice-reference conditioning, not transcription. Qwen3-TTS
Base requires `ref_text` with its reference audio; VibeVoice does not. Voxtral
and Qwen3-TTS CustomVoice remain output-only. A client can use
`audio_input || audio_output` for a general audio badge, with a directional label
such as “Audio output” or “Speech synthesis”. Use `vision` for image support.
When capability metadata is absent, as with older servers or a file-only card,
the capability is unknown; do not interpret absence as `false`.

Every served card receives the runtime capability snapshot, including the
fallback card for a folder without `model-card.json`. A `capabilities` object
saved by a user or publisher never overrides that snapshot. No fields need to
be added to existing card files, and names and descriptions retain their
existing meaning. These fields let clients choose capability labels; they do
not add badges to existing client applications by themselves.

Edit the file and reload the model through the existing load operation, or
restart its server. A loaded model retains its card snapshot, including during
an in-flight request or model replacement. A client should continue routing by
`id` and may display `model_card.name`. Display names need not be unique.

Cards are read from the selected model folder; for adapter bundles, use the
bundle root next to `base-model` and `adapter`. Existing folders without a card
continue to load and use the served ID as the fallback display name. Downloads
create a starter card when none was supplied, and preserve valid publisher cards.
Manually imported models can use the example above.

The file must be a JSON object no larger than 64 KiB. `name` must be a nonblank
string of at most 256 characters without control characters. Surrounding
whitespace is trimmed. Unknown fields are ignored by this version. An invalid
card produces a preflight error before replacing a currently loaded model.

Treat descriptions as user/publisher metadata, not trusted instructions or
independently verified claims about capabilities. Cards never enter the chat
prompt automatically.
