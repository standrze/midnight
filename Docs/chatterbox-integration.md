# Chatterbox integration: plan, implementation, and verification

Implemented on 6 September 2026. Midnight can serve Chatterbox Turbo and a
Multilingual V3 checkpoint through its existing OpenAI-compatible speech route.
The extra controls are model-local configuration, not HTTP extensions.

## Plan and outcome

1. Keep the existing HTTP handler and Voxtral implementation unchanged.
   Completed: no modifications to ModelHTTPServer.swift, AudioSpeechWire.swift,
   LocalSpeechSynthesizer.swift, or the Voxtral implementation files.
2. Add a native Swift/MLX adapter selected by a model-local chatterbox.json.
   Completed, with a pinned audio dependency and narrow compatibility patches.
3. Put model-specific settings and reference voices in configuration.
   Completed; settings are loaded at startup, validated, and passed to inference.
4. Verify both actual checkpoints and existing API behavior.
   Completed functional tests described below. Acoustic parity with the official
   implementation is not established.

## API surface

**New endpoints: none. New request fields: none.**

| Existing API | Chatterbox behavior |
| --- | --- |
| POST /v1/audio/speech | Standard model, input, voice, response_format, speed, and stream_format fields |
| GET /v1/models | Lists the model loaded by that process |
| GET /v1/models/{model} | Existing loaded-model descriptor |
| Existing Mistral speech/voice routes | Same handler and schema; the adapter supplies configured voice descriptors |
| Existing chat routes | No changes; a speech-only process does not become a chat model server |

The input limit remains 1–4096 characters. The voice may be a string or the
existing object form {"id":"..."}. Voice names come from configuration;
"default" denotes checkpoint-provided conditioning when available. OpenAI's
hosted voices such as "alloy" are not shipped or impersonated.

The default response format is MP3, as before. Chatterbox supports MP3, Opus,
AAC, FLAC, WAV and PCM when FFmpeg is installed. WAV and PCM at speed 1 work
without FFmpeg. OpenAI PCM remains headerless signed 16-bit mono at 24 kHz.
Speed 0.25–4 uses FFmpeg's tempo filter, not a change to the model's emotion
setting. FFmpeg is optional and is not installed by this change.

Binary audio and the existing speech.audio.delta / speech.audio.done SSE shapes
are preserved. Synthesis is buffered: the model completes the waveform before
the adapter emits it. SSE availability does not imply incremental model
generation. The backend does not expose token usage; completion usage is zero
rather than counting audio samples as tokens.

Unsupported free-form instructions are rejected by the existing OpenAI error
path. Emotion and guidance belong in chatterbox.json. Invalid voice, model,
input and speed retain the existing error envelope and status handling.

## Installed local layout

Models were downloaded into:

- ~/.midnight/models/chatterbox-turbo
- ~/.midnight/models/chatterbox-multilingual
- ~/.midnight/models/s3-tokenizer

Each speech model has its own chatterbox.json. The multilingual model also has
reference.wav: a generated sample of Turbo's built-in voice, used for the
reference-conditioning smoke test. Replace this with the desired reference
recording and restart to change the voice.

The source checkpoint config.json remains separate from Midnight settings.
The loader can create missing tokenizer JSON metadata in the model directory,
so the initial load needs a writable model folder. Inference does not download
the speech tokenizer: it uses the explicitly configured local directory.

Generation samples are saved in:

- ~/.midnight/samples/chatterbox-turbo.wav
- ~/.midnight/samples/chatterbox-polish.wav

Total installed model data is approximately 3.3 GiB. No existing model folders
were moved or replaced.

## Configuration

Copy Examples/chatterbox-turbo.json or Examples/chatterbox-multilingual.json
into the relevant model directory as chatterbox.json. These templates have
also been installed for the downloaded checkpoints.

Turbo example:

```json
{
  "variant": "turbo",
  "language": "en",
  "temperature": 0.8,
  "top_p": 0.95,
  "voices": {}
}
```

Multilingual example:

```json
{
  "variant": "multilingual",
  "language": "pl",
  "temperature": 0.8,
  "exaggeration": 0.5,
  "cfg_weight": 0.5,
  "speech_tokenizer": "../s3-tokenizer",
  "voices": {
    "reference": "reference.wav"
  }
}
```

| Setting | Default / behavior |
| --- | --- |
| variant | Required: turbo or multilingual; checked against checkpoint metadata |
| language | en; Turbo accepts only en; multilingual includes pl |
| temperature | 0.8; greater than 0 and at most 2 |
| top_p | Turbo 0.95, multilingual 1; greater than 0 and at most 1 |
| top_k | Turbo 1000; multilingual requires 0 |
| min_p | Multilingual 0.05; Turbo requires 0 |
| repetition_penalty | 1.2; positive |
| max_tokens | 1000 speech tokens; 1–4096, also capped by the checkpoint/backend ceiling |
| exaggeration | Multilingual optional override in [0,1]; otherwise backend conditioning default |
| cfg_weight | Multilingual optional override in [0,1]; backend default 0.5 |
| voices | Map of public voice names to local reference audio files |
| speech_tokenizer | Local S3TokenizerV2 directory, required when voices are configured |
| ffmpeg_path | Optional executable override; otherwise checks /opt/homebrew/bin, /usr/local/bin, /usr/bin |

Reference audio and speech-tokenizer paths can be absolute, tilde-prefixed,
or relative to the model directory. "default" is reserved for built-in
conditioning. Unknown settings and controls unsupported by the selected
variant cause a startup error instead of being silently ignored.
Changes take effect after restart.

The config changes language conditioning before tokenization: the reference
language prefix, lowercase/NFKD preparation and [SPACE] tokens are applied.
Japanese and Chinese are currently rejected because their additional text
preparation is not implemented. The other configured language codes are:
ar, da, de, el, en, es, fi, fr, he, hi, it, ko, ms, nl, no, pl, pt, ru, sv, sw, tr.
Polish was exercised with a real checkpoint; this is not a quality certification
of every language. Optional upstream Hebrew diacritization and Russian stress
models are not included.

## Running

Use the existing build/launch script:

```bash
./run.sh --model chatterbox-turbo --port 8080
# Or, in a separate process:
./run.sh --model chatterbox-multilingual --port 8081
```

Midnight continues to load one model per process. To serve both at once, run
two processes on different ports. No model-switching router was added.

A standard speech request needs no extra fields:

```bash
curl http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"chatterbox-turbo","input":"Hello from Midnight.","voice":"default"}' \
  --output speech.mp3
```

For the installed multilingual config, use model "chatterbox-multilingual",
voice "reference", the other port, and Polish input. An OpenAI SDK can use the
same body with its base URL set to the corresponding /v1 URL.

Chatterbox's speech-token limit is configured in chatterbox.json. The CLI's
chat-oriented --max-tokens / mlxRunner.maximumTokens setting is rejected for
this adapter rather than silently ignored.

## Files changed for this integration

| File(s) | Purpose |
| --- | --- |
| Package.swift, Package.resolved | Add the pinned macOS audio dependency; existing dependency revisions unchanged |
| Sources/ModelRunner/main.swift | Select the new adapter when chatterbox.json exists; reject incompatible engine/chat options |
| Sources/ModelRunnerProtocol/ChatterboxSettings.swift | Strict local settings, variant validation and multilingual text preparation |
| Sources/ModelRunnerCore/ChatterboxSpeechSynthesizer.swift | Native loading, configured voices, serialized requests, pinned MLX execution and resource limits |
| Sources/ModelRunnerCore/ChatterboxAudioEncoding.swift | Chatterbox-only optional FFmpeg encoding/tempo support |
| Examples/chatterbox-*.json | Two installable config templates |
| Patches/mlx-audio-midnight-platform.patch | Match macOS 15 and compile only Chatterbox plus the shared generation protocol from the TTS target |
| Patches/mlx-audio-chatterbox-controls.patch | Wire sampling/token controls and explicit local speech-tokenizer loading |
| Patches/mlx-audio-chatterbox-cache-dtype.patch | Keep quantized Turbo decode embeddings in the prefill dtype; fixes an observed KV-cache assertion |
| prepare-dependencies.sh | Apply and verify the patches against the exact pinned dependency revision |
| Tests/ModelRunnerProtocolTests/ChatterboxSettingsTests.swift | Settings and text-conditioning checks |
| Tests/ModelRunnerProtocolTests/ChatterboxAudioEncodingTests.swift | PCM bytes and all six codecs |
| Tests/Integration/ChatterboxHTTP.swift | Reusable live OpenAI HTTP checks |
| README.md, THIRD_PARTY_NOTICES.md, this report | Usage, attribution and implementation report |

The previous ~/.runner to ~/.midnight discovery changes predate this
integration and are not new Chatterbox changes. Unrelated research documents
and existing worktree edits were preserved.

## Verification results

- Native debug build passed.
- The normal build-metal.sh dependency-preparation and Metal-kernel workflow
  passed. Dependency patches are revision-checked and idempotent.
- 26 focused tests across 7 suites passed, including existing OpenAI/Mistral
  wire formats, HTTP wiring and Voxtral audio/catalog regressions.
- Both real models returned HTTP 200 audio through /v1/audio/speech.
- Turbo generated an English WAV and default MP3; Multilingual V3 generated
  a Polish WAV with configured reference audio.
- The live HTTP harness passed against both models: discovery; all six audio
  formats; default MP3; SSE delta/done; speeds 0.25 and 4; errors for wrong
  model, unknown voice, instructions, invalid speed and empty input.
- The Polish sample was independently inspected as mono 24-kHz signed PCM WAV
  with a nonempty waveform. Listening-based intelligibility and full numerical
  parity have not been certified.

Reproduce the HTTP checks against running servers:

```bash
swift Tests/Integration/ChatterboxHTTP.swift \
  http://127.0.0.1:8080 chatterbox-turbo default /tmp/chatterbox-turbo-http
swift Tests/Integration/ChatterboxHTTP.swift \
  http://127.0.0.1:8081 chatterbox-multilingual reference /tmp/chatterbox-multilingual-http
```

## Remaining differences from hosted OpenAI / upstream Chatterbox

- Native Chatterbox is enabled only for macOS Metal. Linux/CUDA and CPU
  serving of existing models are unchanged.
- No free-form instruction following, hosted OpenAI voices, voice-registration
  endpoint, or simultaneous model switching within a process was added.
- Reference voices and model-specific controls are startup configuration.
- Streaming is buffered, and token usage is not reported by the backend.
- FFmpeg is required for compressed formats and speed changes.
- The upstream native port does not apply Perth watermarking; this integration
  does not claim watermark parity.
- Japanese/Chinese text preparation and the optional language-specific
  preprocessing models noted above remain unavailable.
- Checkpoint loading and functional audio tests are not proof of acoustic
  equivalence to the official implementation.

## Source provenance

- [Swift audio implementation](https://github.com/Blaizzy/mlx-audio-swift/tree/bf14ae0c26e4e85553dd989571cae29d70fa6735), MIT.
- [Turbo checkpoint](https://huggingface.co/mlx-community/chatterbox-turbo-4bit/tree/c63817725071d7b5269c7b558772d6e8cbf59cec).
- [Multilingual V3 checkpoint](https://huggingface.co/mlx-community/chatterbox-multilingual-v3/tree/03565773edd72e949572557597af8063bb49a18a).
- [S3 tokenizer checkpoint](https://huggingface.co/mlx-community/S3TokenizerV2/tree/e0c9886f0e1c35ae85b1f27277416fb19fc72bec).
- [Official multilingual reference](https://github.com/resemble-ai/chatterbox/blob/master/src/chatterbox/mtl_tts.py).
