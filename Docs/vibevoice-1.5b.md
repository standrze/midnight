# Microsoft VibeVoice 1.5B local runtime

This setup uses `microsoft/VibeVoice-1.5B` at revision
`c00898d257e6b46004e3e2866a47534085fb685a`. The original weights are
converted locally using Hugging Face Transformers' maintained converter at
revision `c587bc884db2c2e31fc2b8102314656b17aa07b1`.
No archived Microsoft fork or community model weights are used.

The Python environment lives in `Optional/vibevoice-1.5b-env`. The converted
checkpoint lives in `Models/VibeVoice-1.5B-hf`; original downloaded weights
are retained under `Models/.huggingface-cache`. Allow approximately 11 GB
for both copies, plus the Python environment. Conversion preserves BF16
weights; the sample runner loads Float32 for Apple MPS.

```sh
bash Scripts/setup-vibevoice-1.5b.sh
Optional/vibevoice-1.5b-env/bin/python Scripts/sample-vibevoice-1.5b.py
```

To condition a sample on an existing local voice recording:

```sh
Optional/vibevoice-1.5b-env/bin/python Scripts/sample-vibevoice-1.5b.py \
  --reference Samples/voxtral-english-voices/neutral-female.wav \
  --output Samples/vibevoice-1.5b/female.wav
```

Samples are saved as WAV with a companion JSON recording input, reference,
device, token count, duration and elapsed time. Inference loads local files
only. The smoke runner rejects non-finite outputs and generation that hits
its token ceiling. Midnight now registers converted `model_type: vibevoice`
checkpoints with its speech loader and runs a persistent private Python worker.

## Serve through Midnight

```sh
bash Scripts/run-vibevoice.sh
```

This builds Midnight and serves `vibevoice-1.5b` on `127.0.0.1:8096` using
Apple MPS. Set `MIDNIGHT_VIBEVOICE_PORT` to change the port. An existing listener
is not stopped. For other launch methods set `MIDNIGHT_VIBEVOICE_ROOT` to the
checkout; `MIDNIGHT_VIBEVOICE_PYTHON` optionally overrides the Python executable.
The normal CLI and `/v1/runtime/load` also accept the converted model directory.
CPU is supported with `engine: "cpu"`; Metal selects PyTorch MPS, not MLX.

```sh
curl http://127.0.0.1:8096/v1/audio/speech \
  -H "Authorization: Bearer $MIDNIGHT_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"vibevoice-1.5b","input":"Hello from Midnight.","voice":"female","response_format":"wav"}' \
  --output /tmp/midnight-vibevoice.wav
```

OpenAI-compatible speech accepts `default`, `female`, `male`, and
`cheerful-female`. Reference presets appear only when their local sample files
exist. Mistral-style speech accepts `voice_id: "clone"`, `ref_audio` containing
base64 audio, and `response_format: "wav"` or `"pcm"`; no `ref_text` is required
for VibeVoice. Do not combine `voice` with Mistral fields. References must be
finite mono 24-kHz audio, 1–30 seconds, at most 8 MiB decoded. Reference data
is held in memory, never saved or fetched from a URL. `clone` is a request-time
selection, not a stored custom voice.

Output is mono 24 kHz: OpenAI WAV/PCM is signed PCM16; Mistral WAV/PCM is
float32, wrapped in JSON `audio_data` base64 for non-streaming Mistral requests.
Explicitly request WAV or PCM: the default MP3 is unsupported. Speed
must be 1; instructions, multi-speaker scripts, transcription, and persistent
voice creation are unsupported. Audio is generated in full before delivery,
including the audio event in Mistral SSE; this is not incremental synthesis.
The worker serializes requests, loads weights once, and shuts down on model
release. Each exchange has a 180-second timeout. Cancellation or timeout stops
and joins that worker so unload can drain promptly; the next request lazily
starts a replacement worker.
Inputs are limited to 4096 characters, model output to at most 4096 tokens
(launch default 512). Outputs hitting the token ceiling fail rather than return
truncated speech. Discovery reports both audio capabilities; audio input means
reference conditioning, not speech recognition.

Validated on 19 September 2026 using Apple MPS: conversion passed weight-key
validation and reload. Female and male reference-conditioned utterances stopped
before the 160-token limit and produced valid 24-kHz mono PCM16 WAV files.
The female sample contains 5.87 seconds of speech generated in 5.97 seconds;
the male sample contains 4.67 seconds generated in approximately 5 seconds.
Timings exclude model loading. References are the existing synthetic Voxtral
samples in this workspace. Numerical validation does not establish subjective
voice similarity or transcription accuracy; listen to the samples to judge.

Sources: [Microsoft weights](https://huggingface.co/microsoft/VibeVoice-1.5B),
[Transformers implementation](https://huggingface.co/docs/transformers/main/model_doc/vibevoice).
