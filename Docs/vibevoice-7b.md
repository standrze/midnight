# VibeVoice 7B community mirror

Selected repository: [vibevoice/VibeVoice-7B-hf](https://huggingface.co/vibevoice/VibeVoice-7B-hf),
pinned revision `aae5684b9755da90bb1c75417a9bca4454dab418`.
This is the Transformers-compatible conversion linked by the
[community preservation project](https://github.com/vibevoice-community/VibeVoice).
It is not a new official Microsoft distribution or an independently authenticated backup.

The original-format community mirror at revision
`505114ae6ad17be74df98e6939707434ec49c187` has the same ten weight-shard
SHA-256 hashes as `aoi-ot/VibeVoice-7B` at
`097bdd0654dd9c48df30732a8bddccdb1ef6f96e`. The latter identifies Microsoft's
ModelScope release as its source. Matching mirrors corroborate the copy chain;
they do not substitute for an original Microsoft-signed manifest.

The selected conversion uses Safetensors and native Transformers support.
No Python from the mirror is downloaded or executed. The downloader verifies
LFS SHA-256 hashes, records hashes for the other files, and saves its manifest
as `Models/VibeVoice-7B-hf/midnight-download-provenance.json`.

## Download and serve

Use the existing pinned runtime installed for 1.5B:

```sh
Optional/vibevoice-1.5b-env/bin/python Scripts/prepare-vibevoice-7b.py
MIDNIGHT_VIBEVOICE_MODEL="$PWD/Models/VibeVoice-7B-hf" \
MIDNIGHT_VIBEVOICE_NAME=vibevoice-7b bash Scripts/run-vibevoice.sh
```

Do not start a second server on an occupied port. An existing Midnight listener
can switch using `POST /v1/runtime/load` with the converted directory,
`name: "vibevoice-7b"`, `engine: "metal"`, and `maxTokens: 512`.
Midnight drains and unloads the previous model before loading the replacement.
Keep the 1.5B files as a fallback.

The download is approximately 19 GB. The MPS worker loads the 7B backbone
in its stored BF16 precision directly onto the device (approximately 19 GB of
weights plus working memory). Full Float32 loading creates substantial memory
pressure on a 64-GB Mac. CPU still uses Float32; 1.5B retains Float32 on MPS.
Avoid loading both VibeVoice sizes simultaneously on a 64-GB Mac.

The same speech API, reference-cloning support, voice presets, and buffered
output limitations apply as documented in [the integration guide](vibevoice-1.5b.md).
The label 7B refers to the language backbone; the entire model is larger.

Run the live regression and sample generation with:

```sh
MIDNIGHT_TEST_MODEL=vibevoice-7b \
  Optional/vibevoice-1.5b-env/bin/python Scripts/test-vibevoice-http.py
```

Samples are saved separately under `Samples/vibevoice-7b/`.

## Local validation, 19 September 2026

All downloaded weight hashes passed. Midnight loaded the checkpoint on MPS
with BF16 and passed discovery, preset listing, OpenAI PCM16 WAV synthesis,
Mistral reference cloning without a transcript, SSE, and rejection of unknown
voices, unsupported formats and unsupported speeds. The female request took
11.24 seconds (first inference); the clone request took 4.90 seconds; the short
male SSE request took 1.83 seconds. These are different utterances and are not
a controlled model-quality or speed comparison. Listen to the saved audio to
assess pronunciation and resemblance.

The shared-worker regression also passed the same six checks with 1.5B.
The lower-memory direct-device loading is restricted to the 7B MPS branch;
1.5B retains its previously verified Float32 loading path.
