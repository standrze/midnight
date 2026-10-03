"""Live installed-model and automatic speech switching regression (no control calls)."""
import io
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import wave

from midnight_api_auth import json_headers

base = os.environ.get("MIDNIGHT_TEST_URL", "http://127.0.0.1:8096")
root = Path(__file__).resolve().parents[1]
output = root / "Samples/automatic-model-switching"
output.mkdir(parents=True, exist_ok=True)

def call(path, body=None):
    request = urllib.request.Request(base + path,
        data=None if body is None else json.dumps(body).encode(),
        headers=json_headers())
    with urllib.request.urlopen(request, timeout=240) as response:
        return response.read()

ids = {m["id"] for m in json.loads(call("/v1/models"))["data"]}
selections = [("VibeVoice-1.5B-hf", "female"),
              ("mlx-community--Voxtral-4B-TTS-2603-mlx-4bit", "neutral_female"),
              ("VibeVoice-7B-hf", "female")]
assert all(model in ids for model, _ in selections), ids
metrics = []
for model, voice in selections:
    start = time.monotonic()
    audio = call("/v1/audio/speech", {
        "model": model, "voice": voice, "input": "Midnight loaded the voice I requested.",
        "response_format": "wav"})
    elapsed = time.monotonic() - start
    with wave.open(io.BytesIO(audio)) as wav:
        assert wav.getnframes() > 0 and wav.getnchannels() == 1
        duration = wav.getnframes() / wav.getframerate()
    (output / (model + ".wav")).write_bytes(audio)
    status = json.loads(call("/v1/runtime"))
    assert status["phase"] == "ready" and status["loadedModel"]["id"] == model
    catalog = json.loads(call("/v1/models"))["data"]
    assert {m["id"] for m in catalog} >= ids
    assert [m["id"] for m in catalog if m["loaded"]] == [model]
    metrics.append({"model": model, "load_and_speech_seconds": elapsed, "audio_seconds": duration})
    print("PASS", metrics[-1], flush=True)
try:
    call("/v1/audio/speech", {"model": "/tmp/not-a-model", "voice": "female", "input": "No."})
except urllib.error.HTTPError as error:
    assert error.code == 404
else:
    raise AssertionError("Unknown model was accepted")
assert json.loads(call("/v1/runtime"))["loadedModel"]["id"] == selections[-1][0]
(output / "validation.json").write_text(json.dumps(metrics, indent=2) + "\n")
print("PASS unknown-model rejection without replacing active model", flush=True)
