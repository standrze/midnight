"""Opt-in live smoke test: run after Scripts/run-vibevoice.sh (stdlib only)."""
import base64
import io
import json
import os
import time
from pathlib import Path
import urllib.error
import urllib.request
import wave

from midnight_api_auth import json_headers

root = Path(__file__).resolve().parents[1]
base = os.environ.get("MIDNIGHT_TEST_URL", "http://127.0.0.1:8096")
expected_model = os.environ.get("MIDNIGHT_TEST_MODEL", "vibevoice-1.5b")
output_dir = root / "Samples" / expected_model
output_dir.mkdir(parents=True, exist_ok=True)
measurements = []

def call(path, payload=None):
    started = time.monotonic()
    request = urllib.request.Request(base + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers=json_headers())
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read()
    if path == "/v1/audio/speech":
        measurements.append({"voice": payload.get("voice", payload.get("voice_id")),
                             "stream": payload.get("stream", False),
                             "elapsed_seconds": time.monotonic() - started})
    return data

model = json.loads(call("/v1/models"))["data"][0]
assert model["id"] == expected_model
assert model["model_card"]["capabilities"] == {
    "vision": False, "audio_input": True, "audio_output": True}
voices = json.loads(call("/v1/audio/voices"))
assert "cheerful-female" in json.dumps(voices)
request = {"model": model["id"], "input": "Hello from Midnight. Your voice is ready.",
           "voice": "female", "response_format": "wav"}
audio = call("/v1/audio/speech", request)
with wave.open(io.BytesIO(audio)) as wav:
    assert wav.getnchannels() == 1 and wav.getframerate() == 24000
    assert wav.getsampwidth() == 2 and wav.getnframes() > 24000
    assert any(wav.readframes(wav.getnframes()))
output = output_dir / "midnight-http-female.wav"
output.write_bytes(audio)
print("PASS discovery, voices, OpenAI PCM16 WAV:", output, flush=True)

reference = base64.b64encode((root / "Samples/voxtral-english-voices/neutral-female.wav").read_bytes()).decode()
cloning = {"model": model["id"], "input": "This is voice cloning through Midnight.",
           "voice_id": "clone", "ref_audio": reference, "response_format": "wav"}
audio = base64.b64decode(json.loads(call("/v1/audio/speech", cloning))["audio_data"])
assert audio[:4] == b"RIFF" and len(audio) > 24000
output = output_dir / "midnight-http-clone.wav"
output.write_bytes(audio)
print("PASS reference cloning without transcript:", output, flush=True)

streamed = call("/v1/audio/speech", {
    "model": model["id"], "input": "Hello again.", "voice_id": "male",
    "response_format": "pcm", "stream": True})
assert b"data:" in streamed and b"error" not in streamed
print("PASS Mistral SSE", flush=True)
for change in [{"voice": "missing"}, {"response_format": "mp3"}, {"speed": 1.5}]:
    try:
        call("/v1/audio/speech", request | change)
    except urllib.error.HTTPError as error:
        assert 400 <= error.code < 500, error.code
    else:
        raise AssertionError("Invalid request accepted: " + str(change))
print("PASS unknown voice, unsupported format and speed rejection", flush=True)
(output_dir / "midnight-http-validation.json").write_text(json.dumps({
    "model": model, "measurements": measurements, "validation": "passed"
}, indent=2) + "\n")
print(json.dumps(measurements, indent=2), flush=True)
