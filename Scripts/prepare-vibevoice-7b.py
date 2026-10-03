"""Download and hash-verify the pinned community Transformers checkpoint."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "vibevoice/VibeVoice-7B-hf"
REVISION = "aae5684b9755da90bb1c75417a9bca4454dab418"
DEST = ROOT / "Models/VibeVoice-7B-hf"

if __name__ == "__main__":
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    snapshot_download(REPO, revision=REVISION, local_dir=DEST,
                      allow_patterns=["*.json", "*.safetensors", "*.jinja", "README.md"],
                      max_workers=3)
    verified = {}
    for entry in info.siblings:
        path = DEST / entry.rfilename
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if entry.lfs and digest != entry.lfs.sha256:
                raise RuntimeError("Hash mismatch: " + entry.rfilename)
            verified[entry.rfilename] = digest
            print("Verified", entry.rfilename, flush=True)
    (DEST / "midnight-download-provenance.json").write_text(json.dumps({
        "repository": REPO, "revision": REVISION, "sha256": verified,
        "provenance_note": "Community preservation/conversion; not independently authenticated by Microsoft."
    }, indent=2) + "\n")
    print("Ready:", DEST, flush=True)
