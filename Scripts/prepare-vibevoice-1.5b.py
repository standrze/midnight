"""Convert Microsoft's original weights with the pinned Transformers converter."""
from pathlib import Path
import transformers.models.vibevoice.convert_vibevoice_to_hf as converter
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REVISION = "c00898d257e6b46004e3e2866a47534085fb685a"

def pinned_download(**kwargs):
    return snapshot_download(**kwargs, revision=REVISION,
                             cache_dir=str(ROOT / "Models" / ".huggingface-cache"))

if __name__ == "__main__":
    converter.snapshot_download = pinned_download
    converter.convert_checkpoint(
        "microsoft/VibeVoice-1.5B",
        str(ROOT / "Models" / "VibeVoice-1.5B-hf"),
        push_to_hub=None, bfloat16=True,
    )
