"""Local VibeVoice 1.5B speech smoke test using the maintained HF runtime."""
import argparse
from pathlib import Path
import time
import json
import torch
from transformers import AutoProcessor, AutoModelForTextToWaveform, set_seed

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Hello from Midnight. This is a local voice sample from Microsoft's VibeVoice model.")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "Samples/vibevoice-1.5b/default.wav")
    parser.add_argument("--max-tokens", type=int, default=160)
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_path = ROOT / "Models/VibeVoice-1.5B-hf"
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForTextToWaveform.from_pretrained(
        model_path, dtype=torch.float32, local_files_only=True,
    ).to(device).eval()
    content = [{"type": "text", "text": args.text}]
    if args.reference:
        content.append({"type": "audio", "path": str(args.reference.resolve())})
    inputs = processor.apply_chat_template(
        [{"role": "0", "content": content}], return_dict=True,
        tokenize=True, add_generation_prompt=True,
    ).to(device, model.dtype)
    set_seed(42)
    start = time.monotonic()
    with torch.inference_mode():
        result = model.generate(**inputs, max_new_tokens=args.max_tokens,
                                monitor_progress=True, return_dict_in_generate=True)
    audio = result.audio
    if not audio or audio[0] is None or not torch.isfinite(audio[0]).all():
        raise RuntimeError("VibeVoice did not produce finite audio samples")
    generated_tokens = result.sequences.shape[-1] - inputs.input_ids.shape[-1]
    if generated_tokens >= args.max_tokens:
        raise RuntimeError("Generation reached the token limit; increase --max-tokens before saving a complete sample")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    duration = audio[0].numel() / 24000
    processor.save_audio(audio, str(args.output))
    metrics = {"source": "microsoft/VibeVoice-1.5B", "device": device,
               "text": args.text, "reference": str(args.reference) if args.reference else None,
               "generated_tokens": generated_tokens,
               "seconds": duration,
               "elapsed_seconds": time.monotonic() - start}
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved {args.output}; device={device}; elapsed={time.monotonic()-start:.1f}s", flush=True)

if __name__ == "__main__":
    main()
