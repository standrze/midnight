"""Persistent, private stdio worker owned by Midnight's speech backend."""
import argparse
import base64
import contextlib
import io
import json
import sys
from pathlib import Path

wire = sys.stdout
sys.stdout = sys.stderr
import numpy as np
import soundfile as sf
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForTextToWaveform, set_seed

def reply(value):
    wire.write(json.dumps(value) + "\n")
    wire.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=["mps", "cpu"], required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--voices", required=True)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 4096:
        raise ValueError("VibeVoice maximum tokens must be between 1 and 4096")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    # The 7B backbone cannot afford an intermediate full Float32 CPU copy on
    # a 64-GB Mac. Preserve its stored BF16 precision and load onto MPS directly.
    large_mps = args.device == "mps" and config.text_config.hidden_size >= 3584
    dtype = torch.bfloat16 if large_mps else torch.float32
    model = AutoModelForTextToWaveform.from_pretrained(
        args.model, dtype=dtype, local_files_only=True,
        **({"device_map": args.device} if large_mps else {}),
    ).eval()
    if not large_mps:
        model = model.to(args.device)
    references = {"female": "neutral-female.wav", "male": "neutral-male.wav",
                  "cheerful-female": "cheerful-female.wav"}
    available = {key: str(Path(args.voices) / name) for key, name in references.items()
                 if (Path(args.voices) / name).is_file()}
    reply({"ready": True, "voices": ["default", "clone"] + list(available)})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            text = request["input"].strip()
            if not 1 <= len(text) <= 4096:
                raise ValueError("input must contain 1 to 4096 characters")
            content = [{"type": "text", "text": text}]
            reference = request.get("reference")
            if reference:
                raw = base64.b64decode(reference, validate=True)
                if len(raw) > 8 * 1024 * 1024:
                    raise ValueError("Reference audio exceeds 8 MiB")
                samples, rate = sf.read(io.BytesIO(raw), dtype="float32")
                if samples.ndim != 1 or rate != 24000 or not 24000 <= samples.size <= 30 * 24000:
                    raise ValueError("Reference must be mono 24-kHz audio lasting 1 to 30 seconds")
                if not np.isfinite(samples).all():
                    raise ValueError("Reference contains non-finite samples")
                # Passing samples directly prevents URL/file resolution from request data.
                content.append({"type": "audio", "audio": samples})
            elif request["voice"] in available:
                content.append({"type": "audio", "path": available[request["voice"]]})
            elif request["voice"] != "default":
                raise ValueError("clone requires reference audio")
            inputs = processor.apply_chat_template(
                [{"role": "0", "content": content}], tokenize=True,
                return_dict=True, add_generation_prompt=True,
            ).to(args.device, model.dtype)
            set_seed(42)
            with torch.inference_mode():
                result = model.generate(**inputs, max_new_tokens=args.max_tokens,
                                        return_dict_in_generate=True)
            count = result.sequences.shape[-1] - inputs.input_ids.shape[-1]
            if count >= args.max_tokens:
                raise ValueError("Speech reached the generation limit; shorten input or increase the model maxTokens")
            audio = result.audio[0]
            if audio is None or not torch.isfinite(audio).all():
                raise ValueError("VibeVoice returned invalid audio")
            audio = audio.detach().float().cpu().numpy().reshape(-1)
            pcm = request["encoding"]
            if request["format"] == "wav":
                output = io.BytesIO()
                sf.write(output, audio, 24000, format="WAV",
                         subtype="PCM_16" if pcm == "signedInt16LittleEndian" else "FLOAT")
                data = output.getvalue()
            elif request["format"] == "pcm":
                data = (np.rint(np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
                        if pcm == "signedInt16LittleEndian" else audio.astype("<f4").tobytes())
            else:
                raise ValueError("Only WAV and PCM are supported")
            reply({"audio": base64.b64encode(data).decode(),
                   "promptTokens": inputs.input_ids.shape[-1], "completionTokens": count})
        except Exception as error:
            reply({"error": str(error)})

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        reply({"error": str(error)})
        sys.exit(1)
