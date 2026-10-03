#!/usr/bin/env python3

import argparse
import json
import os
import shutil
from pathlib import Path


def normalize_name(name: str) -> str:
    prefix = "language_model."
    return name[len(prefix) :] if name.startswith(prefix) else name


def expand_fused_name(name: str) -> tuple[str, ...]:
    marker = ".gate_up_proj."
    if marker not in name:
        return (name,)
    return (
        name.replace(marker, ".gate_proj."),
        name.replace(marker, ".up_proj."),
    )


def normalize_quantization(policy: dict) -> dict:
    normalized = {}
    for key, value in policy.items():
        if key in {"bits", "group_size", "mode"}:
            normalized[key] = value
            continue
        for output_key in expand_fused_name(normalize_name(key)):
            normalized[output_key] = value
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    if not source.is_dir():
        raise SystemExit(f"Source model directory does not exist: {source}")
    if destination.exists():
        raise SystemExit(f"Destination already exists: {destination}")
    destination.mkdir(parents=True)

    for item in source.iterdir():
        if not item.is_file() or item.name in {"config.json", "model.safetensors.index.json"}:
            continue
        target = destination / item.name
        if item.suffix == ".safetensors":
            os.symlink(item.resolve(), target)
        else:
            shutil.copy2(item, target)

    config = json.loads((source / "config.json").read_text())
    config["quantization"] = normalize_quantization(config["quantization"])
    config.pop("quantization_config", None)
    (destination / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )

    source_index = source / "model.safetensors.index.json"
    if source_index.exists():
        index = json.loads(source_index.read_text())
        normalized_map = {}
        for key, shard in index["weight_map"].items():
            for output_key in expand_fused_name(normalize_name(key)):
                normalized_map[output_key] = shard
        index["weight_map"] = normalized_map
        (destination / source_index.name).write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n"
        )

    provenance = {
        "source": str(source),
        "method": "symlinked exact safetensors; namespace/fused views normalized at load",
        "requantized": False,
    }
    (destination / "vllm-compat.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
