#!/usr/bin/env python3
"""Build a deterministic JSONL language-model benchmark from plain text or JSON text rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() != ".json":
        return raw

    value = json.loads(raw)
    if isinstance(value, dict) and isinstance(value.get("instances"), list):
        rows = value["instances"]
    elif isinstance(value, list):
        rows = value
    else:
        raise ValueError("JSON input must be a list or contain an 'instances' list")

    texts: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError(f"JSON text row {index} does not contain a string 'text'")
        text = row["text"].strip()
        if text:
            texts.append(text)
    return "\n\n".join(texts)


def paragraph_chunks(text: str, target_characters: int) -> list[str]:
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    pending: list[str] = []
    pending_size = 0
    for paragraph in paragraphs:
        addition = len(paragraph) + (2 if pending else 0)
        if pending and pending_size + addition > target_characters:
            chunks.append("\n\n".join(pending))
            pending = []
            pending_size = 0
        if len(paragraph) > target_characters and not pending:
            for start in range(0, len(paragraph), target_characters):
                piece = paragraph[start : start + target_characters].strip()
                if piece:
                    chunks.append(piece)
            continue
        pending.append(paragraph)
        pending_size += addition
    if pending:
        chunks.append("\n\n".join(pending))
    return chunks


def evenly_spaced(items: list[str], count: int) -> list[str]:
    if len(items) < count:
        raise ValueError(f"requested {count} samples but only built {len(items)} chunks")
    if count == 1:
        return [items[0]]
    return [items[(index * (len(items) - 1)) // (count - 1)] for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--target-characters", type=int, default=2_500)
    parser.add_argument("--id-prefix", default="heldout")
    parser.add_argument("--category", default="language-modeling")
    args = parser.parse_args()

    if args.samples <= 0 or args.target_characters < 128:
        parser.error("--samples must be positive and --target-characters must be at least 128")
    if args.source.resolve() == args.output.resolve():
        parser.error("source and output must differ")

    source_bytes = args.source.read_bytes()
    text = load_text(args.source)
    chunks = evenly_spaced(paragraph_chunks(text, args.target_characters), args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for index, chunk in enumerate(chunks):
            json.dump(
                {
                    "id": f"{args.id_prefix}-{index + 1:03d}",
                    "category": args.category,
                    "text": chunk,
                },
                stream,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            stream.write("\n")

    output_bytes = args.output.read_bytes()
    print(f"samples={len(chunks)} characters={sum(map(len, chunks))}")
    print(f"source_sha256={hashlib.sha256(source_bytes).hexdigest()}")
    print(f"output_sha256={hashlib.sha256(output_bytes).hexdigest()}")


if __name__ == "__main__":
    main()
