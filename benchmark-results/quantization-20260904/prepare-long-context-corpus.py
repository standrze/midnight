#!/usr/bin/env python3
"""Recreate the eight-record long-context corpus from the pinned 64-record source."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Existing 64-record WikiText-2 test JSONL")
    parser.add_argument("--output", required=True, type=Path, help="New eight-record JSONL path")
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).parent / "corpora" / "wikitext2-long-8.provenance.json").read_text())
    payload = args.source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["source_sha256"]:
        parser.error("Source SHA256 does not match the archived provenance")
    records = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if len(records) != manifest["source_records"] or len(records) != 64:
        parser.error("Expected exactly 64 source records")
    if any(not isinstance(record.get("text"), str) for record in records):
        parser.error("Every source record must contain text")
    output_records = [
        {
            "id": f"wikitext2-heldout-long-{offset // 8}",
            "category": "long-prose-reference",
            "text": "\n\n".join(record["text"] for record in records[offset:offset + 8]),
        }
        for offset in range(0, 64, 8)
    ]
    # UTF-8, ordinary json.dumps spacing, insertion-order keys, final newline.
    output = ("\n".join(json.dumps(record, ensure_ascii=False) for record in output_records) + "\n").encode("utf-8")
    digest = hashlib.sha256(output).hexdigest()
    if len(output_records) != manifest["output_records"] or digest != manifest["output_sha256"]:
        parser.error("Reconstructed corpus does not match the archived output SHA256")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(output)
    print(f"Wrote {len(output_records)} records; SHA256 {digest}")


if __name__ == "__main__":
    main()
