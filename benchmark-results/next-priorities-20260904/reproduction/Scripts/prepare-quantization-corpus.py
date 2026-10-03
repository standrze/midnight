#!/usr/bin/env python3
"""Prepare the pinned 128-record code/math reference-likelihood corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

PURPOSE = "Teacher-forced reference likelihood; not generated-code pass@1 or math accuracy."
SOURCES = (
    {"name": "mbpp", "repo": "google-research/google-research",
     "revision": "f82046ba5aabbbb427dbfd38a254d26bff08b533", "path": "mbpp/mbpp.jsonl",
     "source_sha256": "ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f",
     "corpus_sha256": "51735211d10adccfb201b6bbb875701b45a60af63aa63b9b986a37d2c2c4b771"},
    {"name": "gsm8k", "repo": "openai/grade-school-math",
     "revision": "b0bb162abedc65e1fdd8e93ed090fd7598ee68bc", "path": "grade_school_math/data/test.jsonl",
     "source_sha256": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
     "corpus_sha256": "d387a74772e4f3dc7329d6158f37ce8084a6f8a1e38a95f277461bb89e74bd8b"},
)
COMBINED_SHA256 = "5fc84a9794338a138e7284f415018323aa6bfec54227d09db8a95ad562e532f6"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify(data, expected, description):
    actual = sha256(data)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {description}: expected {expected}, got {actual}")
    return data


def source_url(source):
    return f"https://raw.githubusercontent.com/{source['repo']}/{source['revision']}/{source['path']}"


def reference_payload(name, data):
    items = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if name == "mbpp":
        items = [item for item in items if 11 <= item["task_id"] <= 74]
        records = [{"id": f"mbpp-test-{item['task_id']}", "category": "code-reference",
                    "text": f"Task: {item['text']}\nSolution:\n{item['code']}"} for item in items]
        if [item["task_id"] for item in items] != list(range(11, 75)):
            raise ValueError("MBPP source must contain ordered tasks 11...74 exactly once")
    elif name == "gsm8k":
        records = [{"id": f"gsm8k-test-{index}", "category": "math-reference",
                    "text": f"Question: {item['question']}\nAnswer: {item['answer']}"}
                   for index, item in enumerate(items[:64])]
    else:
        raise ValueError(f"Unknown source: {name}")
    if len(records) != 64:
        raise ValueError(f"Expected exactly 64 records for {name}, got {len(records)}")
    # Preserve the original experiment's UTF-8 bytes, spaces and terminal newlines.
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode("utf-8")


def read_source(source, output, cache, offline):
    filename = f"{source['name']}-source.jsonl"
    candidates = [output / filename]
    if cache is not None and cache.resolve() != output.resolve():
        candidates.append(cache / filename)
    for candidate in candidates:
        if candidate.exists():
            # Never silently replace an incorrect cache entry with a network response.
            return verify(candidate.read_bytes(), source["source_sha256"], str(candidate))
    if offline:
        raise ValueError(f"Offline source missing: {filename}; provide --cache-dir with verified source files")
    request = urllib.request.Request(source_url(source), headers={"User-Agent": "midnight-pinned-corpus/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    return verify(data, source["source_sha256"], source_url(source))


def prepare(output, cache=None, offline=False, sources=SOURCES, combined_sha256=COMBINED_SHA256):
    output = Path(output)
    cache = Path(cache) if cache is not None else None
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output is not a directory: {output}")
    if cache is not None and not cache.is_dir():
        raise ValueError(f"Cache is not a directory: {cache}")
    files, entries, payloads = {}, [], []
    for source in sources:
        data = read_source(source, output, cache, offline)
        payload = verify(reference_payload(source["name"], data), source["corpus_sha256"], f"{source['name']} corpus")
        files[f"{source['name']}-source.jsonl"] = data
        files[f"{source['name']}-64.jsonl"] = payload
        payloads.append(payload)
        # Same ordering and bytes as the provenance used by the original experiment.
        entries.append({"name": source["name"], "repo": source["repo"], "revision": source["revision"],
                        "path": source["path"], "url": source_url(source),
                        "source_sha256": source["source_sha256"], "corpus_sha256": source["corpus_sha256"],
                        "sample_count": 64, "purpose": PURPOSE})
    combined = verify(b"".join(payloads), combined_sha256, "code-math-128.jsonl")
    files["code-math-128.jsonl"] = combined
    files["provenance.json"] = (json.dumps(entries, indent=2) + "\n").encode("utf-8")
    # Check all existing outputs before writing anything, so a conflict leaves them intact.
    for name, payload in files.items():
        target = output / name
        if target.exists() and (not target.is_file() or target.read_bytes() != payload):
            raise ValueError(f"Refusing to overwrite differing output: {target}")
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        target = output / name
        if target.exists():
            continue
        # Exclusive creation also prevents overwriting a file created after preflight.
        with target.open("xb") as stream:
            stream.write(payload)
    return {"corpus": str((output / "code-math-128.jsonl").resolve()), "sample_count": 128,
            "sha256": combined_sha256, "purpose": PURPOSE,
            "files": {name: sha256(payload) for name, payload in files.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, help="Reuse pinned *-source.jsonl files; hashes are always checked")
    parser.add_argument("--offline", action="store_true", help="Never use the network; verify and reuse cached sources")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(prepare(args.output_dir, args.cache_dir, args.offline), indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Corpus preparation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
