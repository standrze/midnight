#!/usr/bin/env python3
"""Reproduce the pinned 192-record Laguna NLL corpus without downloading data."""
import argparse
import hashlib
import json
from pathlib import Path

PROVENANCE = Path(__file__).parent / 'corpora/laguna-heldout-192.provenance.json'


def prepare(wikitext, code_math, output, provenance=None):
    metadata = json.loads(PROVENANCE.read_text()) if provenance is None else provenance
    sources = [Path(wikitext), Path(code_math)]
    records = []
    for path, expected, count in zip(sources, metadata['source_files'], (64, 128)):
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected['sha256']:
            raise ValueError(f'Source SHA256 mismatch: {path}')
        source_records = [json.loads(line) for line in payload.decode('utf-8').splitlines() if line.strip()]
        if len(source_records) != count:
            raise ValueError(f'Expected {count} source records: {path}')
        records.extend(source_records)
    if len(records) != metadata['record_count'] or len({r['id'] for r in records}) != len(records):
        raise ValueError('Record count or unique IDs do not match provenance')
    if any(not isinstance(r.get('text'), str) or not r['text'].strip() for r in records):
        raise ValueError('Every record must have nonempty text')
    payload = ''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in records).encode('utf-8')
    if hashlib.sha256(payload).hexdigest() != metadata['output_sha256']:
        raise ValueError('Reconstructed payload SHA256 does not match provenance')
    output = Path(output)
    if output.exists():
        if output.read_bytes() != payload:
            raise ValueError('Output already exists with differing content')
        return metadata['output_sha256']
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(payload)
    return metadata['output_sha256']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wikitext', required=True, type=Path, help='Pinned 64-record WikiText-2 test JSONL')
    parser.add_argument('--code-math', required=True, type=Path, help='Pinned 128-record MBPP/GSM8K reference JSONL')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        digest = prepare(args.wikitext, args.code_math, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f'Verified 192 records; SHA256 {digest}')


if __name__ == '__main__':
    main()
