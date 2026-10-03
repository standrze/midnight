#!/usr/bin/env python3
"""Verify compressed XML hashes and reproduce saved CPU profile totals/symbol weights.

Run from any directory with Python 3. Uses only the archived XML and frozen parser;
no Instruments, native inference, GPU, downloads, or writes are performed. Equal-
weight symbols may appear in a different order because the parser uses Python sets;
comparison therefore uses symbol-keyed weights rather than incidental tie ordering.
"""
import copy
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def canonical(report):
    result = copy.deepcopy(report)
    for table in ('leaf', 'inclusive'):
        values = result[table]
        mapped = {item['symbol']: item for item in values}
        if len(mapped) != len(values):
            raise ValueError('Duplicate symbols in ' + table)
        result[table] = mapped
    return result


def main():
    sys.dont_write_bytecode = True
    root = Path(__file__).resolve().parent
    parser_path = root.parent / 'reproduction/Scripts/analyze-time-profiler.py'
    snapshot = json.loads((root.parent / 'reproduction/snapshot-provenance.json').read_text())
    expected = next(row['sha256'] for row in snapshot['files']
                    if row['archive_path'] == 'Scripts/analyze-time-profiler.py')
    if hashlib.sha256(parser_path.read_bytes()).hexdigest() != expected:
        raise ValueError('Frozen analyzer hash mismatch')
    spec = importlib.util.spec_from_file_location('archived_time_profiler', parser_path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    manifest = json.loads((root / 'compression-provenance.json').read_text())
    for profile in manifest['profiles']:
        compressed = (root / profile['archive_name']).read_bytes()
        if (len(compressed) != profile['compressed_bytes'] or
                hashlib.sha256(compressed).hexdigest() != profile['compressed_sha256']):
            raise ValueError('Compressed XML hash/size mismatch')
        raw = gzip.decompress(compressed)
        if (len(raw) != profile['original_bytes'] or
                hashlib.sha256(raw).hexdigest() != profile['original_sha256']):
            raise ValueError('Original XML hash/size mismatch')
        interval = profile['analysis_interval_seconds']
        with gzip.open(root / profile['archive_name'], 'rb') as stream:
            actual = parser.analyze(stream, interval['start'], interval['end'])
        expected = json.loads((root / profile['analysis_name']).read_text())
        if canonical(actual) != canonical(expected):
            raise ValueError('Saved analysis mismatch: ' + profile['analysis_name'])
        print(profile['analysis_name'] + ': exact totals and per-symbol weights reproduced')


if __name__ == '__main__':
    main()
