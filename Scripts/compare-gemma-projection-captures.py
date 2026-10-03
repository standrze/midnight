#!/usr/bin/env python3
"""CPU-only comparison of real teacher-forced projection captures; requires numpy."""
import argparse
import json
from pathlib import Path
import re
import struct
import numpy as np


def arrays(path):
    with path.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(size))
    output = {}
    for key, item in header.items():
        if key == '__metadata__':
            continue
        dtype = {'BF16': '<u2', 'F16': '<f2', 'F32': '<f4'}[item['dtype']]
        data = np.memmap(path, mode='r', dtype=dtype, offset=8 + size + item['data_offsets'][0],
                         shape=tuple(item['shape']))
        output[key] = (data.astype(np.uint32) << 16).view(np.float32) if item['dtype'] == 'BF16' else data.astype(np.float32)
    return output


def order(name):
    layer = re.search(r'layers\.(\d+)\.', name)
    rank = next((i for i, key in enumerate(('q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj')) if key in name), 99)
    return (int(layer.group(1)) if layer else 10000, rank, name)


def compare(left, right):
    manifest_a = json.loads((left / 'manifest.json').read_text())
    manifest_b = json.loads((right / 'manifest.json').read_text())
    assert manifest_a['token_ids'] == manifest_b['token_ids']
    assert manifest_a['checkpoint'] == manifest_b['checkpoint']
    files = sorted(left.glob('*.safetensors'))
    assert [path.name for path in files] == sorted(path.name for path in right.glob('*.safetensors'))
    assert len(files) == len(manifest_a['token_ids'])
    rows = []
    for a_path in files:
        a, b = arrays(a_path), arrays(right / a_path.name)
        assert set(a) == set(b)
        for name in sorted((key[:-7] for key in a if key.endswith('.output')), key=order):
            x_a, x_b = a[name + '.input'], b[name + '.input']
            y_a, y_b = a[name + '.output'], b[name + '.output']
            assert x_a.shape == x_b.shape and y_a.shape == y_b.shape
            if not all(np.all(np.isfinite(value)) for value in (x_a, x_b, y_a, y_b)):
                raise ValueError(f'Nonfinite capture: {a_path.name} {name}')
            input_equal = np.array_equal(x_a, x_b)
            delta = y_b - y_a
            count = int(np.count_nonzero(delta))
            if not count:
                continue
            rows.append({
                'position': int(a_path.stem), 'projection': name, 'identical_input': input_equal,
                'input_max_abs': float(np.max(np.abs(x_a))),
                'input_max_change': float(np.max(np.abs(x_b - x_a))),
                'output_changed': count, 'output_elements': y_a.size,
                'output_max_change': float(np.max(np.abs(delta))),
                'output_relative_rms': float(np.sqrt(np.mean(delta.astype(np.float64) ** 2)
                    / max(np.mean(y_a.astype(np.float64) ** 2), 1e-30))),
            })
    local = [row for row in rows if row['identical_input']]
    return {'scope': 'Fixed-prefix projection localization; not throughput',
            'first_divergence': rows[0] if rows else None,
            'first_divergence_with_identical_input': local[0] if local else None,
            'different_projection_calls': len(rows), 'identical_input_different_output_calls': len(local),
            'rows': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}, indent=2))
