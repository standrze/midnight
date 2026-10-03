#!/usr/bin/env python3
"""Preflight the optional Metal telemetry patch on copied files, then apply once."""
import argparse
import difflib
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PIN = '1f8e74e3f12f31365464a6867c6579f0e9b29d85'
TARGETS = ('mlx/backend/metal/device.cpp', 'mlx/backend/metal/command_timing.h')


def git(checkout, *args, input_text=None):
    return subprocess.run(['git', '-C', str(checkout), *args], input=input_text,
                          capture_output=True, text=True, check=False)


def require(result, operation):
    if result.returncode:
        raise ValueError(f'{operation}: {result.stderr.strip()}')
    return result.stdout


def identity(path):
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected a regular dependency file: {path}')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(checkout, patch, expected_revision=PIN):
    checkout, patch = Path(checkout).resolve(), Path(patch).resolve()
    revision = require(git(checkout, 'rev-parse', 'HEAD'), 'Read dependency revision').strip()
    if revision != expected_revision:
        raise ValueError(f'Refusing telemetry patch at unexpected MLX revision: {revision}')
    actual_targets = {line[6:].split('\t')[0] for line in patch.read_text().splitlines()
                      if line.startswith(('--- a/', '+++ b/'))}
    if actual_targets != set(TARGETS):
        raise ValueError('Telemetry patch target inventory differs from reviewed files')
    before = {name: identity(checkout / name) for name in TARGETS}
    difference = []
    with tempfile.TemporaryDirectory(prefix='midnight-metal-timing-prepare-') as temporary:
        fixture = Path(temporary)
        for name in TARGETS:
            source, target = checkout / name, fixture / name
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        forward = git(fixture, 'apply', '--check', '--whitespace=error-all', str(patch)).returncode == 0
        reverse = git(fixture, 'apply', '--reverse', '--check', str(patch)).returncode == 0
        if forward == reverse:
            raise ValueError('Ambiguous or drifted telemetry patch; no dependency files changed')
        if reverse:
            require(git(fixture, 'apply', '--reverse', str(patch)), 'Peel telemetry patch')
        require(git(fixture, 'apply', '--check', '--whitespace=error-all', str(patch)), 'Preflight telemetry replay')
        require(git(fixture, 'apply', str(patch)), 'Replay telemetry patch')
        for name in TARGETS:
            source, target = checkout / name, fixture / name
            a = source.read_text().splitlines(True) if source.exists() else []
            b = target.read_text().splitlines(True)
            difference.extend(difflib.unified_diff(a, b, fromfile='a/' + name if source.exists() else '/dev/null', tofile='b/' + name))
    payload = ''.join(difference)
    if {name: identity(checkout / name) for name in TARGETS} != before:
        raise ValueError('Dependency files changed during preflight; refusing concurrent mutation')
    if not payload:
        return False
    require(git(checkout, 'apply', '--check', '--whitespace=error-all', '-', input_text=payload), 'Preflight combined telemetry diff')
    # A single git apply avoids the partial state caused by applying files one at a time.
    require(git(checkout, 'apply', '-', input_text=payload), 'Apply combined telemetry diff')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True, help='Pinned MLX repository (not the outer mlx-swift checkout)')
    args = parser.parse_args()
    patch = Path(__file__).resolve().parents[1] / 'Patches/mlx-metal-command-timing.patch'
    try:
        changed = prepare(args.checkout, patch)
    except (ValueError, OSError) as error:
        print(f'Metal telemetry preparation failed: {error}', file=sys.stderr)
        return 1
    print('Metal telemetry source applied; runtime remains opt-in.' if changed else 'Metal telemetry source already applied; exact replay verified.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
