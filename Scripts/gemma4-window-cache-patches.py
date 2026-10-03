#!/usr/bin/env python3
"""Prepare the overlapping Gemma 4 model overlays after an exact copied-tree replay."""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile

PIN = "14414441fa44f45eee35a61e9fa0bab577cf9734"
PATCHES = (
    "mlx-swift-lm-gemma4-nonrotating-cache.patch",
    "mlx-swift-lm-gemma4-dense-fusion.patch",
    "mlx-swift-lm-gemma4-window-mask.patch",
    "mlx-swift-lm-gemma4-window-slicing.patch",
    "mlx-swift-lm-gemma4-text-mtp.patch",
    "mlx-swift-lm-gemma4-text-assistant.patch",
    "mlx-swift-lm-gemma4-expert-gate-up.patch",
    "mlx-swift-lm-gemma4-bounded-window-cache.patch",
    "mlx-swift-lm-gemma-assistant-unmasked.patch",
)


def git(checkout, *arguments, input_text=None):
    return subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        input=input_text, text=True, capture_output=True, check=False,
    )


def require(result, operation):
    if result.returncode:
        raise ValueError(f"{operation}: {result.stderr.strip()}")
    return result.stdout


def patch_paths(patch):
    paths = set()
    for line in patch.read_text().splitlines():
        if not line.startswith(("--- a/", "+++ b/")):
            continue
        relative = line[6:].split("\t", 1)[0]
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts or not relative.startswith("Libraries/"):
            raise ValueError(f"Unsupported dependency patch path: {relative}")
        paths.add(relative)
    if not paths:
        raise ValueError(f"Dependency patch has no source paths: {patch}")
    return paths


def prepare(root, checkout, patches=PATCHES, expected_revision=PIN):
    """Preflight all overlays without changing the checkout, then apply one combined diff."""
    root, checkout = Path(root).resolve(), Path(checkout).resolve()
    revision = require(git(checkout, "rev-parse", "HEAD"), "Read dependency revision").strip()
    if revision != expected_revision:
        raise ValueError(f"Refusing Gemma 4 overlays for unexpected mlx-swift-lm revision: {revision}")
    overlays = [root / "Patches" / name for name in patches]
    paths = sorted(set().union(*(patch_paths(patch) for patch in overlays)))
    changes = []
    with tempfile.TemporaryDirectory(prefix="midnight-gemma4-preflight-") as temporary:
        fixture = Path(temporary)
        for relative in paths:
            source, destination = checkout / relative, fixture / relative
            if source.exists():
                if source.is_symlink() or not source.is_file():
                    raise ValueError(f"Expected a regular source file: {source}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)

        # Later overlays deliberately alter earlier hunks. Remove exact overlays
        # in reverse order, then prove the complete forward sequence. A partial
        # or drifted overlay survives this peel and makes replay fail closed.
        for patch in reversed(overlays):
            if git(fixture, "apply", "--reverse", "--check", str(patch)).returncode == 0:
                require(git(fixture, "apply", "--reverse", str(patch)), f"Peel {patch.name}")
        for patch in overlays:
            require(git(fixture, "apply", str(patch)), f"Replay {patch.name}")

        for relative in paths:
            source, destination = checkout / relative, fixture / relative
            before = source.read_text().splitlines(True) if source.exists() else []
            after = destination.read_text().splitlines(True) if destination.exists() else []
            changes.extend(difflib.unified_diff(
                before, after,
                fromfile="a/" + relative if source.exists() else "/dev/null",
                tofile="b/" + relative if destination.exists() else "/dev/null",
            ))

    difference = "".join(changes)
    if not difference:
        return False
    require(git(checkout, "apply", "--check", "-", input_text=difference), "Preflight combined Gemma 4 diff")
    require(git(checkout, "apply", "-", input_text=difference), "Apply combined Gemma 4 diff")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    arguments = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    try:
        changed = prepare(root, arguments.checkout)
    except (OSError, ValueError) as error:
        print(f"Gemma 4 patch preparation failed: {error}", file=sys.stderr)
        return 1
    print("Gemma 4 model overlays applied (bounded storage remains opt-in)." if changed
          else "Gemma 4 model overlays already applied and replay verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
