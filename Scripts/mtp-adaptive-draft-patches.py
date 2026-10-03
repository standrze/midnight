#!/usr/bin/env python3
"""Replay the pinned MTP overlay stack on copies before applying a combined diff."""

import argparse
from pathlib import Path
import runpy
import sys

PATCHES = (
    "mlx-swift-lm-mtp-prompt-hidden-window.patch",
    "mlx-swift-lm-mtp-decode-scheduling.patch",
    "mlx-swift-lm-mtp-first-rejection-diagnostic.patch",
    "mlx-swift-lm-mtp-adaptive-drafts.patch",
    "mlx-swift-lm-mtp-stateless-adaptation.patch",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    arguments = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    # Reuse the existing copied-tree preflight implementation, with this
    # independently owned ordered stack and the same immutable LM revision.
    prepare = runpy.run_path(str(root / "Scripts/gemma4-window-cache-patches.py"))["prepare"]
    try:
        changed = prepare(root, arguments.checkout, patches=PATCHES)
    except (OSError, ValueError) as error:
        print(f"MTP patch preparation failed: {error}", file=sys.stderr)
        return 1
    print("MTP overlays applied; adaptive drafts remain opt-in." if changed
          else "MTP overlays already applied and replay verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
