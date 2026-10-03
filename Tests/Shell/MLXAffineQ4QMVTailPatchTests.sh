#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
python3 "$PACKAGE_ROOT/Tests/Python/test_affine_q4_qmv_tail_patch.py"
