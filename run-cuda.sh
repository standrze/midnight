#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd)"
CUDA_PROFILE="native"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "CUDA runs require Linux. Use ./run.sh --engine metal on macOS." >&2
  exit 1
fi

if [[ $# -gt 0 && "$1" != -* ]]; then
  CUDA_PROFILE="$1"
  shift
fi

SPM_CUDA=1 MLX_CUDA_PROFILE="$CUDA_PROFILE" exec "$PACKAGE_ROOT/run.sh" \
  --engine cuda "$@"
