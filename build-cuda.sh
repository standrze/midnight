#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd)"
CUDA_PROFILE="native"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "CUDA builds require Linux. Use ./build.sh on macOS for Metal." >&2
  exit 1
fi

if [[ $# -gt 0 && "$1" != -* ]]; then
  CUDA_PROFILE="$1"
  shift
fi

if [[ $# -gt 0 ]]; then
  echo "build-cuda.sh accepts only an optional profile." >&2
  exit 2
fi

SPM_CUDA=1 MLX_CUDA_PROFILE="$CUDA_PROFILE" exec "$PACKAGE_ROOT/build.sh"
