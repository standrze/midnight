#!/usr/bin/env bash
set -euo pipefail

# Compatibility entrypoint; implementation and build outputs belong to the
# separate Midnight Quantization project.
RUNTIME_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QUANTIZATION_ROOT="${MIDNIGHT_QUANTIZATION_ROOT:-$RUNTIME_ROOT/../midnight-quantization}"
ENTRYPOINT="$QUANTIZATION_ROOT/Scripts/rescore-laguna-q4r8.sh"
if [[ ! -f "$ENTRYPOINT" ]]; then
  echo "This command moved to Midnight Quantization: $ENTRYPOINT" >&2
  exit 2
fi
exec bash "$ENTRYPOINT" "$@"
