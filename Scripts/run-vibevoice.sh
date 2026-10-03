#!/usr/bin/env bash
set -euo pipefail
VIBEVOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$VIBEVOICE_ROOT"
export MIDNIGHT_VIBEVOICE_ROOT="$VIBEVOICE_ROOT"
swift build --product midnight
VIBEVOICE_BINARY="$(swift build --show-bin-path)/midnight"
exec "$VIBEVOICE_BINARY" --model "${MIDNIGHT_VIBEVOICE_MODEL:-$VIBEVOICE_ROOT/Models/VibeVoice-1.5B-hf}" \
  --name "${MIDNIGHT_VIBEVOICE_NAME:-vibevoice-1.5b}" --host 127.0.0.1 \
  --port "${MIDNIGHT_VIBEVOICE_PORT:-8096}" --max-tokens 512 "$@"
