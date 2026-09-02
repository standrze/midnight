#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-run}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

cd "$PACKAGE_ROOT"

case "$MODE" in
  run|debug|verify)
    ;;
  logs|telemetry)
    echo "Midnight Runner writes runtime logs and telemetry to its standard output." >&2
    exit 0
    ;;
  *)
    echo "usage: $0 [run|debug|verify|logs|telemetry] [midnight arguments...]" >&2
    exit 64
    ;;
esac

CONFIGURATION=release
if [[ "$MODE" == "debug" ]]; then
  CONFIGURATION=debug
fi

MODEL_RUNNER_BUILD_CONFIGURATION="$CONFIGURATION" \
MODEL_RUNNER_BUILD_PRODUCT=midnight \
  "$PACKAGE_ROOT/build.sh"
BIN_PATH="$(swift build -c "$CONFIGURATION" --show-bin-path)/midnight"

if [[ "$#" -eq 0 ]]; then
  set -- --list-models
fi

"$BIN_PATH" "$@"
