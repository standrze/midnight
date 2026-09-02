#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODEL_RUNNER_BUILD_CONFIGURATION=release \
MODEL_RUNNER_BUILD_PRODUCT=model-runner-metal-quant-bench \
  "$PACKAGE_ROOT/build-metal.sh"

BIN_DIR="$(
  cd "$PACKAGE_ROOT"
  swift build --configuration release --show-bin-path
)"
exec "$BIN_DIR/model-runner-metal-quant-bench" "$@"
