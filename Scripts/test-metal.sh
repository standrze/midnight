#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MODEL_RUNNER_BUILD_CONFIGURATION="${MODEL_RUNNER_BUILD_CONFIGURATION:-release}"
./build-metal.sh
swift build --configuration "$MODEL_RUNNER_BUILD_CONFIGURATION" --build-tests -Xswiftc -enable-testing
BIN_DIR="$(swift build --configuration "$MODEL_RUNNER_BUILD_CONFIGURATION" --show-bin-path)"
# SwiftPM may emit one bundle per test target or one combined package bundle.
for TEST_BUNDLE in "$BIN_DIR"/*.xctest; do
  [[ -d "$TEST_BUNDLE/Contents/MacOS" ]] || continue
  cp "$BIN_DIR/mlx.metallib" "$TEST_BUNDLE/Contents/MacOS/mlx.metallib"
done
swift test --configuration "$MODEL_RUNNER_BUILD_CONFIGURATION" --skip-build --no-parallel "$@"
