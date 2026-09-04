#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MODEL_RUNNER_BUILD_CONFIGURATION="${MODEL_RUNNER_BUILD_CONFIGURATION:-release}"
./build-metal.sh
swift build --configuration "$MODEL_RUNNER_BUILD_CONFIGURATION" --build-tests -Xswiftc -enable-testing
BIN_DIR="$(swift build --configuration "$MODEL_RUNNER_BUILD_CONFIGURATION" --show-bin-path)"
cp "$BIN_DIR/mlx.metallib" "$BIN_DIR/MidnightPackageTests.xctest/Contents/MacOS/mlx.metallib"
swift test --configuration "$MODEL_RUNNER_BUILD_CONFIGURATION" --skip-build --no-parallel "$@"
