#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$PACKAGE_ROOT/Scripts/swiftpm-scratch-path.sh"

TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/model-runner-scratch.XXXXXX")"
TEST_ROOT="$(cd "$TEST_ROOT" && pwd -P)"
cleanup() {
  local status=$?
  trap - EXIT
  rm -rf "$TEST_ROOT"
  exit "$status"
}
trap cleanup EXIT
FAKE_PACKAGE_ROOT="$TEST_ROOT/package"
mkdir -p "$FAKE_PACKAGE_ROOT"

(
  unset MODEL_RUNNER_SCRATCH_PATH
  model_runner_configure_swiftpm_scratch "$FAKE_PACKAGE_ROOT" Linux
  [[ "$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH" == "$FAKE_PACKAGE_ROOT/.build" ]]
  [[ "${#MODEL_RUNNER_SWIFT_BUILD_SCRATCH_ARGS[@]}" -eq 0 ]]
  [[ "${#MODEL_RUNNER_SWIFT_PACKAGE_SCRATCH_ARGS[@]}" -eq 0 ]]
)

ABSOLUTE_SCRATCH="$TEST_ROOT/isolated build"
(
  MODEL_RUNNER_SCRATCH_PATH="$ABSOLUTE_SCRATCH"
  model_runner_configure_swiftpm_scratch "$FAKE_PACKAGE_ROOT" Linux
  [[ "$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH" == "$ABSOLUTE_SCRATCH" ]]
  [[ "${MODEL_RUNNER_SWIFT_BUILD_SCRATCH_ARGS[*]}" == "--scratch-path $ABSOLUTE_SCRATCH" ]]
  [[ "${MODEL_RUNNER_SWIFT_PACKAGE_SCRATCH_ARGS[*]}" == "--scratch-path $ABSOLUTE_SCRATCH" ]]
)

(
  MODEL_RUNNER_SCRATCH_PATH="scratch-relative"
  model_runner_configure_swiftpm_scratch "$FAKE_PACKAGE_ROOT" Linux
  [[ "$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH" == "$FAKE_PACKAGE_ROOT/scratch-relative" ]]
)

(
  MODEL_RUNNER_SCRATCH_PATH="$ABSOLUTE_SCRATCH"
  model_runner_configure_swiftpm_scratch "$FAKE_PACKAGE_ROOT" Darwin
  [[ "$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH" == "$FAKE_PACKAGE_ROOT/.build" ]]
  [[ "${#MODEL_RUNNER_SWIFT_BUILD_SCRATCH_ARGS[@]}" -eq 0 ]]
)

for unsafe_path in / /tmp /var/tmp "$FAKE_PACKAGE_ROOT"; do
  if (
    MODEL_RUNNER_SCRATCH_PATH="$unsafe_path"
    model_runner_configure_swiftpm_scratch "$FAKE_PACKAGE_ROOT" Linux
  ) >/dev/null 2>&1; then
    echo "unsafe scratch path unexpectedly passed: $unsafe_path" >&2
    exit 1
  fi
done

grep -Fq 'PROFILE_MARKER="$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH/.model-runner-profile"' \
  "$PACKAGE_ROOT/build.sh"
grep -Fq 'swift package "${MODEL_RUNNER_SWIFT_PACKAGE_SCRATCH_ARGS[@]}" clean' \
  "$PACKAGE_ROOT/build.sh"
grep -Fq 'BIN_DIR="$(swift build "${SWIFT_BUILD_ARGS[@]}" --show-bin-path)"' \
  "$PACKAGE_ROOT/build.sh"
grep -Fq 'MLX_SWIFT_CHECKOUT="$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH/checkouts/mlx-swift"' \
  "$PACKAGE_ROOT/prepare-dependencies.sh"
grep -Fq 'MLX_SWIFT_LM_CHECKOUT="$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH/checkouts/mlx-swift-lm"' \
  "$PACKAGE_ROOT/prepare-dependencies.sh"
grep -Fq 'swift package "${MODEL_RUNNER_SWIFT_PACKAGE_SCRATCH_ARGS[@]}" resolve' \
  "$PACKAGE_ROOT/prepare-dependencies.sh"
grep -Fq 'swift build --configuration release \' \
  "$PACKAGE_ROOT/run.sh"

echo "SwiftPM scratch-path checks passed"
