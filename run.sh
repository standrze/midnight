#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd)"
"$PACKAGE_ROOT/build.sh"

cd "$PACKAGE_ROOT"
source "$PACKAGE_ROOT/Scripts/swiftpm-scratch-path.sh"
source "$PACKAGE_ROOT/Scripts/release-publisher.sh"
model_runner_configure_swiftpm_scratch "$PACKAGE_ROOT" "$(uname -s)"
if [[ "$(uname -s)" == "Linux" ]]; then
  CUDA_PROFILE="${MLX_CUDA_PROFILE:-native}"
  if [[ "$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH" != "$PACKAGE_ROOT/.build" \
    && "${SPM_CUDA:-1}" == "1" \
    && ( "$CUDA_PROFILE" == "rtx-4090" || "$CUDA_PROFILE" == "4090" ) ]]; then
    model_runner_verify_rtx4090_publication \
      "$PACKAGE_ROOT" "$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH"
    exec "$MODEL_RUNNER_RTX4090_COMPAT_PATH" "$@"
  fi
  BIN_DIR="$(swift build --configuration release \
    "${MODEL_RUNNER_SWIFT_BUILD_SCRATCH_ARGS[@]}" --show-bin-path)"
else
  # Bash 3.2 reports an empty array expansion as unbound under `set -u`.
  # macOS normally uses the default scratch tree, so keep that path explicit.
  if [[ -n "${MODEL_RUNNER_SWIFT_BUILD_SCRATCH_ARGS+configured}" ]]; then
    BIN_DIR="$(swift build "${MODEL_RUNNER_SWIFT_BUILD_SCRATCH_ARGS[@]}" --show-bin-path)"
  else
    BIN_DIR="$(swift build --show-bin-path)"
  fi
fi
exec "$BIN_DIR/midnight" "$@"
