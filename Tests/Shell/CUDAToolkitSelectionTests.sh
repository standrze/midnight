#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_SCRIPT="$PACKAGE_ROOT/build.sh"
TOOLKIT_SCRIPT="$PACKAGE_ROOT/Scripts/cuda-toolkit.sh"

# shellcheck source=../../Scripts/cuda-toolkit.sh
source "$TOOLKIT_SCRIPT"

TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/model-runner-cuda-toolkit.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT
  rm -rf "$TEST_ROOT"
  exit "$status"
}
trap cleanup EXIT

TOOLKIT="$TEST_ROOT/cuda-13.0"
WRONG_BIN="$TEST_ROOT/wrong-bin"
mkdir -p "$TOOLKIT/bin" "$WRONG_BIN"
printf '#!/usr/bin/env bash\nprintf "toolkit nvcc\\n"\n' > "$TOOLKIT/bin/nvcc"
printf '#!/usr/bin/env bash\nprintf "wrong nvcc\\n"\n' > "$WRONG_BIN/nvcc"
chmod 0755 "$TOOLKIT/bin/nvcc" "$WRONG_BIN/nvcc"
ln -s "$TOOLKIT" "$TEST_ROOT/cuda"

PATH="$WRONG_BIN:$PATH"
export PATH
[[ "$(command -v nvcc)" == "$WRONG_BIN/nvcc" ]]
model_runner_resolve_cuda_toolkit "$TEST_ROOT/cuda"
[[ "$CUDA_TOOLKIT_PATH" == "$TOOLKIT" ]]
[[ "$NVCC_PATH" == "$TOOLKIT/bin/nvcc" ]]
[[ "$(readlink -f "$(command -v nvcc)")" == "$NVCC_PATH" ]]
[[ "$(nvcc)" == "toolkit nvcc" ]]

if model_runner_resolve_cuda_toolkit relative/cuda >/dev/null 2>&1; then
  echo "Relative CUDA toolkit path unexpectedly passed" >&2
  exit 1
fi
if model_runner_resolve_cuda_toolkit "$TEST_ROOT/missing" >/dev/null 2>&1; then
  echo "Missing CUDA toolkit unexpectedly passed" >&2
  exit 1
fi

OUTSIDE="$TEST_ROOT/outside-nvcc"
printf '#!/usr/bin/env bash\nexit 0\n' > "$OUTSIDE"
chmod 0755 "$OUTSIDE"
mv "$TOOLKIT/bin/nvcc" "$TOOLKIT/bin/nvcc.saved"
ln -s "$OUTSIDE" "$TOOLKIT/bin/nvcc"
if model_runner_resolve_cuda_toolkit "$TEST_ROOT/cuda" >/dev/null 2>&1; then
  echo "Out-of-toolkit nvcc symlink unexpectedly passed" >&2
  exit 1
fi

grep -Fq 'source "$PACKAGE_ROOT/Scripts/cuda-toolkit.sh"' "$BUILD_SCRIPT"
grep -Fq 'model_runner_resolve_cuda_toolkit /usr/local/cuda' "$BUILD_SCRIPT"
grep -Fq 'NVCC_VERSION_FINGERPRINT="$("$NVCC_PATH" --version' "$BUILD_SCRIPT"
grep -Fq 'NVCC_GPU_CODES="$("$NVCC_PATH" --list-gpu-code)' "$BUILD_SCRIPT"

echo "CUDA toolkit compiler selection checks passed"
