#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNTIME_SCRIPT="$PACKAGE_ROOT/Scripts/cuda-runtime-environment.sh"

# shellcheck source=../../Scripts/cuda-runtime-environment.sh
source "$RUNTIME_SCRIPT"

TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/model-runner-cuda-runtime.XXXXXX")"
TEST_ROOT="$(cd "$TEST_ROOT" && pwd -P)"
cleanup() {
  local status=$?
  trap - EXIT
  rm -rf "$TEST_ROOT"
  exit "$status"
}
trap cleanup EXIT

write_file() {
  local path="$1"
  shift
  mkdir -p "$(dirname "$path")"
  printf '%s\n' "$@" > "$path"
}

fixture_sha256() {
  local path="$1"

  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  else
    shasum -a 256 "$path" | awk '{print $1}'
  fi
}

make_clang_fixture() {
  local path="$1"

  write_file "$path" \
    '#!/usr/bin/env bash' \
    'if [[ " $* " == *" -dM -E -x c++ /dev/null "* ]]; then' \
    '  echo "#define __clang__ 1"' \
    '  echo "#define __clang_major__ 18"' \
    '  exit 0' \
    'fi' \
    'echo "clang version 18.1.0 fixture"'
  chmod 0755 "$path"
}

TOOLKIT="$TEST_ROOT/cuda-13.0"
mkdir -p "$TOOLKIT/bin" "$TOOLKIT/include" "$TOOLKIT/targets/x86_64-linux/lib"
write_file "$TOOLKIT/bin/nvcc" '#!/usr/bin/env bash' 'echo "Cuda compilation tools, release 13.0"'
chmod 0755 "$TOOLKIT/bin/nvcc"
touch "$TOOLKIT/targets/x86_64-linux/lib/libcudart.so"
touch "$TOOLKIT/targets/x86_64-linux/lib/libnvrtc.so"
ln -s targets/x86_64-linux/lib "$TOOLKIT/lib64"

CUDNN_ROOT="$TEST_ROOT/cudnn-frontend"
write_file "$CUDNN_ROOT/include/cudnn_frontend.h" '#pragma once'
write_file "$CUDNN_ROOT/CMakeLists.txt" \
  'project(cudnn_frontend VERSION 1.16.0 LANGUAGES CXX)'

CUTLASS_ROOT_FIXTURE="$TEST_ROOT/cutlass"
write_file "$CUTLASS_ROOT_FIXTURE/include/cutlass/cutlass.h" '#pragma once' '// exact cutlass fixture'
write_file "$CUTLASS_ROOT_FIXTURE/include/cutlass/version.h" \
  '#define CUTLASS_MAJOR 4' \
  '#define CUTLASS_MINOR 3' \
  '#define CUTLASS_PATCH 5'
write_file "$CUTLASS_ROOT_FIXTURE/include/cutlass/numeric_conversion.h" '#pragma once'
write_file "$CUTLASS_ROOT_FIXTURE/include/cutlass/detail/runtime-extra.h" '#define CUTLASS_RUNTIME_EXTRA 1'
write_file "$CUTLASS_ROOT_FIXTURE/include/cute/tensor.hpp" '#pragma once'
write_file "$CUTLASS_ROOT_FIXTURE/include/cute/numeric/numeric_types.hpp" '#pragma once'
write_file "$CUTLASS_ROOT_FIXTURE/include/cute/algorithm/runtime-extra.hpp" '#pragma once'

HOST_CXX="$TEST_ROOT/clang++-18"
make_clang_fixture "$HOST_CXX"

CUDNN_FRONTEND_ROOT="$CUDNN_ROOT"
CUTLASS_ROOT="$CUTLASS_ROOT_FIXTURE"
MLX_CUDA_HOST_CXX="$HOST_CXX"
export CUDNN_FRONTEND_ROOT CUTLASS_ROOT MLX_CUDA_HOST_CXX

model_runner_resolve_cuda_runtime_environment "$TOOLKIT"
[[ "$CUDA_TOOLKIT_PATH" == "$TOOLKIT" ]]
[[ "$CUDA_RUNTIME_BIN_DIR" == "$TOOLKIT/bin" ]]
[[ "$CUDA_RUNTIME_INCLUDE_DIR" == "$TOOLKIT/include" ]]
[[ "$CUDA_RUNTIME_LIBRARY_DIR" == "$TOOLKIT/targets/x86_64-linux/lib" ]]
[[ "$CUDA_HOME" == "$TOOLKIT" && "$CUDA_PATH" == "$TOOLKIT" ]]
[[ "$CUDACXX" == "$TOOLKIT/bin/nvcc" ]]
[[ "$PATH" == "$TOOLKIT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" ]]
[[ "$LD_LIBRARY_PATH" == "$CUDA_RUNTIME_LIBRARY_DIR" ]]
[[ "$LIBRARY_PATH" == "$CUDA_RUNTIME_LIBRARY_DIR" ]]
[[ "$CPATH" == "$CUTLASS_ROOT_FIXTURE/include:$CUDNN_ROOT/include:$TOOLKIT/include" ]]
[[ "$MLX_CUDA_INCLUDE_PATHS" == "$CUTLASS_ROOT_FIXTURE/include:$CUDNN_ROOT/include" ]]
[[ "$MLX_CUDA_HOST_CXX" == "$(readlink -f "$HOST_CXX")" ]]
[[ "$CUDAHOSTCXX" == "$MLX_CUDA_HOST_CXX" ]]

model_runner_cuda_runtime_systemd_setenv_args
SYSTEMD_ENV="$(printf '%s\n' "${MODEL_RUNNER_CUDA_RUNTIME_SYSTEMD_SETENV_ARGS[@]}")"
grep -Fqx -- "--setenv=CUDA_HOME=$TOOLKIT" <<< "$SYSTEMD_ENV"
grep -Fqx -- "--setenv=LD_LIBRARY_PATH=$CUDA_RUNTIME_LIBRARY_DIR" <<< "$SYSTEMD_ENV"
grep -Fqx -- "--setenv=CPATH=$CPATH" <<< "$SYSTEMD_ENV"
grep -Fqx -- "--setenv=MLX_CUDA_INCLUDE_PATHS=$MLX_CUDA_INCLUDE_PATHS" <<< "$SYSTEMD_ENV"
grep -Fqx -- "--setenv=MLX_CUDA_HOST_CXX=$MLX_CUDA_HOST_CXX" <<< "$SYSTEMD_ENV"
grep -Fqx -- "--setenv=CUDAHOSTCXX=$MLX_CUDA_HOST_CXX" <<< "$SYSTEMD_ENV"

PUBLISH_PACKAGE="$TEST_ROOT/publish-package"
mkdir -p "$PUBLISH_PACKAGE"
model_runner_publish_cuda_runtime_headers "$PUBLISH_PACKAGE"
model_runner_verify_cuda_runtime_headers "$PUBLISH_PACKAGE"
[[ -f "$PUBLISH_PACKAGE/include/cutlass/detail/runtime-extra.h" ]]
[[ -f "$PUBLISH_PACKAGE/include/cute/algorithm/runtime-extra.hpp" ]]
[[ ! -L "$PUBLISH_PACKAGE/include" ]]
if find "$PUBLISH_PACKAGE/include" -type l -print -quit | grep -q .; then
  echo "runtime header publication unexpectedly retained a symlink" >&2
  exit 1
fi
MANIFEST="$PUBLISH_PACKAGE/include/.model-runner-cuda-runtime-headers.manifest"
grep -Fqx 'manifest_version=1' "$MANIFEST"
grep -Fqx 'cutlass_version=4.3.5' "$MANIFEST"
grep -Eq '^validated_headers_sha256=[0-9a-f]{64}$' "$MANIFEST"
grep -Eq '^cutlass_tree_fingerprint=[0-9a-f]{64}$' "$MANIFEST"
grep -Eq '^cute_tree_fingerprint=[0-9a-f]{64}$' "$MANIFEST"
[[ "$(find "$PUBLISH_PACKAGE" -maxdepth 1 -name '.cuda-runtime-include.stage.*' -print)" == "" ]]

# A byte of published-tree drift must fail verification and must never be
# silently repaired by a subsequent publisher call.
printf '// drift\n' >> "$PUBLISH_PACKAGE/include/cute/algorithm/runtime-extra.hpp"
if model_runner_verify_cuda_runtime_headers "$PUBLISH_PACKAGE" >/dev/null 2>&1; then
  echo "drifted runtime headers unexpectedly verified" >&2
  exit 1
fi
if model_runner_publish_cuda_runtime_headers "$PUBLISH_PACKAGE" >/dev/null 2>&1; then
  echo "publisher unexpectedly overwrote drifted runtime headers" >&2
  exit 1
fi
grep -Fq '// drift' "$PUBLISH_PACKAGE/include/cute/algorithm/runtime-extra.hpp"

# The standalone publishing path migrates an already-published stable runner
# without replacing or rewriting either existing release artifact.
MIGRATION_PACKAGE="$TEST_ROOT/migration-package"
mkdir -p "$MIGRATION_PACKAGE/bin"
write_file "$MIGRATION_PACKAGE/bin/midnight-rtx4090" '#!/usr/bin/env bash' 'exit 0'
chmod 0755 "$MIGRATION_PACKAGE/bin/midnight-rtx4090"
write_file "$MIGRATION_PACKAGE/bin/midnight-rtx4090.manifest" 'existing=release-manifest'
MIGRATION_BINARY_SHA="$(fixture_sha256 "$MIGRATION_PACKAGE/bin/midnight-rtx4090")"
MIGRATION_MANIFEST_SHA="$(fixture_sha256 "$MIGRATION_PACKAGE/bin/midnight-rtx4090.manifest")"
CUDNN_FRONTEND_ROOT="$CUDNN_ROOT" \
CUTLASS_ROOT="$CUTLASS_ROOT_FIXTURE" \
MLX_CUDA_HOST_CXX="$HOST_CXX" \
bash "$RUNTIME_SCRIPT" \
  --toolkit "$TOOLKIT" \
  --package-root "$MIGRATION_PACKAGE" \
  --publish-only
[[ "$(fixture_sha256 "$MIGRATION_PACKAGE/bin/midnight-rtx4090")" == "$MIGRATION_BINARY_SHA" ]]
[[ "$(fixture_sha256 "$MIGRATION_PACKAGE/bin/midnight-rtx4090.manifest")" == "$MIGRATION_MANIFEST_SHA" ]]
model_runner_verify_cuda_runtime_headers "$MIGRATION_PACKAGE"
model_runner_verify_cuda_runtime_headers_for_runner \
  "$MIGRATION_PACKAGE" "$MIGRATION_PACKAGE/bin/midnight-rtx4090"
mkdir -p "$MIGRATION_PACKAGE/.build/release"
ln -s "$MIGRATION_PACKAGE/bin/midnight-rtx4090" \
  "$MIGRATION_PACKAGE/.build/release/midnight"
model_runner_verify_cuda_runtime_headers_for_runner \
  "$MIGRATION_PACKAGE" "$MIGRATION_PACKAGE/.build/release/midnight"
[[ "$MODEL_RUNNER_CUDA_RUNTIME_RUNNER_REALPATH" \
  == "$MIGRATION_PACKAGE/bin/midnight-rtx4090" ]]
[[ "$MODEL_RUNNER_CUDA_RUNTIME_RUNNER_INCLUDE_DIR" \
  == "$MIGRATION_PACKAGE/include" ]]
CUDNN_FRONTEND_ROOT="$CUDNN_ROOT" \
CUTLASS_ROOT="$CUTLASS_ROOT_FIXTURE" \
MLX_CUDA_HOST_CXX="$HOST_CXX" \
bash "$RUNTIME_SCRIPT" \
  --toolkit "$TOOLKIT" \
  --package-root "$MIGRATION_PACKAGE" \
  --verify-only

# Source-tree drift is also detected even when the published bytes are intact.
printf '// source drift\n' >> "$CUTLASS_ROOT_FIXTURE/include/cutlass/detail/runtime-extra.h"
if model_runner_verify_cuda_runtime_headers "$MIGRATION_PACKAGE" >/dev/null 2>&1; then
  echo "runtime headers unexpectedly verified against a drifted source tree" >&2
  exit 1
fi
write_file "$CUTLASS_ROOT_FIXTURE/include/cutlass/detail/runtime-extra.h" \
  '#define CUTLASS_RUNTIME_EXTRA 1'
model_runner_resolve_cuda_runtime_environment "$TOOLKIT"
model_runner_verify_cuda_runtime_headers "$MIGRATION_PACKAGE"

# A real executable outside package bin/ and a SwiftPM scratch executable both
# derive a different MLX include root. They fail without receiving headers.
ARBITRARY_ROOT="$TEST_ROOT/arbitrary-runner-root"
mkdir -p "$ARBITRARY_ROOT/bin"
write_file "$ARBITRARY_ROOT/bin/midnight" '#!/usr/bin/env bash' 'exit 0'
chmod 0755 "$ARBITRARY_ROOT/bin/midnight"
if model_runner_resolve_cuda_runtime_runner_include \
  "$MIGRATION_PACKAGE" "$ARBITRARY_ROOT/bin/midnight" >/dev/null 2>&1; then
  echo "arbitrary external runner unexpectedly accepted managed package headers" >&2
  exit 1
fi
[[ ! -e "$ARBITRARY_ROOT/include" ]]

SCRATCH_RUNNER="$MIGRATION_PACKAGE/.build/custom/x86_64-unknown-linux-gnu/release/midnight"
mkdir -p "$(dirname "$SCRATCH_RUNNER")"
write_file "$SCRATCH_RUNNER" '#!/usr/bin/env bash' 'exit 0'
chmod 0755 "$SCRATCH_RUNNER"
if model_runner_resolve_cuda_runtime_runner_include \
  "$MIGRATION_PACKAGE" "$SCRATCH_RUNNER" >/dev/null 2>&1; then
  echo "SwiftPM scratch runner unexpectedly accepted package-root runtime headers" >&2
  exit 1
fi
[[ ! -e "$MIGRATION_PACKAGE/.build/custom/x86_64-unknown-linux-gnu/include" ]]
[[ "$(fixture_sha256 "$MIGRATION_PACKAGE/bin/midnight-rtx4090")" == "$MIGRATION_BINARY_SHA" ]]
[[ "$(fixture_sha256 "$MIGRATION_PACKAGE/bin/midnight-rtx4090.manifest")" == "$MIGRATION_MANIFEST_SHA" ]]

# The launcher passes its exact verified environment to an arbitrary command.
CAPTURE_PACKAGE="$TEST_ROOT/capture-package"
CAPTURE_FILE="$TEST_ROOT/captured-environment.txt"
CAPTURE_COMMAND="$TEST_ROOT/capture-command.sh"
mkdir -p "$CAPTURE_PACKAGE"
write_file "$CAPTURE_COMMAND" \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  ': "${CAPTURE_FILE:?}"' \
  'printf "%s\n" "$CUDA_HOME" "$CPATH" "$MLX_CUDA_HOST_CXX" "$CUDAHOSTCXX" > "$CAPTURE_FILE"'
chmod 0755 "$CAPTURE_COMMAND"
CUDNN_FRONTEND_ROOT="$CUDNN_ROOT" \
CUTLASS_ROOT="$CUTLASS_ROOT_FIXTURE" \
MLX_CUDA_HOST_CXX="$HOST_CXX" \
CAPTURE_FILE="$CAPTURE_FILE" \
bash "$RUNTIME_SCRIPT" \
  --toolkit "$TOOLKIT" \
  --package-root "$CAPTURE_PACKAGE" \
  -- "$CAPTURE_COMMAND"
sed -n '1p' "$CAPTURE_FILE" | grep -Fqx "$TOOLKIT"
sed -n '2p' "$CAPTURE_FILE" | grep -Fqx "$CPATH"
sed -n '3p' "$CAPTURE_FILE" | grep -Fqx "$MLX_CUDA_HOST_CXX"
sed -n '4p' "$CAPTURE_FILE" | grep -Fqx "$MLX_CUDA_HOST_CXX"
model_runner_verify_cuda_runtime_headers "$CAPTURE_PACKAGE"

# An unmanaged include directory is preserved and fails closed.
UNMANAGED_PACKAGE="$TEST_ROOT/unmanaged-package"
mkdir -p "$UNMANAGED_PACKAGE/include"
write_file "$UNMANAGED_PACKAGE/include/owner.txt" 'do not replace'
if model_runner_publish_cuda_runtime_headers "$UNMANAGED_PACKAGE" >/dev/null 2>&1; then
  echo "publisher unexpectedly replaced an unmanaged include directory" >&2
  exit 1
fi
grep -Fqx 'do not replace' "$UNMANAGED_PACKAGE/include/owner.txt"

BUILD_SCRIPT="$PACKAGE_ROOT/build.sh"
SMOKE_SCRIPT="$PACKAGE_ROOT/Scripts/smoke-cuda-model.sh"
grep -Fq 'model_runner_publish_cuda_runtime_headers "$PACKAGE_ROOT"' "$BUILD_SCRIPT"
grep -Fq 'model_runner_verify_cuda_runtime_headers "$PACKAGE_ROOT"' "$BUILD_SCRIPT"
grep -Fq 'model_runner_resolve_cuda_runtime_environment /usr/local/cuda' "$SMOKE_SCRIPT"
grep -Fq 'model_runner_publish_cuda_runtime_headers "$PACKAGE_ROOT"' "$SMOKE_SCRIPT"
grep -Fq 'model_runner_resolve_cuda_runtime_runner_include "$PACKAGE_ROOT" "$RUNNER_PATH"' "$SMOKE_SCRIPT"
grep -Fq 'model_runner_verify_cuda_runtime_headers_for_runner "$PACKAGE_ROOT" "$RUNNER_PATH"' "$SMOKE_SCRIPT"
grep -Fq 'MODEL_RUNNER_CUDA_RUNTIME_SYSTEMD_SETENV_ARGS' "$SMOKE_SCRIPT"
grep -Fq 'MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID' "$SMOKE_SCRIPT"
if grep -Eq '(kill|systemctl[^#]*stop).*MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID' "$SMOKE_SCRIPT"; then
  echo "runtime environment wiring introduced a protected-PID termination path" >&2
  exit 1
fi
RUNNER_CHECK_LINE="$(grep -nF \
  'model_runner_resolve_cuda_runtime_runner_include "$PACKAGE_ROOT" "$RUNNER_PATH"' \
  "$SMOKE_SCRIPT" | cut -d: -f1)"
PUBLISH_LINE="$(grep -nF \
  'model_runner_publish_cuda_runtime_headers "$PACKAGE_ROOT"' \
  "$SMOKE_SCRIPT" | cut -d: -f1)"
if (( RUNNER_CHECK_LINE >= PUBLISH_LINE )); then
  echo "smoke harness publishes runtime headers before validating runner realpath" >&2
  exit 1
fi

echo "CUDA runtime environment and header publication checks passed"
