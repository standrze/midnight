#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$PACKAGE_ROOT/Scripts/cuda-dependency-checks.sh"

TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/model-runner-cuda-deps.XXXXXX")"
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

make_cudnn_cmake_fixture() {
  local root="$1"
  local version="$2"
  write_file "$root/include/cudnn_frontend.h" '#pragma once'
  write_file "$root/CMakeLists.txt" "project(cudnn_frontend VERSION $version LANGUAGES CXX)"
}

make_cudnn_header_fixture() {
  local root="$1"
  local patch="$2"
  write_file "$root/include/cudnn_frontend.h" '#pragma once'
  write_file "$root/include/cudnn_frontend_version.h" \
    '#define CUDNN_FRONTEND_VERSION_MAJOR 1' \
    '#define CUDNN_FRONTEND_VERSION_MINOR 16' \
    "#define CUDNN_FRONTEND_VERSION_PATCH $patch"
}

make_cutlass_fixture() {
  local root="$1"
  local patch="$2"
  write_file "$root/include/cutlass/cutlass.h" '#pragma once'
  write_file "$root/include/cutlass/version.h" \
    '#define CUTLASS_MAJOR 4' \
    '#define CUTLASS_MINOR 3' \
    "#define CUTLASS_PATCH $patch"
  write_file "$root/include/cutlass/numeric_conversion.h" '#pragma once'
  write_file "$root/include/cute/tensor.hpp" '#pragma once'
  write_file "$root/include/cute/numeric/numeric_types.hpp" '#pragma once'
}

VALID_CUDNN="$TEST_ROOT/cudnn-valid"
HEADER_CUDNN="$TEST_ROOT/cudnn-header-valid"
OLD_CUDNN="$TEST_ROOT/cudnn-old"
MISLEADING_CUDNN="$TEST_ROOT/cudnn-misleading"
VALID_CUTLASS="$TEST_ROOT/cutlass-valid"
OLD_CUTLASS="$TEST_ROOT/cutlass-old"
make_cudnn_cmake_fixture "$VALID_CUDNN" 1.16.0
make_cudnn_header_fixture "$HEADER_CUDNN" 0
make_cudnn_cmake_fixture "$OLD_CUDNN" 0.9.2
make_cudnn_cmake_fixture "$MISLEADING_CUDNN" 0.9.2
printf '%s\n' '# upgrade note: target cudnn-frontend 1.16.0 later' \
  >> "$MISLEADING_CUDNN/CMakeLists.txt"
make_cutlass_fixture "$VALID_CUTLASS" 5
make_cutlass_fixture "$OLD_CUTLASS" 4

(
  CUDNN_FRONTEND_ROOT="$VALID_CUDNN"
  CUTLASS_ROOT="$VALID_CUTLASS"
  model_runner_resolve_cuda_dependencies
  [[ "$CUDNN_FRONTEND_VERSION" == 1.16.0 ]]
  [[ "$CUTLASS_VERSION" == 4.3.5 ]]
  [[ "$CUDNN_FRONTEND_INCLUDE_DIR" == "$(readlink -f "$VALID_CUDNN/include")" ]]
  [[ "$CUTLASS_INCLUDE_DIR" == "$(readlink -f "$VALID_CUTLASS/include")" ]]
  [[ -n "$CUDNN_FRONTEND_HEADERS_FINGERPRINT" ]]
  [[ -n "$CUTLASS_HEADERS_FINGERPRINT" ]]
)

(
  CUDNN_FRONTEND_INCLUDE_DIR="$HEADER_CUDNN/include"
  CUTLASS_INCLUDE_DIR="$VALID_CUTLASS/include"
  model_runner_resolve_cuda_dependencies
  [[ "$CUDNN_FRONTEND_VERSION_SOURCE" == header-macros:* ]]
)

if (
  CUDNN_FRONTEND_ROOT="$OLD_CUDNN"
  CUTLASS_ROOT="$VALID_CUTLASS"
  model_runner_resolve_cuda_dependencies
) >/dev/null 2>&1; then
  echo "old cudnn-frontend metadata unexpectedly passed" >&2
  exit 1
fi

if (
  CUDNN_FRONTEND_ROOT="$MISLEADING_CUDNN"
  CUTLASS_ROOT="$VALID_CUTLASS"
  model_runner_resolve_cuda_dependencies
) >/dev/null 2>&1; then
  echo "a comment mentioning 1.16.0 unexpectedly passed old CMake metadata" >&2
  exit 1
fi

if (
  CUDNN_FRONTEND_ROOT="$VALID_CUDNN"
  CUTLASS_ROOT="$OLD_CUTLASS"
  model_runner_resolve_cuda_dependencies
) >/dev/null 2>&1; then
  echo "old CUTLASS version macros unexpectedly passed" >&2
  exit 1
fi

if (
  CUDNN_FRONTEND_ROOT="$VALID_CUDNN"
  CUDNN_FRONTEND_INCLUDE_DIR="$HEADER_CUDNN/include"
  CUTLASS_ROOT="$VALID_CUTLASS"
  model_runner_resolve_cuda_dependencies
) >/dev/null 2>&1; then
  echo "metadata/header tree mismatch unexpectedly passed" >&2
  exit 1
fi

BUILD_SCRIPT="$PACKAGE_ROOT/build.sh"
grep -Fq 'cudnn-frontend-v1.16.0' "$PACKAGE_ROOT/Scripts/cuda-dependency-checks.sh"
grep -Fq 'cutlass-v4.3.5' "$PACKAGE_ROOT/Scripts/cuda-dependency-checks.sh"
grep -Fq 'cudnn-headers=$CUDNN_FRONTEND_HEADERS_FINGERPRINT' "$BUILD_SCRIPT"
grep -Fq 'cutlass-headers=$CUTLASS_HEADERS_FINGERPRINT' "$BUILD_SCRIPT"

echo "CUDA dependency checks passed"
