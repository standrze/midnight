#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MLX_PATCH="$PACKAGE_ROOT/Patches/mlx-affine-q4-qmv-specialization.patch"
MLX_SWIFT_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-affine-q4-qmv-jit.patch"
PREPARE_SCRIPT="$PACKAGE_ROOT/prepare-dependencies.sh"
BENCHMARK_SOURCE="$PACKAGE_ROOT/Benchmarks/MetalQuantization/main.swift"
MLX_SWIFT_CHECKOUT="$PACKAGE_ROOT/.build/checkouts/mlx-swift"
MLX_CHECKOUT="$MLX_SWIFT_CHECKOUT/Source/Cmlx/mlx"
MLX_SWIFT_REVISION="72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798"
MLX_REVISION="1f8e74e3f12f31365464a6867c6579f0e9b29d85"

require_exact_targets() {
  local patch_file="$1"
  shift
  local target_count
  target_count="$(grep -c '^diff --git ' "$patch_file" || true)"
  if [[ "$target_count" -ne "$#" ]]; then
    echo "$patch_file must modify exactly $# files; found $target_count" >&2
    exit 1
  fi
  local expected
  for expected in "$@"; do
    if ! grep -Fqx "diff --git a/$expected b/$expected" "$patch_file"; then
      echo "$patch_file does not modify required file $expected" >&2
      exit 1
    fi
  done
}

require_added_count() {
  local expected="$1"
  local pattern="$2"
  local patch_file="$3"
  local actual
  actual="$(grep -F "$pattern" "$patch_file" | grep -c '^+' || true)"
  if [[ "$actual" -ne "$expected" ]]; then
    echo "Expected $expected added occurrence(s) of '$pattern' in $patch_file; found $actual" >&2
    exit 1
  fi
}

check_patch_round_trip() {
  local label="$1"
  local checkout="$2"
  local revision="$3"
  local patch_file="$4"
  local original="$TEST_ROOT/$label-original"
  local patched="$TEST_ROOT/$label-patched"

  git -C "$checkout" cat-file -e "$revision^{commit}"
  mkdir -p "$original" "$patched"
  git -C "$checkout" archive "$revision" | tar -xf - -C "$original"
  git -C "$checkout" archive "$revision" | tar -xf - -C "$patched"

  git -C "$patched" apply --check --whitespace=error-all "$patch_file"
  git -C "$patched" apply "$patch_file"
  git -C "$patched" apply --reverse --check "$patch_file"
  git -C "$patched" apply --reverse "$patch_file"
  diff -qr "$original" "$patched" >/dev/null
}

[[ -s "$MLX_PATCH" ]]
[[ -s "$MLX_SWIFT_PATCH" ]]

require_exact_targets \
  "$MLX_PATCH" \
  "mlx/backend/metal/kernels/quantized.h" \
  "mlx/backend/metal/kernels/quantized.metal" \
  "mlx/backend/metal/quantized.cpp"
require_exact_targets \
  "$MLX_SWIFT_PATCH" \
  "Source/Cmlx/mlx-generated/metal/quantized.h" \
  "Source/Cmlx/mlx-generated/quantized.cpp"

require_added_count 1 'MLX_METAL_AFFINE_QMV_RESULTS_PER_SIMDGROUP' "$MLX_PATCH"
require_added_count 1 'MLX_METAL_AFFINE_QMV_MIN_OUTPUTS' "$MLX_PATCH"
require_added_count 3 'affine_qmv_results_per_simdgroup(' "$MLX_PATCH"
require_added_count 1 'bn = 2 * results_per_simdgroup;' "$MLX_PATCH"
require_added_count 4 'results_suffix,' "$MLX_PATCH"
require_added_count 1 'global_scale.has_value(),' "$MLX_PATCH"
require_added_count 2 'qmv_fast_impl<T, group_size, bits, results_per_simdgroup>(' "$MLX_PATCH"
require_added_count 5 'instantiate_affine_qmv_fast_results(type,' "$MLX_PATCH"
require_added_count 3 'instantiate_affine_gather_qmv_fast_results(type,' "$MLX_PATCH"
require_added_count 4 'instantiate_affine_qmv_fast_specializations(' "$MLX_PATCH"

require_added_count 4 'int results_per_simdgroup = 4>' "$MLX_SWIFT_PATCH"
require_added_count 4 'qmv_fast_impl<T, group_size, bits, results_per_simdgroup>(' \
  "$MLX_SWIFT_PATCH"

grep -Fq \
  'MLX_SOURCE_AFFINE_Q4_QMV_PATCH="$PACKAGE_ROOT/Patches/mlx-affine-q4-qmv-specialization.patch"' \
  "$PREPARE_SCRIPT"
grep -Fq \
  'MLX_SWIFT_AFFINE_Q4_QMV_JIT_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-affine-q4-qmv-jit.patch"' \
  "$PREPARE_SCRIPT"
grep -Fq 'if [[ "$HOST_OS" == "Darwin" ]]; then' "$PREPARE_SCRIPT"
grep -Fq '"mlx affine Q4 QMV specialization"' "$PREPARE_SCRIPT"
grep -Fq '"mlx-swift affine Q4 QMV generated JIT"' "$PREPARE_SCRIPT"

grep -Fq 'CommandLine.arguments.contains("--qmv-specialization")' "$BENCHMARK_SOURCE"
grep -Fq 'MLX_METAL_AFFINE_QMV_RESULTS_PER_SIMDGROUP' "$BENCHMARK_SOURCE"
grep -Fq 'dense-lm-head-2048x100352' "$BENCHMARK_SOURCE"
grep -Fq 'gather-gate-up-2048x1024' "$BENCHMARK_SOURCE"
grep -Fq 'gather-down-512x2048' "$BENCHMARK_SOURCE"
grep -Fq 'outputHash &*= 1_099_511_628_211' "$BENCHMARK_SOURCE"

TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/model-runner-affine-q4-qmv.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT
  rm -rf "$TEST_ROOT"
  exit "$status"
}
trap cleanup EXIT

check_patch_round_trip "mlx" "$MLX_CHECKOUT" "$MLX_REVISION" "$MLX_PATCH"
check_patch_round_trip \
  "mlx-swift" "$MLX_SWIFT_CHECKOUT" "$MLX_SWIFT_REVISION" "$MLX_SWIFT_PATCH"

echo "MLX affine Q4 QMV specialization patch tests passed."
