#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MLX_SWIFT_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-direct-slice-update.patch"
MLX_SWIFT_LM_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-direct-kv-slice-update.patch"
PREPARE_SCRIPT="$PACKAGE_ROOT/prepare-dependencies.sh"
MLX_SWIFT_CHECKOUT="$PACKAGE_ROOT/.build/checkouts/mlx-swift"
MLX_SWIFT_LM_CHECKOUT="$PACKAGE_ROOT/.build/checkouts/mlx-swift-lm"
MLX_SWIFT_DARWIN_EXPECTED_REVISION="72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798"
MLX_SWIFT_LINUX_EXPECTED_REVISION="2d2724006b62855c6c2a71df633baf4ee4ad8a0f"
MLX_SWIFT_LM_EXPECTED_REVISION="14414441fa44f45eee35a61e9fa0bab577cf9734"

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

require_removed_count() {
  local expected="$1"
  local pattern="$2"
  local patch_file="$3"
  local actual
  actual="$(grep -F "$pattern" "$patch_file" | grep -c '^-' || true)"
  if [[ "$actual" -ne "$expected" ]]; then
    echo "Expected $expected removed occurrence(s) of '$pattern' in $patch_file; found $actual" >&2
    exit 1
  fi
}

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

check_patch_round_trip() {
  local label="$1"
  local checkout="$2"
  local revision="$3"
  local patch_file="$4"
  local original="$TEST_ROOT/$label-original"
  local patched="$TEST_ROOT/$label-patched"

  if ! git -C "$checkout" cat-file -e "$revision^{commit}" >/dev/null 2>&1; then
    echo "Missing pinned $label revision $revision in $checkout" >&2
    exit 1
  fi

  mkdir -p "$original" "$patched"
  git -C "$checkout" archive "$revision" | tar -xf - -C "$original"
  git -C "$checkout" archive "$revision" | tar -xf - -C "$patched"

  git -C "$patched" apply --check --whitespace=error-all "$patch_file"
  git -C "$patched" apply "$patch_file"
  git -C "$patched" apply --reverse --check "$patch_file"
  git -C "$patched" apply --reverse "$patch_file"

  if ! diff -qr "$original" "$patched" >/dev/null; then
    echo "$label patch did not reverse back to its immutable pinned baseline" >&2
    exit 1
  fi
}

[[ -s "$MLX_SWIFT_PATCH" ]]
[[ -s "$MLX_SWIFT_LM_PATCH" ]]

require_exact_targets \
  "$MLX_SWIFT_PATCH" \
  "Source/MLX/MLXArray+Indexing.swift" \
  "Tests/MLXTests/MLXArray+IndexingTests.swift"
require_exact_targets \
  "$MLX_SWIFT_LM_PATCH" \
  "Libraries/MLXLMCommon/KVCache.swift" \
  "Tests/MLXLMTests/KVCacheTests.swift"

# The lower-level API must call mlx_slice_update directly, use fixed inline
# bounds, and update the existing C wrapper. Allocating generic Swift bounds
# arrays or a temporary MLXArray would reintroduce per-layer decode overhead.
require_added_count 1 'public func updateSliceAxis2(' "$MLX_SWIFT_PATCH"
require_added_count 1 'with update: MLXArray' "$MLX_SWIFT_PATCH"
require_added_count 1 'var bounds = SIMD16<Int32>(' "$MLX_SWIFT_PATCH"
require_added_count 1 'withUnsafePointer(to: &bounds)' "$MLX_SWIFT_PATCH"
require_added_count 1 'withMemoryRebound(to: Int32.self, capacity: 16)' "$MLX_SWIFT_PATCH"
require_added_count 1 'mlx_slice_update(' "$MLX_SWIFT_PATCH"
require_added_count 1 'let source = ctx' "$MLX_SWIFT_PATCH"
require_added_count 1 '&ctx, source, update.ctx' "$MLX_SWIFT_PATCH"
require_added_count 0 'withUnsafeTemporaryAllocation' "$MLX_SWIFT_PATCH"
require_added_count 0 'MLXArray(result)' "$MLX_SWIFT_PATCH"
require_added_count 0 '_updateInternal(' "$MLX_SWIFT_PATCH"
require_added_count 0 'mlx_array_new()' "$MLX_SWIFT_PATCH"
require_added_count 0 'mlx_array_set(' "$MLX_SWIFT_PATCH"
require_added_count 0 'mlx_array_free(' "$MLX_SWIFT_PATCH"
require_added_count 0 'starts: [Int32]' "$MLX_SWIFT_PATCH"
require_added_count 0 'ends: [Int32]' "$MLX_SWIFT_PATCH"
require_added_count 0 'strides: [Int32]' "$MLX_SWIFT_PATCH"
require_added_count 1 'testDirectSliceUpdatePreservesIdentityAndAvoidsShapeOperations' \
  "$MLX_SWIFT_PATCH"
require_added_count 1 'XCTAssertEqual(ObjectIdentifier(a), identity)' "$MLX_SWIFT_PATCH"
require_added_count 1 'XCTAssertEqual(a.ctx.ctx, cWrapperIdentity)' "$MLX_SWIFT_PATCH"
require_added_count 1 'description.contains("SliceUpdate")' "$MLX_SWIFT_PATCH"
require_added_count 1 'description.contains("Broadcast")' "$MLX_SWIFT_PATCH"
require_added_count 1 'description.contains("Reshape")' "$MLX_SWIFT_PATCH"

# Both hot cache writers must leave the generic NumPy-style setter. These four
# removals cover K/V writes in KVCacheSimple and RotatingKVCache.updateInPlace.
require_removed_count 1 'self.keys?[.ellipsis, previous ..< self.offset, 0...] = keys' \
  "$MLX_SWIFT_LM_PATCH"
require_removed_count 1 'self.values?[.ellipsis, previous ..< self.offset, 0...] = values' \
  "$MLX_SWIFT_LM_PATCH"
require_removed_count 1 'self.keys![.ellipsis, idx ..< (idx + S), 0...] = keys' \
  "$MLX_SWIFT_LM_PATCH"
require_removed_count 1 'self.values![.ellipsis, idx ..< (idx + S), 0...] = values' \
  "$MLX_SWIFT_LM_PATCH"

require_added_count 5 'updateKVCacheSlice(' "$MLX_SWIFT_LM_PATCH"
require_added_count 2 'range: previous ..< self.offset' "$MLX_SWIFT_LM_PATCH"
require_added_count 2 'range: idx ..< (idx + S)' "$MLX_SWIFT_LM_PATCH"
require_added_count 1 '.updateSliceAxis2(' "$MLX_SWIFT_LM_PATCH"
require_added_count 0 'starts: [' "$MLX_SWIFT_LM_PATCH"
require_added_count 0 'ends: [' "$MLX_SWIFT_LM_PATCH"
require_added_count 0 'strides: [' "$MLX_SWIFT_LM_PATCH"
require_added_count 1 'testSimpleCacheDirectSliceUpdatePreservesIdentityUntilGrowth' \
  "$MLX_SWIFT_LM_PATCH"
require_added_count 1 'testRotatingCacheDirectSliceUpdatePreservesIdentityAcrossWrap' \
  "$MLX_SWIFT_LM_PATCH"
require_added_count 1 'initial[0] === filled[0]' "$MLX_SWIFT_LM_PATCH"
require_added_count 1 'initial[1] === filled[1]' "$MLX_SWIFT_LM_PATCH"
require_added_count 1 'grown[0] === refilled[0]' "$MLX_SWIFT_LM_PATCH"
require_added_count 1 'grown[1] === refilled[1]' "$MLX_SWIFT_LM_PATCH"
require_added_count 1 'storage[0] === wrapped[0]' "$MLX_SWIFT_LM_PATCH"
require_added_count 1 'storage[1] === wrapped[1]' "$MLX_SWIFT_LM_PATCH"
require_added_count 1 'capacity growth must install a larger key array' "$MLX_SWIFT_LM_PATCH"
require_added_count 1 'capacity growth must install a larger value array' "$MLX_SWIFT_LM_PATCH"
require_added_count 2 '== [6, 7, 8, 9]' "$MLX_SWIFT_LM_PATCH"

grep -Fq \
  'MLX_SWIFT_DIRECT_SLICE_UPDATE_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-direct-slice-update.patch"' \
  "$PREPARE_SCRIPT"
grep -Fq \
  'MLX_SWIFT_LM_DIRECT_KV_SLICE_UPDATE_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-direct-kv-slice-update.patch"' \
  "$PREPARE_SCRIPT"
grep -Fq \
  "MLX_SWIFT_DARWIN_EXPECTED_REVISION=\"$MLX_SWIFT_DARWIN_EXPECTED_REVISION\"" \
  "$PREPARE_SCRIPT"
grep -Fq \
  "MLX_SWIFT_LINUX_EXPECTED_REVISION=\"$MLX_SWIFT_LINUX_EXPECTED_REVISION\"" \
  "$PREPARE_SCRIPT"
grep -Fq \
  "MLX_SWIFT_LM_EXPECTED_REVISION=\"$MLX_SWIFT_LM_EXPECTED_REVISION\"" \
  "$PREPARE_SCRIPT"

TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/model-runner-direct-kv-patches.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT
  rm -rf "$TEST_ROOT"
  exit "$status"
}
trap cleanup EXIT

# Join shell continuations so each apply_dependency_patch call can be checked
# as one exact tuple instead of accepting three unrelated string occurrences.
PREPARE_FLAT="$TEST_ROOT/prepare-dependencies.flat"
awk '
  {
    line = $0
    continued = sub(/[[:space:]]*\\[[:space:]]*$/, "", line)
    if (buffer == "") {
      buffer = line
    } else {
      buffer = buffer " " line
    }
    if (!continued) {
      gsub(/[[:space:]]+/, " ", buffer)
      print buffer
      buffer = ""
    }
  }
  END {
    if (buffer != "") print buffer
  }
' "$PREPARE_SCRIPT" > "$PREPARE_FLAT"

MLX_SWIFT_CALL='apply_dependency_patch "mlx-swift direct slice update" "$MLX_SWIFT_CHECKOUT" "$MLX_SWIFT_DIRECT_SLICE_UPDATE_PATCH"'
MLX_SWIFT_LM_CALL='apply_dependency_patch "mlx-swift-lm direct KV slice update" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_DIRECT_KV_SLICE_UPDATE_PATCH"'
MLX_SWIFT_VERIFY='verify_checkout_revision "mlx-swift" "$MLX_SWIFT_CHECKOUT" "$MLX_SWIFT_EXPECTED_REVISION"'
MLX_SWIFT_LM_VERIFY='verify_checkout_revision "mlx-swift-lm" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_EXPECTED_REVISION"'
[[ "$(grep -Fxc "$MLX_SWIFT_CALL" "$PREPARE_FLAT" || true)" -eq 1 ]]
[[ "$(grep -Fxc "$MLX_SWIFT_LM_CALL" "$PREPARE_FLAT" || true)" -eq 1 ]]
[[ "$(grep -Fxc "$MLX_SWIFT_VERIFY" "$PREPARE_FLAT" || true)" -eq 1 ]]
[[ "$(grep -Fxc "$MLX_SWIFT_LM_VERIFY" "$PREPARE_FLAT" || true)" -eq 1 ]]

MLX_SWIFT_CALL_LINE="$(grep -Fn "$MLX_SWIFT_CALL" "$PREPARE_FLAT" | cut -d: -f1)"
MLX_SWIFT_LM_CALL_LINE="$(grep -Fn "$MLX_SWIFT_LM_CALL" "$PREPARE_FLAT" | cut -d: -f1)"
MLX_SWIFT_VERIFY_LINE="$(grep -Fn "$MLX_SWIFT_VERIFY" "$PREPARE_FLAT" | cut -d: -f1)"
MLX_SWIFT_LM_VERIFY_LINE="$(grep -Fn "$MLX_SWIFT_LM_VERIFY" "$PREPARE_FLAT" | cut -d: -f1)"
if [[ "$MLX_SWIFT_VERIFY_LINE" -ge "$MLX_SWIFT_CALL_LINE" ]] \
  || [[ "$MLX_SWIFT_LM_VERIFY_LINE" -ge "$MLX_SWIFT_LM_CALL_LINE" ]]; then
  echo "Pinned dependency revisions must be verified before direct slice patches are applied" >&2
  exit 1
fi
if [[ "$MLX_SWIFT_CALL_LINE" -ge "$MLX_SWIFT_LM_CALL_LINE" ]]; then
  echo "mlx-swift direct slice API must be prepared before the mlx-swift-lm consumer" >&2
  exit 1
fi

check_patch_round_trip \
  "mlx-swift-darwin" \
  "$MLX_SWIFT_CHECKOUT" \
  "$MLX_SWIFT_DARWIN_EXPECTED_REVISION" \
  "$MLX_SWIFT_PATCH"
check_patch_round_trip \
  "mlx-swift-linux" \
  "$MLX_SWIFT_CHECKOUT" \
  "$MLX_SWIFT_LINUX_EXPECTED_REVISION" \
  "$MLX_SWIFT_PATCH"
check_patch_round_trip \
  "mlx-swift-lm" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_EXPECTED_REVISION" "$MLX_SWIFT_LM_PATCH"

echo "MLX direct KV slice-update patch persistence checks passed"
