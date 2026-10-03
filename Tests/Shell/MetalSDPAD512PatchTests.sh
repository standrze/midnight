#!/usr/bin/env bash
set -euo pipefail

SDPA_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SDPA_SWIFT="$SDPA_ROOT/.build/checkouts/mlx-swift"
SDPA_CORE="$SDPA_SWIFT/Source/Cmlx/mlx"
SDPA_TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/midnight-sdpa-d512.XXXXXX")"
trap 'rm -rf "$SDPA_TEST_ROOT"' EXIT

round_trip() {
  local sdpa_name="$1" sdpa_checkout="$2" sdpa_revision="$3" sdpa_patch="$4"
  shift 4
  local sdpa_original="$SDPA_TEST_ROOT/$sdpa_name-original"
  local sdpa_patched="$SDPA_TEST_ROOT/$sdpa_name-patched"
  local sdpa_target_count sdpa_target
  sdpa_target_count="$(grep -c '^diff --git ' "$sdpa_patch")"
  [[ "$sdpa_target_count" -eq "$#" ]]
  for sdpa_target in "$@"; do
    grep -Fqx "diff --git a/$sdpa_target b/$sdpa_target" "$sdpa_patch"
  done
  mkdir -p "$sdpa_original" "$sdpa_patched"
  git -C "$sdpa_checkout" archive "$sdpa_revision" "$@" | tar -xf - -C "$sdpa_original"
  cp -R "$sdpa_original/." "$sdpa_patched/"
  git -C "$sdpa_patched" apply --check --whitespace=error-all "$sdpa_patch"
  git -C "$sdpa_patched" apply "$sdpa_patch"
  git -C "$sdpa_patched" apply --reverse --check "$sdpa_patch"
}

round_trip core "$SDPA_CORE" 1f8e74e3f12f31365464a6867c6579f0e9b29d85 \
  "$SDPA_ROOT/Patches/mlx-metal-sdpa-d512-decode.patch" \
  mlx/backend/metal/kernels/scaled_dot_product_attention.metal \
  mlx/backend/metal/scaled_dot_product_attention.cpp
round_trip swift "$SDPA_SWIFT" 72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798 \
  "$SDPA_ROOT/Patches/mlx-swift-metal-sdpa-d512-generated.patch" \
  Source/Cmlx/mlx-generated/metal/scaled_dot_product_attention.metal

# The generated copy only shortens include paths. Detect omissions in either library.
sed 's@mlx/backend/metal/kernels/@@g' \
  "$SDPA_TEST_ROOT/core-patched/mlx/backend/metal/kernels/scaled_dot_product_attention.metal" \
  > "$SDPA_TEST_ROOT/normalized.metal"
cmp "$SDPA_TEST_ROOT/normalized.metal" \
  "$SDPA_TEST_ROOT/swift-patched/Source/Cmlx/mlx-generated/metal/scaled_dot_product_attention.metal"

git -C "$SDPA_TEST_ROOT/core-patched" apply --reverse "$SDPA_ROOT/Patches/mlx-metal-sdpa-d512-decode.patch"
git -C "$SDPA_TEST_ROOT/swift-patched" apply --reverse "$SDPA_ROOT/Patches/mlx-swift-metal-sdpa-d512-generated.patch"
diff -qr "$SDPA_TEST_ROOT/core-original" "$SDPA_TEST_ROOT/core-patched" >/dev/null
diff -qr "$SDPA_TEST_ROOT/swift-original" "$SDPA_TEST_ROOT/swift-patched" >/dev/null

# Non-Mac preparation must not access a checkout or require the patches.
source "$SDPA_ROOT/Scripts/metal-sdpa-d512-patch.sh"
model_runner_prepare_metal_sdpa_d512 Linux /does-not-exist /does-not-exist
echo "D512 SDPA pinned-patch round trips and generated Metal consistency passed."
