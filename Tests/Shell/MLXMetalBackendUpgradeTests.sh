#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PACKAGE_MANIFEST="$PACKAGE_ROOT/Package.swift"
PACKAGE_RESOLUTION="$PACKAGE_ROOT/Package.resolved"
PREPARE_SCRIPT="$PACKAGE_ROOT/prepare-dependencies.sh"
METAL_BUILD_SCRIPT="$PACKAGE_ROOT/build-metal.sh"
MLX_SWIFT_CHECKOUT="$PACKAGE_ROOT/.build/checkouts/mlx-swift"

MLX_SWIFT_REVISION="72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798"
MLX_CORE_REVISION="1f8e74e3f12f31365464a6867c6579f0e9b29d85"
MLX_C_REVISION="c74db5307cc8ce122f48d97ef951b30578674e7f"

grep -Fq "revision: \"$MLX_SWIFT_REVISION\"" "$PACKAGE_MANIFEST"
grep -Fq "\"revision\" : \"$MLX_SWIFT_REVISION\"" "$PACKAGE_RESOLUTION"
grep -Fq "MLX_SWIFT_DARWIN_EXPECTED_REVISION=\"$MLX_SWIFT_REVISION\"" \
  "$PREPARE_SCRIPT"
grep -Fq "MLX_SOURCE_DARWIN_EXPECTED_REVISION=\"$MLX_CORE_REVISION\"" \
  "$PREPARE_SCRIPT"
grep -Fq "MLX_C_SOURCE_DARWIN_EXPECTED_REVISION=\"$MLX_C_REVISION\"" \
  "$PREPARE_SCRIPT"

# Compile the authoritative static Metal sources. The generated JIT directory
# intentionally does not contain every 0.32.2 standalone kernel.
grep -Fq 'KERNEL_ROOT="$MLX_SOURCE_ROOT/mlx/backend/metal/kernels"' \
  "$METAL_BUILD_SCRIPT"
grep -Fq '"$KERNEL_ROOT/dot.metal"' "$METAL_BUILD_SCRIPT"
grep -Fq '"$KERNEL_ROOT/fence.metal"' "$METAL_BUILD_SCRIPT"
grep -Fq -- '-I "$MLX_SOURCE_ROOT"' "$METAL_BUILD_SCRIPT"
if grep -Fq 'mlx-generated' "$METAL_BUILD_SCRIPT"; then
  echo "Metal build must not depend on the incomplete generated static-kernel copy" >&2
  exit 1
fi

if [[ -d "$MLX_SWIFT_CHECKOUT" ]]; then
  [[ "$(git -C "$MLX_SWIFT_CHECKOUT" rev-parse HEAD)" == "$MLX_SWIFT_REVISION" ]]
  [[ "$(git -C "$MLX_SWIFT_CHECKOUT/Source/Cmlx/mlx" rev-parse HEAD)" \
    == "$MLX_CORE_REVISION" ]]
  [[ "$(git -C "$MLX_SWIFT_CHECKOUT/Source/Cmlx/mlx-c" rev-parse HEAD)" \
    == "$MLX_C_REVISION" ]]
fi

echo "MLX Metal backend provenance checks passed"
