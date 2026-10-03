#!/usr/bin/env bash
set -euo pipefail
VISION_PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$VISION_PACKAGE_ROOT"
# Extra SwiftPM flags permit offline dependency mirrors or a custom cache.
swift build -c release --product midnight-vision-worker "$@"
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Built the compatibility-error executable; native vision requires Apple silicon macOS."
  exit 0
fi
VISION_BIN_DIR="$(swift build -c release --show-bin-path "$@")"
VISION_SCRATCH_DIR="$(cd "$VISION_BIN_DIR/../.." && pwd)"
VISION_MLX_ROOT="$VISION_SCRATCH_DIR/checkouts/mlx-swift/Source/Cmlx/mlx"
VISION_KERNEL_ROOT="$VISION_MLX_ROOT/mlx/backend/metal/kernels"
VISION_AIR_DIR="$VISION_SCRATCH_DIR/vision-metal"
mkdir -p "$VISION_AIR_DIR"
if ! xcrun -sdk macosx --find metal >/dev/null 2>&1; then
  echo "Metal compiler is required; install the Xcode MetalToolchain component." >&2
  exit 1
fi
VISION_AIR_FILES=()
for source in steel/attn/kernels/steel_attention arg_reduce conv dot fence rms_norm random scaled_dot_product_attention layer_norm rope; do
  output="$VISION_AIR_DIR/$(basename "$source").air"
  xcrun -sdk macosx metal -std=metal4.0 -Wno-c++20-extensions -I "$VISION_MLX_ROOT" \
    -c "$VISION_KERNEL_ROOT/$source.metal" -o "$output"
  VISION_AIR_FILES+=("$output")
done
xcrun -sdk macosx metallib "${VISION_AIR_FILES[@]}" -o "$VISION_BIN_DIR/mlx.metallib"
echo "Built $VISION_BIN_DIR/midnight-vision-worker and its separate mlx.metallib."
