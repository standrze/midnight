#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PACKAGE_ROOT"

"$PACKAGE_ROOT/prepare-dependencies.sh"
BUILD_CONFIGURATION="${MODEL_RUNNER_BUILD_CONFIGURATION:-debug}"
BUILD_PRODUCT="${MODEL_RUNNER_BUILD_PRODUCT:-midnight}"
PINNED_MLX="${MODEL_RUNNER_PINNED_MLX:-0}"
case "$BUILD_CONFIGURATION" in
  debug|release) ;;
  *)
    echo "MODEL_RUNNER_BUILD_CONFIGURATION must be debug or release." >&2
    exit 2
    ;;
esac

case "$PINNED_MLX" in
  0|1) ;;
  *)
    echo "MODEL_RUNNER_PINNED_MLX must be 0 or 1." >&2
    exit 2
    ;;
esac

# Experimental only: the matched Q4R8 A/B did not show a repeatable speed win.
if [[ "$PINNED_MLX" == "1" ]]; then
  swift build --configuration "$BUILD_CONFIGURATION" \
    -Xswiftc -DMODEL_RUNNER_PINNED_MLX \
    --product "$BUILD_PRODUCT"
else
  swift build --configuration "$BUILD_CONFIGURATION" --product "$BUILD_PRODUCT"
fi

BIN_DIR="$(swift build --configuration "$BUILD_CONFIGURATION" --show-bin-path)"
MLX_SOURCE_ROOT="$PACKAGE_ROOT/.build/checkouts/mlx-swift/Source/Cmlx/mlx"
KERNEL_ROOT="$MLX_SOURCE_ROOT/mlx/backend/metal/kernels"
AIR_DIR="$PACKAGE_ROOT/.build/metal"

if ! xcrun -sdk macosx --find metal >/dev/null 2>&1; then
  echo "The Metal compiler is missing. Install it once with:"
  echo "  xcodebuild -downloadComponent MetalToolchain"
  exit 1
fi

mkdir -p "$AIR_DIR"

SOURCES=(
  "$KERNEL_ROOT/steel/attn/kernels/steel_attention.metal"
  "$KERNEL_ROOT/arg_reduce.metal"
  "$KERNEL_ROOT/conv.metal"
  "$KERNEL_ROOT/dot.metal"
  "$KERNEL_ROOT/fence.metal"
  "$KERNEL_ROOT/rms_norm.metal"
  "$KERNEL_ROOT/random.metal"
  "$KERNEL_ROOT/scaled_dot_product_attention.metal"
  "$KERNEL_ROOT/layer_norm.metal"
  "$KERNEL_ROOT/rope.metal"
)

AIR_FILES=()
for source in "${SOURCES[@]}"; do
  name="$(basename "$source" .metal)"
  air_file="$AIR_DIR/$name.air"
  xcrun -sdk macosx metal \
    -std=metal4.0 \
    -Wno-c++20-extensions \
    -I "$MLX_SOURCE_ROOT" \
    -c "$source" \
    -o "$air_file"
  AIR_FILES+=("$air_file")
done

xcrun -sdk macosx metallib "${AIR_FILES[@]}" -o "$BIN_DIR/mlx.metallib"
echo "Built $BIN_DIR/$BUILD_PRODUCT and $BIN_DIR/mlx.metallib"
