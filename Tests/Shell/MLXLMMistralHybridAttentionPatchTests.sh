#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PATCH_FILE="$ROOT/Patches/mlx-swift-lm-mistral-hybrid-attention.patch"
PREPARE="$ROOT/prepare-dependencies.sh"
CHECKOUT="$ROOT/.build/checkouts/mlx-swift-lm"

grep -Fq 'mlx-swift-lm-mistral-hybrid-attention.patch' "$PREPARE"
grep -Fq '"mlx-swift-lm Mistral hybrid attention"' "$PREPARE"
grep -Fq 'case layerTypes = "layer_types"' "$PATCH_FILE"
grep -Fq 'case slidingWindow = "sliding_window"' "$PATCH_FILE"
grep -Fq 'modelType == "mistral"' "$PATCH_FILE"
grep -Fq 'usesSlidingWindow: layerType == "sliding_attention"' "$PATCH_FILE"
grep -Fq 'windowSize: slidingWindow' "$PATCH_FILE"
grep -Fq 'slidingWindow == nil ? "full_attention" : "sliding_attention"' "$PATCH_FILE"

if [[ -d "$CHECKOUT/.git" ]]; then
  git -C "$CHECKOUT" apply --reverse --check "$PATCH_FILE"
fi

echo "mlx-swift-lm Mistral hybrid-attention patch checks passed"
