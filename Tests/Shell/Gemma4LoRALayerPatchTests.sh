#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PATCH_FILE="$PACKAGE_ROOT/Patches/mlx-swift-lm-gemma4-lora-layers.patch"
PREPARE_SCRIPT="$PACKAGE_ROOT/prepare-dependencies.sh"

grep -Fq 'model.layers.map { $0.selfAttn }' "$PATCH_FILE"
grep -Fq 'model.layers' "$PATCH_FILE"
grep -Fq 'mlx-swift-lm-gemma4-lora-layers.patch' "$PREPARE_SCRIPT"
grep -Fq '"mlx-swift-lm Gemma 4 LoRA layer coverage"' "$PREPARE_SCRIPT"

echo "Gemma 4 LoRA layer coverage patch checks passed"
