#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PATCH_FILE="$PACKAGE_ROOT/Patches/mlx-swift-lm-gemma4-nonrotating-cache.patch"
PREPARE_SCRIPT="$PACKAGE_ROOT/prepare-dependencies.sh"

grep -Fq 'makeHybridAttentionKVCache(' "$PATCH_FILE"
grep -Fq 'makeAttentionKVCache(parameters: parameters)' "$PATCH_FILE"
grep -Fq 'mlx-swift-lm-gemma4-nonrotating-cache.patch' "$PREPARE_SCRIPT"
grep -Fq '"mlx-swift-lm Gemma 4 non-rotating cache"' "$PREPARE_SCRIPT"

echo "Gemma 4 non-rotating cache patch checks passed"
