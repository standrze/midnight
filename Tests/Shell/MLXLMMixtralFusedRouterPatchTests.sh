#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PATCH_FILE="$ROOT/Patches/mlx-swift-lm-mixtral-fused-router.patch"
PREPARE="$ROOT/prepare-dependencies.sh"
CHECKOUT="$ROOT/.build/checkouts/mlx-swift-lm"

grep -Fq 'mlx-swift-lm-mixtral-fused-router.patch' "$PREPARE"
grep -Fq '"mlx-swift-lm Mixtral fused decode router"' "$PREPARE"
grep -Fq '#if os(macOS)' "$PATCH_FILE"
grep -Fq 'return false' "$PATCH_FILE"
grep -Fq 'Device.defaultDevice().deviceType == .gpu' "$PATCH_FILE"
grep -Fq 'shouldUseFusedMixtralRouter(gates, k: k, isTraining: training)' "$PATCH_FILE"
grep -Fq 'selection.dtype == .float16' "$PATCH_FILE"
grep -Fq 'MLX.argPartition(-gates' "$PATCH_FILE"
grep -Fq 'MLX.softmax(selectedLogits' "$PATCH_FILE"
grep -Fq 'testFusedDescendingRouterMatchesGenericMixtralExpression' "$PATCH_FILE"
grep -Fq 'testFusedRouterRanksZeroAboveNegativeLogits' "$PATCH_FILE"
grep -Fq 'testFusedRouterIsDisabledOnCPU' "$PATCH_FILE"
grep -Fq 'testMixtralRouterGateIsInferenceOnlyAndRejectsUnsupportedDTypes' "$PATCH_FILE"
grep -Fq 'testFusedRouterMatchesGenericAcrossDeterministicFixtures' "$PATCH_FILE"
grep -Fq 'map(\.bitPattern)' "$PATCH_FILE"

if [[ -d "$CHECKOUT/.git" ]]; then
  git -C "$CHECKOUT" apply --reverse --check "$PATCH_FILE"
fi

echo "mlx-swift-lm Mixtral fused-router patch checks passed"
