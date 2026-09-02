#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PATCH_FILE="$ROOT/Patches/mlx-swift-lm-mtp-prompt-hidden-window.patch"
PREPARE="$ROOT/prepare-dependencies.sh"
CHECKOUT="$ROOT/.build/checkouts/mlx-swift-lm"

grep -Fq 'mlx-swift-lm-mtp-prompt-hidden-window.patch' "$PREPARE"
grep -Fq 'var promptHiddenStateWindow: Int? { get }' "$PATCH_FILE"
grep -Fq 'var supportsChunkedPromptPrefill: Bool { get }' "$PATCH_FILE"
grep -Fq 'drafter.promptHiddenStateWindow ?? promptLength' "$PATCH_FILE"
grep -Fq '!drafter.supportsChunkedPromptPrefill' "$PATCH_FILE"

if [[ -d "$CHECKOUT/.git" ]]; then
  git -C "$CHECKOUT" apply --reverse --check "$PATCH_FILE"
fi

echo "mlx-swift-lm MTP prompt hidden-window patch checks passed"
