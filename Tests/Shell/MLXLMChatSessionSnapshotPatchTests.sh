#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PATCH_FILE="$ROOT/Patches/mlx-swift-lm-chat-session-snapshot.patch"
PREPARE="$ROOT/prepare-dependencies.sh"
CHECKOUT="$ROOT/.build/checkouts/mlx-swift-lm"

grep -Fq 'mlx-swift-lm-chat-session-snapshot.patch' "$PREPARE"
grep -Fq 'public struct ChatSessionSnapshot: @unchecked Sendable' "$PATCH_FILE"
grep -Fq 'public func snapshot() async throws -> ChatSessionSnapshot' "$PATCH_FILE"
grep -Fq 'restoring snapshot: borrowing ChatSessionSnapshot' "$PATCH_FILE"
grep -Fq 'let main = snapshot.main.copy()' "$PATCH_FILE"
grep -Fq 'let mainCopy = stored.main.copy()' "$PATCH_FILE"
grep -Fq 'main: main,' "$PATCH_FILE"
grep -Fq 'uncommittedTokens: snapshot.uncommittedTokens ?? []' "$PATCH_FILE"

if [[ -d "$CHECKOUT/.git" ]]; then
  git -C "$CHECKOUT" apply --reverse --check "$PATCH_FILE"
fi

echo "mlx-swift-lm ChatSession snapshot patch checks passed"
