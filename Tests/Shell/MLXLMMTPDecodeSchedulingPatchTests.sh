#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROMPT_PATCH="$ROOT/Patches/mlx-swift-lm-mtp-prompt-hidden-window.patch"
SCHEDULING_PATCH="$ROOT/Patches/mlx-swift-lm-mtp-decode-scheduling.patch"
DIAGNOSTIC_PATCH="$ROOT/Patches/mlx-swift-lm-mtp-first-rejection-diagnostic.patch"
CHECKOUT="$ROOT/.build/checkouts/mlx-swift-lm"
EXPECTED_REVISION="14414441fa44f45eee35a61e9fa0bab577cf9734"
SOURCE_RELATIVE="Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift"

require_added_count() {
  local expected="$1"
  local pattern="$2"
  local patch_file="$3"
  local actual
  actual="$(grep -F "$pattern" "$patch_file" | grep -c '^+' || true)"
  if [[ "$actual" -ne "$expected" ]]; then
    echo "Expected $expected added occurrence(s) of '$pattern' in $patch_file; found $actual" >&2
    exit 1
  fi
}

# Guard the performance-sensitive structure as well as patch applicability.
require_added_count 1 'private var passthroughPipelinePrimed = false' "$SCHEDULING_PATCH"
require_added_count 1 'if processor == nil, sampler is ArgMaxSampler {' "$SCHEDULING_PATCH"
require_added_count 1 'eval(mainTokens, flatDraftTokens)' "$SCHEDULING_PATCH"
require_added_count 1 'asyncEval(token)' "$SCHEDULING_PATCH"
require_added_count 1 'eval([token] + mainCache.flatMap { $0.state })' "$SCHEDULING_PATCH"
require_added_count 1 'eval([firstToken] + mainCache.flatMap { $0.state })' "$SCHEDULING_PATCH"
require_added_count 1 'let firstToken = advancePassthrough()' "$SCHEDULING_PATCH"
require_added_count 1 'let followingToken = advancePassthrough()' "$SCHEDULING_PATCH"
require_added_count 1 'pendingTokens.append(materializedTargetStep())' "$SCHEDULING_PATCH"

require_added_count 1 'MODEL_RUNNER_DFLASH_FIRST_REJECTION_DIAGNOSTIC' "$DIAGNOSTIC_PATCH"
require_added_count 3 'reportFirstRejection(' "$DIAGNOSTIC_PATCH"
require_added_count 1 '[MTP_FIRST_REJECTION]' "$DIAGNOSTIC_PATCH"
require_added_count 1 'let topTwo = top(logits.asType(.float32), k: 2, axis: -1).flattened()' \
  "$DIAGNOSTIC_PATCH"

if ! git -C "$CHECKOUT" rev-parse --git-dir >/dev/null 2>&1; then
  echo "Missing mlx-swift-lm checkout at $CHECKOUT" >&2
  exit 1
fi

ACTUAL_REVISION="$(git -C "$CHECKOUT" rev-parse HEAD)"
if [[ "$ACTUAL_REVISION" != "$EXPECTED_REVISION" ]]; then
  echo "Expected mlx-swift-lm $EXPECTED_REVISION; found $ACTUAL_REVISION" >&2
  exit 1
fi

TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/model-runner-mtp-decode-patches.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT
  rm -rf "$TEST_ROOT"
  exit "$status"
}
trap cleanup EXIT

SOURCE_FILE="$TEST_ROOT/$SOURCE_RELATIVE"
mkdir -p "$(dirname "$SOURCE_FILE")"

# Reconstruct the prompt-window-overlaid baseline from the immutable pinned
# tree, independent of any patches currently present in the working checkout.
git -C "$CHECKOUT" show "$EXPECTED_REVISION:$SOURCE_RELATIVE" > "$SOURCE_FILE"
git -C "$TEST_ROOT" apply --include="$SOURCE_RELATIVE" "$PROMPT_PATCH"
cp "$SOURCE_FILE" "$TEST_ROOT/prompt-window-baseline.swift"

# The scheduling patch is defined against the prompt-window baseline; the
# diagnostic patch is intentionally defined against the scheduling result.
git -C "$TEST_ROOT" apply --check "$SCHEDULING_PATCH"
git -C "$TEST_ROOT" apply "$SCHEDULING_PATCH"
git -C "$TEST_ROOT" apply --check "$DIAGNOSTIC_PATCH"
git -C "$TEST_ROOT" apply "$DIAGNOSTIC_PATCH"

grep -Fq 'if processor == nil, sampler is ArgMaxSampler {' "$SOURCE_FILE"
grep -Fq 'let followingToken = advancePassthrough()' "$SOURCE_FILE"
grep -Fq 'MODEL_RUNNER_DFLASH_FIRST_REJECTION_DIAGNOSTIC' "$SOURCE_FILE"
grep -Fq '[MTP_FIRST_REJECTION]' "$SOURCE_FILE"

# Reverse checks must run newest-first. This models dependency preparation's
# already-applied detection without modifying the actual dependency checkout.
git -C "$TEST_ROOT" apply --reverse --check "$DIAGNOSTIC_PATCH"
git -C "$TEST_ROOT" apply --reverse "$DIAGNOSTIC_PATCH"
if grep -Fq 'MODEL_RUNNER_DFLASH_FIRST_REJECTION_DIAGNOSTIC' "$SOURCE_FILE"; then
  echo "Diagnostic patch content remained after its reverse application" >&2
  exit 1
fi
grep -Fq 'if processor == nil, sampler is ArgMaxSampler {' "$SOURCE_FILE"

git -C "$TEST_ROOT" apply --reverse --check "$SCHEDULING_PATCH"
git -C "$TEST_ROOT" apply --reverse "$SCHEDULING_PATCH"
cmp "$SOURCE_FILE" "$TEST_ROOT/prompt-window-baseline.swift"

echo "mlx-swift-lm MTP scheduling and first-rejection patch checks passed"
