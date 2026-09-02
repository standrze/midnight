#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PATCH_FILE="$PACKAGE_ROOT/Patches/mlx-swift-lm-backend-token-evaluation.patch"
PREPARE_SCRIPT="$PACKAGE_ROOT/prepare-dependencies.sh"

grep -Fq 'mlx-swift-lm backend-aware token evaluation' "$PREPARE_SCRIPT"
grep -Fq 'eval([token] + cache.flatMap { $0.state })' "$PREPARE_SCRIPT"

grep -Fq 'mlx-swift-lm-backend-token-evaluation.patch' "$PREPARE_SCRIPT"
grep -Fq '"mlx-swift-lm backend-aware token evaluation"' "$PREPARE_SCRIPT"
grep -Fq '+            #if os(macOS)' "$PATCH_FILE"
grep -Fq '+                asyncEval(token)' "$PATCH_FILE"
grep -Fq '+                eval([token] + cache.flatMap { $0.state })' "$PATCH_FILE"

TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/model-runner-token-eval.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT
  rm -rf "$TEST_ROOT"
  exit "$status"
}
trap cleanup EXIT

SOURCE_DIR="$TEST_ROOT/Libraries/MLXLMCommon"
mkdir -p "$SOURCE_DIR"
cat > "$SOURCE_DIR/Evaluate.swift" <<'EOF'
public struct TokenIterator: TokenIteratorProtocol {
    mutating public func next() -> Int? {
        return autoreleasepool {
            let previousY = y

            // compute the next state and async eval the next token
            let token = step(previous: previousY)
            y = .init(tokens: token)
            // Evaluate the cache state together with the token: caches update
            // through functional ops (concatenation, slice assignment), and an
            // unevaluated chain of those updates keeps every prior step's
            // intermediates alive. Python mlx-lm settles cache state the same
            // way in its generation loop.
            asyncEval([token] + cache.flatMap { $0.state })

            tokenCount += 1

            // Periodically return freed buffers that cannot be reused (odd or
            // monotonically growing sizes accumulate in the pool otherwise).
            // Matches mlx-lm's clear cadence.
            if tokenCount % 256 == 0 {
                MLX.Memory.clearCache()
            }

            return previousY.tokens.item(Int.self)
        }
    }
}
EOF

git -C "$TEST_ROOT" apply --check "$PATCH_FILE"
git -C "$TEST_ROOT" apply "$PATCH_FILE"
git -C "$TEST_ROOT" apply --reverse --check "$PATCH_FILE"

grep -Fq '#if os(macOS)' "$SOURCE_DIR/Evaluate.swift"
grep -Fq 'asyncEval(token)' "$SOURCE_DIR/Evaluate.swift"
grep -Fq 'eval([token] + cache.flatMap { $0.state })' "$SOURCE_DIR/Evaluate.swift"
if grep -Fq 'MLX.Memory.clearCache()' "$SOURCE_DIR/Evaluate.swift"; then
  echo "Token evaluation patch must not clear the allocator cache during decode." >&2
  exit 1
fi

echo "mlx-swift-lm backend-aware token evaluation patch checks passed"
