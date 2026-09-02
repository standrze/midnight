#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PATCH_FILE="$PACKAGE_ROOT/Patches/mlx-swift-lm-corefoundation-linux.patch"
PREPARE_SCRIPT="$PACKAGE_ROOT/prepare-dependencies.sh"

grep -Fq 'MLX_SWIFT_LM_EXPECTED_REVISION="14414441fa44f45eee35a61e9fa0bab577cf9734"' \
  "$PREPARE_SCRIPT"
grep -Fq 'mlx-swift-lm-corefoundation-linux.patch' "$PREPARE_SCRIPT"
grep -Fq '"mlx-swift-lm" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_EXPECTED_REVISION"' \
  "$PREPARE_SCRIPT"
grep -Fq '"mlx-swift-lm CoreFoundation import" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_COREFOUNDATION_PATCH"' \
  "$PREPARE_SCRIPT"

if [[ "$(grep -F 'import CoreFoundation' "$PATCH_FILE" | grep -c '^+')" -ne 1 ]]; then
  echo "CoreFoundation patch must add exactly one import" >&2
  exit 1
fi

TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/model-runner-corefoundation.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT
  rm -rf "$TEST_ROOT"
  exit "$status"
}
trap cleanup EXIT

SOURCE_DIR="$TEST_ROOT/Libraries/MLXLMCommon/Tool"
mkdir -p "$SOURCE_DIR"
cat > "$SOURCE_DIR/Value.swift" <<'EOF'
// Copyright © 2025 Apple Inc.

import Foundation

/// Type-safe representation of JSON values
public enum JSONValue: Hashable, Codable, Sendable {}
EOF

git -C "$TEST_ROOT" apply --check "$PATCH_FILE"
git -C "$TEST_ROOT" apply "$PATCH_FILE"
git -C "$TEST_ROOT" apply --reverse --check "$PATCH_FILE"

if git -C "$TEST_ROOT" apply --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "CoreFoundation patch unexpectedly remained forward-applicable" >&2
  exit 1
fi
[[ "$(grep -F -c 'import CoreFoundation' "$SOURCE_DIR/Value.swift")" -eq 1 ]]
[[ "$(grep -F -c 'import Foundation' "$SOURCE_DIR/Value.swift")" -eq 1 ]]

echo "mlx-swift-lm CoreFoundation compatibility patch checks passed"
