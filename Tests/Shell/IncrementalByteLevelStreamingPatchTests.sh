#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TRANSFORMERS_PATCH="$ROOT/Patches/swift-transformers-incremental-bytelevel-decoder.patch"
MLXLM_PATCH="$ROOT/Patches/mlx-swift-lm-incremental-bytelevel-streaming.patch"
PREPARE="$ROOT/prepare-dependencies.sh"
TRANSFORMERS_CHECKOUT="$ROOT/.build/checkouts/swift-transformers"
MLXLM_CHECKOUT="$ROOT/.build/checkouts/mlx-swift-lm"
TRANSFORMERS_REVISION="2fa33e1f5e7131a7fc64c28e6d161dcec0d24820"
MLXLM_REVISION="14414441fa44f45eee35a61e9fa0bab577cf9734"

[[ -s "$TRANSFORMERS_PATCH" ]]
[[ -s "$MLXLM_PATCH" ]]
[[ "$(grep -c '^diff --git ' "$TRANSFORMERS_PATCH" || true)" -eq 3 ]]
[[ "$(grep -c '^diff --git ' "$MLXLM_PATCH" || true)" -eq 5 ]]

grep -Fq 'public protocol IncrementalTokenDecoder: Sendable' "$TRANSFORMERS_PATCH"
grep -Fq 'final class ByteLevelIncrementalDecodingTable: Sendable' "$TRANSFORMERS_PATCH"
grep -Fq 'extension PreTrainedTokenizer: IncrementalTokenDecoderProviding' "$TRANSFORMERS_PATCH"
grep -Fq 'Copied decoder values keep independent UTF-8 state' "$TRANSFORMERS_PATCH"
grep -Fq 'private struct IncrementalTokenDecoderBridge' "$MLXLM_PATCH"
grep -Fq 'var incrementalPendingText = ""' "$MLXLM_PATCH"
grep -Fq 'testMultipleAppendsAreCombinedBeforeNext' "$MLXLM_PATCH"
grep -Fq 'incremental payload decoders are used and reset between Harmony frames' "$MLXLM_PATCH"

if ! git -C "$TRANSFORMERS_CHECKOUT" cat-file -e "${TRANSFORMERS_REVISION}^{commit}" \
  >/dev/null 2>&1; then
  echo "Missing pinned swift-transformers revision $TRANSFORMERS_REVISION" >&2
  exit 1
fi
if ! git -C "$MLXLM_CHECKOUT" cat-file -e "${MLXLM_REVISION}^{commit}" >/dev/null 2>&1; then
  echo "Missing pinned mlx-swift-lm revision $MLXLM_REVISION" >&2
  exit 1
fi

TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/model-runner-incremental-bytelevel-patches.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT
  rm -rf "$TEST_ROOT"
  exit "$status"
}
trap cleanup EXIT

PREPARE_FLAT="$TEST_ROOT/prepare-dependencies.flat"
awk '
  {
    line = $0
    continued = sub(/[[:space:]]*\\[[:space:]]*$/, "", line)
    buffer = buffer == "" ? line : buffer " " line
    if (!continued) {
      gsub(/[[:space:]]+/, " ", buffer)
      print buffer
      buffer = ""
    }
  }
  END { if (buffer != "") print buffer }
' "$PREPARE" > "$PREPARE_FLAT"

TRANSFORMERS_CHECKOUT_DECL='SWIFT_TRANSFORMERS_CHECKOUT="$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH/checkouts/swift-transformers"'
TRANSFORMERS_PATCH_DECL='SWIFT_TRANSFORMERS_INCREMENTAL_BYTELEVEL_PATCH="$PACKAGE_ROOT/Patches/swift-transformers-incremental-bytelevel-decoder.patch"'
TRANSFORMERS_REVISION_DECL='SWIFT_TRANSFORMERS_EXPECTED_REVISION="2fa33e1f5e7131a7fc64c28e6d161dcec0d24820"'
TRANSFORMERS_VERIFY='verify_checkout_revision "swift-transformers" "$SWIFT_TRANSFORMERS_CHECKOUT" "$SWIFT_TRANSFORMERS_EXPECTED_REVISION"'
TRANSFORMERS_APPLY='apply_dependency_patch "swift-transformers incremental ByteLevel decoder" "$SWIFT_TRANSFORMERS_CHECKOUT" "$SWIFT_TRANSFORMERS_INCREMENTAL_BYTELEVEL_PATCH"'
MLXLM_PATCH_DECL='MLX_SWIFT_LM_INCREMENTAL_BYTELEVEL_STREAMING_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-incremental-bytelevel-streaming.patch"'
MLXLM_REVISION_DECL='MLX_SWIFT_LM_EXPECTED_REVISION="14414441fa44f45eee35a61e9fa0bab577cf9734"'
MLXLM_VERIFY='verify_checkout_revision "mlx-swift-lm" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_EXPECTED_REVISION"'
MLXLM_APPLY='apply_dependency_patch "mlx-swift-lm incremental ByteLevel streaming" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_INCREMENTAL_BYTELEVEL_STREAMING_PATCH"'

for statement in \
  "$TRANSFORMERS_CHECKOUT_DECL" \
  "$TRANSFORMERS_PATCH_DECL" \
  "$TRANSFORMERS_REVISION_DECL" \
  "$TRANSFORMERS_VERIFY" \
  "$TRANSFORMERS_APPLY" \
  "$MLXLM_PATCH_DECL" \
  "$MLXLM_REVISION_DECL" \
  "$MLXLM_VERIFY" \
  "$MLXLM_APPLY"; do
  if [[ "$(grep -Fxc "$statement" "$PREPARE_FLAT" || true)" -ne 1 ]]; then
    echo "Expected exactly one prepare-dependencies statement: $statement" >&2
    exit 1
  fi
done

line_of() {
  grep -Fn "$1" "$PREPARE_FLAT" | cut -d: -f1
}

if [[ "$(line_of "$TRANSFORMERS_PATCH_DECL")" -ge "$(line_of "$TRANSFORMERS_APPLY")" ]] \
  || [[ "$(line_of "$TRANSFORMERS_REVISION_DECL")" -ge "$(line_of "$TRANSFORMERS_VERIFY")" ]] \
  || [[ "$(line_of "$TRANSFORMERS_VERIFY")" -ge "$(line_of "$TRANSFORMERS_APPLY")" ]] \
  || [[ "$(line_of "$MLXLM_PATCH_DECL")" -ge "$(line_of "$MLXLM_APPLY")" ]] \
  || [[ "$(line_of "$MLXLM_REVISION_DECL")" -ge "$(line_of "$MLXLM_VERIFY")" ]] \
  || [[ "$(line_of "$MLXLM_VERIFY")" -ge "$(line_of "$MLXLM_APPLY")" ]] \
  || [[ "$(line_of "$TRANSFORMERS_APPLY")" -ge "$(line_of "$MLXLM_APPLY")" ]]; then
  echo "Pinned revisions and the upstream tokenizer patch must precede the MLX bridge patch" >&2
  exit 1
fi

TRANSFORMERS_ROOT="$TEST_ROOT/swift-transformers"
TRANSFORMERS_BASELINE="$TEST_ROOT/swift-transformers-baseline"
MLXLM_ROOT="$TEST_ROOT/mlx-swift-lm"
MLXLM_BASELINE="$TEST_ROOT/mlx-swift-lm-baseline"
mkdir -p "$TRANSFORMERS_ROOT" "$TRANSFORMERS_BASELINE" "$MLXLM_ROOT" "$MLXLM_BASELINE"

for relative in \
  Sources/Tokenizers/Decoder.swift \
  Sources/Tokenizers/Tokenizer.swift; do
  mkdir -p "$TRANSFORMERS_ROOT/$(dirname "$relative")"
  mkdir -p "$TRANSFORMERS_BASELINE/$(dirname "$relative")"
  git -C "$TRANSFORMERS_CHECKOUT" show "$TRANSFORMERS_REVISION:$relative" \
    > "$TRANSFORMERS_ROOT/$relative"
  cp "$TRANSFORMERS_ROOT/$relative" "$TRANSFORMERS_BASELINE/$relative"
done

for relative in \
  Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift \
  Libraries/MLXLMCommon/Tokenizer.swift \
  Tests/MLXLMTests/HarmonyOutputRouterTests.swift \
  Tests/MLXLMTests/StopStringTests.swift \
  Tests/MLXLMTests/StreamingDetokenizerTests.swift; do
  mkdir -p "$MLXLM_ROOT/$(dirname "$relative")"
  mkdir -p "$MLXLM_BASELINE/$(dirname "$relative")"
  git -C "$MLXLM_CHECKOUT" show "$MLXLM_REVISION:$relative" > "$MLXLM_ROOT/$relative"
  cp "$MLXLM_ROOT/$relative" "$MLXLM_BASELINE/$relative"
done

git -C "$TRANSFORMERS_ROOT" apply --check --whitespace=error-all "$TRANSFORMERS_PATCH"
git -C "$TRANSFORMERS_ROOT" apply "$TRANSFORMERS_PATCH"
git -C "$MLXLM_ROOT" apply --check --whitespace=error-all "$MLXLM_PATCH"
git -C "$MLXLM_ROOT" apply "$MLXLM_PATCH"

[[ -f "$TRANSFORMERS_ROOT/Tests/TokenizersTests/IncrementalByteLevelDecoderTests.swift" ]]
grep -Fq 'maximumDecodedByteCount = 32 * 1_024 * 1_024' \
  "$TRANSFORMERS_ROOT/Sources/Tokenizers/Decoder.swift"
grep -Fq 'let uniqueConfiguredTokenIDs = Set(configuredTokenIDs)' \
  "$TRANSFORMERS_ROOT/Sources/Tokenizers/Tokenizer.swift"
grep -Fq 'private struct IncrementalTokenDecoderBridge' \
  "$MLXLM_ROOT/Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift"
grep -Fq 'var incrementalOutputIsStable = true' \
  "$MLXLM_ROOT/Libraries/MLXLMCommon/Tokenizer.swift"

git -C "$MLXLM_ROOT" apply --reverse --check "$MLXLM_PATCH"
git -C "$MLXLM_ROOT" apply --reverse "$MLXLM_PATCH"
git -C "$TRANSFORMERS_ROOT" apply --reverse --check "$TRANSFORMERS_PATCH"
git -C "$TRANSFORMERS_ROOT" apply --reverse "$TRANSFORMERS_PATCH"

[[ ! -e "$TRANSFORMERS_ROOT/Tests/TokenizersTests/IncrementalByteLevelDecoderTests.swift" ]]
diff -ru "$TRANSFORMERS_BASELINE" "$TRANSFORMERS_ROOT"
diff -ru "$MLXLM_BASELINE" "$MLXLM_ROOT"

echo "incremental ByteLevel streaming patch checks passed"
