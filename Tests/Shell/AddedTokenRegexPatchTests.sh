#!/usr/bin/env bash
set -euo pipefail
TOKEN_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TOKEN_CHECKOUT="$TOKEN_ROOT/.build/checkouts/swift-transformers"
TOKEN_REV=2fa33e1f5e7131a7fc64c28e6d161dcec0d24820
TOKEN_FILE=Sources/Tokenizers/Tokenizer.swift
TOKEN_PATCH="$TOKEN_ROOT/Patches/swift-transformers-added-token-regex.patch"
TOKEN_MIGRATION="$TOKEN_ROOT/Patches/swift-transformers-added-token-regex-empty-token-migration.patch"
TOKEN_PRIOR="$TOKEN_ROOT/Patches/swift-transformers-incremental-bytelevel-decoder.patch"
TOKEN_TEST="$(mktemp -d "${TMPDIR:-/tmp}/midnight-added-token-regex.XXXXXX")"
trap 'rm -rf "$TOKEN_TEST"' EXIT
mkdir -p "$TOKEN_TEST/pinned" "$TOKEN_TEST/current/Sources/Tokenizers"
git -C "$TOKEN_CHECKOUT" archive "$TOKEN_REV" | tar -xf - -C "$TOKEN_TEST/pinned"
cp "$TOKEN_CHECKOUT/$TOKEN_FILE" "$TOKEN_TEST/current/$TOKEN_FILE"
if git -C "$TOKEN_TEST/current" apply --reverse --check "$TOKEN_PATCH" >/dev/null 2>&1; then
  git -C "$TOKEN_TEST/current" apply --reverse "$TOKEN_PATCH"
elif git -C "$TOKEN_TEST/current" apply --check "$TOKEN_MIGRATION" >/dev/null 2>&1; then
  git -C "$TOKEN_TEST/current" apply "$TOKEN_MIGRATION"
  git -C "$TOKEN_TEST/current" apply --reverse "$TOKEN_PATCH"
fi
printf 'retain unrelated data\n' > "$TOKEN_TEST/current/unrelated.txt"
cp -R "$TOKEN_TEST/pinned" "$TOKEN_TEST/stack"
git -C "$TOKEN_TEST/stack" apply --whitespace=error-all "$TOKEN_PRIOR"
source "$TOKEN_ROOT/Scripts/added-token-regex-patch.sh"

# Mock only the revision lookup; all patch validation and application is real git.
git() {
  if [[ "$#" -eq 4 && "$1" == -C && "$3" == rev-parse && "$4" == HEAD && "$2" == "$TOKEN_TEST/"* ]]; then
    printf '%s\n' "${TOKEN_TEST_REVISION:-$TOKEN_REV}"
  else
    command git "$@"
  fi
}

for fixture in pinned current stack; do
  cp -R "$TOKEN_TEST/$fixture" "$TOKEN_TEST/$fixture-before"
  model_runner_prepare_added_token_regex "$TOKEN_ROOT" "$TOKEN_TEST/$fixture"
  cp -R "$TOKEN_TEST/$fixture" "$TOKEN_TEST/$fixture-applied"
  model_runner_prepare_added_token_regex "$TOKEN_ROOT" "$TOKEN_TEST/$fixture"
  diff -qr "$TOKEN_TEST/$fixture-applied" "$TOKEN_TEST/$fixture" >/dev/null
  git -C "$TOKEN_TEST/$fixture" apply --reverse "$TOKEN_PATCH"
  diff -qr "$TOKEN_TEST/$fixture-before" "$TOKEN_TEST/$fixture" >/dev/null
done

# Reconstruct the previous complete overlay without installing it in production.
cp -R "$TOKEN_TEST/current-before" "$TOKEN_TEST/legacy"
git -C "$TOKEN_TEST/legacy" apply "$TOKEN_PATCH"
git -C "$TOKEN_TEST/legacy" apply --reverse "$TOKEN_MIGRATION"
cp -R "$TOKEN_TEST/legacy" "$TOKEN_TEST/legacy-before"
model_runner_prepare_added_token_regex "$TOKEN_ROOT" "$TOKEN_TEST/legacy"
diff -qr "$TOKEN_TEST/current-applied" "$TOKEN_TEST/legacy" >/dev/null
model_runner_prepare_added_token_regex "$TOKEN_ROOT" "$TOKEN_TEST/legacy"
diff -qr "$TOKEN_TEST/current-applied" "$TOKEN_TEST/legacy" >/dev/null

# Both independently managed patches must remain reversible after both are applied.
model_runner_prepare_added_token_regex "$TOKEN_ROOT" "$TOKEN_TEST/stack"
git -C "$TOKEN_TEST/stack" apply --reverse --check "$TOKEN_PRIOR"
git -C "$TOKEN_TEST/stack" apply --reverse "$TOKEN_PRIOR"
git -C "$TOKEN_TEST/stack" apply --check --whitespace=error-all "$TOKEN_PRIOR"
git -C "$TOKEN_TEST/stack" apply --whitespace=error-all "$TOKEN_PRIOR"
model_runner_prepare_added_token_regex "$TOKEN_ROOT" "$TOKEN_TEST/stack"

token_assert_rejected() {
  local fixture="$1"
  cp -R "$TOKEN_TEST/$fixture" "$TOKEN_TEST/$fixture-before-rejection"
  if model_runner_prepare_added_token_regex "$TOKEN_ROOT" "$TOKEN_TEST/$fixture"; then
    echo "Expected added-token fixture $fixture to be rejected." >&2
    exit 1
  fi
  diff -qr "$TOKEN_TEST/$fixture-before-rejection" "$TOKEN_TEST/$fixture" >/dev/null
}

cp -R "$TOKEN_TEST/current-before" "$TOKEN_TEST/conflict"
printf '// incompatible tokenizer source\n' > "$TOKEN_TEST/conflict/$TOKEN_FILE"
token_assert_rejected conflict
cp -R "$TOKEN_TEST/legacy-before" "$TOKEN_TEST/partial-legacy"
python3 - "$TOKEN_TEST/partial-legacy/$TOKEN_FILE" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
assert s.count('if !needsAddedTokenCaptures {') == 1
p.write_text(s.replace('if !needsAddedTokenCaptures {', 'if needsAddedTokenCaptures {', 1))
PY
token_assert_rejected partial-legacy
cp -R "$TOKEN_TEST/current-before" "$TOKEN_TEST/stale-pin"
TOKEN_TEST_REVISION=0000000000000000000000000000000000000000 token_assert_rejected stale-pin

echo 'Added-token regex passes replay, prior-overlay compatibility, legacy migration, idempotence, reversal and rejection checks.'
