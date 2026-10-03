#!/usr/bin/env bash
set -euo pipefail
CLEANUP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLEANUP_LM="$CLEANUP_ROOT/.build/checkouts/swift-transformers"
CLEANUP_REV=2fa33e1f5e7131a7fc64c28e6d161dcec0d24820
CLEANUP_FILE=Sources/Tokenizers/Tokenizer.swift
CLEANUP_PATCH="$CLEANUP_ROOT/Patches/swift-transformers-gemma-cleanup-default.patch"
CLEANUP_TEST="$(mktemp -d "${TMPDIR:-/tmp}/midnight-extra-cleanup.XXXXXX")"
trap 'rm -rf "$CLEANUP_TEST"' EXIT
mkdir -p "$CLEANUP_TEST/current/Sources/Tokenizers" "$CLEANUP_TEST/pinned"
cp "$CLEANUP_LM/$CLEANUP_FILE" "$CLEANUP_TEST/current/$CLEANUP_FILE"
if git -C "$CLEANUP_TEST/current" apply --reverse --check "$CLEANUP_PATCH" >/dev/null 2>&1; then
  git -C "$CLEANUP_TEST/current" apply --reverse "$CLEANUP_PATCH"
fi
printf 'retain unrelated data\n' > "$CLEANUP_TEST/current/unrelated.txt"
git -C "$CLEANUP_LM" archive "$CLEANUP_REV" "$CLEANUP_FILE" | tar -xf - -C "$CLEANUP_TEST/pinned"
cp -R "$CLEANUP_TEST/current" "$CLEANUP_TEST/current-original"
cp -R "$CLEANUP_TEST/pinned" "$CLEANUP_TEST/pinned-original"
source "$CLEANUP_ROOT/Scripts/gemma-cleanup-default-patch.sh"

# Only revision lookup is mocked; every forward/reverse application is real git.
git() {
  if [[ "$#" -eq 4 && "$1" == -C && "$3" == rev-parse && "$4" == HEAD && "$2" == "$CLEANUP_TEST/"* ]]; then
    printf '%s\n' "${CLEANUP_TEST_REVISION:-$CLEANUP_REV}"
  else
    command git "$@"
  fi
}

for fixture in current pinned; do
  model_runner_prepare_gemma_cleanup_default "$CLEANUP_ROOT" "$CLEANUP_TEST/$fixture"
  git -C "$CLEANUP_TEST/$fixture" apply --reverse --check "$CLEANUP_PATCH"
  cp -R "$CLEANUP_TEST/$fixture" "$CLEANUP_TEST/$fixture-applied"
  model_runner_prepare_gemma_cleanup_default "$CLEANUP_ROOT" "$CLEANUP_TEST/$fixture"
  diff -qr "$CLEANUP_TEST/$fixture-applied" "$CLEANUP_TEST/$fixture" >/dev/null
  git -C "$CLEANUP_TEST/$fixture" apply --reverse "$CLEANUP_PATCH"
  diff -qr "$CLEANUP_TEST/$fixture-original" "$CLEANUP_TEST/$fixture" >/dev/null
done

cleanup_assert_rejected() {
  local fixture="$1"
  cp -R "$CLEANUP_TEST/$fixture" "$CLEANUP_TEST/$fixture-before"
  if model_runner_prepare_gemma_cleanup_default "$CLEANUP_ROOT" "$CLEANUP_TEST/$fixture"; then
    echo "Expected Gemma cleanup fixture $fixture to be rejected." >&2
    exit 1
  fi
  diff -qr "$CLEANUP_TEST/$fixture-before" "$CLEANUP_TEST/$fixture" >/dev/null
}

cp -R "$CLEANUP_TEST/current-original" "$CLEANUP_TEST/partial"
python3 - "$CLEANUP_TEST/partial/$CLEANUP_FILE" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
p.write_text(p.read_text().replace('cleanUpTokenizationSpaces.boolean(or: true)', 'cleanUpTokenizationSpaces.boolean(or: false)', 1))
PY
cleanup_assert_rejected partial
cp -R "$CLEANUP_TEST/current-original" "$CLEANUP_TEST/conflict"
printf '// unrelated incompatible source\n' > "$CLEANUP_TEST/conflict/$CLEANUP_FILE"
cleanup_assert_rejected conflict
cp -R "$CLEANUP_TEST/current-original" "$CLEANUP_TEST/stale-pin"
CLEANUP_TEST_REVISION=0000000000000000000000000000000000000000 cleanup_assert_rejected stale-pin

echo 'Gemma cleanup overlay passes pinned/current replay, idempotence, reversal, partial/conflict rejection and revision checks.'
