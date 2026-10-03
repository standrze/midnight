#!/usr/bin/env bash
set -euo pipefail
GEMMA_TAIL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GEMMA_TAIL_LM="$GEMMA_TAIL_ROOT/.build/checkouts/mlx-swift-lm"
GEMMA_TAIL_REV=14414441fa44f45eee35a61e9fa0bab577cf9734
GEMMA_TAIL_PATCH="$GEMMA_TAIL_ROOT/Patches/mlx-swift-lm-gemma3-compiled-tail.patch"
GEMMA_TAIL_LAYOUT="$GEMMA_TAIL_ROOT/Patches/mlx-swift-lm-gemma3-attention-layout.patch"
GEMMA_TAIL_TEXT=Libraries/MLXLLM/Models/Gemma3Text.swift
GEMMA_TAIL_NEW=Libraries/MLXLLM/Models/Gemma3CompiledTail.swift
GEMMA_TAIL_TEST="$(mktemp -d "${TMPDIR:-/tmp}/midnight-gemma3-compiled-tail.XXXXXX")"
trap 'rm -rf "$GEMMA_TAIL_TEST"' EXIT

# Verify the candidate is a narrow overlay, not a copy of the existing layout patch.
awk '/^\+\+\+ b\// {sub(/^\+\+\+ b\//, ""); print}' "$GEMMA_TAIL_PATCH" \
  | sort > "$GEMMA_TAIL_TEST/patch-targets.txt"
printf '%s\n' "$GEMMA_TAIL_TEXT" "$GEMMA_TAIL_NEW" \
  | sort > "$GEMMA_TAIL_TEST/expected-targets.txt"
cmp "$GEMMA_TAIL_TEST/expected-targets.txt" "$GEMMA_TAIL_TEST/patch-targets.txt"

# Prefer the current prepared source. Peel the candidate on this COPY only if
# it is already installed, making the test valid before and after preparation.
mkdir -p "$GEMMA_TAIL_TEST/base/Libraries/MLXLLM/Models"
cp "$GEMMA_TAIL_LM/$GEMMA_TAIL_TEXT" "$GEMMA_TAIL_TEST/base/$GEMMA_TAIL_TEXT"
if [[ -e "$GEMMA_TAIL_LM/$GEMMA_TAIL_NEW" ]]; then
  cp "$GEMMA_TAIL_LM/$GEMMA_TAIL_NEW" "$GEMMA_TAIL_TEST/base/$GEMMA_TAIL_NEW"
fi
if git -C "$GEMMA_TAIL_TEST/base" apply --reverse --check "$GEMMA_TAIL_PATCH" >/dev/null 2>&1; then
  git -C "$GEMMA_TAIL_TEST/base" apply --reverse "$GEMMA_TAIL_PATCH"
else
  git -C "$GEMMA_TAIL_TEST/base" apply --check --whitespace=error-all "$GEMMA_TAIL_PATCH"
fi
test ! -e "$GEMMA_TAIL_TEST/base/$GEMMA_TAIL_NEW"
printf 'preserve unrelated source\n' > "$GEMMA_TAIL_TEST/base/unrelated-local-edit.txt"
git -C "$GEMMA_TAIL_TEST/base" apply --reverse --check "$GEMMA_TAIL_LAYOUT"

# Independently reconstruct the pinned source plus its prior Gemma 3 overlay.
mkdir -p "$GEMMA_TAIL_TEST/pinned-layout"
git -C "$GEMMA_TAIL_LM" archive "$GEMMA_TAIL_REV" "$GEMMA_TAIL_TEXT" \
  | tar -xf - -C "$GEMMA_TAIL_TEST/pinned-layout"
git -C "$GEMMA_TAIL_TEST/pinned-layout" apply --whitespace=error-all "$GEMMA_TAIL_LAYOUT"
cp -R "$GEMMA_TAIL_TEST/pinned-layout" "$GEMMA_TAIL_TEST/pinned-layout-before"
cp -R "$GEMMA_TAIL_TEST/base" "$GEMMA_TAIL_TEST/current"

source "$GEMMA_TAIL_ROOT/Scripts/gemma3-compiled-tail-patch.sh"
# Fixtures have no Git database. Mock only the exact revision lookup on fixture
# paths; every apply/check/reverse operation is the actual git command.
git() {
  if [[ "$#" -eq 4 && "$1" == -C && "$3" == rev-parse && "$4" == HEAD && "$2" == "$GEMMA_TAIL_TEST/"* ]]; then
    printf '%s\n' "${GEMMA_TAIL_TEST_REVISION:-$GEMMA_TAIL_REV}"
  else
    command git "$@"
  fi
}
model_runner_prepare_gemma3_compiled_tail Linux /does-not-exist /does-not-exist

for fixture in current pinned-layout; do
  model_runner_prepare_gemma3_compiled_tail Darwin "$GEMMA_TAIL_ROOT" "$GEMMA_TAIL_TEST/$fixture"
  test -f "$GEMMA_TAIL_TEST/$fixture/$GEMMA_TAIL_NEW"
  git -C "$GEMMA_TAIL_TEST/$fixture" apply --reverse --check "$GEMMA_TAIL_PATCH"
  git -C "$GEMMA_TAIL_TEST/$fixture" apply --reverse --check "$GEMMA_TAIL_LAYOUT"
  cp -R "$GEMMA_TAIL_TEST/$fixture" "$GEMMA_TAIL_TEST/$fixture-applied"
  model_runner_prepare_gemma3_compiled_tail Darwin "$GEMMA_TAIL_ROOT" "$GEMMA_TAIL_TEST/$fixture"
  diff -qr "$GEMMA_TAIL_TEST/$fixture-applied" "$GEMMA_TAIL_TEST/$fixture" >/dev/null
  git -C "$GEMMA_TAIL_TEST/$fixture" apply --reverse "$GEMMA_TAIL_PATCH"
  test ! -e "$GEMMA_TAIL_TEST/$fixture/$GEMMA_TAIL_NEW"
done
diff -qr "$GEMMA_TAIL_TEST/base" "$GEMMA_TAIL_TEST/current" >/dev/null
diff -qr "$GEMMA_TAIL_TEST/pinned-layout-before" "$GEMMA_TAIL_TEST/pinned-layout" >/dev/null
git -C "$GEMMA_TAIL_TEST/pinned-layout" apply --reverse "$GEMMA_TAIL_LAYOUT"
git -C "$GEMMA_TAIL_LM" show "$GEMMA_TAIL_REV:$GEMMA_TAIL_TEXT" > "$GEMMA_TAIL_TEST/original.swift"
cmp "$GEMMA_TAIL_TEST/original.swift" "$GEMMA_TAIL_TEST/pinned-layout/$GEMMA_TAIL_TEXT"

gemma_tail_assert_rejected_without_changes() {
  local gemma_tail_fixture="$1"
  cp -R "$GEMMA_TAIL_TEST/$gemma_tail_fixture" "$GEMMA_TAIL_TEST/$gemma_tail_fixture-before"
  if model_runner_prepare_gemma3_compiled_tail Darwin "$GEMMA_TAIL_ROOT" "$GEMMA_TAIL_TEST/$gemma_tail_fixture"; then
    echo "Expected Gemma 3 compiled tail fixture $gemma_tail_fixture to be refused." >&2
    exit 1
  fi
  diff -qr "$GEMMA_TAIL_TEST/$gemma_tail_fixture-before" "$GEMMA_TAIL_TEST/$gemma_tail_fixture" >/dev/null
}

# A new file alone and a changed decoder alone are both incomplete overlays.
cp -R "$GEMMA_TAIL_TEST/base" "$GEMMA_TAIL_TEST/new-file-only"
cp "$GEMMA_TAIL_TEST/current-applied/$GEMMA_TAIL_NEW" "$GEMMA_TAIL_TEST/new-file-only/$GEMMA_TAIL_NEW"
gemma_tail_assert_rejected_without_changes new-file-only
cp -R "$GEMMA_TAIL_TEST/base" "$GEMMA_TAIL_TEST/decoder-only"
cp "$GEMMA_TAIL_TEST/current-applied/$GEMMA_TAIL_TEXT" "$GEMMA_TAIL_TEST/decoder-only/$GEMMA_TAIL_TEXT"
gemma_tail_assert_rejected_without_changes decoder-only

cp -R "$GEMMA_TAIL_TEST/current-applied" "$GEMMA_TAIL_TEST/drifted-new-file"
printf '\n// intentional fixture drift\n' >> "$GEMMA_TAIL_TEST/drifted-new-file/$GEMMA_TAIL_NEW"
gemma_tail_assert_rejected_without_changes drifted-new-file
cp -R "$GEMMA_TAIL_TEST/base" "$GEMMA_TAIL_TEST/conflicting-decoder"
printf '// intentional fixture replacement\n' > "$GEMMA_TAIL_TEST/conflicting-decoder/$GEMMA_TAIL_TEXT"
gemma_tail_assert_rejected_without_changes conflicting-decoder

cp -R "$GEMMA_TAIL_TEST/base" "$GEMMA_TAIL_TEST/stale-pin"
GEMMA_TAIL_TEST_REVISION=0000000000000000000000000000000000000000 \
  gemma_tail_assert_rejected_without_changes stale-pin

echo 'Gemma 3 compiled tail replay, layout compatibility, idempotence, partial-state rejection, and stale-pin checks passed.'
