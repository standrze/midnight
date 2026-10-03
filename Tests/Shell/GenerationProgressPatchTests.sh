#!/usr/bin/env bash
set -euo pipefail
PROGRESS_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROGRESS_LM="$PROGRESS_ROOT/.build/checkouts/mlx-swift-lm"
PROGRESS_REV=14414441fa44f45eee35a61e9fa0bab577cf9734
PROGRESS_FILE=Libraries/MLXLMCommon/Evaluate.swift
PROGRESS_PATCH="$PROGRESS_ROOT/Patches/mlx-swift-lm-generation-progress.patch"
PROGRESS_TEST="$(mktemp -d "${TMPDIR:-/tmp}/midnight-generation-progress.XXXXXX")"
trap 'rm -rf "$PROGRESS_TEST"' EXIT
mkdir -p "$PROGRESS_TEST/current/Libraries/MLXLMCommon" "$PROGRESS_TEST/pinned"
cp "$PROGRESS_LM/$PROGRESS_FILE" "$PROGRESS_TEST/current/$PROGRESS_FILE"
if git -C "$PROGRESS_TEST/current" apply --reverse --check "$PROGRESS_PATCH" >/dev/null 2>&1; then
  git -C "$PROGRESS_TEST/current" apply --reverse "$PROGRESS_PATCH"
fi
printf 'retain unrelated data\n' > "$PROGRESS_TEST/current/unrelated.txt"
git -C "$PROGRESS_LM" archive "$PROGRESS_REV" "$PROGRESS_FILE" | tar -xf - -C "$PROGRESS_TEST/pinned"
cp -R "$PROGRESS_TEST/current" "$PROGRESS_TEST/current-original"
cp -R "$PROGRESS_TEST/pinned" "$PROGRESS_TEST/pinned-original"
source "$PROGRESS_ROOT/Scripts/generation-progress-patch.sh"

# Only revision lookup is mocked; every forward/reverse application is real git.
git() {
  if [[ "$#" -eq 4 && "$1" == -C && "$3" == rev-parse && "$4" == HEAD && "$2" == "$PROGRESS_TEST/"* ]]; then
    printf '%s\n' "${PROGRESS_TEST_REVISION:-$PROGRESS_REV}"
  else
    command git "$@"
  fi
}

for fixture in current pinned; do
  model_runner_prepare_generation_progress "$PROGRESS_ROOT" "$PROGRESS_TEST/$fixture"
  git -C "$PROGRESS_TEST/$fixture" apply --reverse --check "$PROGRESS_PATCH"
  cp -R "$PROGRESS_TEST/$fixture" "$PROGRESS_TEST/$fixture-applied"
  model_runner_prepare_generation_progress "$PROGRESS_ROOT" "$PROGRESS_TEST/$fixture"
  diff -qr "$PROGRESS_TEST/$fixture-applied" "$PROGRESS_TEST/$fixture" >/dev/null
  git -C "$PROGRESS_TEST/$fixture" apply --reverse "$PROGRESS_PATCH"
  diff -qr "$PROGRESS_TEST/$fixture-original" "$PROGRESS_TEST/$fixture" >/dev/null
done

progress_assert_rejected() {
  local fixture="$1"
  cp -R "$PROGRESS_TEST/$fixture" "$PROGRESS_TEST/$fixture-before"
  if model_runner_prepare_generation_progress "$PROGRESS_ROOT" "$PROGRESS_TEST/$fixture"; then
    echo "Expected generation progress fixture $fixture to be rejected." >&2
    exit 1
  fi
  diff -qr "$PROGRESS_TEST/$fixture-before" "$PROGRESS_TEST/$fixture" >/dev/null
}

cp -R "$PROGRESS_TEST/current-original" "$PROGRESS_TEST/partial"
python3 - "$PROGRESS_TEST/partial/$PROGRESS_FILE" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
p.write_text(p.read_text().replace('import MLXNN\n', 'import MLXNN\n\n/// Passive request-local observation of emitted token counts. This does not alter sampling.\npublic enum GenerationTokenObserver {\n    @TaskLocal public static var onToken: (@Sendable (Int) -> Void)?\n}\n', 1))
PY
progress_assert_rejected partial
cp -R "$PROGRESS_TEST/current-original" "$PROGRESS_TEST/conflict"
printf '// unrelated incompatible source\n' > "$PROGRESS_TEST/conflict/$PROGRESS_FILE"
progress_assert_rejected conflict
cp -R "$PROGRESS_TEST/current-original" "$PROGRESS_TEST/stale-pin"
PROGRESS_TEST_REVISION=0000000000000000000000000000000000000000 progress_assert_rejected stale-pin

echo 'Generation progress overlay passes pinned/current replay, idempotence, reversal, partial/conflict rejection and revision checks.'
