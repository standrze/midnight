#!/usr/bin/env bash
set -euo pipefail
EOS_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EOS_LM="$EOS_ROOT/.build/checkouts/mlx-swift-lm"
EOS_REV=14414441fa44f45eee35a61e9fa0bab577cf9734
EOS_FILE=Libraries/MLXLMCommon/Evaluate.swift
EOS_PATCH="$EOS_ROOT/Patches/mlx-swift-lm-extra-eos-token-identity.patch"
EOS_TEST="$(mktemp -d "${TMPDIR:-/tmp}/midnight-extra-eos.XXXXXX")"
trap 'rm -rf "$EOS_TEST"' EXIT
mkdir -p "$EOS_TEST/current/Libraries/MLXLMCommon" "$EOS_TEST/pinned"
cp "$EOS_LM/$EOS_FILE" "$EOS_TEST/current/$EOS_FILE"
if git -C "$EOS_TEST/current" apply --reverse --check "$EOS_PATCH" >/dev/null 2>&1; then
  git -C "$EOS_TEST/current" apply --reverse "$EOS_PATCH"
fi
printf 'retain unrelated data\n' > "$EOS_TEST/current/unrelated.txt"
git -C "$EOS_LM" archive "$EOS_REV" "$EOS_FILE" | tar -xf - -C "$EOS_TEST/pinned"
cp -R "$EOS_TEST/current" "$EOS_TEST/current-original"
cp -R "$EOS_TEST/pinned" "$EOS_TEST/pinned-original"
source "$EOS_ROOT/Scripts/extra-eos-token-identity-patch.sh"

# Only revision lookup is mocked; every forward/reverse application is real git.
git() {
  if [[ "$#" -eq 4 && "$1" == -C && "$3" == rev-parse && "$4" == HEAD && "$2" == "$EOS_TEST/"* ]]; then
    printf '%s\n' "${EOS_TEST_REVISION:-$EOS_REV}"
  else
    command git "$@"
  fi
}

for fixture in current pinned; do
  model_runner_prepare_extra_eos_token_identity "$EOS_ROOT" "$EOS_TEST/$fixture"
  git -C "$EOS_TEST/$fixture" apply --reverse --check "$EOS_PATCH"
  cp -R "$EOS_TEST/$fixture" "$EOS_TEST/$fixture-applied"
  model_runner_prepare_extra_eos_token_identity "$EOS_ROOT" "$EOS_TEST/$fixture"
  diff -qr "$EOS_TEST/$fixture-applied" "$EOS_TEST/$fixture" >/dev/null
  git -C "$EOS_TEST/$fixture" apply --reverse "$EOS_PATCH"
  diff -qr "$EOS_TEST/$fixture-original" "$EOS_TEST/$fixture" >/dev/null
done

eos_assert_rejected() {
  local fixture="$1"
  cp -R "$EOS_TEST/$fixture" "$EOS_TEST/$fixture-before"
  if model_runner_prepare_extra_eos_token_identity "$EOS_ROOT" "$EOS_TEST/$fixture"; then
    echo "Expected extra EOS fixture $fixture to be rejected." >&2
    exit 1
  fi
  diff -qr "$EOS_TEST/$fixture-before" "$EOS_TEST/$fixture" >/dev/null
}

cp -R "$EOS_TEST/current-original" "$EOS_TEST/partial"
python3 - "$EOS_TEST/partial/$EOS_FILE" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
p.write_text(p.read_text().replace('private func buildStopTokenIds(', 'func buildStopTokenIds(', 1))
PY
eos_assert_rejected partial
cp -R "$EOS_TEST/current-original" "$EOS_TEST/conflict"
printf '// unrelated incompatible source\n' > "$EOS_TEST/conflict/$EOS_FILE"
eos_assert_rejected conflict
cp -R "$EOS_TEST/current-original" "$EOS_TEST/stale-pin"
EOS_TEST_REVISION=0000000000000000000000000000000000000000 eos_assert_rejected stale-pin

echo 'Extra EOS overlay passes pinned/current replay, idempotence, reversal, partial/conflict rejection and revision checks.'
