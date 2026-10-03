#!/usr/bin/env bash
set -euo pipefail
GROUPED_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GROUPED_LM="$GROUPED_ROOT/.build/checkouts/mlx-swift-lm"
GROUPED_REV=14414441fa44f45eee35a61e9fa0bab577cf9734
GROUPED_PATCH="$GROUPED_ROOT/Patches/mlx-swift-lm-gemma-grouped-expert-verification.patch"
GROUPED_TEST="$(mktemp -d "${TMPDIR:-/tmp}/midnight-grouped-expert.XXXXXX")"
trap 'rm -rf "$GROUPED_TEST"' EXIT
mkdir -p "$GROUPED_TEST/pinned" "$GROUPED_TEST/current"
git -C "$GROUPED_LM" archive "$GROUPED_REV" Libraries/MLXLMCommon/SwitchLayers.swift \
  | tar -xf - -C "$GROUPED_TEST/pinned"
cp -R "$GROUPED_TEST/pinned/." "$GROUPED_TEST/current/"
# Existing gate/up views and calibration constructor overlays must still reverse
# cleanly after this overlay. Their target hunks are independent.
for patch in mlx-swift-lm-gate-up-slices.patch mlx-swift-lm-q4-affine-scale-search.patch; do
  git -C "$GROUPED_TEST/current" apply --include=Libraries/MLXLMCommon/SwitchLayers.swift \
    "$GROUPED_ROOT/Patches/$patch"
done
for fixture in pinned current; do
  git -C "$GROUPED_TEST/$fixture" apply --check --whitespace=error-all "$GROUPED_PATCH"
  git -C "$GROUPED_TEST/$fixture" apply "$GROUPED_PATCH"
  git -C "$GROUPED_TEST/$fixture" apply --reverse --check "$GROUPED_PATCH"
done
for patch in mlx-swift-lm-gate-up-slices.patch mlx-swift-lm-q4-affine-scale-search.patch; do
  git -C "$GROUPED_TEST/current" apply --reverse --check \
    --include=Libraries/MLXLMCommon/SwitchLayers.swift "$GROUPED_ROOT/Patches/$patch"
done

source "$GROUPED_ROOT/Scripts/gemma-grouped-expert-patch.sh"
# The fixture deliberately contains no Git database; mock only the exact pin lookup.
# All patch checks and mutations remain real git commands on copied source.
git() {
  if [[ "$#" -eq 4 && "$1" == -C && "$3" == rev-parse && "$4" == HEAD ]]; then
    printf '%s\n' "$GROUPED_REV"
  else
    command git "$@"
  fi
}
model_runner_prepare_gemma_grouped_expert Linux /does-not-exist /does-not-exist
model_runner_prepare_gemma_grouped_expert Darwin "$GROUPED_ROOT" "$GROUPED_TEST/current"
git -C "$GROUPED_TEST/current" apply --reverse "$GROUPED_PATCH"
model_runner_prepare_gemma_grouped_expert Darwin "$GROUPED_ROOT" "$GROUPED_TEST/current"
model_runner_prepare_gemma_grouped_expert Darwin "$GROUPED_ROOT" "$GROUPED_TEST/current"

# Partial new-file installation is source drift: refuse without changing either file.
cp -R "$GROUPED_TEST/current" "$GROUPED_TEST/corrupt"
printf '\n// intentional fixture drift\n' >> \
  "$GROUPED_TEST/corrupt/Libraries/MLXLMCommon/GemmaGroupedExpertVerification.swift"
cp -R "$GROUPED_TEST/corrupt" "$GROUPED_TEST/corrupt-before"
if model_runner_prepare_gemma_grouped_expert Darwin "$GROUPED_ROOT" "$GROUPED_TEST/corrupt"; then
  echo 'Expected a conflicting grouped expert overlay to fail.' >&2
  exit 1
fi
diff -qr "$GROUPED_TEST/corrupt-before" "$GROUPED_TEST/corrupt" >/dev/null

# The original pinned source round-trips exactly.
git -C "$GROUPED_TEST/pinned" apply --reverse "$GROUPED_PATCH"
git -C "$GROUPED_LM" show "$GROUPED_REV:Libraries/MLXLMCommon/SwitchLayers.swift" \
  > "$GROUPED_TEST/original.swift"
cmp "$GROUPED_TEST/original.swift" "$GROUPED_TEST/pinned/Libraries/MLXLMCommon/SwitchLayers.swift"
test ! -e "$GROUPED_TEST/pinned/Libraries/MLXLMCommon/GemmaGroupedExpertVerification.swift"
echo 'Grouped expert patch round trips, overlay compatibility, drift rejection, and idempotence passed.'
