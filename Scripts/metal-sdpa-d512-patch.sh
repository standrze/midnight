#!/usr/bin/env bash
# Pinned Darwin SDPA experiments. D256 extends D512's host dispatch hunk.
# Preflight the full ordered patch stack on copies before touching either repo.
model_runner_prepare_metal_sdpa_d512() (
  local sdpa_host="$1"
  [[ "$sdpa_host" == "Darwin" ]] || return 0
  local sdpa_root="$2"
  local sdpa_swift_checkout="$3"
  local sdpa_repositories=("$sdpa_swift_checkout/Source/Cmlx/mlx" "$sdpa_swift_checkout")
  local sdpa_revisions=(
    "1f8e74e3f12f31365464a6867c6579f0e9b29d85"
    "72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798"
  )
  local sdpa_patch_names=(
    mlx-metal-sdpa-d512-decode.patch
    mlx-metal-sdpa-d256-mask-bounds.patch
    mlx-swift-metal-sdpa-d512-generated.patch
    mlx-swift-metal-sdpa-d256-mask-bounds-generated.patch
  )
  local sdpa_states=()
  local sdpa_temp sdpa_index sdpa_step sdpa_patch_index sdpa_actual sdpa_target sdpa_patch sdpa_staged
  sdpa_temp="$(mktemp -d "${TMPDIR:-/tmp}/midnight-sdpa-prepare.XXXXXX")" || return 1
  trap 'rm -rf "$sdpa_temp"' EXIT
  for sdpa_index in 0 1; do
    sdpa_actual="$(git -C "${sdpa_repositories[$sdpa_index]}" rev-parse HEAD)" || return 1
    if [[ "$sdpa_actual" != "${sdpa_revisions[$sdpa_index]}" ]]; then
      echo "Refusing SDPA patch stack for unexpected revision: $sdpa_actual" >&2
      return 1
    fi
    mkdir -p "$sdpa_temp/$sdpa_index" || return 1
    # Copy only the stack's target files; other pending dependency changes stay untouched.
    for sdpa_step in 0 1; do
      sdpa_patch_index=$((sdpa_index * 2 + sdpa_step))
      sdpa_patch="$sdpa_root/Patches/${sdpa_patch_names[$sdpa_patch_index]}"
      [[ -s "$sdpa_patch" ]] || return 1
      while IFS= read -r sdpa_target; do
        sdpa_staged="$sdpa_temp/$sdpa_index/$sdpa_target"
        if [[ ! -f "$sdpa_staged" ]]; then
          mkdir -p "$(dirname "$sdpa_staged")" || return 1
          cp "${sdpa_repositories[$sdpa_index]}/$sdpa_target" "$sdpa_staged" || return 1
          # SwiftPM sources may be 0444, and overlays can share a target.
          # Only private copies need to be writable for preflight/replay.
          chmod u+w "$sdpa_staged" || return 1
        fi
      done < <(sed -n 's@^+++ b/@@p' "$sdpa_patch")
    done
    # Peel applied overlays in reverse order. A missing overlay is checked by
    # replaying the full stack below, after earlier prerequisites are present.
    for sdpa_step in 1 0; do
      sdpa_patch_index=$((sdpa_index * 2 + sdpa_step))
      sdpa_patch="$sdpa_root/Patches/${sdpa_patch_names[$sdpa_patch_index]}"
      if git -C "$sdpa_temp/$sdpa_index" apply --reverse --check "$sdpa_patch" >/dev/null 2>&1; then
        git -C "$sdpa_temp/$sdpa_index" apply --reverse "$sdpa_patch" || return 1
        sdpa_states[$sdpa_patch_index]=applied
      else
        sdpa_states[$sdpa_patch_index]=pending
      fi
    done
    if [[ "${sdpa_states[$((sdpa_index * 2))]}" == pending &&
      "${sdpa_states[$((sdpa_index * 2 + 1))]}" == applied ]]; then
      echo "Inconsistent SDPA patch order; no checkouts were changed." >&2
      return 1
    fi
    for sdpa_step in 0 1; do
      sdpa_patch_index=$((sdpa_index * 2 + sdpa_step))
      sdpa_patch="$sdpa_root/Patches/${sdpa_patch_names[$sdpa_patch_index]}"
      if ! git -C "$sdpa_temp/$sdpa_index" apply --check --whitespace=error-all "$sdpa_patch" >/dev/null 2>&1; then
        echo "SDPA patch stack conflicts with ${sdpa_repositories[$sdpa_index]}; no checkouts were changed." >&2
        return 1
      fi
      git -C "$sdpa_temp/$sdpa_index" apply "$sdpa_patch" || return 1
    done
  done
  for sdpa_index in 0 1; do
    for sdpa_step in 0 1; do
      sdpa_patch_index=$((sdpa_index * 2 + sdpa_step))
      sdpa_patch="$sdpa_root/Patches/${sdpa_patch_names[$sdpa_patch_index]}"
      if [[ "${sdpa_states[$sdpa_patch_index]}" == pending ]]; then
        git -C "${sdpa_repositories[$sdpa_index]}" apply "$sdpa_patch" || return 1
        echo "SDPA patch applied: ${sdpa_patch_names[$sdpa_patch_index]}"
      else
        echo "SDPA patch already applied: ${sdpa_patch_names[$sdpa_patch_index]}"
      fi
    done
  done
)
