#!/usr/bin/env bash
# The ordered stack preserves the stage3 16-value experiment and upgrades it
# to the generic-order eight-value variant. Preflight both source repositories
# completely before changing either checkout; unrelated edits remain intact.
model_runner_prepare_affine_q4_qmv_tail() (
  local tail_host="$1"
  [[ "$tail_host" == "Darwin" ]] || return 0
  local tail_root="$2" tail_swift="$3"
  local tail_repositories=("$tail_swift/Source/Cmlx/mlx" "$tail_swift")
  local tail_revisions=(
    "1f8e74e3f12f31365464a6867c6579f0e9b29d85"
    "72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798"
  )
  local tail_patch_names=(
    mlx-affine-q4-qmv-specialization.patch
    mlx-affine-q4-qmv-tail.patch
    mlx-affine-q4-qmv-tail-order.patch
    mlx-swift-affine-q4-qmv-jit.patch
    mlx-swift-affine-q4-qmv-tail-jit.patch
    mlx-swift-affine-q4-qmv-tail-order-jit.patch
  )
  local tail_states=()
  local tail_temp tail_index tail_step tail_patch_index tail_actual tail_path tail_patch tail_seen_pending tail_staged
  tail_temp="$(mktemp -d "${TMPDIR:-/tmp}/midnight-q4-tail-stack.XXXXXX")" || return 1
  trap 'rm -rf "$tail_temp"' EXIT
  for tail_index in 0 1; do
    tail_actual="$(git -C "${tail_repositories[$tail_index]}" rev-parse HEAD)" || return 1
    if [[ "$tail_actual" != "${tail_revisions[$tail_index]}" ]]; then
      echo "Refusing affine Q4 QMV tails for unexpected dependency revision: $tail_actual" >&2
      return 1
    fi
    for tail_step in 0 1 2; do
      tail_patch_index=$((tail_index * 3 + tail_step))
      tail_patch="$tail_root/Patches/${tail_patch_names[$tail_patch_index]}"
      [[ -s "$tail_patch" ]] || return 1
      while IFS= read -r tail_path; do
        tail_staged="$tail_temp/$tail_index/$tail_path"
        if [[ ! -f "$tail_staged" ]]; then
          mkdir -p "$(dirname "$tail_staged")" || return 1
          cp "${tail_repositories[$tail_index]}/$tail_path" "$tail_staged" || return 1
          # SwiftPM may mark sources 0444. Overlays share these targets;
          # copy each once and make only the private staging file writable.
          chmod u+w "$tail_staged" || return 1
        fi
      done < <(sed -n 's@^+++ b/@@p' "$tail_patch")
    done
    for tail_step in 2 1 0; do
      tail_patch_index=$((tail_index * 3 + tail_step))
      tail_patch="$tail_root/Patches/${tail_patch_names[$tail_patch_index]}"
      if git -C "$tail_temp/$tail_index" apply --reverse --check "$tail_patch" >/dev/null 2>&1; then
        git -C "$tail_temp/$tail_index" apply --reverse "$tail_patch" || return 1
        tail_states[$tail_patch_index]=applied
      else
        tail_states[$tail_patch_index]=pending
      fi
    done
    tail_seen_pending=0
    for tail_step in 0 1 2; do
      tail_patch_index=$((tail_index * 3 + tail_step))
      tail_patch="$tail_root/Patches/${tail_patch_names[$tail_patch_index]}"
      if [[ "${tail_states[$tail_patch_index]}" == pending ]]; then
        tail_seen_pending=1
      elif [[ "$tail_seen_pending" == 1 ]]; then
        echo "Inconsistent affine Q4 QMV overlay order; no checkouts were changed." >&2
        return 1
      fi
      if ! git -C "$tail_temp/$tail_index" apply --check --whitespace=error-all "$tail_patch" >/dev/null 2>&1; then
        echo "Affine Q4 QMV stack conflicts with dependency $tail_index; no checkouts were changed." >&2
        return 1
      fi
      git -C "$tail_temp/$tail_index" apply "$tail_patch" || return 1
    done
  done
  for tail_index in 0 1; do
    for tail_step in 0 1 2; do
      tail_patch_index=$((tail_index * 3 + tail_step))
      tail_patch="$tail_root/Patches/${tail_patch_names[$tail_patch_index]}"
      if [[ "${tail_states[$tail_patch_index]}" == pending ]]; then
        git -C "${tail_repositories[$tail_index]}" apply "$tail_patch" || return 1
        echo "Affine Q4 QMV overlay applied: ${tail_patch_names[$tail_patch_index]}"
      else
        echo "Affine Q4 QMV overlay already applied: ${tail_patch_names[$tail_patch_index]}"
      fi
    done
  done
)
