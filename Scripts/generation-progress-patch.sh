#!/usr/bin/env bash
# Apply one atomic overlay without disturbing the other Evaluate.swift patches.
model_runner_prepare_generation_progress() {
  local progress_root="$1" progress_checkout="$2"
  local progress_revision progress_patch progress_forward=0 progress_reverse=0
  progress_revision="$(git -C "$progress_checkout" rev-parse HEAD)" || return 1
  if [[ "$progress_revision" != 14414441fa44f45eee35a61e9fa0bab577cf9734 ]]; then
    echo "Refusing generation progress patch for unexpected mlx-swift-lm revision: $progress_revision" >&2
    return 1
  fi
  progress_patch="$progress_root/Patches/mlx-swift-lm-generation-progress.patch"
  if [[ ! -r "$progress_patch" ]]; then
    echo "Generation progress patch is missing; checkout was not changed." >&2
    return 1
  fi
  if git -C "$progress_checkout" apply --check --whitespace=error-all "$progress_patch" >/dev/null 2>&1; then
    progress_forward=1
  fi
  if git -C "$progress_checkout" apply --reverse --check "$progress_patch" >/dev/null 2>&1; then
    progress_reverse=1
  fi
  if [[ "$progress_forward" == "$progress_reverse" ]]; then
    echo "Refusing conflicting or ambiguous generation progress patch; checkout was not changed." >&2
    return 1
  fi
  if [[ "$progress_reverse" == 1 ]]; then
    echo "mlx-swift-lm generation progress patch already applied."
    return 0
  fi
  # git apply validates the complete patch before writing; never use --reject.
  git -C "$progress_checkout" apply --whitespace=error-all "$progress_patch" || return 1
  echo "mlx-swift-lm generation progress patch applied."
}
