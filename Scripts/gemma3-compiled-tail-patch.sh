#!/usr/bin/env bash
# Opt-in fixed-shape Gemma 3 post-attention graph; pinned MLX-LM only.
model_runner_prepare_gemma3_compiled_tail() {
  local gemma_tail_host="$1"
  [[ "$gemma_tail_host" == "Darwin" ]] || return 0
  local gemma_tail_root="$2" gemma_tail_checkout="$3"
  local gemma_tail_actual gemma_tail_patch
  gemma_tail_actual="$(git -C "$gemma_tail_checkout" rev-parse HEAD)" || return 1
  if [[ "$gemma_tail_actual" != "14414441fa44f45eee35a61e9fa0bab577cf9734" ]]; then
    echo "Refusing Gemma 3 compiled tail patch for unexpected mlx-swift-lm revision: $gemma_tail_actual" >&2
    return 1
  fi
  gemma_tail_patch="$gemma_tail_root/Patches/mlx-swift-lm-gemma3-compiled-tail.patch"
  if [[ ! -r "$gemma_tail_patch" ]]; then
    echo "Gemma 3 compiled tail patch is missing; checkout was not changed." >&2
    return 1
  fi
  if git -C "$gemma_tail_checkout" apply --reverse --check "$gemma_tail_patch" >/dev/null 2>&1; then
    echo "mlx-swift-lm Gemma 3 compiled tail patch already applied."
  elif git -C "$gemma_tail_checkout" apply --check --whitespace=error-all "$gemma_tail_patch" >/dev/null 2>&1; then
    # git apply checks the entire patch before changing files. Never use --reject:
    # either both source files are applicable or a partial installation is refused.
    git -C "$gemma_tail_checkout" apply --whitespace=error-all "$gemma_tail_patch" || return 1
    echo "mlx-swift-lm Gemma 3 compiled tail patch applied (runtime remains opt-in)."
  else
    echo "Gemma 3 compiled tail patch conflicts with mlx-swift-lm; checkout was not changed." >&2
    return 1
  fi
}
