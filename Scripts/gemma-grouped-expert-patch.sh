#!/usr/bin/env bash
# Narrow opt-in Gemma A4B Q4/G64 verification kernel; pinned MLX-LM only.
model_runner_prepare_gemma_grouped_expert() {
  local grouped_host="$1"
  [[ "$grouped_host" == "Darwin" ]] || return 0
  local grouped_root="$2" grouped_checkout="$3"
  local grouped_actual grouped_patch
  grouped_actual="$(git -C "$grouped_checkout" rev-parse HEAD)" || return 1
  if [[ "$grouped_actual" != "14414441fa44f45eee35a61e9fa0bab577cf9734" ]]; then
    echo "Refusing grouped expert patch for unexpected mlx-swift-lm revision: $grouped_actual" >&2
    return 1
  fi
  grouped_patch="$grouped_root/Patches/mlx-swift-lm-gemma-grouped-expert-verification.patch"
  if git -C "$grouped_checkout" apply --reverse --check "$grouped_patch" >/dev/null 2>&1; then
    echo "mlx-swift-lm grouped expert verification patch already applied."
  elif git -C "$grouped_checkout" apply --check --whitespace=error-all "$grouped_patch" >/dev/null 2>&1; then
    git -C "$grouped_checkout" apply "$grouped_patch" || return 1
    echo "mlx-swift-lm grouped expert verification patch applied."
  else
    echo "Grouped expert verification patch conflicts with mlx-swift-lm; checkout was not changed." >&2
    return 1
  fi
}
