#!/usr/bin/env bash

# Keep the unvalidated expert kernel out of ordinary builds. An isolated CUDA
# benchmark build opts into this source overlay; its runtime flag can then
# compare stock and candidate kernels using the same executable.
# Source optional-dependency-patch.sh before calling this helper.
model_runner_prepare_cuda_experts() {
  local expert_host="$1"
  local expert_package_root="$2"
  local expert_swift_checkout="$3"
  local expert_requested="${MODEL_RUNNER_ENABLE_CUDA_EXPERT_QMV_OVERLAY:-0}"
  local expert_state
  local expert_checkout="$expert_swift_checkout/Source/Cmlx/mlx"
  local expert_expected="7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247"
  local expert_actual

  [[ "$expert_host" == "Linux" ]] || return 0
  case "$expert_requested" in
    0) expert_state=off ;;
    1) expert_state=on ;;
    *)
      echo "MODEL_RUNNER_ENABLE_CUDA_EXPERT_QMV_OVERLAY must be unset, 0, or 1." >&2
      return 2
      ;;
  esac

  expert_actual="$(git -C "$expert_checkout" rev-parse HEAD)" || return 1
  if [[ "$expert_actual" != "$expert_expected" ]]; then
    echo "Refusing CUDA expert overlay for unexpected MLX revision: $expert_actual" >&2
    return 1
  fi

  model_runner_reconcile_optional_dependency_patch \
    "MLX CUDA expert QMV" \
    "$expert_checkout" \
    "$expert_package_root/Patches/mlx-cuda-expert-qmv-fp32.patch" \
    "$expert_state"
}
