#!/usr/bin/env bash
# Apply one atomic overlay without disturbing the other Evaluate.swift patches.
model_runner_prepare_extra_eos_token_identity() {
  local eos_root="$1" eos_checkout="$2"
  local eos_revision eos_patch eos_forward=0 eos_reverse=0
  eos_revision="$(git -C "$eos_checkout" rev-parse HEAD)" || return 1
  if [[ "$eos_revision" != 14414441fa44f45eee35a61e9fa0bab577cf9734 ]]; then
    echo "Refusing extra EOS token identity patch for unexpected mlx-swift-lm revision: $eos_revision" >&2
    return 1
  fi
  eos_patch="$eos_root/Patches/mlx-swift-lm-extra-eos-token-identity.patch"
  if [[ ! -r "$eos_patch" ]]; then
    echo "Extra EOS token identity patch is missing; checkout was not changed." >&2
    return 1
  fi
  if git -C "$eos_checkout" apply --check --whitespace=error-all "$eos_patch" >/dev/null 2>&1; then
    eos_forward=1
  fi
  if git -C "$eos_checkout" apply --reverse --check "$eos_patch" >/dev/null 2>&1; then
    eos_reverse=1
  fi
  if [[ "$eos_forward" == "$eos_reverse" ]]; then
    echo "Refusing conflicting or ambiguous extra EOS token identity patch; checkout was not changed." >&2
    return 1
  fi
  if [[ "$eos_reverse" == 1 ]]; then
    echo "mlx-swift-lm extra EOS token identity patch already applied."
    return 0
  fi
  # git apply validates the complete patch before writing; never use --reject.
  git -C "$eos_checkout" apply --whitespace=error-all "$eos_patch" || return 1
  echo "mlx-swift-lm extra EOS token identity patch applied."
}
