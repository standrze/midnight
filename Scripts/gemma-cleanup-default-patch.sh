#!/usr/bin/env bash
# Apply the scoped Gemma decode default without disturbing other overlays.
model_runner_prepare_gemma_cleanup_default() {
  local cleanup_root="$1" cleanup_checkout="$2"
  local cleanup_revision cleanup_patch cleanup_forward=0 cleanup_reverse=0
  cleanup_revision="$(git -C "$cleanup_checkout" rev-parse HEAD)" || return 1
  if [[ "$cleanup_revision" != 2fa33e1f5e7131a7fc64c28e6d161dcec0d24820 ]]; then
    echo "Refusing Gemma cleanup default patch for unexpected swift-transformers revision: $cleanup_revision" >&2
    return 1
  fi
  cleanup_patch="$cleanup_root/Patches/swift-transformers-gemma-cleanup-default.patch"
  if [[ ! -r "$cleanup_patch" ]]; then
    echo "Gemma cleanup default patch is missing; checkout was not changed." >&2
    return 1
  fi
  if git -C "$cleanup_checkout" apply --check --whitespace=error-all "$cleanup_patch" >/dev/null 2>&1; then
    cleanup_forward=1
  fi
  if git -C "$cleanup_checkout" apply --reverse --check "$cleanup_patch" >/dev/null 2>&1; then
    cleanup_reverse=1
  fi
  if [[ "$cleanup_forward" == "$cleanup_reverse" ]]; then
    echo "Refusing conflicting or ambiguous Gemma cleanup default patch; checkout was not changed." >&2
    return 1
  fi
  if [[ "$cleanup_reverse" == 1 ]]; then
    echo "swift-transformers Gemma cleanup default patch already applied."
    return 0
  fi
  git -C "$cleanup_checkout" apply --whitespace=error-all "$cleanup_patch" || return 1
  echo "swift-transformers Gemma cleanup default patch applied."
}
