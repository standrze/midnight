#!/usr/bin/env bash
# Preserve added-token segmentation while avoiding unnecessary capture groups.
model_runner_prepare_added_token_regex() {
  local token_regex_root="$1" token_regex_checkout="$2"
  local token_regex_revision token_regex_patch token_regex_migration
  local token_regex_forward=0 token_regex_reverse=0 token_regex_upgrade=0
  token_regex_revision="$(git -C "$token_regex_checkout" rev-parse HEAD)" || return 1
  if [[ "$token_regex_revision" != 2fa33e1f5e7131a7fc64c28e6d161dcec0d24820 ]]; then
    echo "Refusing added-token regex patch for unexpected swift-transformers revision: $token_regex_revision" >&2
    return 1
  fi
  token_regex_patch="$token_regex_root/Patches/swift-transformers-added-token-regex.patch"
  token_regex_migration="$token_regex_root/Patches/swift-transformers-added-token-regex-empty-token-migration.patch"
  if [[ ! -r "$token_regex_patch" || ! -r "$token_regex_migration" ]]; then
    echo "Added-token regex patch or migration is missing; checkout was not changed." >&2
    return 1
  fi
  if git -C "$token_regex_checkout" apply --check --whitespace=error-all "$token_regex_patch" >/dev/null 2>&1; then
    token_regex_forward=1
  fi
  if git -C "$token_regex_checkout" apply --reverse --check "$token_regex_patch" >/dev/null 2>&1; then
    token_regex_reverse=1
  fi
  # The migration checks the complete previous fast-path block, including its
  # fallback. Never strip or reapply an unrecognized partially changed overlay.
  if git -C "$token_regex_checkout" apply --check --whitespace=error-all "$token_regex_migration" >/dev/null 2>&1; then
    token_regex_upgrade=1
  fi
  if [[ $((token_regex_forward + token_regex_reverse + token_regex_upgrade)) != 1 ]]; then
    echo "Refusing conflicting or ambiguous added-token regex patch; checkout was not changed." >&2
    return 1
  fi
  if [[ "$token_regex_reverse" == 1 ]]; then
    echo "swift-transformers added-token regex patch already applied."
    return 0
  fi
  if [[ "$token_regex_upgrade" == 1 ]]; then
    git -C "$token_regex_checkout" apply --whitespace=error-all "$token_regex_migration" || return 1
    echo "swift-transformers added-token regex patch migrated to preserve empty tokens."
    return 0
  fi
  git -C "$token_regex_checkout" apply --whitespace=error-all "$token_regex_patch" || return 1
  echo "swift-transformers added-token regex patch applied."
}
