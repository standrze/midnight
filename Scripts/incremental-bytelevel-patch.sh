#!/usr/bin/env bash
# Verify the complete earlier patch after accounting for the scoped follow-up.
model_runner_bytelevel_present_with_gemma_cleanup() (
  local patch_root="$1" patch_checkout="$2" replay_dir
  [[ "$(git -C "$patch_checkout" rev-parse HEAD)" == 2fa33e1f5e7131a7fc64c28e6d161dcec0d24820 ]] || return 1
  replay_dir="$(mktemp -d "${TMPDIR:-/tmp}/midnight-bytelevel-replay.XXXXXX")" || return 1
  trap 'rm -rf "$replay_dir"' EXIT
  mkdir -p "$replay_dir/Sources/Tokenizers" "$replay_dir/Tests/TokenizersTests" || return 1
  local replay_file
  for replay_file in Sources/Tokenizers/Decoder.swift Sources/Tokenizers/Tokenizer.swift \
    Tests/TokenizersTests/IncrementalByteLevelDecoderTests.swift; do
    cp "$patch_checkout/$replay_file" "$replay_dir/$replay_file" || return 1
  done
  # Reverse only in the disposable copy. Check every hunk of both patches;
  # a partial decoder or a modified regression test must still fail closed.
  git -C "$replay_dir" apply --reverse --check \
    "$patch_root/Patches/swift-transformers-gemma-cleanup-default.patch" >/dev/null 2>&1 || return 1
  git -C "$replay_dir" apply --reverse \
    "$patch_root/Patches/swift-transformers-gemma-cleanup-default.patch" >/dev/null 2>&1 || return 1
  git -C "$replay_dir" apply --reverse --check \
    "$patch_root/Patches/swift-transformers-incremental-bytelevel-decoder.patch" >/dev/null 2>&1
)
