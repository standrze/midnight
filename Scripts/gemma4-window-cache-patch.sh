#!/usr/bin/env bash
# This helper owns the complete overlapping Gemma 4 model overlay stack.
# Replace its individual apply_dependency_patch calls with this one call.
model_runner_prepare_gemma4_window_cache() {
  local gemma_root="$1"
  local gemma_checkout="$2"
  python3 "$gemma_root/Scripts/gemma4-window-cache-patches.py" --checkout "$gemma_checkout"
}
