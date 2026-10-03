#!/usr/bin/env bash
# Own the overlapping MTP scheduling/diagnostic/adaptive overlay stack.
model_runner_prepare_mtp_adaptive_drafts() {
  local mtp_root="$1"
  local mtp_checkout="$2"
  python3 "$mtp_root/Scripts/mtp-adaptive-draft-patches.py" --checkout "$mtp_checkout"
}
