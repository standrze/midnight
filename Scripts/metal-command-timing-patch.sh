#!/usr/bin/env bash
# Explicit integration hook; sourcing does not apply anything.
model_runner_prepare_metal_command_timing() (
  local timing_host="$1"
  [[ "$timing_host" == "Darwin" ]] || return 0
  local timing_root="$2"
  local timing_mlx_checkout="$3"
  python3 "$timing_root/Scripts/metal-command-timing-patch.py" --checkout "$timing_mlx_checkout"
)
