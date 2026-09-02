#!/usr/bin/env bash

# Resolve the optional authorization for one already-running NVIDIA compute
# process. Presence is intentional: setting both variables to empty strings is
# an invalid opt-in, not the same thing as leaving both variables unset.
model_runner_smoke_resolve_compute_authorization() {
  local pid_is_set=0
  local hash_is_set=0

  if [[ "${MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID+is-set}" == "is-set" ]]; then
    pid_is_set=1
  fi
  if [[ "${MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_CMDLINE_SHA256+is-set}" == "is-set" ]]; then
    hash_is_set=1
  fi

  if [[ "$pid_is_set" != "$hash_is_set" ]]; then
    echo "MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID and MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_CMDLINE_SHA256 must be provided together or both left unset." >&2
    return 2
  fi

  if [[ "$pid_is_set" == "0" ]]; then
    MODEL_RUNNER_SMOKE_COMPUTE_OVERLAP_MODE=deny
    export MODEL_RUNNER_SMOKE_COMPUTE_OVERLAP_MODE
    return 0
  fi

  if [[ ! "$MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID" =~ ^[1-9][0-9]*$ ]]; then
    echo "MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID must be a positive decimal PID." >&2
    return 2
  fi
  if [[ ! "$MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_CMDLINE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_CMDLINE_SHA256 must be exactly 64 lowercase hexadecimal characters." >&2
    return 2
  fi

  MODEL_RUNNER_SMOKE_COMPUTE_OVERLAP_MODE=allow-one
  export MODEL_RUNNER_SMOKE_COMPUTE_OVERLAP_MODE
}

model_runner_smoke_validate_allowed_compute_identity() {
  local allowed_pid="$1"
  local expected_hash="$2"
  local proc_root="${3:-/proc}"
  local cmdline_path="$proc_root/$allowed_pid/cmdline"
  local actual_hash
  local empty_hash=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

  # procfs cmdline files normally report a stat size of zero even when reads
  # return data, so -s is not a valid nonempty check here.
  if [[ ! -d "$proc_root/$allowed_pid" || ! -r "$cmdline_path" ]]; then
    echo "Protected compute PID $allowed_pid has no readable $cmdline_path." >&2
    return 1
  fi
  if ! actual_hash="$(sha256sum "$cmdline_path" 2>/dev/null | awk '{print $1}')"; then
    echo "Could not hash the command line for protected compute PID $allowed_pid." >&2
    return 1
  fi
  if [[ "$actual_hash" == "$empty_hash" ]]; then
    echo "Protected compute PID $allowed_pid has an empty command line; refusing authorization." >&2
    return 1
  fi
  if [[ ! "$actual_hash" =~ ^[0-9a-f]{64}$ || "$actual_hash" != "$expected_hash" ]]; then
    echo "Protected compute PID $allowed_pid command-line SHA-256 does not match the explicit authorization." >&2
    return 1
  fi
}

model_runner_smoke_assert_idle_host() {
  local proc_root="${1:-/proc}"
  local runner_pids
  local pgrep_status
  local compute_processes
  local compute_row
  local compute_pid
  local compute_row_count=0
  local gpu_count
  local gpu_free_mib
  local required_gpu_free_mib
  local listeners
  local overlap_mode="${MODEL_RUNNER_SMOKE_COMPUTE_OVERLAP_MODE:-deny}"
  local allowed_pid="${MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID:-}"
  local allowed_hash="${MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_CMDLINE_SHA256:-}"

  case "$overlap_mode" in
    deny)
      ;;
    allow-one)
      model_runner_smoke_validate_allowed_compute_identity \
        "$allowed_pid" "$allowed_hash" "$proc_root" || return 1
      ;;
    *)
      echo "Internal error: unresolved compute-overlap policy '$overlap_mode'." >&2
      return 1
      ;;
  esac

  if runner_pids="$(pgrep -x midnight)"; then
    while IFS= read -r compute_pid; do
      compute_pid="$(printf '%s\n' "$compute_pid" | awk '{$1=$1};1')"
      if [[ ! "$compute_pid" =~ ^[1-9][0-9]*$ ]] \
        || [[ "$overlap_mode" != "allow-one" ]] \
        || [[ "$compute_pid" != "$allowed_pid" ]]
      then
        echo "An existing Midnight Runner process is active; refusing to overlap: $runner_pids" >&2
        return 1
      fi
    done <<< "$runner_pids"
  else
    pgrep_status=$?
    if [[ "$pgrep_status" != "1" ]]; then
      echo "Could not inspect existing Midnight Runner processes (pgrep status $pgrep_status)." >&2
      return 1
    fi
  fi

  if ! gpu_count="$(nvidia-smi --list-gpus | sed '/^[[:space:]]*$/d' | awk 'END {print NR}')"; then
    echo "Could not enumerate visible NVIDIA GPUs." >&2
    return 1
  fi
  if [[ "$gpu_count" != "1" ]]; then
    echo "The smoke harness requires exactly one visible NVIDIA GPU; found $gpu_count." >&2
    return 1
  fi

  if ! compute_processes="$(
    nvidia-smi \
      --query-compute-apps=pid,process_name,used_gpu_memory \
      --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'
  )"; then
    echo "Could not inspect existing NVIDIA compute processes." >&2
    return 1
  fi

  if [[ "$overlap_mode" == "deny" ]]; then
    if [[ -n "$compute_processes" ]]; then
      echo "Existing NVIDIA compute processes were found; refusing to overlap:" >&2
      printf '%s\n' "$compute_processes" >&2
      return 1
    fi
  else
    if [[ -z "$compute_processes" ]]; then
      echo "Protected compute PID $allowed_pid is not present in NVIDIA's compute-process table." >&2
      return 1
    fi
    while IFS= read -r compute_row; do
      [[ -n "$compute_row" ]] || continue
      compute_pid="$(printf '%s\n' "${compute_row%%,*}" | awk '{$1=$1};1')"
      if [[ ! "$compute_pid" =~ ^[1-9][0-9]*$ ]]; then
        echo "Could not parse NVIDIA compute-process row: $compute_row" >&2
        return 1
      fi
      if [[ "$compute_pid" != "$allowed_pid" ]]; then
        echo "NVIDIA compute PID $compute_pid is not the explicitly protected PID $allowed_pid; refusing to overlap." >&2
        return 1
      fi
      compute_row_count=$((compute_row_count + 1))
    done <<< "$compute_processes"
    if [[ "$compute_row_count" == "0" ]]; then
      echo "No NVIDIA compute row matched protected PID $allowed_pid." >&2
      return 1
    fi
  fi

  if ! gpu_free_mib="$(
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
      | sed '/^[[:space:]]*$/d'
  )"; then
    echo "Could not inspect free NVIDIA GPU memory." >&2
    return 1
  fi
  gpu_free_mib="$(printf '%s\n' "$gpu_free_mib" | awk '{$1=$1};1')"
  if [[ ! "$gpu_free_mib" =~ ^[0-9]+$ ]]; then
    echo "Expected one integer NVIDIA memory.free value; received '$gpu_free_mib'." >&2
    return 1
  fi
  required_gpu_free_mib=$((10#$MODEL_RUNNER_SMOKE_MLX_MEMORY_GIB * 1024 + 2048))
  if (( 10#$gpu_free_mib < required_gpu_free_mib )); then
    echo "Insufficient free GPU memory: ${gpu_free_mib} MiB available; ${required_gpu_free_mib} MiB required for the MLX cap plus a 2 GiB reserve." >&2
    return 1
  fi

  if ! listeners="$(ss -H -ltn)"; then
    echo "Could not inspect listening TCP ports with ss." >&2
    return 1
  fi
  if printf '%s\n' "$listeners" | awk -v port="$MODEL_RUNNER_SMOKE_PORT" \
    '$4 ~ (":" port "$") {found=1} END {exit(found ? 0 : 1)}'
  then
    echo "TCP port $MODEL_RUNNER_SMOKE_PORT is already listening." >&2
    return 1
  fi
}
