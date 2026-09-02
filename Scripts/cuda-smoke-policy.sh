#!/usr/bin/env bash

model_runner_smoke_require_integer() {
  local name="$1"
  local value="$2"
  local minimum="$3"
  local maximum="$4"

  if [[ ! "$value" =~ ^[0-9]+$ ]] \
    || (( 10#$value < minimum || 10#$value > maximum ))
  then
    echo "$name must be an integer in $minimum...$maximum; received '$value'." >&2
    return 2
  fi
}

model_runner_smoke_resolve_policy() {
  local profile="$1"
  local default_mlx_memory
  local default_mlx_cache
  local default_host_high
  local default_host_max
  local default_runtime
  local default_health_timeout
  local default_http_timeout
  local default_cpu_quota

  case "$profile" in
    qwen)
      default_mlx_memory=8
      default_mlx_cache=64
      default_host_high=12
      default_host_max=16
      default_runtime=600
      default_health_timeout=300
      default_http_timeout=180
      default_cpu_quota=200
      ;;
    gemma)
      default_mlx_memory=18
      default_mlx_cache=128
      default_host_high=24
      default_host_max=32
      default_runtime=1500
      default_health_timeout=900
      default_http_timeout=300
      default_cpu_quota=400
      ;;
    *)
      echo "Unknown smoke profile '$profile'; choose qwen or gemma." >&2
      return 2
      ;;
  esac

  MODEL_RUNNER_SMOKE_PROFILE="$profile"
  MODEL_RUNNER_SMOKE_MLX_MEMORY_GIB="${MODEL_RUNNER_SMOKE_MLX_MEMORY_GIB:-$default_mlx_memory}"
  MODEL_RUNNER_SMOKE_MLX_CACHE_MIB="${MODEL_RUNNER_SMOKE_MLX_CACHE_MIB:-$default_mlx_cache}"
  MODEL_RUNNER_SMOKE_HOST_MEMORY_HIGH_GIB="${MODEL_RUNNER_SMOKE_HOST_MEMORY_HIGH_GIB:-$default_host_high}"
  MODEL_RUNNER_SMOKE_HOST_MEMORY_MAX_GIB="${MODEL_RUNNER_SMOKE_HOST_MEMORY_MAX_GIB:-$default_host_max}"
  MODEL_RUNNER_SMOKE_SWAP_MAX_GIB="${MODEL_RUNNER_SMOKE_SWAP_MAX_GIB:-1}"
  MODEL_RUNNER_SMOKE_RUNTIME_SECONDS="${MODEL_RUNNER_SMOKE_RUNTIME_SECONDS:-$default_runtime}"
  MODEL_RUNNER_SMOKE_HEALTH_TIMEOUT_SECONDS="${MODEL_RUNNER_SMOKE_HEALTH_TIMEOUT_SECONDS:-$default_health_timeout}"
  MODEL_RUNNER_SMOKE_HTTP_TIMEOUT_SECONDS="${MODEL_RUNNER_SMOKE_HTTP_TIMEOUT_SECONDS:-$default_http_timeout}"
  MODEL_RUNNER_SMOKE_MAX_TOKENS="${MODEL_RUNNER_SMOKE_MAX_TOKENS:-96}"
  MODEL_RUNNER_SMOKE_PORT="${MODEL_RUNNER_SMOKE_PORT:-8080}"
  MODEL_RUNNER_SMOKE_CPU_QUOTA_PERCENT="${MODEL_RUNNER_SMOKE_CPU_QUOTA_PERCENT:-$default_cpu_quota}"

  # These ceilings are part of the safety contract, not profile suggestions.
  model_runner_smoke_require_integer MODEL_RUNNER_SMOKE_MLX_MEMORY_GIB \
    "$MODEL_RUNNER_SMOKE_MLX_MEMORY_GIB" 1 20 || return 2
  model_runner_smoke_require_integer MODEL_RUNNER_SMOKE_MLX_CACHE_MIB \
    "$MODEL_RUNNER_SMOKE_MLX_CACHE_MIB" 0 128 || return 2
  model_runner_smoke_require_integer MODEL_RUNNER_SMOKE_HOST_MEMORY_HIGH_GIB \
    "$MODEL_RUNNER_SMOKE_HOST_MEMORY_HIGH_GIB" 1 32 || return 2
  model_runner_smoke_require_integer MODEL_RUNNER_SMOKE_HOST_MEMORY_MAX_GIB \
    "$MODEL_RUNNER_SMOKE_HOST_MEMORY_MAX_GIB" 1 32 || return 2
  model_runner_smoke_require_integer MODEL_RUNNER_SMOKE_SWAP_MAX_GIB \
    "$MODEL_RUNNER_SMOKE_SWAP_MAX_GIB" 0 2 || return 2
  model_runner_smoke_require_integer MODEL_RUNNER_SMOKE_RUNTIME_SECONDS \
    "$MODEL_RUNNER_SMOKE_RUNTIME_SECONDS" 60 1800 || return 2
  model_runner_smoke_require_integer MODEL_RUNNER_SMOKE_HEALTH_TIMEOUT_SECONDS \
    "$MODEL_RUNNER_SMOKE_HEALTH_TIMEOUT_SECONDS" 10 900 || return 2
  model_runner_smoke_require_integer MODEL_RUNNER_SMOKE_HTTP_TIMEOUT_SECONDS \
    "$MODEL_RUNNER_SMOKE_HTTP_TIMEOUT_SECONDS" 10 600 || return 2
  model_runner_smoke_require_integer MODEL_RUNNER_SMOKE_MAX_TOKENS \
    "$MODEL_RUNNER_SMOKE_MAX_TOKENS" 1 96 || return 2
  model_runner_smoke_require_integer MODEL_RUNNER_SMOKE_PORT \
    "$MODEL_RUNNER_SMOKE_PORT" 1024 65535 || return 2
  model_runner_smoke_require_integer MODEL_RUNNER_SMOKE_CPU_QUOTA_PERCENT \
    "$MODEL_RUNNER_SMOKE_CPU_QUOTA_PERCENT" 1 400 || return 2

  if (( 10#$MODEL_RUNNER_SMOKE_HOST_MEMORY_HIGH_GIB > 10#$MODEL_RUNNER_SMOKE_HOST_MEMORY_MAX_GIB )); then
    echo "Host MemoryHigh may not exceed MemoryMax." >&2
    return 2
  fi
  if (( 10#$MODEL_RUNNER_SMOKE_MLX_MEMORY_GIB > 10#$MODEL_RUNNER_SMOKE_HOST_MEMORY_MAX_GIB )); then
    echo "The MLX memory ceiling may not exceed the whole-process host memory ceiling." >&2
    return 2
  fi
  if (( 10#$MODEL_RUNNER_SMOKE_HEALTH_TIMEOUT_SECONDS >= 10#$MODEL_RUNNER_SMOKE_RUNTIME_SECONDS )); then
    echo "The health timeout must be shorter than the unit runtime ceiling." >&2
    return 2
  fi
  if (( 10#$MODEL_RUNNER_SMOKE_HTTP_TIMEOUT_SECONDS >= 10#$MODEL_RUNNER_SMOKE_RUNTIME_SECONDS )); then
    echo "The HTTP timeout must be shorter than the unit runtime ceiling." >&2
    return 2
  fi
  if ((
    10#$MODEL_RUNNER_SMOKE_HEALTH_TIMEOUT_SECONDS
      + 10#$MODEL_RUNNER_SMOKE_HTTP_TIMEOUT_SECONDS + 15
      >= 10#$MODEL_RUNNER_SMOKE_RUNTIME_SECONDS
  )); then
    echo "Health plus HTTP timeouts need at least 15 seconds below the runtime ceiling." >&2
    return 2
  fi

  export MODEL_RUNNER_SMOKE_PROFILE
  export MODEL_RUNNER_SMOKE_MLX_MEMORY_GIB MODEL_RUNNER_SMOKE_MLX_CACHE_MIB
  export MODEL_RUNNER_SMOKE_HOST_MEMORY_HIGH_GIB MODEL_RUNNER_SMOKE_HOST_MEMORY_MAX_GIB
  export MODEL_RUNNER_SMOKE_SWAP_MAX_GIB MODEL_RUNNER_SMOKE_RUNTIME_SECONDS
  export MODEL_RUNNER_SMOKE_HEALTH_TIMEOUT_SECONDS MODEL_RUNNER_SMOKE_HTTP_TIMEOUT_SECONDS
  export MODEL_RUNNER_SMOKE_MAX_TOKENS MODEL_RUNNER_SMOKE_PORT
  export MODEL_RUNNER_SMOKE_CPU_QUOTA_PERCENT
}
