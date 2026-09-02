#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$PACKAGE_ROOT/Scripts/cuda-smoke-policy.sh"
source "$PACKAGE_ROOT/Scripts/cuda-smoke-host-guard.sh"
source "$PACKAGE_ROOT/Scripts/swiftpm-scratch-path.sh"
source "$PACKAGE_ROOT/Scripts/release-publisher.sh"
source "$PACKAGE_ROOT/Scripts/cuda-runtime-environment.sh"

usage() {
  cat <<'USAGE'
Usage:
  Scripts/smoke-cuda-model.sh --model ABSOLUTE_MODEL_DIR --name SERVED_NAME [options]

Options:
  --profile qwen|gemma   Bounded resource profile (default: qwen)
  --runner PATH          Existing midnight binary; otherwise use SwiftPM bin path
  --help                 Show this help

The model path and served name are always required explicitly. Resource ceilings
may be lowered with the MODEL_RUNNER_SMOKE_* variables documented in README.md.
USAGE
}

MODEL_PATH=""
MODEL_NAME=""
SMOKE_PROFILE="qwen"
RUNNER_OVERRIDE="${MODEL_RUNNER_SMOKE_RUNNER:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -ge 2 ]] || { echo "--model requires a value." >&2; exit 2; }
      MODEL_PATH="$2"
      shift 2
      ;;
    --name)
      [[ $# -ge 2 ]] || { echo "--name requires a value." >&2; exit 2; }
      MODEL_NAME="$2"
      shift 2
      ;;
    --profile)
      [[ $# -ge 2 ]] || { echo "--profile requires qwen or gemma." >&2; exit 2; }
      SMOKE_PROFILE="$2"
      shift 2
      ;;
    --runner)
      [[ $# -ge 2 ]] || { echo "--runner requires a path." >&2; exit 2; }
      RUNNER_OVERRIDE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "The CUDA smoke harness runs only on Linux." >&2
  exit 1
fi
if [[ -z "$MODEL_PATH" || -z "$MODEL_NAME" ]]; then
  echo "Both --model and --name are required; settings-file discovery is disabled." >&2
  usage >&2
  exit 2
fi
if [[ "$MODEL_NAME" =~ ^[[:space:]]*$ ]] \
  || printf '%s' "$MODEL_NAME" | LC_ALL=C grep -q '[[:cntrl:]]'
then
  echo "--name must be nonblank and contain no control characters." >&2
  exit 2
fi
if (( ${#MODEL_NAME} > 128 )); then
  echo "--name may not exceed 128 characters." >&2
  exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory does not exist: $MODEL_PATH" >&2
  exit 1
fi
MODEL_PATH="$(cd "$MODEL_PATH" && pwd -P)"
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Model directory is missing config.json: $MODEL_PATH" >&2
  exit 1
fi
if [[ -z "$(find "$MODEL_PATH" -maxdepth 1 -type f -name '*.safetensors' -print -quit)" ]]; then
  echo "Model directory contains no top-level .safetensors weights: $MODEL_PATH" >&2
  exit 1
fi

model_runner_smoke_resolve_policy "$SMOKE_PROFILE"
model_runner_smoke_resolve_compute_authorization

REQUIRED_COMMANDS=(
  awk cksum cp curl date find grep journalctl jq mv nvidia-smi pgrep ps
  readlink rm sed sort ss stdbuf systemctl systemd-run
)
if [[ "$MODEL_RUNNER_SMOKE_COMPUTE_OVERLAP_MODE" == "allow-one" ]]; then
  REQUIRED_COMMANDS+=(sha256sum)
fi
for required_command in "${REQUIRED_COMMANDS[@]}"; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "Required smoke-test command was not found: $required_command" >&2
    exit 1
  fi
done
if ! systemctl --user show-environment >/dev/null 2>&1; then
  echo "A working systemd user manager is required for the memory-capped unit." >&2
  exit 1
fi

# Runtime CUDA kernel compilation is separate from the Swift build. Reuse the
# same exact toolkit, dependency, and Clang validators now; publication waits
# until the selected runner's real path proves it uses this package's include.
model_runner_resolve_cuda_runtime_environment /usr/local/cuda
model_runner_cuda_runtime_systemd_setenv_args

# Check before any SwiftPM executable lookup and again immediately before the
# service starts to reduce the race window with another local GPU job.
model_runner_smoke_assert_idle_host

if [[ -n "$RUNNER_OVERRIDE" ]]; then
  if [[ ! -x "$RUNNER_OVERRIDE" ]]; then
    echo "--runner is not executable: $RUNNER_OVERRIDE" >&2
    exit 1
  fi
  RUNNER_PATH="$(cd "$(dirname "$RUNNER_OVERRIDE")" && pwd -P)/$(basename "$RUNNER_OVERRIDE")"
else
  model_runner_release_paths "$PACKAGE_ROOT"
  if [[ -e "$MODEL_RUNNER_RTX4090_MANIFEST_PATH" \
    || -L "$MODEL_RUNNER_RTX4090_MANIFEST_PATH" ]]; then
    model_runner_verify_rtx4090_publication "$PACKAGE_ROOT"
    RUNNER_PATH="$MODEL_RUNNER_RTX4090_COMPAT_PATH"
  else
    if ! command -v swift >/dev/null 2>&1; then
      echo "swift is required to locate the active SwiftPM binary; use --runner to provide one." >&2
      exit 1
    fi
    model_runner_configure_swiftpm_scratch "$PACKAGE_ROOT" Linux
    BIN_DIR="$(
      cd "$PACKAGE_ROOT"
      swift build --configuration release \
        "${MODEL_RUNNER_SWIFT_BUILD_SCRATCH_ARGS[@]}" --show-bin-path
    )"
    RUNNER_PATH="$BIN_DIR/midnight"
  fi
  if [[ ! -x "$RUNNER_PATH" ]]; then
    echo "Midnight Runner is not built at $RUNNER_PATH; build it before the smoke test." >&2
    exit 1
  fi
fi

# Pinned MLX derives its runtime include root from the real executable path.
# Refuse an arbitrary override or SwiftPM scratch binary instead of copying
# headers into an unmanaged parent tree. A compatibility symlink resolving to
# the stable package bin/ publication intentionally passes this check.
model_runner_resolve_cuda_runtime_runner_include "$PACKAGE_ROOT" "$RUNNER_PATH"
model_runner_publish_cuda_runtime_headers "$PACKAGE_ROOT"
model_runner_verify_cuda_runtime_headers_for_runner "$PACKAGE_ROOT" "$RUNNER_PATH"

OUTPUT_ROOT="${MODEL_RUNNER_SMOKE_OUTPUT_ROOT:-$PACKAGE_ROOT/.smoke-results}"
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd -P)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
RESULT_DIR="$OUTPUT_ROOT/$RUN_ID"
mkdir -p "$RESULT_DIR"
UNIT_NAME="midnight-smoke-${UID}-$$-${RANDOM}.service"
UNIT_STARTED=0
SAMPLER_PID=0

capture_gpu_metrics() {
  local stage="$1"
  {
    printf 'stage=%s\n' "$stage"
    nvidia-smi \
      --query-gpu=timestamp,index,name,memory.total,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits
    nvidia-smi \
      --query-compute-apps=pid,process_name,used_gpu_memory \
      --format=csv,noheader,nounits || true
  } >> "$RESULT_DIR/gpu-metrics.txt"
}

capture_host_metrics() {
  local stage="$1"
  {
    printf 'stage=%s\n' "$stage"
    grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo
  } >> "$RESULT_DIR/host-memory.txt"
}

capture_unit_metrics() {
  local stage="$1"
  {
    printf 'stage=%s\n' "$stage"
    systemctl --user show "$UNIT_NAME" \
      --property=ActiveState \
      --property=SubState \
      --property=MainPID \
      --property=MemoryCurrent \
      --property=MemoryPeak \
      --property=MemorySwapCurrent \
      --no-pager || true
    local main_pid
    main_pid="$(systemctl --user show "$UNIT_NAME" --property=MainPID --value 2>/dev/null || true)"
    if [[ "$main_pid" =~ ^[1-9][0-9]*$ ]]; then
      ps -p "$main_pid" -o pid=,ppid=,etime=,rss=,vsz=,comm= || true
    fi
  } >> "$RESULT_DIR/unit-memory.txt" 2>&1
}

sample_resource_timeline() {
  local timestamp
  local gpu_sample
  local gpu_used
  local gpu_free
  local gpu_utilization
  local host_available
  local swap_free
  local unit_memory

  set +e
  while true; do
    timestamp="$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
    gpu_sample="$(
      nvidia-smi \
        --query-gpu=memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null
    )"
    if [[ -n "$gpu_sample" ]]; then
      IFS=',' read -r gpu_used gpu_free gpu_utilization <<< "$gpu_sample"
    else
      gpu_used=NA
      gpu_free=NA
      gpu_utilization=NA
    fi
    host_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    swap_free="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
    unit_memory="$(
      systemctl --user show "$UNIT_NAME" --property=MemoryCurrent --value 2>/dev/null
    )"
    unit_memory="${unit_memory:-NA}"
    printf '%s,%s,%s,%s,%s,%s,%s\n' \
      "$timestamp" \
      "${gpu_used//[[:space:]]/}" \
      "${gpu_free//[[:space:]]/}" \
      "${gpu_utilization//[[:space:]]/}" \
      "${host_available:-NA}" \
      "${swap_free:-NA}" \
      "$unit_memory" \
      >> "$RESULT_DIR/resource-timeline.csv"
    sleep 1
  done
}

stop_sampler() {
  local sampler_parent

  if [[ "${MODEL_RUNNER_SMOKE_COMPUTE_OVERLAP_MODE:-deny}" == "allow-one" ]] \
    && [[ "${SAMPLER_PID:-0}" == "${MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID:-}" ]]
  then
    echo "Internal error: sampler PID equals the protected compute PID; refusing to signal it." >&2
    SAMPLER_PID=0
    return 1
  fi
  if [[ "${SAMPLER_PID:-0}" =~ ^[1-9][0-9]*$ ]] && kill -0 "$SAMPLER_PID" 2>/dev/null; then
    sampler_parent="$(ps -p "$SAMPLER_PID" -o ppid= | awk '{$1=$1};1')"
    if [[ "$sampler_parent" == "$$" ]]; then
      kill "$SAMPLER_PID" 2>/dev/null || true
    fi
  fi
  if [[ "${SAMPLER_PID:-0}" =~ ^[1-9][0-9]*$ ]]; then
    # wait is restricted to this shell's child processes and also reaps a
    # sampler that happened to exit between kill -0 and the ownership check.
    wait "$SAMPLER_PID" 2>/dev/null || true
  fi
  SAMPLER_PID=0
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP
  stop_sampler || true
  if [[ -d "${RESULT_DIR:-}" ]]; then
    capture_host_metrics before-cleanup || true
    capture_gpu_metrics before-cleanup || true
  fi
  if [[ "${UNIT_STARTED:-0}" == "1" ]]; then
    capture_unit_metrics before-stop || true
    systemctl --user stop "$UNIT_NAME" >/dev/null 2>&1 || true
    journalctl --user --unit "$UNIT_NAME" --no-pager --output=short-precise \
      > "$RESULT_DIR/server.log" 2>&1 || true
    capture_unit_metrics after-stop || true
  fi
  if [[ -d "${RESULT_DIR:-}" ]]; then
    printf '%s\n' "$status" > "$RESULT_DIR/exit-status.txt"
    if [[ "$status" == "0" ]]; then
      printf 'PASS\n' > "$RESULT_DIR/result.txt"
    else
      printf 'FAIL\n' > "$RESULT_DIR/result.txt"
    fi
    echo "Smoke artifacts: $RESULT_DIR" >&2
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

capture_host_metrics preflight
capture_gpu_metrics preflight
model_runner_smoke_assert_idle_host

UNIT_LOAD_STATE="$(
  systemctl --user show "$UNIT_NAME" --property=LoadState --value 2>/dev/null || true
)"
if [[ -n "$UNIT_LOAD_STATE" && "$UNIT_LOAD_STATE" != "not-found" ]]; then
  echo "Generated unit name unexpectedly already exists: $UNIT_NAME" >&2
  exit 1
fi

printf '%s\n' \
  'timestamp,gpu_memory_used_mib,gpu_memory_free_mib,gpu_utilization_percent,host_mem_available_kib,host_swap_free_kib,unit_memory_current_bytes' \
  > "$RESULT_DIR/resource-timeline.csv"
sample_resource_timeline &
SAMPLER_PID=$!

START_NANOSECONDS="$(date +%s%N)"
if systemd-run --user \
    --unit="$UNIT_NAME" \
    --collect \
    --quiet \
    --working-directory="$PACKAGE_ROOT" \
    --property="Description=Bounded Midnight Runner CUDA smoke $RUN_ID" \
    --property=MemoryAccounting=yes \
    --property="MemoryHigh=${MODEL_RUNNER_SMOKE_HOST_MEMORY_HIGH_GIB}G" \
    --property="MemoryMax=${MODEL_RUNNER_SMOKE_HOST_MEMORY_MAX_GIB}G" \
    --property="MemorySwapMax=${MODEL_RUNNER_SMOKE_SWAP_MAX_GIB}G" \
    --property="RuntimeMaxSec=${MODEL_RUNNER_SMOKE_RUNTIME_SECONDS}s" \
    --property="CPUQuota=${MODEL_RUNNER_SMOKE_CPU_QUOTA_PERCENT}%" \
    --property=KillMode=control-group \
    --property=TimeoutStopSec=15s \
    "${MODEL_RUNNER_CUDA_RUNTIME_SYSTEMD_SETENV_ARGS[@]}" \
    --setenv="MODEL_RUNNER_MLX_MEMORY_LIMIT_GIB=$MODEL_RUNNER_SMOKE_MLX_MEMORY_GIB" \
    --setenv="MODEL_RUNNER_MLX_CACHE_LIMIT_MIB=$MODEL_RUNNER_SMOKE_MLX_CACHE_MIB" \
    --setenv="MODEL_RUNNER_SMOKE_RUN_ID=$RUN_ID" \
    stdbuf --output=L --error=L \
    "$RUNNER_PATH" \
    --model "$MODEL_PATH" \
    --name "$MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$MODEL_RUNNER_SMOKE_PORT" \
    --max-tokens "$MODEL_RUNNER_SMOKE_MAX_TOKENS" \
    --engine cuda
then
  UNIT_STARTED=1
else
  UNIT_LOAD_STATE="$(
    systemctl --user show "$UNIT_NAME" --property=LoadState --value 2>/dev/null || true
  )"
  if [[ -n "$UNIT_LOAD_STATE" && "$UNIT_LOAD_STATE" != "not-found" ]]; then
    # The unique name was absent immediately before systemd-run, so a newly
    # materialized unit with this name belongs to this invocation.
    UNIT_STARTED=1
  fi
  echo "systemd-run failed to launch the bounded smoke unit." >&2
  exit 1
fi

BASE_URL="http://127.0.0.1:$MODEL_RUNNER_SMOKE_PORT"
HEALTH_DEADLINE=$((SECONDS + 10#$MODEL_RUNNER_SMOKE_HEALTH_TIMEOUT_SECONDS))
HEALTH_READY=0
while (( SECONDS < HEALTH_DEADLINE )); do
  UNIT_STATE="$(systemctl --user is-active "$UNIT_NAME" 2>/dev/null || true)"
  if [[ "$UNIT_STATE" != "active" && "$UNIT_STATE" != "activating" ]]; then
    echo "Smoke unit exited before becoming healthy (state=$UNIT_STATE)." >&2
    capture_unit_metrics startup-failure
    exit 1
  fi
  if curl --fail --silent --show-error \
    --connect-timeout 2 --max-time 5 \
    --output "$RESULT_DIR/models-readiness.body" \
    --write-out '{"http_code":%{http_code},"time_total":%{time_total},"time_starttransfer":%{time_starttransfer}}\n' \
    "$BASE_URL/v1/models" > "$RESULT_DIR/models-readiness-timing.json" 2>> "$RESULT_DIR/models-readiness-errors.log"
  then
    HEALTH_READY=1
    break
  fi
  sleep 1
done
if [[ "$HEALTH_READY" != "1" ]]; then
  echo "Timed out waiting for /v1/models after $MODEL_RUNNER_SMOKE_HEALTH_TIMEOUT_SECONDS seconds." >&2
  exit 1
fi
HEALTH_NANOSECONDS="$(date +%s%N)"
LOAD_MILLISECONDS=$(( (HEALTH_NANOSECONDS - START_NANOSECONDS) / 1000000 ))
printf '%s\n' "$LOAD_MILLISECONDS" > "$RESULT_DIR/load-milliseconds.txt"
capture_host_metrics healthy
capture_gpu_metrics healthy
capture_unit_metrics healthy

EXPECTED_MLX_MEMORY_BYTES=$((10#$MODEL_RUNNER_SMOKE_MLX_MEMORY_GIB * 1073741824))
EXPECTED_MLX_CACHE_BYTES=$((10#$MODEL_RUNNER_SMOKE_MLX_CACHE_MIB * 1048576))
EXPECTED_GUARD_LINE="MLX resource guard: memory=$EXPECTED_MLX_MEMORY_BYTES bytes cache=$EXPECTED_MLX_CACHE_BYTES bytes"
GUARD_LINE_FOUND=0
for _ in {1..10}; do
  journalctl --user --unit "$UNIT_NAME" --no-pager --output=cat \
    > "$RESULT_DIR/startup-journal.txt" 2>&1 || true
  if grep -Fqx "$EXPECTED_GUARD_LINE" "$RESULT_DIR/startup-journal.txt"; then
    GUARD_LINE_FOUND=1
    break
  fi
  sleep 1
done
if [[ "$GUARD_LINE_FOUND" != "1" ]]; then
  echo "The exact MLX resource-guard line was not observed in the line-buffered unit journal." >&2
  echo "Expected: $EXPECTED_GUARD_LINE" >&2
  exit 1
fi
printf '%s\n' "$EXPECTED_GUARD_LINE" > "$RESULT_DIR/verified-mlx-resource-guard.txt"

curl --fail-with-body --silent --show-error \
  --connect-timeout 2 --max-time "$MODEL_RUNNER_SMOKE_HTTP_TIMEOUT_SECONDS" \
  --output "$RESULT_DIR/models.json" \
  --write-out '{"http_code":%{http_code},"time_total":%{time_total},"time_starttransfer":%{time_starttransfer}}\n' \
  "$BASE_URL/v1/models" > "$RESULT_DIR/models-timing.json"
jq --exit-status --arg name "$MODEL_NAME" \
  '.object == "list" and any(.data[]?; .id == $name)' \
  "$RESULT_DIR/models.json" >/dev/null

jq --null-input \
  --arg model "$MODEL_NAME" \
  --arg prompt 'Reply briefly with the words: CUDA smoke test OK' \
  --argjson max_tokens "$MODEL_RUNNER_SMOKE_MAX_TOKENS" \
  '{model:$model,messages:[{role:"user",content:$prompt}],stream:true,max_tokens:$max_tokens,temperature:0}' \
  > "$RESULT_DIR/chat-request.json"

curl --fail-with-body --silent --show-error --no-buffer \
  --connect-timeout 2 --max-time "$MODEL_RUNNER_SMOKE_HTTP_TIMEOUT_SECONDS" \
  --header 'Content-Type: application/json' \
  --data-binary "@$RESULT_DIR/chat-request.json" \
  --output "$RESULT_DIR/chat-stream.txt" \
  --write-out '{"http_code":%{http_code},"time_total":%{time_total},"time_starttransfer":%{time_starttransfer}}\n' \
  "$BASE_URL/v1/chat/completions" > "$RESULT_DIR/chat-timing.json"

grep -Fqx 'data: [DONE]' "$RESULT_DIR/chat-stream.txt"
sed -n 's/^data: //p' "$RESULT_DIR/chat-stream.txt" \
  | grep -Fvx '[DONE]' \
  | jq --slurp --exit-status \
    'length > 0 and all(.[]; has("error") | not) and any(.[]; ((.choices[0].delta.content? // "") | length) > 0)' \
    >/dev/null

capture_host_metrics after-chat
capture_gpu_metrics after-chat
capture_unit_metrics after-chat
stop_sampler

jq --null-input \
  --arg result PASS \
  --arg run_id "$RUN_ID" \
  --arg profile "$MODEL_RUNNER_SMOKE_PROFILE" \
  --arg model_path "$MODEL_PATH" \
  --arg model_name "$MODEL_NAME" \
  --arg unit "$UNIT_NAME" \
  --argjson load_milliseconds "$LOAD_MILLISECONDS" \
  --argjson mlx_memory_gib "$MODEL_RUNNER_SMOKE_MLX_MEMORY_GIB" \
  --argjson mlx_cache_mib "$MODEL_RUNNER_SMOKE_MLX_CACHE_MIB" \
  --argjson host_memory_max_gib "$MODEL_RUNNER_SMOKE_HOST_MEMORY_MAX_GIB" \
  --argjson cpu_quota_percent "$MODEL_RUNNER_SMOKE_CPU_QUOTA_PERCENT" \
  --argjson max_tokens "$MODEL_RUNNER_SMOKE_MAX_TOKENS" \
  --arg compute_overlap_mode "$MODEL_RUNNER_SMOKE_COMPUTE_OVERLAP_MODE" \
  --arg protected_compute_pid "${MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID:-}" \
  '{result:$result,run_id:$run_id,profile:$profile,model_path:$model_path,model_name:$model_name,unit:$unit,load_milliseconds:$load_milliseconds,mlx_memory_gib:$mlx_memory_gib,mlx_cache_mib:$mlx_cache_mib,host_memory_max_gib:$host_memory_max_gib,cpu_quota_percent:$cpu_quota_percent,max_tokens:$max_tokens,compute_overlap_mode:$compute_overlap_mode,protected_compute_pid:(if $protected_compute_pid == "" then null else $protected_compute_pid end)}' \
  > "$RESULT_DIR/summary.json"

echo "CUDA smoke test passed in ${LOAD_MILLISECONDS} ms load-to-health."
