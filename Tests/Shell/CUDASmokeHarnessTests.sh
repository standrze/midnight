#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$PACKAGE_ROOT/Scripts/cuda-smoke-policy.sh"
source "$PACKAGE_ROOT/Scripts/cuda-smoke-host-guard.sh"

TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/model-runner-smoke-policy.XXXXXX")"
TEST_ROOT="$(cd "$TEST_ROOT" && pwd -P)"
cleanup() {
  local status=$?
  trap - EXIT
  rm -rf "$TEST_ROOT"
  exit "$status"
}
trap cleanup EXIT

clear_smoke_overrides() {
  unset MODEL_RUNNER_SMOKE_MLX_MEMORY_GIB
  unset MODEL_RUNNER_SMOKE_MLX_CACHE_MIB
  unset MODEL_RUNNER_SMOKE_HOST_MEMORY_HIGH_GIB
  unset MODEL_RUNNER_SMOKE_HOST_MEMORY_MAX_GIB
  unset MODEL_RUNNER_SMOKE_SWAP_MAX_GIB
  unset MODEL_RUNNER_SMOKE_RUNTIME_SECONDS
  unset MODEL_RUNNER_SMOKE_HEALTH_TIMEOUT_SECONDS
  unset MODEL_RUNNER_SMOKE_HTTP_TIMEOUT_SECONDS
  unset MODEL_RUNNER_SMOKE_MAX_TOKENS
  unset MODEL_RUNNER_SMOKE_PORT
  unset MODEL_RUNNER_SMOKE_CPU_QUOTA_PERCENT
}

(
  clear_smoke_overrides
  model_runner_smoke_resolve_policy qwen
  [[ "$MODEL_RUNNER_SMOKE_MLX_MEMORY_GIB" == 8 ]]
  [[ "$MODEL_RUNNER_SMOKE_MLX_CACHE_MIB" == 64 ]]
  [[ "$MODEL_RUNNER_SMOKE_HOST_MEMORY_HIGH_GIB" == 12 ]]
  [[ "$MODEL_RUNNER_SMOKE_HOST_MEMORY_MAX_GIB" == 16 ]]
  [[ "$MODEL_RUNNER_SMOKE_RUNTIME_SECONDS" == 600 ]]
  [[ "$MODEL_RUNNER_SMOKE_CPU_QUOTA_PERCENT" == 200 ]]
  [[ "$MODEL_RUNNER_SMOKE_MAX_TOKENS" == 96 ]]
)

(
  clear_smoke_overrides
  model_runner_smoke_resolve_policy gemma
  [[ "$MODEL_RUNNER_SMOKE_MLX_MEMORY_GIB" == 18 ]]
  [[ "$MODEL_RUNNER_SMOKE_MLX_CACHE_MIB" == 128 ]]
  [[ "$MODEL_RUNNER_SMOKE_HOST_MEMORY_HIGH_GIB" == 24 ]]
  [[ "$MODEL_RUNNER_SMOKE_HOST_MEMORY_MAX_GIB" == 32 ]]
  [[ "$MODEL_RUNNER_SMOKE_RUNTIME_SECONDS" == 1500 ]]
  [[ "$MODEL_RUNNER_SMOKE_CPU_QUOTA_PERCENT" == 400 ]]
  [[ "$MODEL_RUNNER_SMOKE_MAX_TOKENS" == 96 ]]
)

(
  clear_smoke_overrides
  MODEL_RUNNER_SMOKE_MLX_MEMORY_GIB=6
  MODEL_RUNNER_SMOKE_MLX_CACHE_MIB=32
  MODEL_RUNNER_SMOKE_HOST_MEMORY_HIGH_GIB=8
  MODEL_RUNNER_SMOKE_HOST_MEMORY_MAX_GIB=12
  MODEL_RUNNER_SMOKE_SWAP_MAX_GIB=0
  MODEL_RUNNER_SMOKE_RUNTIME_SECONDS=300
  MODEL_RUNNER_SMOKE_HEALTH_TIMEOUT_SECONDS=120
  MODEL_RUNNER_SMOKE_HTTP_TIMEOUT_SECONDS=60
  MODEL_RUNNER_SMOKE_MAX_TOKENS=48
  model_runner_smoke_resolve_policy qwen
  [[ "$MODEL_RUNNER_SMOKE_HOST_MEMORY_MAX_GIB" == 12 ]]
  [[ "$MODEL_RUNNER_SMOKE_MAX_TOKENS" == 48 ]]
)

expect_policy_failure() {
  local variable="$1"
  local value="$2"
  local profile="${3:-qwen}"
  if (
    clear_smoke_overrides
    export "$variable=$value"
    model_runner_smoke_resolve_policy "$profile"
  ) >/dev/null 2>&1; then
    echo "$variable=$value unexpectedly passed the smoke policy" >&2
    exit 1
  fi
}

expect_policy_failure MODEL_RUNNER_SMOKE_MLX_MEMORY_GIB 21
expect_policy_failure MODEL_RUNNER_SMOKE_MLX_CACHE_MIB 129
expect_policy_failure MODEL_RUNNER_SMOKE_HOST_MEMORY_MAX_GIB 33
expect_policy_failure MODEL_RUNNER_SMOKE_SWAP_MAX_GIB 3
expect_policy_failure MODEL_RUNNER_SMOKE_RUNTIME_SECONDS 1801
expect_policy_failure MODEL_RUNNER_SMOKE_MAX_TOKENS 97
expect_policy_failure MODEL_RUNNER_SMOKE_PORT 80
expect_policy_failure MODEL_RUNNER_SMOKE_CPU_QUOTA_PERCENT 401

if (
  clear_smoke_overrides
  MODEL_RUNNER_SMOKE_RUNTIME_SECONDS=600
  MODEL_RUNNER_SMOKE_HEALTH_TIMEOUT_SECONDS=500
  MODEL_RUNNER_SMOKE_HTTP_TIMEOUT_SECONDS=100
  model_runner_smoke_resolve_policy qwen
) >/dev/null 2>&1; then
  echo "timeouts without runtime cleanup margin unexpectedly passed" >&2
  exit 1
fi

MOCK_BIN="$TEST_ROOT/mock-bin"
MOCK_PROC="$TEST_ROOT/proc"
mkdir -p "$MOCK_BIN" "$MOCK_PROC/4242"
printf 'python\0-m\0protected-server\0--port\08080\0' > "$MOCK_PROC/4242/cmdline"

cat > "$MOCK_BIN/sha256sum" <<'MOCK'
#!/usr/bin/env bash
if [[ -x /usr/bin/sha256sum ]]; then
  exec /usr/bin/sha256sum "$@"
fi
if [[ -x /usr/bin/shasum ]]; then
  exec /usr/bin/shasum -a 256 "$@"
fi
echo "No system SHA-256 utility is available for the test fixture." >&2
exit 1
MOCK
cat > "$MOCK_BIN/pgrep" <<'MOCK'
#!/usr/bin/env bash
if [[ "$*" != "-x midnight" ]]; then
  exit 2
fi
if [[ -n "${MOCK_RUNNER_PIDS:-}" ]]; then
  printf '%b\n' "$MOCK_RUNNER_PIDS"
  exit 0
fi
exit "${MOCK_PGREP_STATUS:-1}"
MOCK
cat > "$MOCK_BIN/nvidia-smi" <<'MOCK'
#!/usr/bin/env bash
case "$*" in
  --list-gpus)
    index=0
    while (( index < ${MOCK_GPU_COUNT:-1} )); do
      printf 'GPU %s: Mock GPU\n' "$index"
      index=$((index + 1))
    done
    ;;
  *--query-compute-apps=pid,process_name,used_gpu_memory*)
    if [[ -n "${MOCK_COMPUTE_ROWS:-}" ]]; then
      printf '%b\n' "$MOCK_COMPUTE_ROWS"
    fi
    ;;
  *--query-gpu=memory.free*)
    printf '%s\n' "${MOCK_GPU_FREE_MIB:-24576}"
    ;;
  *)
    echo "Unexpected mocked nvidia-smi arguments: $*" >&2
    exit 2
    ;;
esac
MOCK
cat > "$MOCK_BIN/ss" <<'MOCK'
#!/usr/bin/env bash
if [[ "$*" != "-H -ltn" ]]; then
  exit 2
fi
if [[ -n "${MOCK_LISTENERS:-}" ]]; then
  printf '%b\n' "$MOCK_LISTENERS"
fi
MOCK
chmod +x "$MOCK_BIN/sha256sum" "$MOCK_BIN/pgrep" "$MOCK_BIN/nvidia-smi" "$MOCK_BIN/ss"
PATH="$MOCK_BIN:$PATH"
export PATH

ALLOWED_HASH="$(sha256sum "$MOCK_PROC/4242/cmdline" | awk '{print $1}')"

mkdir -p "$MOCK_PROC/4243"
: > "$MOCK_PROC/4243/cmdline"
if model_runner_smoke_validate_allowed_compute_identity \
  4243 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 \
  "$MOCK_PROC" >/dev/null 2>&1
then
  echo "compute authorization unexpectedly accepted an empty command line" >&2
  exit 1
fi

clear_compute_authorization() {
  unset MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID
  unset MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_CMDLINE_SHA256
  unset MODEL_RUNNER_SMOKE_COMPUTE_OVERLAP_MODE
}

configure_mock_host() {
  MOCK_GPU_COUNT=1
  MOCK_GPU_FREE_MIB=24576
  MOCK_COMPUTE_ROWS=""
  MOCK_RUNNER_PIDS=""
  MOCK_PGREP_STATUS=1
  MOCK_LISTENERS=""
  MODEL_RUNNER_SMOKE_MLX_MEMORY_GIB=8
  MODEL_RUNNER_SMOKE_PORT=8080
  export MOCK_GPU_COUNT MOCK_GPU_FREE_MIB MOCK_COMPUTE_ROWS
  export MOCK_RUNNER_PIDS MOCK_PGREP_STATUS MOCK_LISTENERS
  export MODEL_RUNNER_SMOKE_MLX_MEMORY_GIB MODEL_RUNNER_SMOKE_PORT
}

(
  clear_compute_authorization
  model_runner_smoke_resolve_compute_authorization
  [[ "$MODEL_RUNNER_SMOKE_COMPUTE_OVERLAP_MODE" == "deny" ]]
)

if (
  clear_compute_authorization
  MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID=4242
  model_runner_smoke_resolve_compute_authorization
) >/dev/null 2>&1; then
  echo "compute authorization unexpectedly accepted PID without command-line hash" >&2
  exit 1
fi

if (
  clear_compute_authorization
  MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_CMDLINE_SHA256="$ALLOWED_HASH"
  model_runner_smoke_resolve_compute_authorization
) >/dev/null 2>&1; then
  echo "compute authorization unexpectedly accepted hash without PID" >&2
  exit 1
fi

if (
  clear_compute_authorization
  MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID=""
  MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_CMDLINE_SHA256=""
  model_runner_smoke_resolve_compute_authorization
) >/dev/null 2>&1; then
  echo "compute authorization unexpectedly accepted empty opt-in values" >&2
  exit 1
fi

if (
  clear_compute_authorization
  MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID=4242
  MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_CMDLINE_SHA256="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
  model_runner_smoke_resolve_compute_authorization
) >/dev/null 2>&1; then
  echo "compute authorization unexpectedly accepted a non-canonical hash" >&2
  exit 1
fi

(
  configure_mock_host
  clear_compute_authorization
  model_runner_smoke_resolve_compute_authorization
  model_runner_smoke_assert_idle_host "$MOCK_PROC"
)

if (
  configure_mock_host
  clear_compute_authorization
  MOCK_COMPUTE_ROWS='4242, protected-server, 1024'
  export MOCK_COMPUTE_ROWS
  model_runner_smoke_resolve_compute_authorization
  model_runner_smoke_assert_idle_host "$MOCK_PROC"
) >/dev/null 2>&1; then
  echo "default smoke mode unexpectedly permitted an NVIDIA compute process" >&2
  exit 1
fi

(
  configure_mock_host
  clear_compute_authorization
  MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID=4242
  MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_CMDLINE_SHA256="$ALLOWED_HASH"
  MOCK_COMPUTE_ROWS='4242, protected-server, 1024'
  export MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID
  export MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_CMDLINE_SHA256 MOCK_COMPUTE_ROWS
  model_runner_smoke_resolve_compute_authorization
  model_runner_smoke_assert_idle_host "$MOCK_PROC"
)

if (
  configure_mock_host
  clear_compute_authorization
  MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID=4242
  MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_CMDLINE_SHA256="$ALLOWED_HASH"
  MOCK_COMPUTE_ROWS='4242, protected-server, 1024\n9999, intruder, 512'
  export MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID
  export MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_CMDLINE_SHA256 MOCK_COMPUTE_ROWS
  model_runner_smoke_resolve_compute_authorization
  model_runner_smoke_assert_idle_host "$MOCK_PROC"
) >/dev/null 2>&1; then
  echo "protected-PID mode unexpectedly permitted another NVIDIA compute PID" >&2
  exit 1
fi

if (
  configure_mock_host
  clear_compute_authorization
  MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID=4242
  MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_CMDLINE_SHA256="$ALLOWED_HASH"
  export MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID
  export MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_CMDLINE_SHA256
  model_runner_smoke_resolve_compute_authorization
  model_runner_smoke_assert_idle_host "$MOCK_PROC"
) >/dev/null 2>&1; then
  echo "protected-PID mode unexpectedly passed without a matching NVIDIA compute row" >&2
  exit 1
fi

if (
  configure_mock_host
  clear_compute_authorization
  MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID=4242
  MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_CMDLINE_SHA256="$(printf '%064d' 0)"
  MOCK_COMPUTE_ROWS='4242, protected-server, 1024'
  export MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID
  export MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_CMDLINE_SHA256 MOCK_COMPUTE_ROWS
  model_runner_smoke_resolve_compute_authorization
  model_runner_smoke_assert_idle_host "$MOCK_PROC"
) >/dev/null 2>&1; then
  echo "protected-PID mode unexpectedly accepted a changed command line" >&2
  exit 1
fi

if (
  configure_mock_host
  clear_compute_authorization
  MOCK_GPU_FREE_MIB=10239
  export MOCK_GPU_FREE_MIB
  model_runner_smoke_resolve_compute_authorization
  model_runner_smoke_assert_idle_host "$MOCK_PROC"
) >/dev/null 2>&1; then
  echo "GPU memory gate unexpectedly passed without the full 2 GiB reserve" >&2
  exit 1
fi

(
  configure_mock_host
  clear_compute_authorization
  MOCK_GPU_FREE_MIB=10240
  export MOCK_GPU_FREE_MIB
  model_runner_smoke_resolve_compute_authorization
  model_runner_smoke_assert_idle_host "$MOCK_PROC"
)

if (
  clear_smoke_overrides
  MODEL_RUNNER_SMOKE_HOST_MEMORY_HIGH_GIB=17
  MODEL_RUNNER_SMOKE_HOST_MEMORY_MAX_GIB=16
  model_runner_smoke_resolve_policy qwen
) >/dev/null 2>&1; then
  echo "MemoryHigh above MemoryMax unexpectedly passed" >&2
  exit 1
fi

HARNESS="$PACKAGE_ROOT/Scripts/smoke-cuda-model.sh"
HOST_GUARD="$PACKAGE_ROOT/Scripts/cuda-smoke-host-guard.sh"
grep -Fq 'Both --model and --name are required' "$HARNESS"
grep -Fq 'pgrep -x midnight' "$HOST_GUARD"
grep -Fq -- '--query-compute-apps=pid,process_name,used_gpu_memory' "$HOST_GUARD"
grep -Fq -- '--query-gpu=memory.free' "$HOST_GUARD"
grep -Fq 'plus a 2 GiB reserve' "$HOST_GUARD"
grep -Fq 'model_runner_smoke_resolve_compute_authorization' "$HARNESS"
grep -Fq 'model_runner_smoke_assert_idle_host' "$HARNESS"
grep -Fq 'model_runner_resolve_cuda_runtime_runner_include "$PACKAGE_ROOT" "$RUNNER_PATH"' "$HARNESS"
grep -Fq 'model_runner_verify_cuda_runtime_headers_for_runner "$PACKAGE_ROOT" "$RUNNER_PATH"' "$HARNESS"
grep -Fq 'MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID' "$HARNESS"
grep -Fq 'SAMPLER_PID:-0' "$HARNESS"
grep -Fq 'sampler PID equals the protected compute PID; refusing to signal it' "$HARNESS"
grep -Fq 'systemd-run --user' "$HARNESS"
grep -Fq -- '--property="MemoryHigh=' "$HARNESS"
grep -Fq -- '--property="MemoryMax=' "$HARNESS"
grep -Fq -- '--property="MemorySwapMax=' "$HARNESS"
grep -Fq -- '--property="RuntimeMaxSec=' "$HARNESS"
grep -Fq -- '--property="CPUQuota=' "$HARNESS"
grep -Fq -- '--host 127.0.0.1' "$HARNESS"
grep -Fq '"$BASE_URL/v1/models"' "$HARNESS"
grep -Fq '"$BASE_URL/v1/chat/completions"' "$HARNESS"
grep -Fq 'systemctl --user stop "$UNIT_NAME"' "$HARNESS"
grep -Fq 'kill "$SAMPLER_PID"' "$HARNESS"
grep -Fq 'wait "$SAMPLER_PID"' "$HARNESS"
grep -Fq 'verified-mlx-resource-guard.txt' "$HARNESS"
grep -Fq 'resource-timeline.csv' "$HARNESS"
grep -Fq 'sleep 1' "$HARNESS"
if grep -Eq '(^|[[:space:]])(pkill|killall)([[:space:]]|$)' "$HARNESS"; then
  echo "smoke harness contains a process-wide termination command" >&2
  exit 1
fi
if grep -Eq '(kill|systemctl[^#]*stop).*MODEL_RUNNER_SMOKE_ALLOWED_COMPUTE_PID' "$HARNESS"; then
  echo "smoke harness contains a termination path targeting the protected PID" >&2
  exit 1
fi

"$HARNESS" --help > "$TEST_ROOT/help.txt"
grep -Fq -- '--profile qwen|gemma' "$TEST_ROOT/help.txt"

echo "CUDA smoke harness checks passed"
