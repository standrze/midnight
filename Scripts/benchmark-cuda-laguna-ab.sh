#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$PACKAGE_ROOT/Scripts/cuda-runtime-environment.sh"
source "$PACKAGE_ROOT/Scripts/cuda-smoke-host-guard.sh"

usage() {
  cat <<'USAGE'
Usage:
  Scripts/benchmark-cuda-laguna-ab.sh \
    --runner ABSOLUTE_RUNNER \
    --baseline ABSOLUTE_MODEL_DIR \
    --candidate ABSOLUTE_MODEL_DIR \
    --output ABSOLUTE_NEW_RESULT_DIR

Environment overrides:
  MODEL_RUNNER_BENCHMARK_TRIALS       Measured trials after one warm-up (default: 5)
  MODEL_RUNNER_BENCHMARK_TOKENS       Tokens per warm-up/trial (default: 256)
  MODEL_RUNNER_BENCHMARK_PORT         Loopback port (default: 18083)
  MODEL_RUNNER_BENCHMARK_ORDER        baseline-first or candidate-first
USAGE
}

RUNNER=""
BASELINE=""
CANDIDATE=""
OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runner) RUNNER="$2"; shift 2 ;;
    --baseline) BASELINE="$2"; shift 2 ;;
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "The Laguna CUDA A/B harness runs only on Linux." >&2
  exit 1
fi
if [[ -z "$RUNNER" || -z "$BASELINE" || -z "$CANDIDATE" || -z "$OUTPUT" ]]; then
  echo "--runner, --baseline, --candidate, and --output are required." >&2
  usage >&2
  exit 2
fi
if [[ "$RUNNER" != /* || ! -x "$RUNNER" ]]; then
  echo "--runner must be an absolute executable path: $RUNNER" >&2
  exit 2
fi
for model_path in "$BASELINE" "$CANDIDATE"; do
  if [[ "$model_path" != /* || ! -d "$model_path" \
    || ! -f "$model_path/config.json" \
    || ! -f "$model_path/model.safetensors.index.json" ]]; then
    echo "Model is not an indexed absolute checkpoint directory: $model_path" >&2
    exit 2
  fi
done
if [[ "$OUTPUT" != /* || -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "--output must be an unused absolute path: $OUTPUT" >&2
  exit 2
fi

TRIALS="${MODEL_RUNNER_BENCHMARK_TRIALS:-5}"
TOKENS="${MODEL_RUNNER_BENCHMARK_TOKENS:-256}"
PORT="${MODEL_RUNNER_BENCHMARK_PORT:-18083}"
ORDER="${MODEL_RUNNER_BENCHMARK_ORDER:-baseline-first}"
if [[ ! "$TRIALS" =~ ^[1-9][0-9]*$ ]] || (( 10#$TRIALS > 20 )); then
  echo "MODEL_RUNNER_BENCHMARK_TRIALS must be in 1...20." >&2
  exit 2
fi
if [[ ! "$TOKENS" =~ ^[1-9][0-9]*$ ]] || (( 10#$TOKENS > 2048 )); then
  echo "MODEL_RUNNER_BENCHMARK_TOKENS must be in 1...2048." >&2
  exit 2
fi
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( 10#$PORT < 1024 || 10#$PORT > 65535 )); then
  echo "MODEL_RUNNER_BENCHMARK_PORT must be in 1024...65535." >&2
  exit 2
fi
if [[ "$ORDER" != "baseline-first" && "$ORDER" != "candidate-first" ]]; then
  echo "MODEL_RUNNER_BENCHMARK_ORDER must be baseline-first or candidate-first." >&2
  exit 2
fi

mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd -P)"
RUNNER="$(cd "$(dirname "$RUNNER")" && pwd -P)/$(basename "$RUNNER")"
BASELINE="$(cd "$BASELINE" && pwd -P)"
CANDIDATE="$(cd "$CANDIDATE" && pwd -P)"

for command_name in curl jq journalctl nvidia-smi pgrep sed ss stdbuf systemctl systemd-run; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "Required command not found: $command_name" >&2
    exit 1
  }
done
systemctl --user show-environment >/dev/null

model_runner_resolve_cuda_runtime_environment /usr/local/cuda
model_runner_cuda_runtime_systemd_setenv_args
model_runner_verify_cuda_runtime_headers_for_runner "$PACKAGE_ROOT" "$RUNNER"

MODEL_RUNNER_SMOKE_COMPUTE_OVERLAP_MODE=deny
MODEL_RUNNER_SMOKE_MLX_MEMORY_GIB=20
MODEL_RUNNER_SMOKE_PORT="$PORT"
export MODEL_RUNNER_SMOKE_COMPUTE_OVERLAP_MODE
export MODEL_RUNNER_SMOKE_MLX_MEMORY_GIB MODEL_RUNNER_SMOKE_PORT

ACTIVE_UNIT=""
ACTIVE_LOG=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP
  if [[ -n "$ACTIVE_UNIT" ]]; then
    systemctl --user stop "$ACTIVE_UNIT" >/dev/null 2>&1 || true
    if [[ -n "$ACTIVE_LOG" ]]; then
      journalctl --user --unit "$ACTIVE_UNIT" --no-pager --output=cat \
        > "$ACTIVE_LOG" 2>&1 || true
    fi
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

PROMPT='Write a long, detailed technical tutorial about implementing a lock-free work-stealing scheduler in Swift. Continue with implementation details and code examples until the output limit; do not conclude or summarize early.'

median_jq='def median: sort as $s | ($s|length) as $n | if ($n % 2) == 1 then $s[($n/2|floor)] else (($s[$n/2-1] + $s[$n/2]) / 2) end; median'

run_variant() {
  local label="$1"
  local model_path="$2"
  local served_name="laguna-q4r8-$label"
  local result_dir="$OUTPUT/$label"
  local unit="model-runner-laguna-bench-${UID}-${label}-$$-${RANDOM}.service"
  local base_url="http://127.0.0.1:$PORT"
  local request_file="$result_dir/request.json"
  local journal_file="$result_dir/server.log"
  local metrics_file="$result_dir/generation-metrics.txt"
  local rates_file="$result_dir/decode-rates.jsonl"
  local prompt_rates_file="$result_dir/prompt-rates.jsonl"
  local attempt
  local request_index
  local completion_tokens
  local expected_metrics=$((10#$TRIALS + 1))

  mkdir -p "$result_dir"
  model_runner_smoke_assert_idle_host
  ACTIVE_UNIT="$unit"
  ACTIVE_LOG="$journal_file"

  systemd-run --user \
    --unit="$unit" \
    --collect \
    --quiet \
    --working-directory="$PACKAGE_ROOT" \
    --property="Description=Matched Laguna Q4R8 benchmark $label" \
    --property=MemoryAccounting=yes \
    --property=MemoryHigh=28G \
    --property=MemoryMax=32G \
    --property=MemorySwapMax=1G \
    --property=RuntimeMaxSec=1200s \
    --property=CPUQuota=400% \
    --property=KillMode=control-group \
    --property=TimeoutStopSec=20s \
    "${MODEL_RUNNER_CUDA_RUNTIME_SYSTEMD_SETENV_ARGS[@]}" \
    --setenv=MODEL_RUNNER_MLX_MEMORY_LIMIT_GIB=20 \
    --setenv=MODEL_RUNNER_MLX_CACHE_LIMIT_MIB=128 \
    stdbuf --output=L --error=L \
    "$RUNNER" \
    --model "$model_path" \
    --name "$served_name" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-tokens "$TOKENS" \
    --engine cuda \
    --verbose

  for attempt in {1..180}; do
    if curl --fail --silent --show-error --max-time 3 \
      "$base_url/v1/models" > "$result_dir/models.json" 2> "$result_dir/readiness-errors.txt"; then
      break
    fi
    if [[ "$(systemctl --user is-active "$unit" 2>/dev/null || true)" != "active" ]]; then
      echo "$label server exited before readiness." >&2
      return 1
    fi
    sleep 1
  done
  if [[ "$attempt" == "180" ]]; then
    echo "$label server did not become ready." >&2
    return 1
  fi

  jq --null-input \
    --arg model "$served_name" \
    --arg prompt "$PROMPT" \
    --argjson tokens "$TOKENS" \
    '{model:$model,messages:[{role:"user",content:$prompt}],stream:false,max_tokens:$tokens,temperature:0,top_p:1}' \
    > "$request_file"

  for ((request_index=0; request_index<=10#$TRIALS; request_index++)); do
    curl --fail-with-body --silent --show-error \
      --connect-timeout 3 --max-time 360 \
      --header 'Content-Type: application/json' \
      --data-binary "@$request_file" \
      --output "$result_dir/response-$request_index.json" \
      --write-out '{"time_total":%{time_total},"time_starttransfer":%{time_starttransfer}}\n' \
      "$base_url/v1/chat/completions" \
      > "$result_dir/http-timing-$request_index.json"
    completion_tokens="$(jq -er '.usage.completion_tokens' "$result_dir/response-$request_index.json")"
    if [[ "$completion_tokens" != "$TOKENS" ]]; then
      echo "$label request $request_index generated $completion_tokens/$TOKENS tokens." >&2
      return 1
    fi
  done

  journalctl --user --unit "$unit" --no-pager --output=cat > "$journal_file"
  grep 'generation prompt_tokens=' "$journal_file" > "$metrics_file"
  if [[ "$(wc -l < "$metrics_file" | tr -d '[:space:]')" != "$expected_metrics" ]]; then
    echo "$label emitted an unexpected number of generation metric lines." >&2
    return 1
  fi
  sed -n '2,$s/.* tokens_per_second=\([0-9.][0-9.]*\) .*/\1/p' "$metrics_file" \
    | jq -R 'tonumber' > "$rates_file"
  sed -n '2,$s/.* prompt_tokens_per_second=\([0-9.][0-9.]*\) .*/\1/p' "$metrics_file" \
    | jq -R 'tonumber' > "$prompt_rates_file"
  if [[ "$(wc -l < "$rates_file" | tr -d '[:space:]')" != "$TRIALS" ]]; then
    echo "$label decode-rate extraction failed." >&2
    return 1
  fi

  systemctl --user stop "$unit"
  journalctl --user --unit "$unit" --no-pager --output=cat > "$journal_file"
  ACTIVE_UNIT=""
  ACTIVE_LOG=""

  jq --null-input \
    --arg label "$label" \
    --arg model_path "$model_path" \
    --arg served_name "$served_name" \
    --argjson tokens "$TOKENS" \
    --argjson trials "$TRIALS" \
    --slurpfile decode_rates "$rates_file" \
    --slurpfile prompt_rates "$prompt_rates_file" \
    --argjson median_decode "$(jq -s "$median_jq" "$rates_file")" \
    --argjson median_prompt "$(jq -s "$median_jq" "$prompt_rates_file")" \
    '{label:$label,model_path:$model_path,served_name:$served_name,tokens_per_trial:$tokens,measured_trials:$trials,decode_tokens_per_second:$decode_rates,median_decode_tokens_per_second:$median_decode,prompt_tokens_per_second:$prompt_rates,median_prompt_tokens_per_second:$median_prompt}' \
    > "$result_dir/summary.json"

  for attempt in {1..30}; do
    if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; then
      break
    fi
    sleep 1
  done
  model_runner_smoke_assert_idle_host
  jq -r '"\(.label): median \(.median_decode_tokens_per_second) tok/s"' "$result_dir/summary.json"
}

case "$ORDER" in
  baseline-first)
    run_variant baseline "$BASELINE"
    run_variant candidate "$CANDIDATE"
    ;;
  candidate-first)
    run_variant candidate "$CANDIDATE"
    run_variant baseline "$BASELINE"
    ;;
esac

jq --slurp \
  --arg order "$ORDER" \
  'map({key:.label,value:.}) | from_entries as $runs | {status:"verified",method:"one 256-token warm-up then matched greedy trials",execution_order:$order,baseline:$runs.baseline,candidate:$runs.candidate,decode_delta_percent:((($runs.candidate.median_decode_tokens_per_second / $runs.baseline.median_decode_tokens_per_second) - 1) * 100)}' \
  "$OUTPUT/baseline/summary.json" "$OUTPUT/candidate/summary.json" \
  > "$OUTPUT/summary.json"

trap - EXIT INT TERM HUP
jq '{baseline_median:.baseline.median_decode_tokens_per_second,candidate_median:.candidate.median_decode_tokens_per_second,decode_delta_percent}' "$OUTPUT/summary.json"
