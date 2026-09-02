#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$PACKAGE_ROOT/Scripts/release-publisher.sh"

TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/model-runner-release-publisher.XXXXXX")"
TEST_ROOT="$(cd "$TEST_ROOT" && pwd -P)"
cleanup() {
  local status=$?
  trap - EXIT
  rm -rf "$TEST_ROOT"
  exit "$status"
}
trap cleanup EXIT

VALID_PROFILE='configuration=release:cuda:sm_89:mlx-cross-thread-stream-overlay=off:nvcc=/usr/local/cuda/bin/nvcc:dependencies=test'

make_source() {
  local scratch="$1"
  local body="$2"
  local source="$scratch/x86_64-unknown-linux-gnu/release/midnight"

  mkdir -p "$(dirname "$source")"
  printf '#!/usr/bin/env bash\nprintf %%s\\n %q\n' "$body" > "$source"
  chmod 0755 "$source"
  printf '%s\n' "$VALID_PROFILE" > "$scratch/.model-runner-profile"
  printf '%s\n' "$source"
}

VALID_PACKAGE="$TEST_ROOT/package"
VALID_SCRATCH="$TEST_ROOT/custom scratch"
mkdir -p "$VALID_PACKAGE" "$VALID_SCRATCH"
VALID_SOURCE="$(make_source "$VALID_SCRATCH" first-build)"

model_runner_publish_rtx4090_release \
  "$VALID_PACKAGE" "$VALID_SCRATCH" "$VALID_SOURCE" "$VALID_PROFILE"
model_runner_verify_rtx4090_publication "$VALID_PACKAGE" "$VALID_SCRATCH"
model_runner_release_paths "$VALID_PACKAGE"

[[ -f "$MODEL_RUNNER_RTX4090_STABLE_PATH" ]]
[[ -x "$MODEL_RUNNER_RTX4090_STABLE_PATH" ]]
[[ ! -L "$MODEL_RUNNER_RTX4090_STABLE_PATH" ]]
[[ -L "$MODEL_RUNNER_RTX4090_COMPAT_PATH" ]]
[[ "$(readlink "$MODEL_RUNNER_RTX4090_COMPAT_PATH")" == "$MODEL_RUNNER_RTX4090_STABLE_PATH" ]]
[[ "$MODEL_RUNNER_RTX4090_COMPAT_PATH" -ef "$MODEL_RUNNER_RTX4090_STABLE_PATH" ]]
cmp -s "$VALID_SOURCE" "$MODEL_RUNNER_RTX4090_STABLE_PATH"
grep -Fqx "build_profile=$VALID_PROFILE" "$MODEL_RUNNER_RTX4090_MANIFEST_PATH"
grep -Fq ':mlx-cross-thread-stream-overlay=off:' "$MODEL_RUNNER_RTX4090_MANIFEST_PATH"
grep -Fqx "source_scratch=$VALID_SCRATCH" "$MODEL_RUNNER_RTX4090_MANIFEST_PATH"
grep -Fqx "source_binary=$VALID_SOURCE" "$MODEL_RUNNER_RTX4090_MANIFEST_PATH"
grep -Fqx "stable_binary=$MODEL_RUNNER_RTX4090_STABLE_PATH" "$MODEL_RUNNER_RTX4090_MANIFEST_PATH"
grep -Fqx "compatibility_link=$MODEL_RUNNER_RTX4090_COMPAT_PATH" "$MODEL_RUNNER_RTX4090_MANIFEST_PATH"

# A subsequent verified build atomically replaces the prior valid publication.
printf '#!/usr/bin/env bash\nprintf second-build\n' > "$VALID_SOURCE"
chmod 0755 "$VALID_SOURCE"
model_runner_publish_rtx4090_release \
  "$VALID_PACKAGE" "$VALID_SCRATCH" "$VALID_SOURCE" "$VALID_PROFILE"
cmp -s "$VALID_SOURCE" "$MODEL_RUNNER_RTX4090_STABLE_PATH"
model_runner_verify_rtx4090_publication "$VALID_PACKAGE" "$VALID_SCRATCH"
if find "$VALID_PACKAGE/bin" "$VALID_PACKAGE/.build/release" \
  -maxdepth 1 -name '.midnight-*.*' -print | grep -q .; then
  echo "publisher left a staged temporary file behind" >&2
  exit 1
fi

expect_publish_failure() {
  local name="$1"
  local mode="$2"
  local package="$TEST_ROOT/$name-package"
  local scratch="$TEST_ROOT/$name-scratch"
  local source
  local profile="$VALID_PROFILE"

  mkdir -p "$package" "$scratch"
  source="$(make_source "$scratch" "$name")"
  case "$mode" in
    missing-source)
      rm -f "$source"
      ;;
    non-executable)
      chmod 0644 "$source"
      ;;
    missing-marker)
      rm -f "$scratch/.model-runner-profile"
      ;;
    mismatched-marker)
      printf '%s\n' 'configuration=release:cuda:sm_89:different=true' \
        > "$scratch/.model-runner-profile"
      ;;
    debug-profile)
      profile='configuration=debug:cuda:sm_89:nvcc=/usr/local/cuda/bin/nvcc'
      printf '%s\n' "$profile" > "$scratch/.model-runner-profile"
      ;;
    outside-scratch)
      source="$TEST_ROOT/outside-$name"
      printf '#!/usr/bin/env bash\nexit 0\n' > "$source"
      chmod 0755 "$source"
      ;;
    regular-compatibility-path)
      mkdir -p "$package/.build/release"
      printf 'unmanaged\n' > "$package/.build/release/midnight"
      chmod 0755 "$package/.build/release/midnight"
      ;;
    *)
      echo "unknown failure fixture: $mode" >&2
      exit 1
      ;;
  esac

  if model_runner_publish_rtx4090_release \
    "$package" "$scratch" "$source" "$profile" >/dev/null 2>&1; then
    echo "$mode unexpectedly published" >&2
    exit 1
  fi
}

expect_publish_failure missing-source missing-source
expect_publish_failure non-executable non-executable
expect_publish_failure missing-marker missing-marker
expect_publish_failure mismatched-marker mismatched-marker
expect_publish_failure debug-profile debug-profile
expect_publish_failure outside-scratch outside-scratch
expect_publish_failure regular-compat regular-compatibility-path

# Corruption must be detected and must not be silently healed by republishing.
model_runner_release_paths "$VALID_PACKAGE"
printf 'corrupt\n' >> "$MODEL_RUNNER_RTX4090_STABLE_PATH"
if model_runner_verify_rtx4090_publication "$VALID_PACKAGE" >/dev/null 2>&1; then
  echo "a hash-mismatched stable binary unexpectedly verified" >&2
  exit 1
fi
if model_runner_publish_rtx4090_release \
  "$VALID_PACKAGE" "$VALID_SCRATCH" "$VALID_SOURCE" "$VALID_PROFILE" \
  >/dev/null 2>&1; then
  echo "publisher unexpectedly overwrote a hash-mismatched prior publication" >&2
  exit 1
fi

BUILD_SCRIPT="$PACKAGE_ROOT/build.sh"
RUN_SCRIPT="$PACKAGE_ROOT/run.sh"
SMOKE_SCRIPT="$PACKAGE_ROOT/Scripts/smoke-cuda-model.sh"
grep -Fq 'SWIFT_BUILD_CONFIGURATION="release"' "$BUILD_SCRIPT"
grep -Fq 'model_runner_publish_rtx4090_release' "$BUILD_SCRIPT"
grep -Fq 'DEFAULT_RELEASE_RUNNER="$PACKAGE_ROOT/.build/release/midnight"' "$BUILD_SCRIPT"
grep -Fq 'model_runner_verify_rtx4090_publication' "$RUN_SCRIPT"
grep -Fq 'model_runner_verify_rtx4090_publication' "$SMOKE_SCRIPT"
grep -Fq 'bin/midnight-rtx4090' "$PACKAGE_ROOT/.gitignore"

echo "Release publisher checks passed"
