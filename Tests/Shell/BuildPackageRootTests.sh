#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FIXTURE="$(mktemp -d "${TMPDIR:-/tmp}/midnight-package-root.XXXXXX")"
FIXTURE="$(cd "$FIXTURE" && pwd -P)"
trap 'rm -rf "$FIXTURE"' EXIT
RUNTIME="$FIXTURE/runtime source"
SIBLING="$FIXTURE/sibling tools"
mkdir -p "$RUNTIME" "$SIBLING" "$FIXTURE/bin"
cp "$ROOT/build.sh" "$ROOT/build-metal.sh" "$ROOT/prepare-dependencies.sh" "$RUNTIME/"
cp -R "$ROOT/Scripts" "$ROOT/Patches" "$RUNTIME/"
printf '// Fixture package: no real Swift compiler or network is used.\n' > "$SIBLING/Package.swift"

# The sibling deliberately has no Scripts or Patches. Use the actual dependency
# preparation script, mocking only external tools, to exercise both ownerships.
export TEST_RUNTIME="$RUNTIME" TEST_SIBLING="$SIBLING"
export TEST_LOG="$FIXTURE/tool.log"
rg '^[A-Z0-9_]+_EXPECTED_REVISION=' "$RUNTIME/prepare-dependencies.sh" > "$FIXTURE/revisions.sh"
export TEST_REVISIONS="$FIXTURE/revisions.sh"

cat > "$FIXTURE/bin/uname" <<'MOCK'
#!/usr/bin/env bash
printf '%s\n' "$TEST_HOST"
MOCK

cat > "$FIXTURE/bin/swift" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
[[ "$PWD" == "$TEST_SIBLING" ]]
printf 'swift|%s|%s\n' "$PWD" "$*" >> "$TEST_LOG"
action="$1"
shift
scratch="$PWD/.build"
configuration=debug
product=midnight
show=0
resolve=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --scratch-path) shift; scratch="$1" ;;
    --configuration) shift; configuration="$1" ;;
    --product) shift; product="$1" ;;
    --show-bin-path) show=1 ;;
    resolve) resolve=1 ;;
  esac
  shift
done
[[ "$scratch" == "$TEST_SCRATCH" ]]
if [[ "$resolve" == 1 ]]; then
  source "$TEST_REVISIONS"
  if [[ "$TEST_HOST" == Darwin ]]; then
    mlx_revision="$MLX_SWIFT_DARWIN_EXPECTED_REVISION"
    source_revision="$MLX_SOURCE_DARWIN_EXPECTED_REVISION"
    c_revision="$MLX_C_SOURCE_DARWIN_EXPECTED_REVISION"
  else
    mlx_revision="$MLX_SWIFT_LINUX_EXPECTED_REVISION"
    source_revision="$MLX_SOURCE_LINUX_EXPECTED_REVISION"
    c_revision="$MLX_C_SOURCE_LINUX_EXPECTED_REVISION"
  fi
  checkout() {
    mkdir -p "$scratch/checkouts/$1"
    printf '%s\n' "$2" > "$scratch/checkouts/$1/.fixture-revision"
  }
  checkout mlx-swift "$mlx_revision"
  checkout mlx-swift/Source/Cmlx/mlx "$source_revision"
  checkout mlx-swift/Source/Cmlx/mlx-c "$c_revision"
  checkout mlx-swift-lm "$MLX_SWIFT_LM_EXPECTED_REVISION"
  checkout swift-transformers "$SWIFT_TRANSFORMERS_EXPECTED_REVISION"
  # The audio revision is an inline verify_checkout_revision argument.
  audio_revision="$(sed -n '/verify_checkout_revision "mlx-audio-swift"/,+2p' "$TEST_RUNTIME/prepare-dependencies.sh" | tail -n 1 | tr -d ' "')"
  checkout mlx-audio-swift "$audio_revision"
  mkdir -p "$scratch/checkouts/mlx-swift-lm/Libraries/MLXLLM/Models"
  mkdir -p "$scratch/checkouts/mlx-swift-lm/Libraries/MLXLMCommon"
  mkdir -p "$scratch/checkouts/mlx-swift/Source/Cmlx/mlx-c/mlx/c"
  touch "$scratch/checkouts/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift"
  touch "$scratch/checkouts/mlx-swift-lm/Libraries/MLXLMCommon/"{Evaluate,ModelConversion,MTPSpeculativeTokenIterator}.swift
  touch "$scratch/checkouts/mlx-swift/Source/Cmlx/mlx-c/mlx/c/stream.cpp"
  printf '{"fixture":true}\n' > "$PWD/Package.resolved"
elif [[ "$action" == build ]]; then
  if [[ "$show" == 1 ]]; then
    printf '%s\n' "$scratch/$configuration"
  else
    mkdir -p "$scratch/$configuration"
    printf '#!/usr/bin/env bash\nexit 0\n' > "$scratch/$configuration/$product"
    chmod +x "$scratch/$configuration/$product"
  fi
fi
MOCK

cat > "$FIXTURE/bin/git" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == -C ]]
checkout="$2"
shift 2
[[ "$checkout" == "$TEST_SCRATCH/checkouts/"* ]]
if [[ "$1" == rev-parse ]]; then
  cat "$checkout/.fixture-revision"
else
  [[ "$1" == apply ]]
  patch="${!#}"
  [[ "$patch" == "$TEST_RUNTIME/Patches/"* && -f "$patch" ]]
  printf 'patch|%s|%s\n' "$checkout" "$patch" >> "$TEST_LOG"
  # Simulate fully patched checkouts. Optional overlays require precisely one
  # successful direction; ordinary patches should also take the no-op path.
  [[ "$*" == *--reverse* ]]
fi
MOCK

cat > "$FIXTURE/bin/xcrun" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
printf 'xcrun|%s|%s\n' "$PWD" "$*" >> "$TEST_LOG"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -I|-c|-o)
      flag="$1"
      shift
      [[ "$1" == "$TEST_SCRATCH/"* ]]
      if [[ "$flag" == -o ]]; then touch "$1"; fi
      ;;
  esac
  shift
done
MOCK
chmod +x "$FIXTURE/bin/"* "$RUNTIME/"*.sh
export PATH="$FIXTURE/bin:$PATH"
unset MODEL_RUNNER_BUILD_PACKAGE_ROOT MODEL_RUNNER_DEPENDENCY_PACKAGE_ROOT
unset MODEL_RUNNER_BUILD_CONFIGURATION MODEL_RUNNER_BUILD_PRODUCT MODEL_RUNNER_BUILD_JOBS
unset MODEL_RUNNER_PINNED_MLX MODEL_RUNNER_SCRATCH_PATH SWIFT_BUILD_JOBS
unset MODEL_RUNNER_ENABLE_MLX_CROSS_THREAD_STREAM_OVERLAY

export MODEL_RUNNER_BUILD_PACKAGE_ROOT="$SIBLING"
export MODEL_RUNNER_BUILD_PRODUCT=midnight-studio-worker
export TEST_HOST=Darwin TEST_SCRATCH="$SIBLING/.build"
: > "$TEST_LOG"
# Exercise Darwin dispatch through build.sh, not only build-metal.sh directly.
"$RUNTIME/build.sh" > "$FIXTURE/metal-output"
[[ -x "$SIBLING/.build/release/midnight-studio-worker" ]]
[[ -f "$SIBLING/.build/release/mlx.metallib" && -f "$SIBLING/Package.resolved" ]]
[[ -f "$SIBLING/.build/metal/steel_attention.air" ]]
grep -Fq "swift|$SIBLING|package resolve" "$TEST_LOG"
grep -Fq "patch|$SIBLING/.build/checkouts/" "$TEST_LOG"
grep -Fq "$RUNTIME/Patches/" "$TEST_LOG"
[[ ! -e "$RUNTIME/.build" && ! -e "$RUNTIME/Package.resolved" ]]

# Linux uses its sibling-owned isolated scratch directory for resolution,
# compilation, profile markers, and executable lookup. CPU mode avoids CUDA.
export TEST_HOST=Linux MODEL_RUNNER_SCRATCH_PATH='isolated cache'
export TEST_SCRATCH="$SIBLING/$MODEL_RUNNER_SCRATCH_PATH" SPM_CUDA=0
: > "$TEST_LOG"
"$RUNTIME/build.sh" > "$FIXTURE/linux-output"
[[ -x "$TEST_SCRATCH/release/midnight-studio-worker" ]]
grep -qx 'configuration=release:cpu' "$TEST_SCRATCH/.model-runner-profile"
grep -qx 'configuration=release:cpu' "$TEST_SCRATCH/.model-runner-cache-profile"
grep -Fq "swift|$SIBLING|package --scratch-path $TEST_SCRATCH resolve" "$TEST_LOG"
grep -Fq "patch|$TEST_SCRATCH/checkouts/" "$TEST_LOG"
grep -Fq "$RUNTIME/Patches/" "$TEST_LOG"
[[ ! -e "$RUNTIME/.build" && ! -e "$RUNTIME/Package.resolved" ]]
[[ ! -e "$SIBLING/Scripts" && ! -e "$SIBLING/Patches" ]]

echo 'Sibling package build, resolution, and runtime patch ownership checks passed'
