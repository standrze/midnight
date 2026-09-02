#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PATCH_FILE="$PACKAGE_ROOT/Patches/mlx-swift-cross-thread-stream.patch"
PREPARE_SCRIPT="$PACKAGE_ROOT/prepare-dependencies.sh"
OPTIONAL_PATCH_SCRIPT="$PACKAGE_ROOT/Scripts/optional-dependency-patch.sh"
OPT_IN_VARIABLE="MODEL_RUNNER_ENABLE_MLX_CROSS_THREAD_STREAM_OVERLAY"

grep -Fq \
  'MLX_SWIFT_LINUX_EXPECTED_REVISION="2d2724006b62855c6c2a71df633baf4ee4ad8a0f"' \
  "$PREPARE_SCRIPT"
grep -Fq \
  'MLX_SWIFT_DARWIN_EXPECTED_REVISION="72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798"' \
  "$PREPARE_SCRIPT"
grep -Fq \
  'MLX_SOURCE_DARWIN_EXPECTED_REVISION="1f8e74e3f12f31365464a6867c6579f0e9b29d85"' \
  "$PREPARE_SCRIPT"
grep -Fq \
  'MLX_C_SOURCE_DARWIN_EXPECTED_REVISION="c74db5307cc8ce122f48d97ef951b30578674e7f"' \
  "$PREPARE_SCRIPT"
grep -Fq 'mlx-swift-cross-thread-stream.patch' "$PREPARE_SCRIPT"
grep -Fq \
  'model_runner_reconcile_optional_dependency_patch \' \
  "$PREPARE_SCRIPT"
grep -Fq '"$MLX_SWIFT_CROSS_THREAD_STREAM_OVERLAY_MODE"' "$PREPARE_SCRIPT"
grep -Fq "$OPT_IN_VARIABLE" "$OPTIONAL_PATCH_SCRIPT"
VERIFY_MLX_SWIFT_LINE="$(
  grep -n -F '"mlx-swift" "$MLX_SWIFT_CHECKOUT" "$MLX_SWIFT_EXPECTED_REVISION"' \
    "$PREPARE_SCRIPT" | cut -d: -f1
)"
RECONCILE_OVERLAY_LINE="$(
  grep -n -F 'model_runner_reconcile_optional_dependency_patch \' \
    "$PREPARE_SCRIPT" | cut -d: -f1
)"
if [[ -z "$VERIFY_MLX_SWIFT_LINE" \
  || -z "$RECONCILE_OVERLAY_LINE" \
  || "$VERIFY_MLX_SWIFT_LINE" -ge "$RECONCILE_OVERLAY_LINE" ]]; then
  echo "Pinned mlx-swift revision must be verified before overlay reconciliation" >&2
  exit 1
fi
if ! awk '
  /if \[\[ "\$APPLY_LINUX_DEPENDENCY_PATCHES" == "1" \]\]; then/ {
    in_linux_block = 1
  }
  in_linux_block && /model_runner_reconcile_optional_dependency_patch/ { found = 1 }
  in_linux_block && /^fi$/ { in_linux_block = 0 }
  END { exit(found ? 0 : 1) }
' "$PREPARE_SCRIPT"; then
  echo "Cross-thread stream overlay reconciliation must remain Linux-only" >&2
  exit 1
fi

# The overlay must restore the known Linux MLX bridge without altering the
# package manifest that selects the synchronized Darwin backend.
grep -Fq \
  'diff --git a/Source/Cmlx/mlx-swift-stream.cpp b/Source/Cmlx/mlx-swift-stream.cpp' \
  "$PATCH_FILE"
grep -Fq \
  'mlx::core::new_thread_unsafe_stream(mlx_device_get_(dev))' \
  "$PATCH_FILE"
grep -Fq \
  'public static let gpu = Stream(newThreadUnsafeStream(MLX_GPU))' \
  "$PATCH_FILE"
grep -Fq \
  'public static let cpu = Stream(newThreadUnsafeStream(MLX_CPU))' \
  "$PATCH_FILE"
if grep -Fq 'diff --git a/Package.swift' "$PATCH_FILE"; then
  echo "Cross-thread stream patch must not rewrite the dependency manifest" >&2
  exit 1
fi
if grep -Fq 'Source/Cmlx/mlx-conditional/mlx-swift-stream.cpp' "$PATCH_FILE"; then
  echo "Linux excludes mlx-conditional; the bridge must be in the Cmlx target root" >&2
  exit 1
fi

TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/model-runner-cross-thread-stream.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT
  rm -rf "$TEST_ROOT"
  exit "$status"
}
trap cleanup EXIT

# shellcheck source=../../Scripts/optional-dependency-patch.sh
source "$OPTIONAL_PATCH_SCRIPT"

unset MODEL_RUNNER_ENABLE_MLX_CROSS_THREAD_STREAM_OVERLAY
[[ "$(model_runner_mlx_cross_thread_stream_overlay_mode Linux)" == "on" ]]
MODEL_RUNNER_ENABLE_MLX_CROSS_THREAD_STREAM_OVERLAY=1
export MODEL_RUNNER_ENABLE_MLX_CROSS_THREAD_STREAM_OVERLAY
[[ "$(model_runner_mlx_cross_thread_stream_overlay_mode Linux)" == "on" ]]
MODEL_RUNNER_ENABLE_MLX_CROSS_THREAD_STREAM_OVERLAY=0
[[ "$(model_runner_mlx_cross_thread_stream_overlay_mode Linux)" == "off" ]]
[[ "$(model_runner_mlx_cross_thread_stream_overlay_mode Darwin)" == "unmanaged" ]]
MODEL_RUNNER_ENABLE_MLX_CROSS_THREAD_STREAM_OVERLAY=true
if model_runner_mlx_cross_thread_stream_overlay_mode Linux >/dev/null 2>&1; then
  echo "Cross-thread stream overlay accepted a non-1 opt-in value" >&2
  exit 1
fi
unset MODEL_RUNNER_ENABLE_MLX_CROSS_THREAD_STREAM_OVERLAY

mkdir -p "$TEST_ROOT/Source/Cmlx/include" "$TEST_ROOT/Source/MLX"
cat > "$TEST_ROOT/Source/Cmlx/include/mlx.h" <<'EOF'
#define REQUIRED_CUDA_PATCH_PRESENT 1
#include "mlx/c/transforms_impl.h"
#include "mlx/c/linalg.h"
#include "mlx/c/fast.h"
EOF
cat > "$TEST_ROOT/Source/MLX/Stream.swift" <<'EOF'
public final class Stream: @unchecked Sendable, Equatable {

    let ctx: mlx_stream

    public static let gpu = Stream(mlx_default_gpu_stream_new())
    public static let cpu = Stream(mlx_default_cpu_stream_new())

    @TaskLocal static var defaultStream: Stream?

    public static func withNewDefaultStream<R>(
        device: Device? = nil, _ body: () async throws -> R
    ) async rethrows -> R {
        let device = device ?? Device.defaultDevice()
        return try await $defaultStream.withValue(Stream(device), operation: body)
    }

    init(_ ctx: mlx_stream) {
        self.ctx = ctx
    }

    /// Default stream on the default device.
    public init() {
        let device = Device.defaultDevice()
        var ctx = mlx_stream_new()
        mlx_get_default_stream(&ctx, device.ctx)
        self.ctx = ctx
    }

    @available(*, deprecated, message: "use init(Device) -- index not supported")
    public init(index: Int32, _ device: Device) {
        self.ctx = evalLock.withLock {
            mlx_stream_new_device(device.ctx)
        }
    }

    /// New stream on the given device.
    ///
    /// See also ``withNewDefaultStream(device:_:)-5bwc3``
    public init(_ device: Device) {
        self.ctx = evalLock.withLock {
            mlx_stream_new_device(device.ctx)
        }
    }

    deinit {
        _ = evalLock.withLock {
            mlx_stream_free(ctx)
        }
    }
}
EOF

cat > "$TEST_ROOT/Package.swift" <<'EOF'
// fixture manifest: must not be changed by the optional overlay
EOF
cat > "$TEST_ROOT/Package.resolved" <<'EOF'
{"fixture":"must remain unchanged"}
EOF
ORIGINAL_MLX_HEADER="$(cat "$TEST_ROOT/Source/Cmlx/include/mlx.h")"
ORIGINAL_STREAM_SWIFT="$(cat "$TEST_ROOT/Source/MLX/Stream.swift")"
ORIGINAL_PACKAGE_SWIFT="$(cat "$TEST_ROOT/Package.swift")"
ORIGINAL_PACKAGE_RESOLVED="$(cat "$TEST_ROOT/Package.resolved")"

# Default-off is an idempotent no-op when the overlay is absent.
model_runner_reconcile_optional_dependency_patch \
  "test cross-thread stream" "$TEST_ROOT" "$PATCH_FILE" off
[[ "$(cat "$TEST_ROOT/Source/Cmlx/include/mlx.h")" == "$ORIGINAL_MLX_HEADER" ]]
[[ "$(cat "$TEST_ROOT/Source/MLX/Stream.swift")" == "$ORIGINAL_STREAM_SWIFT" ]]

# Exact opt-in applies the overlay and a repeated preparation is idempotent.
model_runner_reconcile_optional_dependency_patch \
  "test cross-thread stream" "$TEST_ROOT" "$PATCH_FILE" on
model_runner_reconcile_optional_dependency_patch \
  "test cross-thread stream" "$TEST_ROOT" "$PATCH_FILE" on
git -C "$TEST_ROOT" apply --reverse --check "$PATCH_FILE"

if git -C "$TEST_ROOT" apply --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "Cross-thread stream patch unexpectedly remained forward-applicable" >&2
  exit 1
fi
[[ -f "$TEST_ROOT/Source/Cmlx/mlx-swift-stream.cpp" ]]
[[ -f "$TEST_ROOT/Source/Cmlx/include/mlx-swift-stream.h" ]]
[[ ! -e "$TEST_ROOT/Source/Cmlx/mlx-conditional/mlx-swift-stream.cpp" ]]
[[ "$(grep -F -c '#include "mlx-swift-stream.h"' \
  "$TEST_ROOT/Source/Cmlx/include/mlx.h")" -eq 1 ]]
[[ "$(grep -F -c 'mlx_default_gpu_stream_new()' \
  "$TEST_ROOT/Source/MLX/Stream.swift" || true)" -eq 0 ]]
[[ "$(grep -F -c 'public static let gpu = Stream(newThreadUnsafeStream(MLX_GPU))' \
  "$TEST_ROOT/Source/MLX/Stream.swift")" -eq 1 ]]
[[ "$(grep -F -c 'self.ctx = Stream.newThreadUnsafeStream(device)' \
  "$TEST_ROOT/Source/MLX/Stream.swift")" -eq 2 ]]

# Darwin is explicitly unmanaged: even opt-in does not reconcile its checkout.
MODEL_RUNNER_ENABLE_MLX_CROSS_THREAD_STREAM_OVERLAY=1
export MODEL_RUNNER_ENABLE_MLX_CROSS_THREAD_STREAM_OVERLAY
DARWIN_MODE="$(model_runner_mlx_cross_thread_stream_overlay_mode Darwin)"
if [[ "$DARWIN_MODE" != "unmanaged" ]]; then
  model_runner_reconcile_optional_dependency_patch \
    "test cross-thread stream" "$TEST_ROOT" "$PATCH_FILE" "$DARWIN_MODE"
fi
git -C "$TEST_ROOT" apply --reverse --check "$PATCH_FILE"
unset MODEL_RUNNER_ENABLE_MLX_CROSS_THREAD_STREAM_OVERLAY

# Returning to the default cleanly removes only this exact overlay. Repeating
# the default-off preparation remains a no-op and manifests stay untouched.
printf '%s\n' "unrelated dependency state" > "$TEST_ROOT/unrelated-change.txt"
model_runner_reconcile_optional_dependency_patch \
  "test cross-thread stream" "$TEST_ROOT" "$PATCH_FILE" off
model_runner_reconcile_optional_dependency_patch \
  "test cross-thread stream" "$TEST_ROOT" "$PATCH_FILE" off
[[ "$(cat "$TEST_ROOT/Source/Cmlx/include/mlx.h")" == "$ORIGINAL_MLX_HEADER" ]]
[[ "$(cat "$TEST_ROOT/Source/MLX/Stream.swift")" == "$ORIGINAL_STREAM_SWIFT" ]]
[[ ! -e "$TEST_ROOT/Source/Cmlx/mlx-swift-stream.cpp" ]]
[[ ! -e "$TEST_ROOT/Source/Cmlx/include/mlx-swift-stream.h" ]]
[[ "$(cat "$TEST_ROOT/Package.swift")" == "$ORIGINAL_PACKAGE_SWIFT" ]]
[[ "$(cat "$TEST_ROOT/Package.resolved")" == "$ORIGINAL_PACKAGE_RESOLVED" ]]
[[ "$(cat "$TEST_ROOT/unrelated-change.txt")" == "unrelated dependency state" ]]

# A partial/drifted applied overlay is neither cleanly forward- nor
# reverse-applicable. Default-off must fail closed instead of guessing.
model_runner_reconcile_optional_dependency_patch \
  "test cross-thread stream" "$TEST_ROOT" "$PATCH_FILE" on
printf '%s\n' "drift" >> "$TEST_ROOT/Source/Cmlx/mlx-swift-stream.cpp"
if model_runner_reconcile_optional_dependency_patch \
  "test cross-thread stream" "$TEST_ROOT" "$PATCH_FILE" off \
  >/dev/null 2>&1; then
  echo "Cross-thread stream overlay removal accepted ambiguous patch drift" >&2
  exit 1
fi

echo "mlx-swift cross-thread stream default/reversal checks passed"
