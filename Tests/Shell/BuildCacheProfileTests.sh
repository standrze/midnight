#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_SCRIPT="$PACKAGE_ROOT/build.sh"
PUBLISHER="$PACKAGE_ROOT/Scripts/release-publisher.sh"
OPTIONAL_PATCH_SCRIPT="$PACKAGE_ROOT/Scripts/optional-dependency-patch.sh"

# shellcheck source=../../Scripts/optional-dependency-patch.sh
source "$OPTIONAL_PATCH_SCRIPT"

grep -Fq 'CACHE_PROFILE_MARKER="$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH/.model-runner-cache-profile"' \
  "$BUILD_SCRIPT"
grep -Fq 'PROFILE_MARKER="$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH/.model-runner-profile"' \
  "$BUILD_SCRIPT"
grep -Fq 'mv -f "$CACHE_PROFILE_TEMP" "$CACHE_PROFILE_MARKER"' "$BUILD_SCRIPT"
grep -Fq 'IFS= read -r PREVIOUS_PROFILE < "$CACHE_PROFILE_MARKER"' "$BUILD_SCRIPT"
grep -Fq 'source "$PACKAGE_ROOT/Scripts/optional-dependency-patch.sh"' "$BUILD_SCRIPT"
grep -Fq \
  ':mlx-cross-thread-stream-overlay=$MLX_SWIFT_CROSS_THREAD_STREAM_OVERLAY_MODE:' \
  "$BUILD_SCRIPT"

unset MODEL_RUNNER_ENABLE_MLX_CROSS_THREAD_STREAM_OVERLAY
default_overlay_mode="$(model_runner_mlx_cross_thread_stream_overlay_mode Linux)"
MODEL_RUNNER_ENABLE_MLX_CROSS_THREAD_STREAM_OVERLAY=1
export MODEL_RUNNER_ENABLE_MLX_CROSS_THREAD_STREAM_OVERLAY
opt_in_overlay_mode="$(model_runner_mlx_cross_thread_stream_overlay_mode Linux)"
MODEL_RUNNER_ENABLE_MLX_CROSS_THREAD_STREAM_OVERLAY=0
opt_out_overlay_mode="$(model_runner_mlx_cross_thread_stream_overlay_mode Linux)"
darwin_overlay_mode="$(model_runner_mlx_cross_thread_stream_overlay_mode Darwin)"
unset MODEL_RUNNER_ENABLE_MLX_CROSS_THREAD_STREAM_OVERLAY

[[ "$default_overlay_mode" == "on" ]]
[[ "$opt_in_overlay_mode" == "on" ]]
[[ "$opt_out_overlay_mode" == "off" ]]
[[ "$darwin_overlay_mode" == "unmanaged" ]]
default_cuda_profile="configuration=release:cuda:sm_89:mlx-cross-thread-stream-overlay=$default_overlay_mode:fixture=true"
opt_out_cuda_profile="configuration=release:cuda:sm_89:mlx-cross-thread-stream-overlay=$opt_out_overlay_mode:fixture=true"
[[ "$default_cuda_profile" != "$opt_out_cuda_profile" ]]

cache_write_line="$(
  grep -nF 'mv -f "$CACHE_PROFILE_TEMP" "$CACHE_PROFILE_MARKER"' "$BUILD_SCRIPT" \
    | cut -d: -f1
)"
build_line="$(
  grep -nF 'swift build "${SWIFT_BUILD_ARGS[@]}" --product "$BUILD_PRODUCT"' "$BUILD_SCRIPT" \
    | head -n 1 | cut -d: -f1
)"
success_write_line="$(
  grep -nF 'printf '\''%s\n'\'' "$BUILD_PROFILE" > "$PROFILE_MARKER"' "$BUILD_SCRIPT" \
    | cut -d: -f1
)"
overlay_resolve_line="$(
  grep -nF 'MLX_SWIFT_CROSS_THREAD_STREAM_OVERLAY_MODE="$(' "$BUILD_SCRIPT" \
    | cut -d: -f1
)"
profile_construct_line="$(
  grep -nF 'BUILD_PROFILE="configuration=$SWIFT_BUILD_CONFIGURATION:cuda:' "$BUILD_SCRIPT" \
    | cut -d: -f1
)"
profile_compare_line="$(
  grep -nF 'elif [[ "$PREVIOUS_PROFILE" != "$BUILD_PROFILE" ]]; then' "$BUILD_SCRIPT" \
    | cut -d: -f1
)"

[[ "$cache_write_line" =~ ^[0-9]+$ ]]
[[ "$build_line" =~ ^[0-9]+$ ]]
[[ "$success_write_line" =~ ^[0-9]+$ ]]
[[ "$overlay_resolve_line" =~ ^[0-9]+$ ]]
[[ "$profile_construct_line" =~ ^[0-9]+$ ]]
[[ "$profile_compare_line" =~ ^[0-9]+$ ]]
(( overlay_resolve_line < profile_construct_line ))
(( profile_construct_line < profile_compare_line ))
(( cache_write_line < build_line ))
(( build_line < success_write_line ))

# Publishing must continue to require the post-build success marker rather
# than accepting the cache-resume marker as evidence of a finished binary.
grep -Fq '.model-runner-profile' "$PUBLISHER"
if grep -Fq '.model-runner-cache-profile' "$PUBLISHER"; then
  echo "Release publisher must not trust the cache-resume marker." >&2
  exit 1
fi

echo "Build cache/success profile separation checks passed"
