#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd)"

case "$(uname -s)" in
  Darwin)
    exec "$PACKAGE_ROOT/build-metal.sh"
    ;;
  Linux)
    cd "$PACKAGE_ROOT"
    source "$PACKAGE_ROOT/Scripts/swiftpm-scratch-path.sh"
    source "$PACKAGE_ROOT/Scripts/optional-dependency-patch.sh"
    source "$PACKAGE_ROOT/Scripts/release-publisher.sh"
    source "$PACKAGE_ROOT/Scripts/cuda-runtime-environment.sh"
    model_runner_configure_swiftpm_scratch "$PACKAGE_ROOT" Linux
    SWIFT_BUILD_JOBS="${SWIFT_BUILD_JOBS:-2}"
    if [[ ! "$SWIFT_BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
      echo "SWIFT_BUILD_JOBS must be a positive integer." >&2
      exit 2
    fi
    BUILD_PRODUCT="${MODEL_RUNNER_BUILD_PRODUCT:-midnight}"
    case "$BUILD_PRODUCT" in
      midnight|model-runner-quantize|model-runner-laguna-quantize|model-runner-laguna-q4r8-rescore|model-runner-laguna-q4r8-verify|model-runner-runtime-bench|model-runner-quality-bench|model-runner-q4-scale-search-audit|model-runner-scale-plan|model-runner-metal-quant-bench) ;;
      *)
        echo "Unsupported MODEL_RUNNER_BUILD_PRODUCT: $BUILD_PRODUCT" >&2
        exit 2
        ;;
    esac
    # Linux/CUDA products are always optimized release builds so executable
    # paths and performance do not depend on SwiftPM's debug default.
    SWIFT_BUILD_CONFIGURATION="release"
    SWIFT_BUILD_ARGS=(
      --configuration "$SWIFT_BUILD_CONFIGURATION"
      --jobs "$SWIFT_BUILD_JOBS"
      "${MODEL_RUNNER_SWIFT_BUILD_SCRATCH_ARGS[@]}"
    )
    export SPM_CUDA="${SPM_CUDA:-1}"
    CUDA_BUILD="$SPM_CUDA"
    FORCE_CLEAN_CUDA_CACHE=0
    PUBLISH_RTX4090_RELEASE=0
    if [[ "$CUDA_BUILD" != "0" && "$CUDA_BUILD" != "1" ]]; then
      echo "SPM_CUDA must be 0 (CPU-only) or 1 (CUDA)." >&2
      exit 2
    fi
    if [[ "$CUDA_BUILD" == "0" ]]; then
      BUILD_PROFILE="configuration=$SWIFT_BUILD_CONFIGURATION:cpu"
    else
      MLX_SWIFT_CROSS_THREAD_STREAM_OVERLAY_MODE="$(
        model_runner_mlx_cross_thread_stream_overlay_mode Linux
      )"
      CUDA_PROFILE="${MLX_CUDA_PROFILE:-native}"
      case "$CUDA_PROFILE" in
        native)
          DEFAULT_CUDA_ARCH="native"
          ;;
        dgx-spark|spark)
          DEFAULT_CUDA_ARCH="sm_121"
          ;;
        rtx-4090|4090)
          DEFAULT_CUDA_ARCH="sm_89"
          if [[ "$BUILD_PRODUCT" == "midnight" ]]; then
            PUBLISH_RTX4090_RELEASE=1
          fi
          ;;
        *)
          echo "Unknown CUDA profile: $CUDA_PROFILE" >&2
          echo "Choose native, dgx-spark, or rtx-4090." >&2
          exit 2
          ;;
      esac
      export CUDA_ARCH="${CUDA_ARCH:-$DEFAULT_CUDA_ARCH}"

      source "$PACKAGE_ROOT/Scripts/cuda-toolkit.sh"
      model_runner_resolve_cuda_toolkit /usr/local/cuda
      if [[ ! -d /usr/local/cuda/include || ! -d /usr/local/cuda/lib64 ]]; then
        echo "MLX Swift expects CUDA headers and libraries under /usr/local/cuda." >&2
        exit 1
      fi

      # Fail closed on exact dependency releases. Header existence alone can
      # silently select Ubuntu's older cudnn-frontend or a stale CUTLASS tree.
      source "$PACKAGE_ROOT/Scripts/cuda-dependency-checks.sh"
      model_runner_resolve_cuda_dependencies
      export CPATH="$CUTLASS_INCLUDE_DIR:$CUDNN_FRONTEND_INCLUDE_DIR${CPATH:+:$CPATH}"
      export MLX_CUDA_INCLUDE_PATHS="$CUTLASS_INCLUDE_DIR:$CUDNN_FRONTEND_INCLUDE_DIR"

      # CUDA 13 rejects the Clang 21 bundled with the Swift 6.3 toolchain.
      # Resolve one supported Clang 18-20 host compiler and pass the same
      # absolute path to both encuda compile and link. Using GCC 13 here is
      # also invalid: its nvcc-generated host C++ contains glibc _FloatN types
      # which Swift Clang 21 cannot parse in the second compilation phase.
      source "$PACKAGE_ROOT/Scripts/cuda-host-cxx.sh"
      model_runner_resolve_cuda_host_cxx

      NVCC_VERSION_FINGERPRINT="$("$NVCC_PATH" --version | cksum | awk '{print $1 "-" $2}')"
      BUILD_PROFILE="configuration=$SWIFT_BUILD_CONFIGURATION:cuda:$CUDA_ARCH:mlx-cross-thread-stream-overlay=$MLX_SWIFT_CROSS_THREAD_STREAM_OVERLAY_MODE:nvcc=$NVCC_PATH:nvcc-version=$NVCC_VERSION_FINGERPRINT:toolkit=$CUDA_TOOLKIT_PATH:host-cxx-source=$CUDA_HOST_CXX_SOURCE:host-cxx=$CUDA_HOST_CXX_PATH:host-cxx-family=$CUDA_HOST_CXX_FAMILY:host-cxx-major=$CUDA_HOST_CXX_MAJOR:host-cxx-version=$CUDA_HOST_CXX_VERSION_FINGERPRINT:cudnn-root=$CUDNN_FRONTEND_ROOT:cudnn-include=$CUDNN_FRONTEND_INCLUDE_DIR:cudnn-version=$CUDNN_FRONTEND_VERSION:cudnn-evidence=$CUDNN_FRONTEND_VERSION_EVIDENCE_FINGERPRINT:cudnn-headers=$CUDNN_FRONTEND_HEADERS_FINGERPRINT:cutlass-root=$CUTLASS_ROOT:cutlass-include=$CUTLASS_INCLUDE_DIR:cutlass-version=$CUTLASS_VERSION:cutlass-headers=$CUTLASS_HEADERS_FINGERPRINT"

      # The MLX Swift build plugin does not record the GPU selected by
      # CUDA_ARCH=native in its incremental-build inputs. A fresh build is the
      # only reliable choice when the visible GPU may have changed.
      if [[ "$CUDA_ARCH" == "native" ]]; then
        FORCE_CLEAN_CUDA_CACHE=1
      fi

      if [[ "$CUDA_ARCH" != "native" ]]; then
        NVCC_GPU_CODES="$("$NVCC_PATH" --list-gpu-code)"
        if [[ "$NVCC_GPU_CODES" != *"$CUDA_ARCH"* ]]; then
          echo "This CUDA toolkit cannot compile $CUDA_ARCH." >&2
          echo "DGX Spark (sm_121) requires CUDA 13; RTX 4090 uses sm_89." >&2
          exit 1
        fi
      fi
    fi

    "$PACKAGE_ROOT/prepare-dependencies.sh"

    PROFILE_MARKER="$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH/.model-runner-profile"
    CACHE_PROFILE_MARKER="$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH/.model-runner-cache-profile"
    PREVIOUS_PROFILE=""
    if [[ -f "$CACHE_PROFILE_MARKER" ]]; then
      IFS= read -r PREVIOUS_PROFILE < "$CACHE_PROFILE_MARKER" || true
    elif [[ -f "$PROFILE_MARKER" ]]; then
      # Migrate a successful cache produced before the cache/success markers
      # were split. The success marker remains authoritative for publishing.
      IFS= read -r PREVIOUS_PROFILE < "$PROFILE_MARKER" || true
    fi
    if [[ "$FORCE_CLEAN_CUDA_CACHE" == "1" ]]; then
      echo "Preparing a fresh build cache for CUDA_ARCH=native"
      swift package "${MODEL_RUNNER_SWIFT_PACKAGE_SCRATCH_ARGS[@]}" clean
    elif [[ "$PREVIOUS_PROFILE" != "$BUILD_PROFILE" ]]; then
      echo "Preparing build cache for $BUILD_PROFILE"
      swift package "${MODEL_RUNNER_SWIFT_PACKAGE_SCRATCH_ARGS[@]}" clean
    fi

    # Record what the cache contains before compiling. An interrupted build is
    # safe to resume when this exact profile still matches, while the separate
    # success marker below remains absent until the executable is complete.
    mkdir -p "$(dirname "$CACHE_PROFILE_MARKER")"
    CACHE_PROFILE_TEMP="$CACHE_PROFILE_MARKER.tmp.$$"
    printf '%s\n' "$BUILD_PROFILE" > "$CACHE_PROFILE_TEMP"
    mv -f "$CACHE_PROFILE_TEMP" "$CACHE_PROFILE_MARKER"

    if [[ "$CUDA_BUILD" != "0" ]]; then
      echo "Building MLX CUDA configuration=$SWIFT_BUILD_CONFIGURATION profile=$CUDA_PROFILE arch=$CUDA_ARCH jobs=$SWIFT_BUILD_JOBS"
    fi
    swift build "${SWIFT_BUILD_ARGS[@]}" --product "$BUILD_PRODUCT"
    mkdir -p "$(dirname "$PROFILE_MARKER")"
    printf '%s\n' "$BUILD_PROFILE" > "$PROFILE_MARKER"

    BIN_DIR="$(swift build "${SWIFT_BUILD_ARGS[@]}" --show-bin-path)"
    BUILT_RUNNER="$BIN_DIR/$BUILD_PRODUCT"
    if [[ ! -f "$BUILT_RUNNER" || ! -x "$BUILT_RUNNER" || -L "$BUILT_RUNNER" ]]; then
      echo "SwiftPM did not produce a regular release executable at $BUILT_RUNNER" >&2
      exit 1
    fi
    if [[ "$BUILD_PRODUCT" == "midnight" \
      && "$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH" == "$PACKAGE_ROOT/.build" ]]; then
      DEFAULT_RELEASE_RUNNER="$PACKAGE_ROOT/.build/release/midnight"
      if [[ ! -x "$DEFAULT_RELEASE_RUNNER" || ! "$DEFAULT_RELEASE_RUNNER" -ef "$BUILT_RUNNER" ]]; then
        echo "SwiftPM release compatibility path is missing or points at a different artifact:" >&2
        echo "$DEFAULT_RELEASE_RUNNER" >&2
        exit 1
      fi
    elif [[ "$BUILD_PRODUCT" == "midnight" \
      && "$PUBLISH_RTX4090_RELEASE" == "1" ]]; then
      # MLX compiles some CUDA kernels after startup and discovers CUTLASS/CuTe
      # relative to the published executable. Publish only the exact header
      # trees already validated for this build, before exposing the binary.
      model_runner_publish_cuda_runtime_headers "$PACKAGE_ROOT"
      model_runner_verify_cuda_runtime_headers "$PACKAGE_ROOT"
      model_runner_publish_rtx4090_release \
        "$PACKAGE_ROOT" \
        "$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH" \
        "$BUILT_RUNNER" \
        "$BUILD_PROFILE"
      model_runner_verify_cuda_runtime_headers_for_runner \
        "$PACKAGE_ROOT" "$MODEL_RUNNER_RTX4090_STABLE_PATH"
      echo "Published verified RTX 4090 release: $MODEL_RUNNER_RTX4090_STABLE_PATH"
      echo "Compatibility path: $MODEL_RUNNER_RTX4090_COMPAT_PATH"
      echo "Provenance manifest: $MODEL_RUNNER_RTX4090_MANIFEST_PATH"
    fi
    if [[ "$CUDA_BUILD" == "0" ]]; then
      echo "Built $BUILT_RUNNER with the MLX CPU backend"
    else
      echo "Built $BUILT_RUNNER with the MLX CUDA backend"
    fi
    ;;
  *)
    echo "Unsupported host: $(uname -s). Use macOS for Metal or Linux for CUDA." >&2
    exit 1
    ;;
esac
