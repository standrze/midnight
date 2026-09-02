#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$PACKAGE_ROOT/Scripts/cuda-host-cxx.sh"

TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/model-runner-host-cxx.XXXXXX")"
TEST_ROOT="$(cd "$TEST_ROOT" && pwd -P)"
cleanup() {
  local status=$?
  trap - EXIT
  rm -rf "$TEST_ROOT"
  exit "$status"
}
trap cleanup EXIT

write_clang_fixture() {
  local path="$1"
  local label="$2"
  local major="$3"
  printf '%s\n' \
    '#!/usr/bin/env bash' \
    'if [[ " $* " == *" -dM -E -x c++ /dev/null "* ]]; then' \
    '  echo "#define __clang__ 1"' \
    "  echo '#define __clang_major__ $major'" \
    '  exit 0' \
    'fi' \
    "echo '$label clang version $major.0.0'" > "$path"
  chmod 0755 "$path"
}

PRIMARY_CXX="$TEST_ROOT/clang++-18-primary"
ALIAS_CXX="$TEST_ROOT/clang++-20-alias"
TOO_NEW_CXX="$TEST_ROOT/clang++-21"
TOO_OLD_CXX="$TEST_ROOT/clang++-17"
GXX_CXX="$TEST_ROOT/g++-13"
NONEXECUTABLE_CXX="$TEST_ROOT/clang++-nonexec"
write_clang_fixture "$PRIMARY_CXX" primary 18
write_clang_fixture "$ALIAS_CXX" alias 20
write_clang_fixture "$TOO_NEW_CXX" too-new 21
write_clang_fixture "$TOO_OLD_CXX" too-old 17
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'if [[ " $* " == *" -dM -E -x c++ /dev/null "* ]]; then' \
  '  echo "#define __GNUC__ 13"' \
  '  exit 0' \
  'fi' \
  'echo "g++ (Ubuntu 13.3.0) 13.3.0"' > "$GXX_CXX"
chmod 0755 "$GXX_CXX"
printf '%s\n' '#!/usr/bin/env bash' 'echo nonexec' > "$NONEXECUTABLE_CXX"
chmod 0644 "$NONEXECUTABLE_CXX"

(
  unset CUDAHOSTCXX
  MLX_CUDA_HOST_CXX="$PRIMARY_CXX"
  model_runner_resolve_cuda_host_cxx
  [[ "$CUDA_HOST_CXX_SOURCE" == "MLX_CUDA_HOST_CXX" ]]
  [[ "$CUDA_HOST_CXX_PATH" == "$(readlink -f "$PRIMARY_CXX")" ]]
  [[ "$MLX_CUDA_HOST_CXX" == "$CUDA_HOST_CXX_PATH" ]]
  [[ "$CUDA_HOST_CXX_FAMILY" == "clang" ]]
  [[ "$CUDA_HOST_CXX_MAJOR" == "18" ]]
  [[ -n "$CUDA_HOST_CXX_VERSION_FINGERPRINT" ]]
)

(
  unset MLX_CUDA_HOST_CXX
  CUDAHOSTCXX="$ALIAS_CXX"
  model_runner_resolve_cuda_host_cxx
  [[ "$CUDA_HOST_CXX_SOURCE" == "CUDAHOSTCXX" ]]
  [[ "$MLX_CUDA_HOST_CXX" == "$(readlink -f "$ALIAS_CXX")" ]]
  [[ "$CUDA_HOST_CXX_MAJOR" == "20" ]]
)

(
  MLX_CUDA_HOST_CXX="$PRIMARY_CXX"
  CUDAHOSTCXX="$ALIAS_CXX"
  model_runner_resolve_cuda_host_cxx
  [[ "$CUDA_HOST_CXX_SOURCE" == "MLX_CUDA_HOST_CXX" ]]
  [[ "$CUDA_HOST_CXX_PATH" == "$(readlink -f "$PRIMARY_CXX")" ]]
)

if (
  MLX_CUDA_HOST_CXX="relative/g++"
  unset CUDAHOSTCXX
  model_runner_resolve_cuda_host_cxx
) >/dev/null 2>&1; then
  echo "relative CUDA host compiler unexpectedly passed" >&2
  exit 1
fi

if (
  MLX_CUDA_HOST_CXX=""
  CUDAHOSTCXX="$ALIAS_CXX"
  model_runner_resolve_cuda_host_cxx
) >/dev/null 2>&1; then
  echo "empty higher-priority CUDA host compiler unexpectedly fell back" >&2
  exit 1
fi

if (
  MLX_CUDA_HOST_CXX="$NONEXECUTABLE_CXX"
  unset CUDAHOSTCXX
  model_runner_resolve_cuda_host_cxx
) >/dev/null 2>&1; then
  echo "non-executable CUDA host compiler unexpectedly passed" >&2
  exit 1
fi

if (
  MLX_CUDA_HOST_CXX="$TEST_ROOT/missing-clang++"
  unset CUDAHOSTCXX
  model_runner_resolve_cuda_host_cxx
) >/dev/null 2>&1; then
  echo "missing CUDA host compiler unexpectedly passed" >&2
  exit 1
fi

for unsupported_compiler in "$GXX_CXX" "$TOO_OLD_CXX" "$TOO_NEW_CXX"; do
  if (
    MLX_CUDA_HOST_CXX="$unsupported_compiler"
    unset CUDAHOSTCXX
    model_runner_resolve_cuda_host_cxx
  ) >/dev/null 2>&1; then
    echo "unsupported CUDA host compiler unexpectedly passed: $unsupported_compiler" >&2
    exit 1
  fi
done

if [[ -x /usr/bin/clang++-18 ]]; then
  (
    unset MLX_CUDA_HOST_CXX CUDAHOSTCXX
    model_runner_resolve_cuda_host_cxx
    [[ "$CUDA_HOST_CXX_SOURCE" == "linux-default" ]]
    [[ "$CUDA_HOST_CXX_PATH" == "$(readlink -f /usr/bin/clang++-18)" ]]
    [[ "$CUDA_HOST_CXX_MAJOR" == "18" ]]
  )
fi

PLUGIN_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-cuda-linux.patch"
PREPARE_SCRIPT="$PACKAGE_ROOT/prepare-dependencies.sh"
BUILD_SCRIPT="$PACKAGE_ROOT/build.sh"

grep -Fq 'environment["MLX_CUDA_HOST_CXX"]' "$PLUGIN_PATCH"
grep -Fq 'environment["CUDAHOSTCXX"]' "$PLUGIN_PATCH"
grep -Fq 'configuredCompiler.path.hasPrefix("/")' "$PLUGIN_PATCH"
grep -Fq 'FileManager.default.isExecutableFile' "$PLUGIN_PATCH"
[[ "$(grep -F -- '"--clangpp", hostCXXPath' "$PLUGIN_PATCH" | grep -c '^+')" -eq 2 ]]
grep -Fq 'MLX_SWIFT_DARWIN_EXPECTED_REVISION="72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798"' "$PREPARE_SCRIPT"
grep -Fq 'MLX_SWIFT_LINUX_EXPECTED_REVISION="2d2724006b62855c6c2a71df633baf4ee4ad8a0f"' "$PREPARE_SCRIPT"
grep -Fq 'mlx-swift-cuda-linux.patch' "$PREPARE_SCRIPT"
grep -Fq 'mlx-swift-cuda-generated-header.patch' "$PREPARE_SCRIPT"
grep -Fq 'environment["MLX_CUDA_INCLUDE_PATHS"]' "$PLUGIN_PATCH"
grep -Fq 'MLX_CUDA_INCLUDE_PATHS="$CUTLASS_INCLUDE_DIR:$CUDNN_FRONTEND_INCLUDE_DIR"' "$BUILD_SCRIPT"
grep -Fq 'host-cxx=$CUDA_HOST_CXX_PATH' "$BUILD_SCRIPT"
grep -Fq 'host-cxx-family=$CUDA_HOST_CXX_FAMILY' "$BUILD_SCRIPT"
grep -Fq 'host-cxx-major=$CUDA_HOST_CXX_MAJOR' "$BUILD_SCRIPT"
grep -Fq 'host-cxx-version=$CUDA_HOST_CXX_VERSION_FINGERPRINT' "$BUILD_SCRIPT"

# Reproduce the second-stage failure independently of CUDA/glibc and prove why
# the generated source must contain the compatible (typedef-expanded) forms.
if command -v clang++ >/dev/null 2>&1; then
  RAW_FLOAT_SOURCE="$TEST_ROOT/gcc13-generated.cpp"
  COMPAT_FLOAT_SOURCE="$TEST_ROOT/clang18-generated.cpp"
  printf '%s\n' \
    'extern _Float32 f32;' \
    'extern _Float64 f64;' \
    'extern _Float128 f128;' \
    'extern _Float32x f32x;' \
    'extern _Float64x f64x;' > "$RAW_FLOAT_SOURCE"
  printf '%s\n' \
    'extern float f32;' \
    'extern double f64;' \
    'extern __float128 f128;' \
    'extern double f32x;' \
    'extern long double f64x;' > "$COMPAT_FLOAT_SOURCE"

  if clang++ --target=x86_64-unknown-linux-gnu -std=gnu++20 -fsyntax-only \
    "$RAW_FLOAT_SOURCE" >/dev/null 2>&1; then
    echo "Clang unexpectedly accepted GCC 13 native _FloatN output" >&2
    exit 1
  fi
  clang++ --target=x86_64-unknown-linux-gnu -std=gnu++20 -fsyntax-only \
    "$COMPAT_FLOAT_SOURCE"
fi

echo "CUDA host compiler checks passed"
