#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PATCH_FILE="$PACKAGE_ROOT/Patches/mlx-cuda-half-fmod.patch"
PREPARE_SCRIPT="$PACKAGE_ROOT/prepare-dependencies.sh"

grep -Fq 'MLX_SOURCE_LINUX_EXPECTED_REVISION="7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247"' \
  "$PREPARE_SCRIPT"
grep -Fq 'mlx-cuda-half-fmod.patch' "$PREPARE_SCRIPT"
grep -Fq '"mlx source" "$MLX_SOURCE_CHECKOUT" "$MLX_SOURCE_EXPECTED_REVISION"' \
  "$PREPARE_SCRIPT"
grep -Fq '"mlx CUDA half fmod" "$MLX_SOURCE_CHECKOUT" "$MLX_SOURCE_PATCH"' \
  "$PREPARE_SCRIPT"

require_count() {
  local expected="$1"
  local pattern="$2"
  local file="$3"
  local actual
  actual="$(grep -F -c "$pattern" "$file" || true)"
  if [[ "$actual" -ne "$expected" ]]; then
    echo "Expected $expected occurrence(s) of '$pattern' in $file; found $actual" >&2
    exit 1
  fi
}

require_added_count() {
  local expected="$1"
  local pattern="$2"
  local actual
  actual="$(grep -F "$pattern" "$PATCH_FILE" | grep -c '^+' || true)"
  if [[ "$actual" -ne "$expected" ]]; then
    echo "Expected $expected added occurrence(s) of '$pattern'; found $actual" >&2
    exit 1
  fi
}

require_added_count 1 'if constexpr (cuda::std::is_same_v<T, __half>)'
require_added_count 1 'cuda::std::is_same_v<T, __nv_bfloat16>'
require_added_count 1 'cuda::std::fmod(__half2float(x), __half2float(y))'
require_added_count 1 'cuda::std::fmod(__bfloat162float(x), __bfloat162float(y))'

TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/model-runner-half-fmod.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT
  rm -rf "$TEST_ROOT"
  exit "$status"
}
trap cleanup EXIT

SOURCE_DIR="$TEST_ROOT/mlx/backend/cuda/device"
mkdir -p "$SOURCE_DIR"
cat > "$SOURCE_DIR/binary_ops.cuh" <<'EOF'
struct Remainder {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (cuda::std::is_integral_v<T>) {
      if constexpr (cuda::std::is_signed_v<T>) {
        auto r = x % y;
        if (r != 0 && (r < 0 != y < 0)) {
          r += y;
        }
        return r;
      } else {
        return x % y;
      }
    } else if constexpr (is_complex_v<T>) {
      return x % y;
    } else {
      T r = cuda::std::fmod(x, y);
      if (r != 0 && (r < 0 != y < 0)) {
        r = r + y;
      }
      return r;
    }
  }
};
EOF

git -C "$TEST_ROOT" apply --check "$PATCH_FILE"
git -C "$TEST_ROOT" apply "$PATCH_FILE"
git -C "$TEST_ROOT" apply --reverse --check "$PATCH_FILE"

# This is the same reverse-check used by prepare-dependencies.sh. Once true,
# a second preparation pass reports the patch as applied instead of applying it
# a second time.
if git -C "$TEST_ROOT" apply --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "fmod patch unexpectedly remained forward-applicable" >&2
  exit 1
fi
require_count 1 '__half2float(x)' "$SOURCE_DIR/binary_ops.cuh"
require_count 1 '__bfloat162float(x)' "$SOURCE_DIR/binary_ops.cuh"
require_count 0 'T r = cuda::std::fmod(x, y);' "$SOURCE_DIR/binary_ops.cuh"

echo "CUDA half/bfloat16 fmod compatibility patch checks passed"
