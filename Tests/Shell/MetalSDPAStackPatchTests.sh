#!/usr/bin/env bash
set -euo pipefail

SDPA_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SDPA_SWIFT="$SDPA_ROOT/.build/checkouts/mlx-swift"
SDPA_CORE="$SDPA_SWIFT/Source/Cmlx/mlx"
SDPA_TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/midnight-sdpa-stack-test.XXXXXX")"
trap 'rm -rf "$SDPA_TEST_ROOT"' EXIT
SDPA_CORE_REVISION=1f8e74e3f12f31365464a6867c6579f0e9b29d85
SDPA_SWIFT_REVISION=72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798

mkdir -p "$SDPA_TEST_ROOT/baseline/Source/Cmlx/mlx"
git -C "$SDPA_CORE" archive "$SDPA_CORE_REVISION" \
  mlx/backend/metal/scaled_dot_product_attention.cpp \
  mlx/backend/metal/kernels/scaled_dot_product_attention.metal \
  mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h \
  | tar -xf - -C "$SDPA_TEST_ROOT/baseline/Source/Cmlx/mlx"
git -C "$SDPA_SWIFT" archive "$SDPA_SWIFT_REVISION" \
  Source/Cmlx/mlx-generated/metal/scaled_dot_product_attention.metal \
  Source/Cmlx/mlx-generated/metal/steel/attn/kernels/steel_attention_nax.h \
  Source/Cmlx/mlx-generated/steel_attention_nax.cpp \
  | tar -xf - -C "$SDPA_TEST_ROOT/baseline"

# Fixtures contain exactly the pinned files but need not copy repository metadata.
# Only revision lookup is mocked; all patch checks/applications use real git.
git() {
  if [[ "$#" -eq 4 && "$1" == -C && "$3" == rev-parse && "$4" == HEAD && "$2" == "$SDPA_TEST_ROOT/"* ]]; then
    case "$2" in
      */Source/Cmlx/mlx) echo "$SDPA_CORE_REVISION" ;;
      *) echo "$SDPA_SWIFT_REVISION" ;;
    esac
  else
    command git "$@"
  fi
}
source "$SDPA_ROOT/Scripts/metal-sdpa-d512-patch.sh"

for sdpa_prefix in clean d512 complete; do
  sdpa_fixture="$SDPA_TEST_ROOT/$sdpa_prefix"
  cp -R "$SDPA_TEST_ROOT/baseline" "$sdpa_fixture"
  if [[ "$sdpa_prefix" != clean ]]; then
    git -C "$sdpa_fixture/Source/Cmlx/mlx" apply "$SDPA_ROOT/Patches/mlx-metal-sdpa-d512-decode.patch"
    git -C "$sdpa_fixture" apply "$SDPA_ROOT/Patches/mlx-swift-metal-sdpa-d512-generated.patch"
  fi
  if [[ "$sdpa_prefix" == complete ]]; then
    git -C "$sdpa_fixture/Source/Cmlx/mlx" apply "$SDPA_ROOT/Patches/mlx-metal-sdpa-d256-mask-bounds.patch"
    git -C "$sdpa_fixture" apply "$SDPA_ROOT/Patches/mlx-swift-metal-sdpa-d256-mask-bounds-generated.patch"
  fi
  model_runner_prepare_metal_sdpa_d512 Darwin "$SDPA_ROOT" "$sdpa_fixture" >/dev/null
  model_runner_prepare_metal_sdpa_d512 Darwin "$SDPA_ROOT" "$sdpa_fixture" >/dev/null
done
diff -qr "$SDPA_TEST_ROOT/clean" "$SDPA_TEST_ROOT/d512" >/dev/null
diff -qr "$SDPA_TEST_ROOT/clean" "$SDPA_TEST_ROOT/complete" >/dev/null

# SwiftPM can make source files read-only. Both overlays target the host
# dispatch file, so staging must neither overwrite a read-only private copy
# nor change checkout permissions just to complete its preflight.
for sdpa_prefix in clean d512 complete; do
  sdpa_fixture="$SDPA_TEST_ROOT/readonly-$sdpa_prefix"
  cp -R "$SDPA_TEST_ROOT/baseline" "$sdpa_fixture"
  if [[ "$sdpa_prefix" != clean ]]; then
    git -C "$sdpa_fixture/Source/Cmlx/mlx" apply "$SDPA_ROOT/Patches/mlx-metal-sdpa-d512-decode.patch"
    git -C "$sdpa_fixture" apply "$SDPA_ROOT/Patches/mlx-swift-metal-sdpa-d512-generated.patch"
  fi
  if [[ "$sdpa_prefix" == complete ]]; then
    git -C "$sdpa_fixture/Source/Cmlx/mlx" apply "$SDPA_ROOT/Patches/mlx-metal-sdpa-d256-mask-bounds.patch"
    git -C "$sdpa_fixture" apply "$SDPA_ROOT/Patches/mlx-swift-metal-sdpa-d256-mask-bounds-generated.patch"
  fi
  python3 - "$sdpa_fixture" <<'PY'
from pathlib import Path
import sys
for path in Path(sys.argv[1]).rglob('*'):
    if path.is_file():
        path.chmod(0o444)
PY
  model_runner_prepare_metal_sdpa_d512 Darwin "$SDPA_ROOT" "$sdpa_fixture" >/dev/null
  diff -qr "$SDPA_TEST_ROOT/complete" "$sdpa_fixture" >/dev/null
  # Applying a pending patch may replace its files. Once the stack is applied,
  # a read-only idempotent invocation must preserve both bytes and modes.
  python3 - "$sdpa_fixture" "$SDPA_TEST_ROOT/readonly-before.json" <<'PY'
from pathlib import Path
import hashlib, json, stat, sys
root = Path(sys.argv[1])
snapshot = {}
for path in root.rglob('*'):
    if path.is_file():
        path.chmod(0o444)
        snapshot[str(path.relative_to(root))] = {
            'mode': stat.S_IMODE(path.stat().st_mode),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }
Path(sys.argv[2]).write_text(json.dumps(snapshot))
PY
  model_runner_prepare_metal_sdpa_d512 Darwin "$SDPA_ROOT" "$sdpa_fixture" >/dev/null
  python3 - "$sdpa_fixture" "$SDPA_TEST_ROOT/readonly-before.json" <<'PY'
from pathlib import Path
import hashlib, json, stat, sys
root = Path(sys.argv[1])
after = {str(path.relative_to(root)): {
    'mode': stat.S_IMODE(path.stat().st_mode),
    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
} for path in root.rglob('*') if path.is_file()}
assert after == json.loads(Path(sys.argv[2]).read_text()), 'Read-only replay changed checkout bytes or modes'
PY
done

# A conflict in the second repository must not partly patch the first one.
cp -R "$SDPA_TEST_ROOT/baseline" "$SDPA_TEST_ROOT/conflict"
sdpa_conflict_file="$SDPA_TEST_ROOT/conflict/Source/Cmlx/mlx-generated/metal/scaled_dot_product_attention.metal"
sed 's/instantiate_sdpa_vector_heads(float16_t)/unsupported_fixture_change/' "$sdpa_conflict_file" > "$sdpa_conflict_file.tmp"
mv "$sdpa_conflict_file.tmp" "$sdpa_conflict_file"
if model_runner_prepare_metal_sdpa_d512 Darwin "$SDPA_ROOT" "$SDPA_TEST_ROOT/conflict" >/dev/null 2>&1; then
  echo "Expected generated-source conflict to reject the whole SDPA stack." >&2
  exit 1
fi
diff -qr "$SDPA_TEST_ROOT/baseline/Source/Cmlx/mlx" "$SDPA_TEST_ROOT/conflict/Source/Cmlx/mlx" >/dev/null

python3 - "$SDPA_TEST_ROOT/complete" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
core = (root / 'Source/Cmlx/mlx/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h').read_text()
metal = (root / 'Source/Cmlx/mlx-generated/metal/steel/attn/kernels/steel_attention_nax.h').read_text()
jit = (root / 'Source/Cmlx/mlx-generated/steel_attention_nax.cpp').read_text()
assert core.replace('mlx/backend/metal/kernels/', '../../../') == metal
# The generated JIT string inlines includes before the same complete kernel body.
body = core[core.index('using namespace mlx::steel;'):].strip()
assert body in jit, 'Generated JIT kernel body differs from authoritative Metal header'
PY
echo "SDPA clean/partial/applied preparation, read-only source modes, conflict atomicity, and all Metal representations passed."
