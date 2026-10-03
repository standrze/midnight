#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${1:?Usage: package-linux-release.sh vVERSION OUTPUT_DIRECTORY}"
OUTPUT="${2:?Output directory required}"
[[ "$(uname -s)/$(uname -m)" == Linux/x86_64 ]]
[[ "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$ ]]
BIN="$ROOT/.build/release"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[[ "$("$BIN/midnight" --version)" == "${VERSION#v}" ]]
source "$ROOT/Scripts/cuda-dependency-checks.sh"
model_runner_resolve_cuda_dependencies
mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
BUNDLE="$STAGE/midnight"
mkdir -p "$BUNDLE/bin" "$BUNDLE/lib" "$BUNDLE/include" "$BUNDLE/licenses" "$BUNDLE/Scripts"
cp "$BIN/midnight" "$BUNDLE/bin/"
for resource in "$BIN"/*.bundle "$BIN"/*.resources; do
  [[ ! -d "$resource" ]] || cp -R "$resource" "$BUNDLE/bin/"
done
ldd "$BIN/midnight" > "$BUNDLE/build-host-libraries.txt"
if grep -q 'not found' "$BUNDLE/build-host-libraries.txt"; then echo 'Unresolved runtime dependency' >&2; exit 1; fi
while IFS= read -r library; do
  [[ ! -f "$library" ]] || cp -L "$library" "$BUNDLE/lib/"
done < <(awk '$3 ~ /\/lib\/swift\/linux\// {print $3}' "$BUNDLE/build-host-libraries.txt")
[[ -f "$BUNDLE/lib/libswiftCore.so" ]]
cp -R "$CUTLASS_INCLUDE_DIR/cutlass" "$CUTLASS_INCLUDE_DIR/cute" "$BUNDLE/include/"
cp "$CUTLASS_ROOT/LICENSE.txt" "$BUNDLE/licenses/CUTLASS-LICENSE.txt"
SWIFT_SHARE="$(swift -print-target-info | python3 -c 'import json,sys,pathlib; print(pathlib.Path(json.load(sys.stdin)["paths"]["runtimeResourcePath"]).parent.parent / "share/swift")')"
cp "$SWIFT_SHARE/LICENSE.txt" "$BUNDLE/licenses/Swift-LICENSE.txt"
cp "$ROOT/install.sh" "$ROOT/LICENSE" "$ROOT/THIRD_PARTY_NOTICES.md" "$ROOT/Package.resolved" "$BUNDLE/"
cp "$ROOT/Scripts/install-path.sh" "$BUNDLE/Scripts/"
cat > "$BUNDLE/README.txt" <<'TEXT'
Midnight Linux x86-64 / CUDA 13 / RTX 4090 (sm_89)

Install: bash install.sh --binary bin/midnight
Run: ~/.midnight/bin/midnight --model /path/to/model

Built on Ubuntu 24.04 with Swift 6.3 and CUDA 13. Requires an sm_89 NVIDIA
GPU, compatible driver, CUDA 13 at /usr/local/cuda, cuDNN 9, OpenBLAS,
BLAS, LAPACK, gfortran and libcurl. Other GPUs/distributions are not verified.
Swift runtime libraries and CUTLASS/CuTe headers are bundled. Keep bin/,
lib/ and include/ together. Model weights are not included.
TEXT
tar -czf "$OUTPUT/midnight-$VERSION-linux-x86_64-cuda13-sm89.tar.gz" -C "$STAGE" midnight
printf '%s\n' "$OUTPUT/midnight-$VERSION-linux-x86_64-cuda13-sm89.tar.gz"
