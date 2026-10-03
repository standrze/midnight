#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${1:?Usage: package-macos-release.sh vVERSION OUTPUT_DIRECTORY}"
OUTPUT="${2:?Output directory required}"
[[ "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$ ]] || exit 2
BIN="$ROOT/.build/release"
[[ -x "$BIN/midnight" && -f "$BIN/mlx.metallib" ]] || { echo 'Run ./build.sh first' >&2; exit 1; }
[[ "$($BIN/midnight --version)" == "${VERSION#v}" ]] || { echo 'Version mismatch' >&2; exit 1; }
mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/midnight/bin" "$STAGE/midnight/Scripts"
cp "$BIN/midnight" "$BIN/mlx.metallib" "$STAGE/midnight/bin/"
for RESOURCE in "$BIN"/swift-*.bundle; do
  [[ ! -d "$RESOURCE" ]] || cp -R "$RESOURCE" "$STAGE/midnight/bin/"
done
# Swift 6.2 span compatibility runtime is a weak-linked dependency. Include it
# beside the executable so the archive does not require the build toolchain.
TOOLCHAIN="$(dirname "$(dirname "$(xcrun --find swift)")")"
COMPAT="$TOOLCHAIN/lib/swift-6.2/macosx/libswiftCompatibilitySpan.dylib"
if [[ -f "$COMPAT" ]]; then cp "$COMPAT" "$STAGE/midnight/bin/"; fi
if otool -l "$STAGE/midnight/bin/midnight" | grep -Fq "$TOOLCHAIN/lib/swift-6.2/macosx"; then
  install_name_tool -delete_rpath "$TOOLCHAIN/lib/swift-6.2/macosx" "$STAGE/midnight/bin/midnight"
fi
codesign --force --sign - "$STAGE/midnight/bin/midnight"
codesign --verify "$STAGE/midnight/bin/midnight"
cp "$ROOT/install.sh" "$ROOT/LICENSE" "$ROOT/THIRD_PARTY_NOTICES.md" "$STAGE/midnight/"
cp "$ROOT/Scripts/install-path.sh" "$STAGE/midnight/Scripts/"
cat > "$STAGE/midnight/README.txt" <<'TEXT'
Midnight Runner prerelease — Apple silicon, macOS 26+

Install from this directory:
  bash install.sh --binary bin/midnight

The installer asks whether to add ~/.midnight/bin to PATH.
Open a new terminal after accepting, then run midnight download.

This prerelease is ad-hoc signed, not Developer ID signed or notarized.
It includes no model weights. Linux/CUDA builds are available in a separate release archive.
TEXT
COPYFILE_DISABLE=1 tar -czf "$OUTPUT/midnight-$VERSION-macos-arm64.tar.gz" -C "$STAGE" midnight
cp "$ROOT/Scripts/install-latest.sh" "$OUTPUT/install.sh"
(cd "$OUTPUT" && shasum -a 256 "midnight-$VERSION-macos-arm64.tar.gz" install.sh > SHA256SUMS)
echo "$OUTPUT/midnight-$VERSION-macos-arm64.tar.gz"
