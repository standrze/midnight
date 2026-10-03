#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd -P)"
PREFIX="${HOME:?HOME must be set}/.midnight"
BINARY=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix|--binary)
      [[ $# -ge 2 && -n "$2" ]] || { echo "$1 requires a path" >&2; exit 2; }
      if [[ "$1" == --prefix ]]; then PREFIX="$2"; else BINARY="$2"; fi
      shift 2 ;;
    --help|-h)
      echo 'Usage: ./install.sh [--prefix PATH] [--binary PATH]'
      echo 'Build and install the release runner into ~/.midnight/bin.'
      echo '--binary installs an existing build with its adjacent runtime resources.'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$BINARY" ]]; then
  MODEL_RUNNER_BUILD_CONFIGURATION=release MODEL_RUNNER_BUILD_PRODUCT=midnight "$PACKAGE_ROOT/build.sh"
  cd "$PACKAGE_ROOT"
  source "$PACKAGE_ROOT/Scripts/swiftpm-scratch-path.sh"
  model_runner_configure_swiftpm_scratch "$PACKAGE_ROOT" "$(uname -s)"
  BINARY="$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH/release/midnight"
fi
[[ -f "$BINARY" && -x "$BINARY" ]] || { echo "Missing executable: $BINARY" >&2; exit 1; }
SOURCE_DIR="$(cd "$(dirname "$BINARY")" && pwd -P)"
BINARY="$SOURCE_DIR/$(basename "$BINARY")"
if [[ "$(uname -s)" == Darwin && ! -f "$SOURCE_DIR/mlx.metallib" ]]; then
  echo "Missing mlx.metallib beside $BINARY; run ./build.sh first." >&2
  exit 1
fi

umask 077
mkdir -p "$PREFIX/bin" "$PREFIX/models" "$PREFIX/logs" "$PREFIX/apps/.runner-versions"
PREFIX="$(cd "$PREFIX" && pwd -P)"
LOCK="$PREFIX/apps/.runner-install.lock"
mkdir "$LOCK" 2>/dev/null || { echo "Install lock exists: $LOCK" >&2; exit 1; }
STAGE=
LAUNCHER=
PUBLISHED=0
cleanup() {
  [[ -z "$LAUNCHER" ]] || rm -f "$LAUNCHER"
  if [[ "$PUBLISHED" == 0 && -n "$STAGE" ]]; then rm -rf "$STAGE"; fi
  rmdir "$LOCK"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
[[ ! -d "$PREFIX/bin/midnight" ]] || { echo 'bin/midnight is a directory' >&2; exit 1; }
STAGE="$(mktemp -d "$PREFIX/apps/.runner-versions/install.XXXXXXXX")"
mkdir -p "$STAGE/bin"
install -m 755 "$BINARY" "$STAGE/bin/midnight"
for RESOURCE in "$SOURCE_DIR"/*.bundle "$SOURCE_DIR"/*.metallib "$SOURCE_DIR"/*.so* "$SOURCE_DIR"/*.dylib; do
  [[ -e "$RESOURCE" ]] || continue
  cp -R "$RESOURCE" "$STAGE/bin/"
done
# CUDA JIT discovers these header trees relative to the executable.
for RESOURCE in "$SOURCE_DIR/../include" "$PACKAGE_ROOT/include"; do
  [[ -d "$RESOURCE" ]] || continue
  cp -R "$RESOURCE" "$STAGE/"
  break
done
printf '%s\n' "$BINARY" > "$STAGE/source-binary.txt"
LAUNCHER="$(mktemp "$PREFIX/bin/.midnight.XXXXXXXX")"
printf '#!/usr/bin/env bash\nexec %q "$@"\n' "$STAGE/bin/midnight" > "$LAUNCHER"
chmod 755 "$LAUNCHER"
# Keep prior versions available to already-running processes.
PUBLISHED=1
if [[ "$(uname -s)" == Darwin ]]; then
  mv -fh "$LAUNCHER" "$PREFIX/bin/midnight"
else
  mv -fT "$LAUNCHER" "$PREFIX/bin/midnight"
fi
LAUNCHER=
echo "Installed Midnight Runner: $PREFIX/bin/midnight"
echo 'Runtime logs remain on stdout/stderr; redirect them into ~/.midnight/logs as needed.'
source "$PACKAGE_ROOT/Scripts/install-path.sh"
midnight_configure_path "$PREFIX/bin"
