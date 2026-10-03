#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo 'Linux installer test requires Linux'; exit 0; }
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FIXTURE="$(mktemp -d)"
trap 'rm -rf "$FIXTURE"' EXIT
PACKAGE="$FIXTURE/archive"
PREFIX="$FIXTURE/install with spaces"
mkdir -p "$PACKAGE/bin" "$PACKAGE/lib" "$PACKAGE/include/cute" "$PACKAGE/Scripts"
cp "$ROOT/install.sh" "$PACKAGE/"
cp "$ROOT/Scripts/install-path.sh" "$PACKAGE/Scripts/"
printf 'runtime\n' > "$PACKAGE/lib/libfixture.so"
printf 'header\n' > "$PACKAGE/include/cute/fixture.h"
cat > "$PACKAGE/bin/midnight" <<'BINARY'
#!/usr/bin/env bash
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
test -f "$root/lib/libfixture.so"
test -f "$root/include/cute/fixture.h"
[[ "$LD_LIBRARY_PATH" == "$root/lib:$root/bin:/usr/local/cuda/lib64" ]]
[[ "$1" == 'argument with spaces' ]]
echo 'Installed resources and arguments verified'
BINARY
chmod +x "$PACKAGE/bin/midnight"
bash "$PACKAGE/install.sh" --prefix "$PREFIX" > "$FIXTURE/install.log"
mv "$PACKAGE" "$FIXTURE/moved-archive"
env -u LD_LIBRARY_PATH "$PREFIX/bin/midnight" 'argument with spaces'
test ! -d "$PREFIX/apps/.runner-install.lock"
