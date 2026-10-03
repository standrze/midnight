#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FIXTURE="$(mktemp -d)"
trap 'rm -rf "$FIXTURE"' EXIT
export MIDNIGHT_INSTALL_FIXTURE="$FIXTURE"
mkdir -p "$FIXTURE/mock" "$FIXTURE/package/midnight/bin" "$FIXTURE/home"
cat > "$FIXTURE/package/midnight/install.sh" <<'MOCK'
#!/bin/bash
mkdir -p "$HOME/.midnight/bin"
cp "$MIDNIGHT_INSTALL_FIXTURE/package/midnight/bin/midnight" "$HOME/.midnight/bin/midnight"
MOCK
printf '#!/bin/bash\necho 9.8.7-beta.6\n' > "$FIXTURE/package/midnight/bin/midnight"
chmod +x "$FIXTURE/package/midnight/bin/midnight"
COPYFILE_DISABLE=1 tar -czf "$FIXTURE/midnight-v9.8.7-beta.6-macos-arm64.tar.gz" -C "$FIXTURE/package" midnight
(cd "$FIXTURE" && shasum -a 256 midnight-v9.8.7-beta.6-macos-arm64.tar.gz > SHA256SUMS)
printf '[{\n  "tag_name": "v9.8.7-beta.6",\n  "prerelease": true\n}]\n' > "$FIXTURE/release.json"
cat > "$FIXTURE/mock/uname" <<'MOCK'
#!/bin/bash
if [[ "$1" == -s ]]; then echo Darwin; else echo arm64; fi
MOCK
printf '#!/bin/bash\necho 26.0\n' > "$FIXTURE/mock/sw_vers"
cat > "$FIXTURE/mock/curl" <<'MOCK'
#!/bin/bash
set -eu
url= out=
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    https://*) url="$1"; shift ;;
    *) shift ;;
  esac
done
case "$url" in
  'https://api.github.com/repos/standrze/midnight/releases?per_page=1') source=release.json ;;
  https://github.com/standrze/midnight/releases/download/v9.8.7-beta.6/*) source="${url##*/}" ;;
  *) exit 9 ;;
esac
cp "$MIDNIGHT_INSTALL_FIXTURE/$source" "$out"
MOCK
chmod +x "$FIXTURE/mock/"*
HOME="$FIXTURE/home" PATH="$FIXTURE/mock:$PATH" bash "$ROOT/Scripts/install-latest.sh" > "$FIXTURE/output"
test -x "$FIXTURE/home/.midnight/bin/midnight"
grep -q '9.8.7-beta.6' "$FIXTURE/output"
rm -rf "$FIXTURE/home/.midnight"
printf '%064d  midnight-v9.8.7-beta.6-macos-arm64.tar.gz\n' 0 > "$FIXTURE/SHA256SUMS"
if HOME="$FIXTURE/home" PATH="$FIXTURE/mock:$PATH" bash "$ROOT/Scripts/install-latest.sh" > "$FIXTURE/failure" 2>&1; then
  echo 'Bad checksum was accepted' >&2; exit 1
fi
test ! -e "$FIXTURE/home/.midnight"
grep -q 'checksum verification failed' "$FIXTURE/failure"
echo 'Latest prerelease selection and checksum rejection passed'
