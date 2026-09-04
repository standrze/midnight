#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FIXTURE="$(mktemp -d)"
trap 'rm -rf "$FIXTURE"' EXIT
mkdir -p "$FIXTURE/Scripts" "$FIXTURE/bin"
cp "$ROOT/run.sh" "$ROOT/build.sh" "$ROOT/build-metal.sh" "$FIXTURE/"
cp "$ROOT/Scripts/swiftpm-scratch-path.sh" "$ROOT/Scripts/release-publisher.sh" "$FIXTURE/Scripts/"
printf '#!/bin/bash\nexit 0\n' > "$FIXTURE/prepare-dependencies.sh"
printf '#!/bin/bash\necho Darwin\n' > "$FIXTURE/bin/uname"
cat > "$FIXTURE/bin/swift" <<'MOCK'
#!/bin/bash
set -eu
configuration=debug
show=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --configuration) shift; configuration="$1" ;;
    --show-bin-path) show=1 ;;
  esac
  shift
done
if [ "$show" = 1 ]; then
  echo "$PWD/.build/$configuration"
else
  mkdir -p ".build/$configuration"
  printf '#!/bin/bash\necho RUNNING_%s\n' "$configuration" > ".build/$configuration/midnight"
  chmod +x ".build/$configuration/midnight"
fi
MOCK
cat > "$FIXTURE/bin/xcrun" <<'MOCK'
#!/bin/bash
set -eu
while [ "$#" -gt 0 ]; do
  if [ "$1" = '-o' ]; then shift; touch "$1"; fi
  shift
done
MOCK
chmod +x "$FIXTURE/prepare-dependencies.sh" "$FIXTURE/bin/"*
export PATH="$FIXTURE/bin:$PATH"
unset MODEL_RUNNER_BUILD_PRODUCT MODEL_RUNNER_PINNED_MLX MODEL_RUNNER_BUILD_CONFIGURATION
# Default must be release, including a clean installation without debug artifacts.
"$FIXTURE/run.sh" > "$FIXTURE/output"
grep -qx RUNNING_release "$FIXTURE/output"
MODEL_RUNNER_BUILD_CONFIGURATION=debug "$FIXTURE/run.sh" > "$FIXTURE/output"
grep -qx RUNNING_debug "$FIXTURE/output"
# A stale debug artifact must never win over the requested release.
MODEL_RUNNER_BUILD_CONFIGURATION=release "$FIXTURE/run.sh" > "$FIXTURE/output"
grep -qx RUNNING_release "$FIXTURE/output"
echo 'Metal build/launch configuration tests passed'
