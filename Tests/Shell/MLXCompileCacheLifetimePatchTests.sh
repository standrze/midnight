#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/midnight-compile-cache-patch.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT
export PACKAGE_ROOT TEST_ROOT

# Reconstruct the old-side hunk contexts so this fixture needs neither network
# access nor a populated SwiftPM checkout. Line gaps are inert text padding.
python3 - <<'PY'
import os
from pathlib import Path
import re
root = Path(os.environ['PACKAGE_ROOT'])
fixture = Path(os.environ['TEST_ROOT']) / 'mlx-swift'
for name, repo in [('mlx', fixture/'Source/Cmlx/mlx'), ('mlx-c', fixture/'Source/Cmlx/mlx-c'), ('mlx-swift', fixture)]:
    lines = (root/f'Patches/{name}-compile-cache-lifetime.patch').read_text().splitlines(True)
    relative = None
    output = []
    def flush():
        if relative is not None:
            path = repo/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(output))
    for line in lines:
        if line.startswith('diff --git '):
            flush()
            relative = line.split(' b/', 1)[1].strip()
            output = []
        elif line.startswith('@@ '):
            start = int(re.match(r'@@ -(\d+)', line).group(1))
            while len(output) < start-1:
                output.append(f'// omitted fixture line {len(output)+1}\n')
        elif relative is not None and not line.startswith(('--- ', '+++ ', 'index ')):
            if line.startswith((' ', '-')):
                output.append(line[1:])
    flush()
# A nearby user edit must survive both application and repeated preparation.
(fixture/'user-owned-note.txt').write_text('keep this unrelated local edit\n')
PY

# The fixtures contain exact patch contexts, not a clone of the pinned repos.
# Stub only revision identity; patch checking/application uses the real git.
git() {
  if [[ "$#" == 4 && "$1" == -C && "$3" == rev-parse && "$4" == HEAD ]]; then
    if [[ "${TEST_BAD_REVISION:-0}" == 1 ]]; then
      echo 0000000000000000000000000000000000000000
      return 0
    fi
    case "$2" in
      */mlx-c) echo c74db5307cc8ce122f48d97ef951b30578674e7f ;;
      */mlx) echo 1f8e74e3f12f31365464a6867c6579f0e9b29d85 ;;
      */mlx-swift) echo 72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798 ;;
      *) return 1 ;;
    esac
  else
    command git "$@"
  fi
}
source "$PACKAGE_ROOT/Scripts/compile-cache-lifetime-patch.sh"
FIXTURE="$TEST_ROOT/mlx-swift"
for repo in "$FIXTURE" "$FIXTURE/Source/Cmlx/mlx" "$FIXTURE/Source/Cmlx/mlx-c"; do
  command git -C "$repo" init -q
 done

snapshot_fixture() {
  python3 - "$FIXTURE" <<'PY'
from pathlib import Path
import hashlib
import sys
root = Path(sys.argv[1])
for path in sorted(root.rglob('*')):
    if path.is_file() and '.git' not in path.relative_to(root).parts:
        print(path.relative_to(root), hashlib.sha256(path.read_bytes()).hexdigest())
PY
}

snapshot_fixture > "$TEST_ROOT/original.sha256"
# Linux must be a strict no-op, even with nonexistent checkouts.
model_runner_prepare_compile_cache_lifetime Linux /does/not/exist /does/not/exist
snapshot_fixture > "$TEST_ROOT/linux.sha256"
cmp "$TEST_ROOT/original.sha256" "$TEST_ROOT/linux.sha256"

# An unexpected revision is rejected before any patch is written.
TEST_BAD_REVISION=1
if model_runner_prepare_compile_cache_lifetime Darwin "$PACKAGE_ROOT" "$FIXTURE" > "$TEST_ROOT/bad-revision.log" 2>&1; then
  echo 'Unexpected revision was accepted' >&2
  exit 1
fi
unset TEST_BAD_REVISION
snapshot_fixture > "$TEST_ROOT/rejected.sha256"
cmp "$TEST_ROOT/original.sha256" "$TEST_ROOT/rejected.sha256"

# Drift in the last patch must not partly apply the two earlier layers.
DRIFT_FILE="$FIXTURE/Source/MLX/Transforms+Compile.swift"
cp "$DRIFT_FILE" "$TEST_ROOT/compile.swift"
python3 - "$DRIFT_FILE" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); s=p.read_text()
s=s.replace('    deinit {', '    deinit { // conflicting local lifetime change', 1)
p.write_text(s)
PY
snapshot_fixture > "$TEST_ROOT/drift-before.sha256"
if model_runner_prepare_compile_cache_lifetime Darwin "$PACKAGE_ROOT" "$FIXTURE" > "$TEST_ROOT/drift.log" 2>&1; then
  echo 'Conflicting source was accepted' >&2
  exit 1
fi
snapshot_fixture > "$TEST_ROOT/drift-after.sha256"
cmp "$TEST_ROOT/drift-before.sha256" "$TEST_ROOT/drift-after.sha256"
cp "$TEST_ROOT/compile.swift" "$DRIFT_FILE"

model_runner_prepare_compile_cache_lifetime Darwin "$PACKAGE_ROOT" "$FIXTURE"
snapshot_fixture > "$TEST_ROOT/applied.sha256"
model_runner_prepare_compile_cache_lifetime Darwin "$PACKAGE_ROOT" "$FIXTURE"
snapshot_fixture > "$TEST_ROOT/reapplied.sha256"
cmp "$TEST_ROOT/applied.sha256" "$TEST_ROOT/reapplied.sha256"

# Preserve old API semantics, register only in TLS construction, erase all
# caches only at Swift destruction, and keep binding copies synchronized.
python3 - <<'PY'
import os
from pathlib import Path
root = Path(os.environ['PACKAGE_ROOT'])
fixture = Path(os.environ['TEST_ROOT'])/'mlx-swift'
cpp=(fixture/'Source/Cmlx/mlx/mlx/compile.cpp').read_text()
swift=(fixture/'Source/MLX/Transforms+Compile.swift').read_text()
assert 'static thread_local ThreadCompileCache local;' in cpp
assert 'static auto* registry = new CompileCacheRegistry;' in cpp
assert 'removed = std::move(it->second);' in cpp
assert 'removed.swap(cache_);' in cpp
assert cpp.index('result.reserve(caches_.size());') < cpp.index('for (auto& weak : caches_)')
assert 'auto caches = compile_cache_registry().snapshot();' in cpp
assert 'mlx_detail_compile_erase_all_threads(functionID)' in swift
assert 'var cache = mlx_compile_cache_new()' not in swift
for relative in ['Source/Cmlx/include/mlx/c/compile.h', 'Source/Cmlx/include-framework/mlx-c-compile.h', 'Source/Cmlx/mlx-c/mlx/c/compile.h']:
    assert 'int mlx_detail_compile_erase_all_threads(uintptr_t fun_id);' in (fixture/relative).read_text()
assert (fixture/'user-owned-note.txt').read_text() == 'keep this unrelated local edit\n'
prepare=(root/'prepare-dependencies.sh').read_text()
assert 'source "$PACKAGE_ROOT/Scripts/compile-cache-lifetime-patch.sh"' in prepare
assert 'if [[ "$HOST_OS" == "Darwin" ]]; then\n  model_runner_prepare_compile_cache_lifetime' in prepare
# All call-path Swift statements stay untouched; the patch changes only the
# deinit body and explanatory comments before call().
patch=(root/'Patches/mlx-swift-compile-cache-lifetime.patch').read_text()
for line in patch.splitlines():
    if line.startswith('+') and not line.startswith('+++'):
        assert not any(token in line for token in ['func call(', 'func innerCall(', 'func withInstanceLock('])
PY
printf '%s\n' 'MLX compiler-cache lifetime patch fixtures passed.'
