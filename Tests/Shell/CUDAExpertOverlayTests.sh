#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/midnight-cuda-expert-overlay.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT
FIXTURE="$TEST_ROOT/mlx-swift"
CORE="$FIXTURE/Source/Cmlx/mlx"
export PACKAGE_ROOT CORE

# Reconstruct the old-side contexts of this optional overlay. Real git applies
# and reverses it; only the pinned commit identity is stubbed below.
python3 - <<'PY'
import os
import re
from pathlib import Path

root = Path(os.environ['CORE'])
patch = Path(os.environ['PACKAGE_ROOT']) / 'Patches/mlx-cuda-expert-qmv-fp32.patch'
relative, lines, new_file = None, [], False

def flush():
    if relative is not None and not new_file:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(lines))

for line in patch.read_text().splitlines(True):
    if line.startswith('diff --git '):
        flush()
        relative = line.split(' b/', 1)[1].strip()
        lines, new_file = [], False
    elif line.startswith('new file mode '):
        new_file = True
    elif line.startswith('@@ '):
        start = int(re.match(r'@@ -(\d+)', line).group(1))
        while len(lines) < start - 1:
            lines.append('// fixture padding\n')
    elif relative and not line.startswith(('--- ', '+++ ', 'index ')):
        if line.startswith((' ', '-')):
            lines.append(line[1:])
flush()
(root / 'unrelated.txt').write_text('preserve this local change\n')
PY

command git -C "$CORE" init -q
git() {
  if [[ "$#" == 4 && "$1" == -C && "$3" == rev-parse && "$4" == HEAD ]]; then
    printf '%s\n' "${TEST_REVISION:-7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247}"
  else
    command git "$@"
  fi
}
source "$PACKAGE_ROOT/Scripts/optional-dependency-patch.sh"
source "$PACKAGE_ROOT/Scripts/cuda-expert-patch.sh"

snapshot() {
  python3 - <<'PY'
import hashlib
import os
from pathlib import Path
root = Path(os.environ['CORE'])
for path in sorted(root.rglob('*')):
    if path.is_file() and '.git' not in path.relative_to(root).parts:
        print(path.relative_to(root), hashlib.sha256(path.read_bytes()).hexdigest())
PY
}

snapshot > "$TEST_ROOT/original"
unset MODEL_RUNNER_ENABLE_CUDA_EXPERT_QMV_OVERLAY
model_runner_prepare_cuda_experts Linux "$PACKAGE_ROOT" "$FIXTURE"
snapshot > "$TEST_ROOT/default"
cmp "$TEST_ROOT/original" "$TEST_ROOT/default"

MODEL_RUNNER_ENABLE_CUDA_EXPERT_QMV_OVERLAY=invalid
model_runner_prepare_cuda_experts Darwin /nonexistent /nonexistent
if model_runner_prepare_cuda_experts Linux "$PACKAGE_ROOT" "$FIXTURE" >/dev/null 2>&1; then
  echo 'Invalid opt-in was accepted' >&2; exit 1
fi
MODEL_RUNNER_ENABLE_CUDA_EXPERT_QMV_OVERLAY=1
TEST_REVISION=unexpected
if model_runner_prepare_cuda_experts Linux "$PACKAGE_ROOT" "$FIXTURE" >/dev/null 2>&1; then
  echo 'Unexpected MLX revision was accepted' >&2; exit 1
fi
unset TEST_REVISION
snapshot > "$TEST_ROOT/rejected"
cmp "$TEST_ROOT/original" "$TEST_ROOT/rejected"

model_runner_prepare_cuda_experts Linux "$PACKAGE_ROOT" "$FIXTURE"
snapshot > "$TEST_ROOT/enabled"
if cmp -s "$TEST_ROOT/original" "$TEST_ROOT/enabled"; then
  echo 'Opt-in did not apply the overlay' >&2; exit 1
fi
model_runner_prepare_cuda_experts Linux "$PACKAGE_ROOT" "$FIXTURE"
snapshot > "$TEST_ROOT/repeated"
cmp "$TEST_ROOT/enabled" "$TEST_ROOT/repeated"

# Disabling restores only our overlay, including removing its added header.
MODEL_RUNNER_ENABLE_CUDA_EXPERT_QMV_OVERLAY=0
model_runner_prepare_cuda_experts Linux "$PACKAGE_ROOT" "$FIXTURE"
snapshot > "$TEST_ROOT/restored"
cmp "$TEST_ROOT/original" "$TEST_ROOT/restored"

printf '%s\n' 'CUDA expert overlay default, revision, opt-in, idempotence, and restoration checks passed.'
