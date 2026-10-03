#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FIXTURE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/midnight-gate-up-slices.XXXXXX")"
trap 'rm -rf "$FIXTURE_ROOT"' EXIT
export PACKAGE_ROOT FIXTURE_ROOT
python3 - <<'PY'
from pathlib import Path
import os,re
root=Path(os.environ['PACKAGE_ROOT']); fixture=Path(os.environ['FIXTURE_ROOT'])
lines=(root/'Patches/mlx-swift-lm-gate-up-slices.patch').read_text().splitlines(True)
output=[]
for line in lines:
    if line.startswith('@@ '):
        start=int(re.match(r'@@ -(\d+)',line).group(1))
        while len(output)<start-1:output.append(f'// omitted fixture line {len(output)+1}\n')
    elif line.startswith((' ', '-')) and not line.startswith('--- '):output.append(line[1:])
p=fixture/'Libraries/MLXLMCommon/SwitchLayers.swift';p.parent.mkdir(parents=True);p.write_text(''.join(output))
(fixture/'user-owned-note.txt').write_text('preserve this unrelated edit\n')
PY
command git -C "$FIXTURE_ROOT" init -q
git() {
  if [[ "$#" == 4 && "$1" == -C && "$3" == rev-parse && "$4" == HEAD ]]; then
    echo "${TEST_REVISION:-14414441fa44f45eee35a61e9fa0bab577cf9734}"
  else
    command git "$@"
  fi
}
source "$PACKAGE_ROOT/Scripts/gate-up-slices-patch.sh"
SOURCE="$FIXTURE_ROOT/Libraries/MLXLMCommon/SwitchLayers.swift"
cp "$SOURCE" "$FIXTURE_ROOT/original.swift"
model_runner_prepare_gate_up_slices Linux /not/a/root /not/a/checkout
cmp "$SOURCE" "$FIXTURE_ROOT/original.swift"
TEST_REVISION=0000000000000000000000000000000000000000
if model_runner_prepare_gate_up_slices Darwin "$PACKAGE_ROOT" "$FIXTURE_ROOT" >/dev/null 2>&1; then
  echo 'Unexpected source revision accepted' >&2; exit 1
fi
unset TEST_REVISION
cmp "$SOURCE" "$FIXTURE_ROOT/original.swift"
python3 - "$SOURCE" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]);p.write_text(p.read_text().replace('parts: 2, axis: -1','parts: 3, axis: -1'))
PY
cp "$SOURCE" "$FIXTURE_ROOT/drift.swift"
if model_runner_prepare_gate_up_slices Darwin "$PACKAGE_ROOT" "$FIXTURE_ROOT" >/dev/null 2>&1; then
  echo 'Conflicting source accepted' >&2; exit 1
fi
cmp "$SOURCE" "$FIXTURE_ROOT/drift.swift"
cp "$FIXTURE_ROOT/original.swift" "$SOURCE"
model_runner_prepare_gate_up_slices Darwin "$PACKAGE_ROOT" "$FIXTURE_ROOT"
cp "$SOURCE" "$FIXTURE_ROOT/first.swift"
model_runner_prepare_gate_up_slices Darwin "$PACKAGE_ROOT" "$FIXTURE_ROOT"
cmp "$SOURCE" "$FIXTURE_ROOT/first.swift"
python3 - <<'PY'
from pathlib import Path
import os
root=Path(os.environ['PACKAGE_ROOT']);fixture=Path(os.environ['FIXTURE_ROOT'])
s=(fixture/'Libraries/MLXLMCommon/SwitchLayers.swift').read_text()
assert '#if os(macOS)' in s
assert 'precondition(gateUp.dim(-1) % 2 == 0' in s
assert '[gateUp[.ellipsis, ..<half], gateUp[.ellipsis, half...]]' in s
assert '#else\n            let parts = MLX.split(gateUp, parts: 2, axis: -1)\n        #endif' in s
assert (fixture/'user-owned-note.txt').read_text()=='preserve this unrelated edit\n'
prep=(root/'prepare-dependencies.sh').read_text()
assert 'source "$PACKAGE_ROOT/Scripts/gate-up-slices-patch.sh"' in prep
assert 'model_runner_prepare_gate_up_slices "$HOST_OS" "$PACKAGE_ROOT" "$MLX_SWIFT_LM_CHECKOUT"' in prep
PY
printf '%s\n' 'Gate/up slice patch fixtures passed: Linux no-op, revision rejection, drift rejection, exact views, idempotency, unrelated edits.'
