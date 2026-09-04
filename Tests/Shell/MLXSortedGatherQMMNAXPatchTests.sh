#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Exercise the real preparation helper on isolated pinned-source fixtures;
# no package resolution, dependency mutation, Swift build, or GPU use.
python3 - "$PACKAGE_ROOT" <<'PY'
import pathlib
import re
import subprocess
import sys
import tempfile

root = pathlib.Path(sys.argv[1])
prepare = (root / "prepare-dependencies.sh").read_text()
helper = prepare[prepare.index("apply_dependency_patch() {"):prepare.index(
    "# Verify every checkout that this host will patch")]
darwin = prepare[prepare.index('if [[ "$HOST_OS" == "Darwin" ]]; then'):]
darwin = darwin[:darwin.index("\nfi")]
fixed = "const short sgp_sm = align_M ? SM : min(int(SM), max(0, M - (y_row + tm)));"
old = "align_M ? SM : min(SM, short(max(0, (M - (y_row + tm)))));"
specs = [
    ("mlx sorted gather QMM NAX row bounds",
     ".build/checkouts/mlx-swift/Source/Cmlx/mlx",
     "1f8e74e3f12f31365464a6867c6579f0e9b29d85",
     "mlx-sorted-gather-qmm-nax-row-bounds.patch",
     "MLX_SOURCE_SORTED_GATHER_QMM_NAX_PATCH",
     ["mlx/backend/metal/kernels/quantized_nax.h"]),
    ("mlx-swift sorted gather QMM NAX generated JIT",
     ".build/checkouts/mlx-swift",
     "72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798",
     "mlx-swift-sorted-gather-qmm-nax-row-bounds-jit.patch",
     "MLX_SWIFT_SORTED_GATHER_QMM_NAX_JIT_PATCH",
     ["Source/Cmlx/mlx-generated/metal/quantized_nax.h",
      "Source/Cmlx/mlx-generated/quantized_nax.cpp"]),
]

def run(*args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, text=True, **kwargs)

with tempfile.TemporaryDirectory(prefix="midnight-nax-row-bounds-") as tmp:
    scratch = pathlib.Path(tmp)
    helper_path = scratch / "patch-helper.sh"
    helper_path.write_text(helper)
    for index, (label, checkout, revision, patch_name, variable, paths) in enumerate(specs):
        patch = root / "Patches" / patch_name
        assert f'{variable}="$PACKAGE_ROOT/Patches/{patch_name}"' in prepare
        assert f'"{label}"' in darwin and f'"${variable}"' in darwin
        targets = re.findall(r"^diff --git a/(.+) b/(.+)$", patch.read_text(), re.M)
        assert targets == [(path, path) for path in paths], targets
        fixture = scratch / str(index)
        fixture.mkdir()
        originals = {}
        for path in paths:
            original = run("git", "-C", str(root / checkout), "show", f"{revision}:{path}").stdout
            assert original.count(old) == 1
            destination = fixture / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(original)
            originals[path] = original

        run("git", "-C", str(fixture), "apply", "--check", "--whitespace=error-all", str(patch))
        command = 'source "$1"; apply_dependency_patch "$2" "$3" "$4"'
        args = ("bash", "-eu", "-c", command, "patch-test", str(helper_path), label, str(fixture), str(patch))
        assert "Applied" in run(*args).stdout
        patched = {path: (fixture / path).read_text() for path in paths}
        for path, content in patched.items():
            assert content.count(fixed) == 1 and old not in content, path
        assert "already applied" in run(*args).stdout
        assert patched == {path: (fixture / path).read_text() for path in paths}
        run("git", "-C", str(fixture), "apply", "--reverse", "--check", str(patch))
        run("git", "-C", str(fixture), "apply", "--reverse", str(patch))
        assert originals == {path: (fixture / path).read_text() for path in paths}

        # An unexpected kernel must fail closed rather than silently claim success.
        first = fixture / paths[0]
        first.write_text(originals[paths[0]].replace(old, "unexpected_row_bound();"))
        refused = subprocess.run(args, capture_output=True, text=True)
        assert refused.returncode != 0 and "Could not apply" in refused.stderr

print("Sorted gather QMM NAX patch tests passed (source/JIT round-trip, idempotence, conflict rejection).")
PY
