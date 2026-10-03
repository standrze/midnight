#!/usr/bin/env python3
"""Generate a pinned MLX overlay without modifying the supplied source tree."""

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent
CUDA = Path("mlx/backend/cuda")
BASELINES = {
    "device.h": "6b0ab1548f9ff57ea6998d737f66e2b28eeaa9c05c58134f6c799b0bcee2a1fd",
    "device.cpp": "7c6480abae4ad6690ea552391cd07f82464442e78259e0d883e6216cf4ac2eb2",
}
HEADERS = ("midnight_replay.h", "midnight_replay_impl.h", "replay_session_core.h")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"expected exactly one overlay anchor: {old!r}")
    return text.replace(old, new, 1)


def guarded(statement):
    return "#if defined(MIDNIGHT_MLX_CUDA_REPLAY)\n" + statement + "\n#endif\n"


def render(sources):
    header = sources["device.h"]
    header = replace_once(header, '#include "mlx/stream.h"\n',
                          '#include "mlx/stream.h"\n' + guarded(
                              '#include "mlx/backend/cuda/midnight_replay.h"'))
    header = replace_once(header, "  void add_temporary(const array& arr) {\n",
                          "  void add_temporary(const array& arr) {\n" + guarded(
                              "    replay_detail::retain(*this, arr);"))
    source = sources["device.cpp"]
    source = replace_once(source,
                          "CommandEncoder::~CommandEncoder() {\n  synchronize();\n",
                          "CommandEncoder::~CommandEncoder() {\n  synchronize();\n" + guarded(
                              "  replay_detail::encoder_destroyed(*this);"))
    for method in ("set_input_array", "set_output_array"):
        anchor = f"void CommandEncoder::{method}(const array& arr) {{\n"
        source = replace_once(source, anchor, anchor + guarded(
            "  replay_detail::retain(*this, arr);"))
    source = replace_once(source, "    // Reset state\n", guarded(
        "    replay_detail::capture_graph(*this, graph_);") + "    // Reset state\n")
    source = replace_once(source, "  return it->second;\n", guarded(
        "  replay_detail::observe_encoder(it->second);") + "  return it->second;\n")
    source += "\n" + guarded('#include "mlx/backend/cuda/midnight_replay_impl.h"')
    return {"device.h": header, "device.cpp": source}


def prepare(source_root, output):
    source_root = Path(source_root).resolve(strict=True)
    output = Path(output).resolve()
    if output == source_root or source_root in output.parents:
        raise ValueError("overlay output must be outside the source tree")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    before = {}
    for name, expected in BASELINES.items():
        data = (source_root / CUDA / name).read_bytes()
        if sha256(data) != expected:
            raise ValueError(f"unreviewed {name} baseline; expected SHA256 {expected}")
        before[name] = data.decode()
    for name in HEADERS:
        if (source_root / CUDA / name).exists():
            raise ValueError(f"overlay header already exists: {name}")
    after = render(before)
    after.update({name: (ROOT / name).read_text() for name in HEADERS})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".mlx-replay-overlay-", dir=output.parent))
    try:
        patch = []
        manifest = {
            "experimental": True,
            "enabled_by_default": False,
            "compile_definition": "MIDNIGHT_MLX_CUDA_REPLAY=1",
            "mlx_core_pin": "7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247",
            "source_baselines": BASELINES,
            "files": {},
        }
        for name, text in after.items():
            relative = CUDA / name
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
            patch.extend(difflib.unified_diff(
                before.get(name, "").splitlines(keepends=True), text.splitlines(keepends=True),
                fromfile=f"a/{relative}" if name in before else "/dev/null",
                tofile=f"b/{relative}"))
            manifest["files"][str(relative)] = sha256(text.encode())
        (temporary / "overlay.patch").write_text("".join(patch))
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output / "overlay.patch"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.mlx_source, args.output))
