#!/usr/bin/env python3
"""Fresh stock/candidate C++ fixture processes; no performance measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess

from probe import cases, validate_trace


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def compare(baseline, candidate, expected_flags=("0", "1")):
    reports = [json.loads((directory / "results.json").read_text()) for directory in (baseline, candidate)]
    expected = cases()
    for report, mode in zip(reports, expected_flags):
        if report.get("runtime_flag") != mode or report.get("timing") != "not_measured":
            raise ValueError("Wrong or instrumented correctness arm")
        if [case["name"] for case in report["cases"]] != [case["name"] for case in expected]:
            raise ValueError("Incomplete or reordered fixture list")
    rows = []
    for fixture, stock, proposed in zip(expected, reports[0]["cases"], reports[1]["cases"]):
        size = fixture["selected"] * fixture["m"] * fixture["n"] * 4
        values = []
        for directory, result in ((baseline, stock), (candidate, proposed)):
            if any(result.get(key) != fixture[key] for key in ("name", "n", "k", "eligible")):
                raise ValueError("Fixture metadata mismatch")
            if any(not math.isfinite(result[key]) or result[key] < 0 for key in ("max_reference_error", "rms_reference_error")):
                raise ValueError("Invalid reference diagnostic")
            data = (directory / (fixture["name"] + ".f32")).read_bytes()
            if len(data) != size or result["output_bytes"] != size:
                raise ValueError("Incomplete output data")
            if any(not math.isfinite(value[0]) for value in struct.iter_unpack("<f", data)):
                raise ValueError("Nonfinite output data")
            values.append(data)
        pairs = list(zip(struct.iter_unpack("<f", values[0]), struct.iter_unpack("<f", values[1])))
        rows.append(dict(name=fixture["name"], eligible=fixture["eligible"], exact=values[0] == values[1],
                         differing_values=sum(values[0][i:i+4] != values[1][i:i+4] for i in range(0, size, 4)),
                         max_absolute_difference=max(abs(a[0] - b[0]) for a, b in pairs),
                         stock_sha256=hashlib.sha256(values[0]).hexdigest(),
                         candidate_sha256=hashlib.sha256(values[1]).hexdigest(),
                         stock_reference_error=stock["max_reference_error"],
                         candidate_reference_error=proposed["max_reference_error"]))
    passed = all(row["exact"] for row in rows)
    return dict(status="exact_fixtures_passed" if passed else "failed_output_parity",
                promotion="blocked_pending_full_model_validation" if passed else "blocked_by_output_parity",
                timing="not_measured", cases=rows)


def gpu_state():
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.free,memory.used", "--format=csv,noheader,nounits"],
                           check=True, capture_output=True, text=True).stdout
    rows = [row.split(",") for row in query.strip().splitlines()]
    if len(rows) != 1 or int(rows[0][2]) < 1024:
        raise RuntimeError("Probe requires exactly one GPU with at least 1 GiB free")
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
                               check=True, capture_output=True, text=True).stdout
    return dict(gpu=query, processes=processes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary, library = args.binary.resolve(strict=True), args.library.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    linkage = subprocess.run(["ldd", str(binary)], check=True, capture_output=True, text=True).stdout
    (output / "ldd.txt").write_text(linkage)
    linked_mlx = [line for line in linkage.splitlines() if "libmlx" in line]
    if len(linked_mlx) != 1 or "=>" not in linked_mlx[0]:
        raise RuntimeError("Cannot identify the linked MLX shared library")
    linked_path = Path(linked_mlx[0].split("=>", 1)[1].strip().split(" ", 1)[0]).resolve(strict=True)
    if linked_path != library:
        raise RuntimeError("Probe does not link the specified isolated MLX library")
    identity = dict(binary=str(binary), library=str(library), binary_sha256=sha256(binary), library_sha256=sha256(library),
                    jit_cache=str(output / "ptx"),
                    probe_source_sha256=sha256(Path(__file__).with_name("probe.cpp")))
    (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    states = []
    traces = {}
    for label, enabled in (("stock", "0"), ("candidate", "1")):
        states.append(dict(arm=label, **gpu_state()))
        (output / "gpu-state.json").write_text(json.dumps(states, indent=2) + "\n")
        if sha256(binary) != identity["binary_sha256"] or sha256(library) != identity["library_sha256"]:
            raise RuntimeError("Probe or library changed between arms")
        env = dict(os.environ, MIDNIGHT_CUDA_EXPERT_QMV=enabled, MIDNIGHT_CUDA_EXPERT_QMV_TRACE="1",
                   MLX_PTX_CACHE_DIR=str(output / "ptx"))
        with (output / (label + ".log")).open("w") as log:
            result = subprocess.run([str(binary), str(output / label)], env=env, stdout=log, stderr=log)
        if result.returncode:
            raise RuntimeError(f"{label} probe failed ({result.returncode}); inspect saved log")
        traces[label] = validate_trace((output / (label + ".log")).read_text(), candidate=enabled == "1")
    if sha256(binary) != identity["binary_sha256"] or sha256(library) != identity["library_sha256"]:
        raise RuntimeError("Probe or library changed while running")
    summary = dict(compare(output / "stock", output / "candidate"), identity=identity, trace=traces)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # A failed candidate gets one independent stock repeat to distinguish a
    # changed reduction from stock nondeterminism, without changing the gate.
    if summary["status"] == "failed_output_parity":
        states.append(dict(arm="stock-repeat", **gpu_state()))
        (output / "gpu-state.json").write_text(json.dumps(states, indent=2) + "\n")
        env = dict(os.environ, MIDNIGHT_CUDA_EXPERT_QMV="0", MIDNIGHT_CUDA_EXPERT_QMV_TRACE="1",
                   MLX_PTX_CACHE_DIR=str(output / "ptx"))
        with (output / "stock-repeat.log").open("w") as log:
            result = subprocess.run([str(binary), str(output / "stock-repeat")], env=env, stdout=log, stderr=log)
        if result.returncode:
            raise RuntimeError(f"stock-repeat probe failed ({result.returncode}); inspect saved log")
        traces["stock-repeat"] = validate_trace((output / "stock-repeat.log").read_text(), candidate=False)
        repeat = compare(output / "stock", output / "stock-repeat", expected_flags=("0", "0"))
        summary["stock_repeat"] = dict(exact=all(case["exact"] for case in repeat["cases"]), cases=repeat["cases"])
    if sha256(binary) != identity["binary_sha256"] or sha256(library) != identity["library_sha256"]:
        raise RuntimeError("Probe or library changed while running")
    states.append(dict(arm="after", **gpu_state()))
    (output / "gpu-state.json").write_text(json.dumps(states, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "exact_fixtures_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
