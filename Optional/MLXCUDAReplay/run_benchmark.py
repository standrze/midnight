#!/usr/bin/env python3
"""Run bounded, balanced synthetic replay timings; never report model tokens/s."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.free,memory.used,utilization.gpu,temperature.gpu,clocks.sm",
        "--format=csv,noheader,nounits"], text=True)
    rows = [line.split(",") for line in gpu.strip().splitlines()]
    if len(rows) != 1 or int(rows[0][2]) < 1024:
        raise RuntimeError("Exactly one GPU with at least 1 GiB free is required")
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"], text=True)
    return {"gpu": gpu, "processes": processes}


def summarize(report):
    if report.get("model_benchmark") is not False or report.get("exact_checks_passed") is not True:
        raise ValueError("A successful synthetic correctness check is required")
    blocks = report["blocks"]
    order = ["ordinary", "replay", "replay", "ordinary"]
    if report["order"] == "BAAB":
        order = ["replay", "ordinary", "ordinary", "replay"]
    elif report["order"] != "ABBA":
        raise ValueError("Invalid order")
    if len(blocks) != 20:
        raise ValueError("Incomplete block count")
    medians = {"ordinary": [], "replay": []}
    round_ratios = []
    for index, block in enumerate(blocks):
        if block["mode"] != order[index % 4] or block["round"] != index // 4:
            raise ValueError("Unbalanced timing order")
        samples = block["microseconds"]
        if len(samples) != 100 or any(not math.isfinite(x) or x <= 0 for x in samples):
            raise ValueError("Invalid timing samples")
        medians[block["mode"]].append(statistics.median(samples))
        if index % 4 == 3:
            a = statistics.median(medians["ordinary"][-2:])
            b = statistics.median(medians["replay"][-2:])
            round_ratios.append(a / b)
    ordinary, replay = (statistics.median(medians[mode]) for mode in ("ordinary", "replay"))
    return {"ordinary_us": ordinary, "replay_us": replay,
            "elapsed_reduction_percent": 100 * (1 - replay / ordinary),
            "speed_ratio": ordinary / replay, "round_speed_ratios": round_ratios,
            "block_medians_us": medians, "capture_ms": report["capture_ms"],
            "retained_bytes": report["retained_bytes"], "peak_mlx_bytes": report["peak_mlx_bytes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary, library = args.binary.resolve(strict=True), args.library.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    linkage = subprocess.check_output(["ldd", str(binary)], text=True)
    mlx_lines = [line for line in linkage.splitlines() if "libmlx" in line]
    if len(mlx_lines) != 1 or "=>" not in mlx_lines[0]:
        raise RuntimeError("Cannot establish MLX linkage")
    linked = Path(mlx_lines[0].split("=>", 1)[1].strip().split(" ", 1)[0]).resolve(strict=True)
    if linked != library:
        raise RuntimeError("Unexpected linked MLX library")
    identity = {"binary": str(binary), "library": str(library),
                "binary_sha256": digest(binary), "library_sha256": digest(library),
                "runner_sha256": digest(__file__),
                "source_sha256": digest(Path(__file__).with_name("benchmark.cpp"))}
    (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    (output / "ldd.txt").write_text(linkage)
    summary = {"model_benchmark": False, "shared_gpu": True, "runs": []}
    env = dict(os.environ, CUDA_HOME="/usr/local/cuda-13.0", MIDNIGHT_CUDA_EXPERT_QMV="0",
               MLX_USE_CUDA_GRAPHS="1", MLX_PTX_CACHE_DIR=str(output / "ptx"),
               CUDA_CACHE_PATH=str(output / "cuda-cache"))
    # Three fresh-process repetitions, with counterbalanced starting order.
    for repetition in range(3):
        for workload in ("small-fp32", "stack-fp32", "stack-fp16"):
            name = f"{workload}-{repetition}"
            order = "BAAB" if repetition % 2 else "ABBA"
            command = [str(binary), workload, order]
            before = snapshot()
            if digest(binary) != identity["binary_sha256"] or digest(library) != identity["library_sha256"]:
                raise RuntimeError("Binary/library changed during campaign")
            with (output / (name + ".gpu.csv")).open("w") as telemetry:
                monitor = subprocess.Popen([
                    "nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,clocks.sm",
                    "--format=csv,noheader,nounits", "-l", "1"], stdout=telemetry, stderr=subprocess.STDOUT)
                try:
                    with (output / (name + ".json")).open("w") as stdout, (output / (name + ".stderr")).open("w") as stderr:
                        result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, timeout=120)
                finally:
                    monitor.terminate()  # Only this runner's own telemetry child.
                    monitor.wait(timeout=10)
            after = snapshot()
            status = {"command": command, "returncode": result.returncode, "before": before, "after": after}
            (output / (name + ".status.json")).write_text(json.dumps(status, indent=2) + "\n")
            if result.returncode:
                raise RuntimeError(f"{name} failed; inspect saved stderr")
            report = json.loads((output / (name + ".json")).read_text())
            if report["workload"] != workload or report["order"] != order:
                raise ValueError("Unexpected workload/order")
            summary["runs"].append(dict(workload=workload, repetition=repetition, **summarize(report)))
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(name, json.dumps(summary["runs"][-1]), flush=True)
    if digest(binary) != identity["binary_sha256"] or digest(library) != identity["library_sha256"]:
        raise RuntimeError("Binary/library changed during campaign")
    summary["complete"] = True
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
