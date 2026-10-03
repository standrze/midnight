#!/usr/bin/env python3
"""CUDA-only stock/candidate probe; no models, services, or defaults are changed."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


def cases():
    result = []
    for dtype in ("float16", "bfloat16"):
        for name, n, k in (("gate", 512, 2048), ("gate_up", 1024, 2048), ("down", 2048, 512)):
            result.append(dict(name=f"{name}-{dtype}", dtype=dtype, n=n, k=k, experts=256, selected=8, m=1, bits=4, group=64, eligible=True))
    # Exercise fallback boundaries independently: selection count, M, E, dtype,
    # projection dimensions, quantization bits, and group size.
    for name, overrides in (
        ("top7", dict(selected=7)), ("top9", dict(selected=9)),
        ("prefill", dict(m=2)), ("expert_count", dict(experts=255)),
        ("fp32", dict(dtype="float32")), ("other_projection", dict(n=128)),
        ("q8", dict(bits=8)), ("group128", dict(group=128)),
    ):
        result.append(dict(dict(name=name, dtype="float16", n=512, k=2048, experts=256, selected=8, m=1, bits=4, group=64, eligible=False), **overrides))
    return result


def rounded(values, dtype):
    """Independent CPU rounding for finite test fixtures (ties-to-even BF16)."""
    import numpy as np
    values = np.asarray(values, dtype=np.float32)
    if dtype == "float16":
        return values.astype(np.float16).astype(np.float32)
    if dtype == "bfloat16":
        words = values.view(np.uint32)
        words = (words + np.uint32(0x7FFF) + ((words >> 16) & 1)) & np.uint32(0xFFFF0000)
        return words.view(np.float32)
    return values


def worker(args):
    import numpy as np
    import mlx.core as mx
    if sys.platform != "linux" or not mx.cuda.is_available():
        raise RuntimeError("This probe requires a CUDA MLX build on Linux")
    mx.set_default_device(mx.gpu)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for ordinal, case in enumerate(cases()):
        if args.trace:
            print(f"midnight_expert_fixture_begin {case['name']}", file=sys.stderr, flush=True)
        rng = np.random.default_rng(260926 + ordinal)
        e, n, k, b, m = (case[key] for key in ("experts", "n", "k", "selected", "m"))
        dtype = case["dtype"]
        bits, group = case["bits"], case["group"]
        packed_per_word = 32 // bits
        # Packed words are constructed directly. Signed, cancellation-heavy
        # values come from negative biases and inputs spanning both signs.
        packed = rng.integers(0, 2**32, size=(e, n, k // packed_per_word), dtype=np.uint32)
        scale_factor = 15 / (2**bits - 1)
        scales = rounded(rng.uniform(0.004, 0.08, (e, n, k // group)) * scale_factor, dtype)
        biases = rounded(-((2**bits - 1) / 2) * scales, dtype)
        x = rounded(rng.standard_normal((b, m, k)) / k**0.5, dtype)
        rhs = np.array(([e - 1, 0, 73, 73, 127, 15, 2, e - 2, 19])[:b], dtype=np.uint32)
        lhs = np.array(([7, 2, 4, 4, 1, 0, 6, 3, 8])[:b], dtype=np.uint32) % b
        if case["name"].startswith("gate"):
            lhs[:] = 0
        mlx_dtype = getattr(mx, dtype)
        xm, wm, sm, bm = mx.array(x, dtype=mlx_dtype), mx.array(packed), mx.array(scales, dtype=mlx_dtype), mx.array(biases, dtype=mlx_dtype)
        li, ri = mx.array(lhs), mx.array(rhs)
        mx.eval(xm, wm, sm, bm, li, ri)

        def operation():
            y = mx.gather_qmm(xm, wm, sm, bm, lhs_indices=li, rhs_indices=ri, transpose=True, group_size=group, bits=bits)
            mx.eval(y)
            return y

        repeats = 1 if args.trace or args.correctness_only else args.iterations
        warmups = 0 if args.trace or args.correctness_only else args.warmups
        for _ in range(warmups):
            operation()
        timings = []
        for _ in range(repeats):
            start = None if args.correctness_only else time.perf_counter_ns()
            y = operation()
            mx.synchronize()
            if start is not None:
                timings.append((time.perf_counter_ns() - start) / 1e6)
        actual = np.array(y.astype(mx.float32))
        if actual.shape != (b, m, n):
            raise RuntimeError(f"Unexpected output shape for {case['name']}: {actual.shape}")
        if not np.isfinite(actual).all():
            raise RuntimeError(f"nonfinite output: {case['name']}")
        # FP64 accumulation of separately rounded dequantized weights is a
        # diagnostic reference, not a substitute for the exact stock gate.
        reference = np.empty((b, m, n), dtype=np.float64)
        for selection in range(b):
            codes = ((packed[rhs[selection], :, :, None] >> np.arange(0, 32, bits, dtype=np.uint32)) & (2**bits - 1)).reshape(n, k).astype(np.float32)
            weight = rounded(codes * np.repeat(scales[rhs[selection]], group, axis=-1), dtype)
            weight = rounded(weight + np.repeat(biases[rhs[selection]], group, axis=-1), dtype)
            reference[selection] = x[lhs[selection]].astype(np.float64) @ weight.astype(np.float64).T
        digest = hashlib.sha256(actual.tobytes()).hexdigest()
        np.save(output / (case["name"] + ".npy"), actual)
        record = dict(case, output_sha256=digest,
                      max_reference_error=float(np.max(np.abs(actual-reference))),
                      rms_reference_error=float(np.sqrt(np.mean((actual-reference)**2))))
        if timings:
            record.update(median_ms=statistics.median(timings), samples_ms=timings)
        records.append(record)
        del xm, wm, sm, bm, li, ri, y, packed, scales, biases, x
        mx.clear_cache()
        if args.trace:
            print(f"midnight_expert_fixture_end {case['name']}", file=sys.stderr, flush=True)
    extension = Path(mx.__file__).resolve()
    metadata = dict(runtime_flag=os.getenv("MIDNIGHT_CUDA_EXPERT_QMV"), trace=args.trace,
                    mlx_version=getattr(mx, "__version__", None), python=sys.executable,
                    mlx_extension=str(extension), mlx_extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
                    timing="not_measured" if args.correctness_only else "host observed synchronized kernel calls; not full-model inference", cases=records)
    (output / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")


def analyze(reports):
    """Strict fixture parity; numeric diagnostics never replace this gate."""
    if len(reports) != 4:
        raise ValueError("Exactly four ABBA reports are required")
    expected_flags = ["0", "1", "1", "0"]
    expected_names = [c["name"] for c in cases()]
    identities = {report.get("mlx_extension_sha256") for report in reports}
    if len(identities) != 1 or not re.fullmatch(r"[0-9a-f]{64}", next(iter(identities)) or ""):
        raise ValueError("Arms must use one identified MLX extension")
    for key in ("mlx_extension", "python", "mlx_version"):
        if any(key not in report for report in reports) or len({r[key] for r in reports}) != 1:
            raise ValueError(f"Mismatched or missing runtime identity: {key}")
    for report, flag in zip(reports, expected_flags):
        if report.get("runtime_flag") != flag or report.get("trace"):
            raise ValueError("Wrong arm or instrumented timing report")
        if [c["name"] for c in report["cases"]] != expected_names:
            raise ValueError("Incomplete or reordered case list")
    rows = []
    for i, name in enumerate(expected_names):
        parts = [r["cases"][i] for r in reports]
        for part in parts:
            if not math.isfinite(part["median_ms"]) or part["median_ms"] <= 0 or len(part["samples_ms"]) < 5:
                raise ValueError("Missing positive measurements")
            if any(not math.isfinite(value) or value <= 0 for value in part["samples_ms"]):
                raise ValueError("Invalid timing sample")
            if part["median_ms"] != statistics.median(part["samples_ms"]):
                raise ValueError("Reported timing median does not match samples")
            if not re.fullmatch(r"[0-9a-f]{64}", part["output_sha256"]):
                raise ValueError("Missing output identity")
        exact = len({p["output_sha256"] for p in parts}) == 1
        baseline = statistics.geometric_mean([parts[0]["median_ms"], parts[3]["median_ms"]])
        candidate = statistics.geometric_mean([parts[1]["median_ms"], parts[2]["median_ms"]])
        rows.append(dict(name=name, exact=exact, baseline_ms=baseline, candidate_ms=candidate,
                         speed_ratio=baseline/candidate,
                         baseline_reference_error=max(parts[j]["max_reference_error"] for j in (0,3)),
                         candidate_reference_error=max(parts[j]["max_reference_error"] for j in (1,2))))
    passed = all(r["exact"] for r in rows)
    return dict(status="exact_fixtures_passed" if passed else "failed_output_parity",
                promotion="blocked_pending_full_model_validation" if passed else "blocked_by_output_parity",
                interpretation="one primitive screen; no production or model-speed claim", cases=rows)


def validate_trace(log, candidate=True):
    """Require each eligible fixture, and reject each fallback, independently."""
    expected = cases()
    completed = []
    current = None
    dispatches = []
    for line in log.splitlines():
        if line.startswith("midnight_expert_fixture_begin "):
            name = line.removeprefix("midnight_expert_fixture_begin ")
            if current is not None or len(completed) >= len(expected) or name != expected[len(completed)]["name"]:
                raise ValueError("Unexpected trace fixture start")
            current, dispatches = expected[len(completed)], []
        elif line.startswith("midnight_cuda_expert_qmv "):
            match = re.fullmatch(r"midnight_cuda_expert_qmv m=1 b=8 n=(\d+) k=(\d+)", line)
            if current is None or match is None:
                raise ValueError("Unscoped or malformed specialized dispatch")
            dispatches.append((int(match[1]), int(match[2])))
        elif line.startswith("midnight_expert_fixture_end "):
            name = line.removeprefix("midnight_expert_fixture_end ")
            if current is None or name != current["name"]:
                raise ValueError("Unexpected trace fixture end")
            required = [(current["n"], current["k"])] if candidate and current["eligible"] else []
            if dispatches != required:
                raise ValueError(f"Incorrect specialized dispatch for fixture {name}: {dispatches}")
            completed.append(dict(name=name, dispatches=len(dispatches)))
            current = None
    if current is not None or len(completed) != len(expected):
        raise ValueError("Incomplete fixture trace")
    return completed


def campaign(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    script = str(Path(__file__).resolve())
    arms = (("a1", 0, False), ("b1", 1, True)) if args.correctness_only else (("trace", 1, True), ("a1", 0, False), ("b1", 1, False), ("b2", 1, False), ("a2", 0, False))
    for label, enabled, trace in arms:
        env = dict(os.environ, MIDNIGHT_CUDA_EXPERT_QMV=str(enabled), MIDNIGHT_CUDA_EXPERT_QMV_TRACE="1" if trace else "0")
        command = [sys.executable, script, "--worker", "--output", str(output / label), "--iterations", str(args.iterations), "--warmups", str(args.warmups)]
        if trace:
            command.append("--trace")
        if args.correctness_only:
            command.append("--correctness-only")
        with (output / (label + ".log")).open("w") as log:
            subprocess.run(command, env=env, check=True, stdout=log, stderr=log)
        if trace:
            verified = validate_trace((output / (label + ".log")).read_text())
            (output / "trace-validation.json").write_text(json.dumps(verified, indent=2) + "\n")
        if not trace or args.correctness_only:
            reports.append(json.loads((output / label / "results.json").read_text()))
    if args.correctness_only:
        # Reuse identity/completeness validation without manufacturing or
        # reporting speed measurements. Timing fields remain absent on disk.
        identities = ("mlx_extension_sha256", "mlx_extension", "python", "mlx_version")
        if any(reports[0].get(key) != reports[1].get(key) for key in identities):
            raise ValueError("Correctness arms use different runtime identities")
        if [r["runtime_flag"] for r in reports] != ["0", "1"]:
            raise ValueError("Wrong correctness arms")
        for report in reports:
            if [c["name"] for c in report["cases"]] != [c["name"] for c in cases()]:
                raise ValueError("Incomplete correctness fixtures")
        comparisons = [dict(name=a["name"], exact=a["output_sha256"] == b["output_sha256"],
                            baseline_reference_error=a["max_reference_error"],
                            candidate_reference_error=b["max_reference_error"])
                       for a, b in zip(reports[0]["cases"], reports[1]["cases"])]
        passed = all(c["exact"] for c in comparisons)
        result = dict(status="exact_fixtures_passed" if passed else "failed_output_parity",
                      promotion="blocked_pending_full_model_validation" if passed else "blocked_by_output_parity",
                      timing="not_measured", cases=comparisons)
    else:
        result = analyze(reports)
    result["patch_sha256"] = hashlib.sha256((Path(__file__).resolve().parents[2] / "Patches/mlx-cuda-expert-qmv-fp32.patch").read_bytes()).hexdigest()
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "exact_fixtures_passed":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=31)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--correctness-only", action="store_true", help="One operation per fixture per arm, no timing measurements")
    args = parser.parse_args()
    if args.iterations < 5 or args.warmups < 1:
        parser.error("Use at least five measurements and one warmup")
    worker(args) if args.worker else campaign(args)


if __name__ == "__main__":
    main()
