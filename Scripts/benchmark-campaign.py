#!/usr/bin/env python3
"""Run paired native Midnight benchmarks. Python standard library only; never builds models."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
RUNTIME_OPTIONS = {"--engine", "--tokens", "--warmups", "--context-length", "--prefill-step-size", "--kv-compression", "--reasoning-effort"}
QUALITY_OPTIONS = {"--max-tokens-per-sample", "--prefill-step-size"}
LIMITATIONS = [
    "Runtime uses greedy natural generation: quantized models can follow different token trajectories. Ratios describe end-to-end native generation, not isolated kernel speed or equal-output quality.",
    "Every arm starts a new process and performs its own warmups; loading is excluded from native decode/TTFT metrics but not wall time. OS file caches are not flushed.",
    "The order seed controls scheduling only. The native runtime has no sampling-seed flag; no unsupported seed argument is injected.",
    "A release directory is required, but directory names and hashes do not prove compiler optimization flags. Build these artifacts with the documented release commands.",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256_file(path, sampled=False):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        if sampled:
            size = Path(path).stat().st_size
            digest.update(str(size).encode() + b"\0")
            digest.update(stream.read(1024 * 1024))
            if size > 1024 * 1024:
                stream.seek(max(1024 * 1024, size - 1024 * 1024))
                digest.update(stream.read())
        else:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def file_info(path, sampled=False):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sample_sha256" if sampled else "sha256": sha256_file(path, sampled)}


def command_output(args, cwd=None):
    try:
        result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=15)
        return {"exit_code": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def git_info(path):
    head = command_output(["git", "rev-parse", "HEAD"], path)
    status = command_output(["git", "status", "--porcelain", "--untracked-files=normal"], path)
    diff = command_output(["git", "diff", "HEAD", "--binary"], path)
    return {"path": str(path), "head": head, "status": status,
            "tracked_diff_sha256": hashlib.sha256(diff.get("stdout", "").encode()).hexdigest() if diff.get("exit_code") == 0 else None}


def machine_state():
    if sys.platform == "darwin":
        return {"memory": command_output(["vm_stat"]),
                "swap": command_output(["sysctl", "vm.swapusage"]),
                "power": command_output(["pmset", "-g", "batt"]),
                "thermal": command_output(["pmset", "-g", "therm"])}
    return {"load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            "meminfo": Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else None}


def model_info(model, full_hash=False):
    path = Path(model["path"])
    weights = sorted(path.glob("*.safetensors"))
    metadata = sorted({*path.glob("*.json"), *path.glob("*.model"), *path.glob("*.jinja"), *path.glob("*.tiktoken")})
    return {"label": model["label"], "path": str(path),
            "metadata": [file_info(p) for p in metadata if p.is_file()],
            "weights": [file_info(p, sampled=not full_hash) for p in weights],
            "weight_hash_policy": "full_sha256" if full_hash else "size + first/last 1 MiB sample SHA256; NOT a full content hash"}


def provenance(manifest, full_hash):
    dependencies = ROOT / ".build/checkouts"
    selected_env = {k: v for k, v in os.environ.items()
                    if (k.startswith(("MODEL_RUNNER_", "MIDNIGHT_", "MLX_")) or k in {"SPM_CUDA", "CUDA_VISIBLE_DEVICES"})
                    and not any(secret in k.upper() for secret in ("TOKEN", "SECRET", "PASSWORD", "KEY"))}
    result = {"created_at": now(), "python": sys.version, "platform": platform.platform(),
              "machine": platform.machine(), "cpu_count": os.cpu_count(), "compiler": command_output(["swift", "--version"]), "environment": selected_env,
              "repository": git_info(ROOT), "orchestrator": file_info(Path(__file__)),
              "dependencies": {name: git_info(path) for name, path in
                               [(name, dependencies / name) for name in ("mlx-swift", "mlx-swift-lm", "swift-transformers")]
                               + [("mlx-core", dependencies / "mlx-swift/Source/Cmlx/mlx"),
                                  ("mlx-c", dependencies / "mlx-swift/Source/Cmlx/mlx-c")]
                               if path.exists()},
              "source_files": [{"relative_path": str(p.relative_to(ROOT)), **file_info(p)}
                               for p in sorted({*(p for folder in ("Sources", "Scripts", "Patches", "Tests")
                               for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts),
                               *ROOT.glob("*.sh"), *ROOT.glob("Package.*")})],
              "models": [model_info(m, full_hash) for m in manifest["models"]],
              "binaries": {}, "corpora": [], "start_state": machine_state()}
    if (ROOT / "Package.resolved").exists():
        result["package_resolved"] = file_info(ROOT / "Package.resolved")
    if sys.platform == "darwin":
        result["hardware"] = command_output(["sysctl", "hw.model", "hw.memsize", "machdep.cpu.brand_string"])
        result["displays"] = command_output(["system_profiler", "SPDisplaysDataType", "-json"])
    for mode, path in manifest["binaries"].items():
        result["binaries"][mode] = {"executable": file_info(path)}
        metal = Path(path).parent / "mlx.metallib"
        result["binaries"][mode]["metallib"] = file_info(metal) if metal.exists() else None
    if manifest.get("process_memory_probe"):
        result["process_memory_probe"] = file_info(manifest["process_memory_probe"])
    for corpus in manifest.get("quality", {}).get("corpora", []):
        result["corpora"].append({"label": corpus["label"], **file_info(corpus["path"])})
    return result


def absolute_path(value, directory=False):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"Expected an explicit absolute local path: {value!r}")
    path = Path(value).resolve(strict=True)
    if (directory and not path.is_dir()) or (not directory and not path.is_file()):
        raise ValueError(f"Wrong path type: {path}")
    return str(path)


def named_items(items, description):
    if not isinstance(items, list) or not items:
        raise ValueError(f"{description} must be a nonempty list")
    labels = set()
    for item in items:
        if not isinstance(item, dict) or not LABEL.fullmatch(item.get("label", "")):
            raise ValueError(f"Invalid {description} label; use letters, digits, ._- (max 64)")
        if item["label"] in labels:
            raise ValueError(f"Duplicate {description} label: {item['label']}")
        labels.add(item["label"])
    return labels


def native_args(section, mode):
    args = section.get("native_args", [])
    if not isinstance(args, list) or any(not isinstance(x, str) for x in args):
        raise ValueError("native_args must be a JSON string array")
    allowed = RUNTIME_OPTIONS if mode == "runtime" else QUALITY_OPTIONS
    flags = {"--allow-early-stop"} if mode == "runtime" else {"--cpu"}
    seen, values, index = set(), {}, 0
    while index < len(args):
        option = args[index]
        if option in seen or option not in allowed | flags:
            raise ValueError(f"Unsupported/duplicate {mode} native argument: {option}. The campaign owns prompt, output and --trials.")
        seen.add(option)
        if option in flags:
            values[option] = True
            index += 1
        else:
            if index + 1 == len(args) or args[index + 1].startswith("--"):
                raise ValueError(f"Missing value for {option}")
            values[option] = args[index + 1]
            index += 2
    bounds = {"--tokens": (1, 2048), "--warmups": (0, 20), "--max-tokens-per-sample": (2, 32768)}
    if mode == "quality":
        bounds["--prefill-step-size"] = (0, 8192)
    for option, (low, high) in bounds.items():
        if option in values and not low <= int(values[option]) <= high:
            raise ValueError(f"{option} must be in {low}...{high}")
    if "--engine" in values and values["--engine"] not in {"auto", "metal", "cuda", "cpu"}:
        raise ValueError("Invalid --engine")
    if "--reasoning-effort" in values and values["--reasoning-effort"] not in {"low", "medium", "high"}:
        raise ValueError("--reasoning-effort must be low, medium, or high")
    return args, values


def arm_args(manifest, mode, model_label):
    """Per-arm settings override shared values without permitting positional arguments."""
    shared_args, shared = native_args(manifest[mode], mode)
    model = next(m for m in manifest["models"] if m["label"] == model_label)
    extra_args, extra = native_args({"native_args": model.get(f"{mode}_native_args", [])}, mode)
    merged = {**shared, **extra}
    output = []
    for key, value in merged.items():
        output.append(key)
        if value is not True:
            output.append(value)
    if mode == "quality" and int(merged.get("--prefill-step-size", 0)) == 0 and int(merged.get("--max-tokens-per-sample", 512)) > 2048:
        raise ValueError("Quality samples above 2048 tokens require positive --prefill-step-size")
    return output, merged


def arm_environment(manifest, model_label):
    model = next(m for m in manifest["models"] if m["label"] == model_label)
    effective = {**manifest.get("environment", {}), **model.get("environment", {})}
    environment = os.environ.copy()
    # Explicitly allowlisted but omitted settings are unset, avoiding ambient opt-ins.
    for key in manifest.get("environment_allowlist", []):
        environment.pop(key, None)
    environment.update(effective)
    return environment, {key: effective.get(key) for key in manifest.get("environment_allowlist", [])}


def validate_manifest(manifest, mode):
    if manifest.get("process_memory_probe"):
        probe = absolute_path(manifest["process_memory_probe"])
        if sys.platform != "darwin" or not os.access(probe, os.X_OK):
            raise ValueError("process_memory_probe requires an executable macOS ledger probe")
        manifest["process_memory_probe"] = probe
    if manifest.get("version") != 1:
        raise ValueError("Manifest version must be 1")
    labels = named_items(manifest.get("models"), "model")
    if len(labels) < 2 or manifest.get("baseline") not in labels:
        raise ValueError("At least two models and a baseline matching one label are required")
    for model in manifest["models"]:
        model["path"] = absolute_path(model["path"], directory=True)
        if not (Path(model["path"]) / "config.json").is_file() or not list(Path(model["path"]).glob("*.safetensors")):
            raise ValueError(f"Model needs config.json and local safetensors: {model['path']}")
    pairs = manifest.get("pairs", 4)
    if type(pairs) is not int or pairs < 2 or pairs > 100 or pairs % 2:
        raise ValueError("pairs must be an even integer in 2...100 (at least 4 for acceptance)")
    manifest["pairs"] = pairs
    allowed_environment = manifest.get("environment_allowlist", [])
    if not isinstance(allowed_environment, list) or any(not isinstance(k, str) or not re.fullmatch(r"(?:MODEL_RUNNER_|MLX_)[A-Z0-9_]+", k) for k in allowed_environment):
        raise ValueError("environment_allowlist must name explicit MODEL_RUNNER_/MLX_ variables")
    for settings in [manifest.get("environment", {}), *(m.get("environment", {}) for m in manifest["models"])]:
        if not isinstance(settings, dict) or any(k not in allowed_environment or not isinstance(v, str) for k, v in settings.items()):
            raise ValueError("Every environment value must be a string and its key explicitly allowlisted")
    if type(manifest.get("seed", 0)) is not int:
        raise ValueError("seed must be an integer")
    manifest.setdefault("seed", 0)
    threshold = manifest.setdefault("maximum_drift_fraction", 0.10)
    if not isinstance(threshold, (float, int)) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("maximum_drift_fraction must be between 0 and 1")
    modes = [m for m in ("runtime", "quality") if m in manifest and (mode == "all" or m == mode)]
    if not modes:
        raise ValueError(f"Manifest has no workloads for mode {mode}")
    binaries = {}
    for current in modes:
        binary = absolute_path(manifest["binaries"][current])
        configuration_parts = {part.casefold() for part in Path(binary).parts}
        if "release" not in configuration_parts or "debug" in configuration_parts or not os.access(binary, os.X_OK):
            raise ValueError(f"Executable must resolve inside a release directory: {binary}")
        binaries[current] = binary
        section = manifest[current]
        native_args(section, current)
        for model in manifest["models"]:
            _, settings = arm_args(manifest, current, model["label"])
            # Fixed work is essential for paired comparisons, even for configuration sweeps.
            _, common = native_args(section, current)
            fixed = {"--tokens": "256"} if current == "runtime" else {"--max-tokens-per-sample": "512"}
            for key, default in fixed.items():
                if settings.get(key, default) != common.get(key, default):
                    raise ValueError(f"Per-arm {key} must match the shared workload setting")
        items_key = "prompts" if current == "runtime" else "corpora"
        named_items(section.get(items_key), items_key)
        for item in section[items_key]:
            if current == "runtime":
                if not isinstance(item.get("text"), str) or not item["text"].strip():
                    raise ValueError("Prompts require nonempty text")
            else:
                item["path"] = absolute_path(item["path"])
    manifest["binaries"] = binaries
    return manifest, modes


def schedule(manifest, modes):
    rng = random.Random(manifest["seed"])
    cases = [(mode, item, candidate["label"]) for mode in modes
             for item in manifest[mode]["prompts" if mode == "runtime" else "corpora"]
             for candidate in manifest["models"] if candidate["label"] != manifest["baseline"]]
    initial = {(mode, item["label"], label): rng.randrange(2) for mode, item, label in cases}
    result = []
    for repetition in range(manifest["pairs"]):
        order = list(cases)
        rng.shuffle(order)
        for mode, item, candidate in order:
            key = (mode, item["label"], candidate)
            arms = [manifest["baseline"], candidate]
            if (initial[key] + repetition) % 2:
                arms.reverse()
            pair_id = f"{mode}-{item['label']}-{candidate}-{repetition:03d}"
            for position, label in enumerate(arms):
                result.append({"pair_id": pair_id, "mode": mode, "workload": item["label"],
                               "candidate": candidate, "repetition": repetition, "position": position,
                               "model": label, "baseline_first": arms[0] == manifest["baseline"]})
    return result


def build_command(manifest, run, raw_path):
    model = next(m for m in manifest["models"] if m["label"] == run["model"])
    mode = run["mode"]
    section = manifest[mode]
    item = next(w for w in section["prompts" if mode == "runtime" else "corpora"] if w["label"] == run["workload"])
    args = [manifest["binaries"][mode], model["path"]]
    if mode == "quality":
        args.append(item["path"])
    args.append(str(raw_path))
    args.extend(arm_args(manifest, mode, run["model"])[0])
    if mode == "runtime":
        args.extend(["--trials", "1", "--prompt", item["text"]])
    return args


def finite_number(value, name, positive=True):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"Invalid {name}: {value!r}")
    return value


def extract_measurement(report, manifest, run):
    if report.get("format") != 1 or report.get("status") != "measured":
        raise ValueError("Native report must have format=1 and status=measured")
    model = next(m for m in manifest["models"] if m["label"] == run["model"])
    if Path(report.get("model_path", "")).resolve() != Path(model["path"]):
        raise ValueError("Native report model_path does not match the scheduled model")
    _, values = arm_args(manifest, run["mode"], run["model"])
    if run["mode"] == "runtime":
        workload = next(p for p in manifest["runtime"]["prompts"] if p["label"] == run["workload"])
        if report.get("prompt") != workload["text"]:
            raise ValueError("Native report prompt differs from manifest")
        reasoning_effort = report.get("reasoning_effort")
        if reasoning_effort != values.get("--reasoning-effort"):
            raise ValueError("Native report reasoning_effort differs from manifest")
        requested = int(values.get("--tokens", 256))
        if report.get("requested_tokens") != requested or report.get("measured_trials") != 1 or len(report.get("trials", [])) != 1:
            raise ValueError("Native report does not match requested token/trial settings")
        trial = report["trials"][0]
        metrics = trial["metrics"]
        if metrics["generation_token_count"] != requested:
            raise ValueError("Incomplete generation: early-stopped output cannot establish a fixed-length speed comparison")
        identity = metrics.get("prompt_token_id_fingerprint", trial.get("prompt_token_id_fingerprint", report.get("prompt_token_id_fingerprint")))
        return {"decode_tokens_per_second": finite_number(metrics["tokens_per_second"], "decode rate"),
                "prefill_tokens_per_second": finite_number(metrics["prompt_tokens_per_second"], "prefill rate"),
                "ttft_ms": finite_number(trial["time_to_first_token_milliseconds"], "TTFT"),
                "prompt_tokens": finite_number(metrics["prompt_token_count"], "prompt count"),
                "generation_tokens": requested, "prompt_token_id_fingerprint": identity,
                "content_sha256": hashlib.sha256(trial.get("content", "").encode()).hexdigest(),
                "engine": report.get("engine"), "native_mode": trial.get("mode"),
                "total_milliseconds": trial.get("total_milliseconds"),
                "peak_active_memory_bytes": trial.get("peak_active_memory_bytes"),
                "context_length": report.get("context_length"),
                "prefill_step_size": report.get("prefill_step_size"),
                "kv_compression": report.get("kv_compression"),
                "reasoning_effort": reasoning_effort,
                "memory_limit_bytes": report.get("memory_limit_bytes")}

    expected_corpus = next(c["path"] for c in manifest["quality"]["corpora"] if c["label"] == run["workload"])
    if Path(report.get("corpus_path", "")).resolve() != Path(expected_corpus):
        raise ValueError("Native report corpus_path differs from manifest")
    if report.get("maximum_tokens_per_sample") != int(values.get("--max-tokens-per-sample", 512)):
        raise ValueError("Native report maximum_tokens_per_sample differs from manifest")
    requested_prefill = int(values.get("--prefill-step-size", 0))
    actual_prefill = report.get("prefill_step_size")
    if actual_prefill is None and "--prefill-step-size" not in values:
        actual_prefill = 0  # Legacy reports used cache:nil all-at-once scoring.
    if actual_prefill != requested_prefill:
        raise ValueError("Native report prefill_step_size differs from manifest")
    if report.get("metric") != "teacher_forced_next_token_nll" or not report.get("token_id_fingerprint") or not report.get("corpus_fingerprint"):
        raise ValueError("Quality report is missing its NLL metric or corpus/token identity")
    nll = finite_number(report["token_weighted_nll"], "NLL", positive=False)
    count = finite_number(report["scored_token_count"], "scored token count")
    nll_sum = finite_number(report["nll_sum"], "NLL sum", positive=False)
    if not math.isclose(nll, nll_sum / count, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("Quality NLL is inconsistent with its token count/sum")
    return {"token_weighted_nll": nll, "perplexity": finite_number(report["perplexity"], "perplexity"),
            "scored_tokens": count, "token_id_fingerprint": report["token_id_fingerprint"],
            "corpus_fingerprint": report["corpus_fingerprint"], "device": report.get("device"),
            "add_special_tokens": report.get("add_special_tokens"), "sample_count": report.get("sample_count"),
            "prefill_step_size": actual_prefill}


def execute(manifest, run, directory, timeout):
    name = f"{run['pair_id']}-{run['position']}-{run['model']}"
    path = directory / name
    path.mkdir()
    raw = path / "native.json"
    args = build_command(manifest, run, raw)
    environment, explicit_environment = arm_environment(manifest, run["model"])
    record = {**run, "environment": explicit_environment, "command": args, "directory": str(path), "started_at": now(), "status": "failed"}
    write_json(path / "invocation.json", record)
    start = time.monotonic()
    with (path / "stdout.log").open("wb") as stdout, (path / "stderr.log").open("wb") as stderr:
        process = None
        memory_process = None
        memory_stream = None
        memory_errors = None
        try:
            process = subprocess.Popen(args, cwd=ROOT, stdout=stdout, stderr=stderr, env=environment, start_new_session=True)
            if manifest.get("process_memory_probe"):
                memory_stream = (path / "process-memory.jsonl").open("xb")
                memory_errors = (path / "process-memory-stderr.log").open("xb")
                memory_process = subprocess.Popen([manifest["process_memory_probe"], str(process.pid), "50"],
                                                  stdout=memory_stream, stderr=memory_errors)
            record["exit_code"] = process.wait(timeout=timeout)
            if record["exit_code"] != 0:
                raise ValueError(f"Native process exited with {record['exit_code']}")
            record["measurement"] = extract_measurement(json.loads(raw.read_text()), manifest, run)
            record["status"] = "measured"
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            record["error"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "timeout"
            record["interrupted"] = isinstance(error, KeyboardInterrupt)
        except (OSError, ValueError, KeyError, TypeError) as error:
            record["error"] = str(error)
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        finally:
            if memory_process is not None:
                try:
                    memory_code = memory_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    memory_process.kill()
                    memory_code = memory_process.wait()
                record["process_memory_probe_exit_code"] = memory_code
            for stream in (memory_stream, memory_errors):
                if stream is not None:
                    stream.close()
    if manifest.get("process_memory_probe"):
        try:
            events = [json.loads(line) for line in (path / "process-memory.jsonl").read_text().splitlines()]
            record["process_memory"] = process_memory_summary(events, record.get("process_memory_probe_exit_code"))
        except (OSError, ValueError, KeyError, TypeError) as error:
            record["process_memory"] = {"status": "unavailable", "error": str(error)}
    record["elapsed_seconds"] = time.monotonic() - start
    record["finished_at"] = now()
    write_json(path / "invocation.json", record)
    return record


def process_memory_summary(events, exit_code):
    samples = [s for s in events if s.get("event") == "sample"]
    if exit_code != 0 or not samples:
        return {"status": "unavailable", "probe_exit_code": exit_code, "samples": len(samples)}
    fields = ("physical_footprint_bytes", "resident_size_bytes", "wired_size_bytes",
              "observed_lifetime_max_physical_footprint_bytes")
    for sample in samples:
        for field in fields:
            value = sample.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(f"Invalid kernel memory observation: {field}")
    return {"status": "observed", "samples": len(samples), "sampling_interval_ms": 50,
            "sampled_peak_physical_footprint_bytes": max(s["physical_footprint_bytes"] for s in samples),
            "observed_lifetime_max_physical_footprint_bytes": max(s["observed_lifetime_max_physical_footprint_bytes"] for s in samples),
            "sampled_peak_resident_bytes": max(s["resident_size_bytes"] for s in samples),
            "sampled_peak_wired_bytes": max(s["wired_size_bytes"] for s in samples),
            "limitations": "Kernel lifetime maximum observed before exit includes load/warmup/trials. The final unsampled interval is not certified; this is not a serving-concurrency memory measurement."}


def pair_errors(base, candidate, mode):
    if mode == "runtime":
        fields = ["engine", "native_mode", "prompt_tokens", "generation_tokens"]
        a, b = base.get("prompt_token_id_fingerprint"), candidate.get("prompt_token_id_fingerprint")
        errors = ["prompt_token_id_fingerprint mismatch"] if (a is not None or b is not None) and a != b else []
    else:
        fields = ["device", "scored_tokens", "sample_count", "add_special_tokens", "token_id_fingerprint", "corpus_fingerprint"]
        errors = []
    return errors + [f"{field} mismatch" for field in fields if base.get(field) != candidate.get(field)]


def metric_summary(valid, key, lower_better, manifest, common_reasons):
    baseline = [a["measurement"][key] for a, _ in valid]
    candidate = [b["measurement"][key] for _, b in valid]
    ratios = [a / b if lower_better else b / a for a, b in zip(baseline, candidate)]
    half = len(valid) // 2
    drifts = [abs(statistics.median(values[half:]) / statistics.median(values[:half]) - 1) if half else None
              for values in (baseline, candidate)]
    orders = [[ratios[i] for i, (a, _) in enumerate(valid) if a["baseline_first"] == value] for value in [True, False]]
    order_effect = abs(statistics.median(orders[0]) / statistics.median(orders[1]) - 1) if all(orders) else None
    reasons = list(common_reasons)
    for arm, drift in zip(("baseline", "candidate"), drifts):
        if drift is None or drift > manifest["maximum_drift_fraction"]:
            reasons.append(f"{arm} drift exceeds threshold or is unavailable")
    if order_effect is None or order_effect > manifest["maximum_drift_fraction"]:
        reasons.append("AB/BA order effect exceeds threshold or is unavailable")
    rng = random.Random(manifest["seed"])
    bootstrap = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
    median = statistics.median(ratios)
    return {"baseline_median": statistics.median(baseline), "candidate_median": statistics.median(candidate),
            "diagnostic_paired_ratios": ratios, "diagnostic_paired_median_ratio": median,
            "accepted_paired_median_ratio": None if reasons else median,
            "paired_ratio_min": min(ratios), "paired_ratio_max": max(ratios),
            "paired_ratio_median_absolute_deviation": statistics.median(abs(r - median) for r in ratios),
            "diagnostic_paired_bootstrap_95_percent_interval": [bootstrap[49], bootstrap[1949]],
            "accepted_paired_bootstrap_95_percent_interval": None if reasons else [bootstrap[49], bootstrap[1949]],
            "baseline_drift_fraction": drifts[0], "candidate_drift_fraction": drifts[1],
            "order_effect_fraction": order_effect, "acceptance_reasons": reasons,
            "ratio_direction": "baseline/candidate (latency)" if lower_better else "candidate/baseline (rate)",
            "interpretation": "Ratio >1 favors candidate. Accepted means eligible measurement, not proven improvement. Bootstrap resamples pairs (2000 draws); few pairs, thermal autocorrelation and multiple comparisons limit inference."}


def summarize(manifest, records, expected_schedule):
    comparisons = []
    cases = sorted({(r["mode"], r["workload"], r["candidate"]) for r in expected_schedule})
    for mode, workload, candidate in cases:
        selected = [r for r in records if (r["mode"], r["workload"], r["candidate"]) == (mode, workload, candidate)]
        valid, rejected = [], []
        for repetition in range(manifest["pairs"]):
            pair = [r for r in selected if r["repetition"] == repetition]
            if len(pair) != 2 or any(r["status"] != "measured" for r in pair):
                rejected.append({"repetition": repetition, "reasons": ["missing or failed arm"]})
                continue
            base = next(r for r in pair if r["model"] == manifest["baseline"])
            other = next(r for r in pair if r["model"] == candidate)
            errors = pair_errors(base["measurement"], other["measurement"], mode)
            if errors:
                rejected.append({"repetition": repetition, "reasons": errors})
            else:
                valid.append((base, other))
        result = {"mode": mode, "workload": workload, "baseline": manifest["baseline"], "candidate": candidate,
                  "expected_pairs": manifest["pairs"], "valid_pairs": len(valid), "rejected_pairs": rejected,
                  "comparable": not rejected and len(valid) == manifest["pairs"]}
        # A matching pair is insufficient if the native tokenization changes across repeats.
        identity_key = "prompt_token_id_fingerprint" if mode == "runtime" else "token_id_fingerprint"
        identities = {arm["measurement"].get(identity_key) for pair in valid for arm in pair}
        if len(identities) > 1:
            result["comparable"] = False
            result["cross_repeat_error"] = "token identities changed between repetitions"
        if valid and mode == "runtime":
            common_reasons = []
            if not result["comparable"]:
                common_reasons.append("failed or incomparable pairs")
            if len(valid) < 4:
                common_reasons.append("fewer than four complete pairs; exploratory only")
            verified = all(isinstance(arm["measurement"].get("prompt_token_id_fingerprint"), str)
                           and arm["measurement"]["prompt_token_id_fingerprint"] for pair in valid for arm in pair)
            if not verified:
                common_reasons.append("exact prompt token identity unavailable")
            result["native_settings_by_model"] = {label: arm_args(manifest, mode, label)[1] for label in [manifest["baseline"], candidate]}
            result["prompt_token_identity_verified"] = verified
            result["outputs_match"] = all(a["measurement"]["content_sha256"] == b["measurement"]["content_sha256"] for a, b in valid)
            metrics = {}
            for metric, key, lower_better in [("decode", "decode_tokens_per_second", False),
                                               ("prefill", "prefill_tokens_per_second", False),
                                               ("ttft", "ttft_ms", True)]:
                metrics[metric] = metric_summary(valid, key, lower_better, manifest, common_reasons)
            result["metrics"] = metrics
            decode = metrics["decode"]
            result.update({"baseline_median_decode_tokens_per_second": decode["baseline_median"],
                           "candidate_median_decode_tokens_per_second": decode["candidate_median"],
                           "diagnostic_paired_decode_ratios": decode["diagnostic_paired_ratios"],
                           "accepted_paired_median_decode_ratio": decode["accepted_paired_median_ratio"],
                           "acceptance_reasons": decode["acceptance_reasons"],
                           "baseline_drift_fraction": decode["baseline_drift_fraction"],
                           "candidate_drift_fraction": decode["candidate_drift_fraction"],
                           "order_effect_fraction": decode["order_effect_fraction"],
                           "baseline_median_ttft_ms": metrics["ttft"]["baseline_median"],
                           "candidate_median_ttft_ms": metrics["ttft"]["candidate_median"],
                           "accepted_paired_median_ttft_speed_ratio": metrics["ttft"]["accepted_paired_median_ratio"],
                           "baseline_median_prefill_tokens_per_second": metrics["prefill"]["baseline_median"],
                           "candidate_median_prefill_tokens_per_second": metrics["prefill"]["candidate_median"]})
            for field in ("total_milliseconds", "peak_active_memory_bytes"):
                for label, position in (("baseline", 0), ("candidate", 1)):
                    measured = [pair[position]["measurement"].get(field) for pair in valid]
                    result[f"{label}_median_{field}"] = statistics.median(measured) if all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in measured) else None
        elif valid:
            result["native_settings_by_model"] = {label: arm_args(manifest, mode, label)[1] for label in [manifest["baseline"], candidate]}
            result["scoring_prefill_step_size_by_model"] = {manifest["baseline"]: valid[0][0]["measurement"].get("prefill_step_size", 0), candidate: valid[0][1]["measurement"].get("prefill_step_size", 0)}
            result.update({"baseline_median_nll": statistics.median(a["measurement"]["token_weighted_nll"] for a, _ in valid),
                           "candidate_median_nll": statistics.median(b["measurement"]["token_weighted_nll"] for _, b in valid),
                           "accepted_paired_median_nll_delta": statistics.median(b["measurement"]["token_weighted_nll"] - a["measurement"]["token_weighted_nll"] for a, b in valid) if result["comparable"] else None})
        comparisons.append(result)
    return {"version": 1, "created_at": now(), "scheduled_runs": len(expected_schedule),
            "completed_runs": len(records), "failed_runs": sum(r["status"] != "measured" for r in records),
            "comparisons": comparisons, "limitations": LIMITATIONS,
            "status": "complete" if len(records) == len(expected_schedule) and all(c["comparable"] for c in comparisons) else "incomplete_or_invalid"}


def example(binary_dir, models_root):
    prefix = "Ministral-3-14B-Instruct-2512-MLX-Q4-"
    return {"version": 1, "baseline": "standard", "seed": 20260904, "pairs": 4,
            "maximum_drift_fraction": 0.10,
            "binaries": {"runtime": str(binary_dir / "model-runner-runtime-bench"), "quality": str(binary_dir / "model-runner-quality-bench")},
            "models": [{"label": label, "path": str(models_root / f"{prefix}{suffix}-3cea74c")} for label, suffix in
                       [("standard", "Standard"), ("ls2", "ScaleSearch-LS2"), ("awss", "AWSS1")]],
            "runtime": {"native_args": ["--engine", "metal", "--tokens", "256", "--warmups", "1"],
                        "prompts": [{"label": "swift-scheduler", "text": "Write a long, detailed technical tutorial about implementing a lock-free work-stealing scheduler in Swift. Continue with implementation details and code examples until the output limit; do not conclude or summarize early."}]},
            "quality": {"native_args": ["--max-tokens-per-sample", "512"],
                        "corpora": [{"label": "general-smoke", "path": str(ROOT / "Benchmarks/Quality/general-smoke.jsonl")}]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init", help="Write a Ministral standard/LS2/AWSS example; no model execution")
    init.add_argument("--binary-dir", type=Path, required=True)
    init.add_argument("--models-root", type=Path, required=True)
    init.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="Run an explicit manifest; never build or download")
    run.add_argument("manifest", type=Path)
    run.add_argument("--output", type=Path, required=True, help="New result directory; never overwritten")
    run.add_argument("--mode", choices=["all", "runtime", "quality"], default="all")
    run.add_argument("--timeout", type=float, default=1800, help="Maximum seconds per native process")
    run.add_argument("--full-weight-hash", action="store_true")
    run.add_argument("--dry-run", action="store_true", help="Validate and retain provenance/schedule/commands without launching")
    args = parser.parse_args(argv)
    try:
        if args.action == "init":
            if args.output.exists():
                raise ValueError("Refusing to overwrite existing manifest")
            write_json(args.output, example(args.binary_dir.resolve(), args.models_root.resolve()))
            print(args.output.resolve())
            return 0
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("--timeout must be positive and finite")
        manifest, modes = validate_manifest(json.loads(args.manifest.read_text()), args.mode)
        args.output = args.output.resolve()
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "manifest.json", manifest)
        runs = schedule(manifest, modes)
        write_json(args.output / "schedule.json", runs)
        prov = provenance(manifest, args.full_weight_hash)
        write_json(args.output / "provenance.json", prov)
        raw = args.output / "runs"
        raw.mkdir()
        if args.dry_run:
            write_json(args.output / "commands.json", [build_command(manifest, r, raw / f"{r['pair_id']}-{r['position']}-{r['model']}.json") for r in runs])
            print(f"Validated {len(runs)} native invocations; no benchmarks launched. {args.output}")
            return 0
        records = []
        for index, item in enumerate(runs):
            print(f"[{index + 1}/{len(runs)}] {item['pair_id']} {item['model']}", flush=True)
            record = execute(manifest, item, raw, args.timeout)
            records.append(record)
            write_json(args.output / "results.json", records)
            write_json(args.output / "summary.json", summarize(manifest, records, runs))
            if record.get("interrupted"):
                break
        summary = summarize(manifest, records, runs)
        prov["end_state"] = machine_state()
        prov["finished_at"] = now()
        write_json(args.output / "provenance.json", prov)
        write_json(args.output / "summary.json", summary)
        print(f"{summary['status']}: {args.output / 'summary.json'}")
        return 130 if any(r.get("interrupted") for r in records) else (0 if summary["status"] == "complete" else 1)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Campaign error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
