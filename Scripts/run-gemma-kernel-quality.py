#!/usr/bin/env python3
"""Plan, then explicitly execute serial paired Gemma kernel-quality processes.

This harness never builds or downloads. Omit --execute for CPU-only preflight.
Quality elapsed times include diagnostics and are not performance measurements.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmark-results/gemma-performance-20260929/quality"
MODELS = {
    "gemma3-270m": (ROOT / "artifacts/gemma3-270m-it-4bit", "MLXLLM.Gemma3TextModel"),
    "gemma4-26b": (Path.home() / ".midnight/models/gemma-4-26B-A4B-it-midnight", "MLXLLM.Gemma4Model"),
    "gemma4-31b": (Path.home() / ".midnight/models/gemma-4-31B-it-midnight", "MLXLLM.Gemma4Model"),
}
SUITES = {"pilot": ("decode-pilot-24.jsonl", 128), "full": ("decode-reference-192.jsonl", 512),
          "wrap": ("decode-window-3.jsonl", 768), "window1024": ("decode-gemma4-window-3.jsonl", 1280)}
FIXED_ENV = {
    "MLX_METAL_AFFINE_Q4_QMV_TAIL": "0", "MIDNIGHT_METAL_SDPA_D512": "0",
    "MIDNIGHT_METAL_SDPA_D256_MASKED": "0", "MIDNIGHT_METAL_SDPA_D256_PRUNE": "1",
    "MIDNIGHT_GEMMA4_BOUNDED_KV": "0", "MIDNIGHT_GEMMA4_EXPERT_GATE_UP": "0",
    "MIDNIGHT_GEMMA4_DENSE_FUSION": "0", "MIDNIGHT_GEMMA4_WINDOW_SLICING": "0",
    "MODEL_RUNNER_PREFIX_CACHE_ENTRIES": "0",
}
ENV_PREFIXES = ("MLX_METAL_", "MIDNIGHT_METAL_", "MIDNIGHT_GEMMA4_", "MIDNIGHT_GEMMA3_")


def module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "Scripts" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def info(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def quantization_bits(value):
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "bits":
                require(type(child) is int and child >= 4, "All quantization metadata must use at least four bits")
                result.append(child)
            else:
                result.extend(quantization_bits(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(quantization_bits(child))
    return result


def model_provenance(path):
    config = json.loads((path / "config.json").read_text())
    bits = quantization_bits(config.get("quantization")) + quantization_bits(config.get("quantization_config"))
    require(bits, "Missing explicit quantization metadata")
    tensors = sorted(path.glob("*.safetensors"))
    require(tensors, "No checkpoint weights found")
    metadata = [info(p) for p in sorted(path.iterdir()) if p.is_file() and p.suffix in (".json", ".jinja")]
    return {"path": str(path), "model_type": config["model_type"], "quantization_bits": sorted(set(bits)),
            "metadata": metadata, "weight_identity_method": "File path, size and mtime only; weights are not content-hashed",
            "weights": [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in tensors]}


def environment(candidate, arm):
    # MLX_DISABLE_COMPILE disables compilation by its presence, even with value 0.
    # A wrapper-body counter cannot prove graph compilation under that override.
    result = {k: v for k, v in os.environ.items()
              if not k.startswith(ENV_PREFIXES) and k != "MLX_DISABLE_COMPILE"}
    result.update(FIXED_ENV)
    # Older experiment binaries predate the Gemma3 report prefix. Scrub inherited
    # values in every arm, but emit this default-off flag only for its own trial.
    if candidate == "gemma3-compiled":
        result["MIDNIGHT_GEMMA3_COMPILED_TAIL"] = "1" if arm == "B" else "0"
    if arm == "B":
        if candidate in ("tail", "combined"):
            result["MLX_METAL_AFFINE_Q4_QMV_TAIL"] = "1"
        if candidate in ("attention", "combined"):
            result["MIDNIGHT_METAL_SDPA_D512"] = "1"
        if candidate == "bounded":
            result["MIDNIGHT_GEMMA4_BOUNDED_KV"] = "1"
    return result


def validate_candidate(model, candidate, mode):
    if candidate == "gemma3-compiled":
        require(model == "gemma3-270m", "The Gemma3 compiled-tail candidate is restricted to Gemma3 270M")
        require(mode == "nll", "The Gemma3 compiled-tail candidate requires NLL reports with trace activation proof")
    elif model == "gemma3-270m":
        require(candidate == "tail", "The attention and bounded-cache experiments do not target Gemma3 270M")


def validate_compiled_tail_activation(report, arm):
    counts = report.get("gemma3_compiled_tail_trace_counts")
    total = report.get("gemma3_compiled_tail_trace_count")
    require(isinstance(counts, list) and counts and all(type(n) is int and n >= 0 for n in counts),
            "Quality binary must report nonnegative Gemma3 per-layer trace counts")
    require(type(total) is int and total == sum(counts), "Gemma3 compiled-tail total trace count does not reconcile")
    require(arm in ("A", "B"), "Unknown Gemma3 compiled-tail experiment arm")
    if arm == "B":
        require(total > 0 and all(n > 0 for n in counts),
                "Gemma3 compiled-tail candidate did not activate every decoder layer")
    else:
        require(total == 0, "Gemma3 compiled-tail baseline unexpectedly activated the experiment")


def command(args, model, corpus, report, binary):
    base = [str(binary), str(model), str(corpus), str(report)]
    if args.mode == "nll":
        return base + ["--max-tokens-per-sample", str(SUITES[args.suite][1]), "--prefill-step-size", "1", "--token-diagnostics"]
    return base + ["--engine", "metal", "--tokens", "512", "--context-length", "8192", "--prefill-step-size", "512", "--kv-compression", "none"]


def run_process(argv, env, log, timeout):
    start = time.monotonic()
    with log.open("xb") as stream:
        process = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            require(code == 0, f"Process exited {code}; inspect {log}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    return time.monotonic() - start


def analyze_nll(reports, plan, output):
    nll = module("analyze-quality-nll.py", "nll_analysis")
    choices = module("analyze-quality-token-diagnostics.py", "choice_analysis")
    loaded = {label: json.loads(path.read_text()) for label, path in reports.items()}
    for run in plan["runs"]:
        report = loaded[run["label"]]
        require(report.get("model_implementation") == plan["expected_model_implementation"], "Wrong native model implementation")
        require(report.get("model_path") == plan["model"]["path"], "Wrong model path")
        require(report.get("tokenization") == "raw_text_add_special_tokens_no_chat_template", "Unexpected NLL tokenization")
        require(report.get("device") == "gpu" and report.get("backend") == "metal", "Expected Metal GPU report")
        expected = {k: v for k, v in run["environment"].items() if k.startswith(ENV_PREFIXES)}
        require(report.get("runtime_environment") == expected, "Native kernel flags do not match process provenance")
        if plan.get("candidate") == "gemma3-compiled":
            require(report.get("model_implementation") == "MLXLLM.Gemma3TextModel", "Expected Gemma3 text model")
            validate_compiled_tail_activation(report, run["arm"])
        if plan.get("candidate") == "bounded":
            cache_types = report.get("cache_implementations")
            require(isinstance(cache_types, list), "Quality binary does not report actual cache implementations")
            bounded = "MLXLMCommon.WindowedKVCache" in cache_types
            require(bounded == (run["arm"] == "B"), "Bounded cache selection does not match the experiment arm")
    paired = nll.analyze(loaded, draws=10000, seed=20260929)
    paired["provenance"] = {"reports": {label: info(path) for label, path in reports.items()}, "plan": info(output / "plan.json")}
    write_new(output / "nll-analysis.json", paired)
    labels = list(loaded)
    write_new(output / "fixed-prefix-analysis.json", choices.analyze(loaded[labels[0]], loaded[labels[1]]))
    repeated = {}
    if len(labels) == 4:
        for arm, left, right in (("A", labels[0], labels[3]), ("B", labels[1], labels[2])):
            a, b = loaded[left], loaded[right]
            result = choices.analyze(a, b)
            repeated[arm] = {"labels": [left, right], "same_sample_nll_exactly": a["samples"] == b["samples"],
                             "same_token_diagnostics_exactly": a["token_diagnostics"] == b["token_diagnostics"],
                             "winner_flip_count": result["winner_flip_count"]}
    write_new(output / "repeatability.json", {"arms": repeated, "interpretation": "Empty for a single pair. Differences require investigation; no quality acceptance threshold is inferred."})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--candidate", choices=("tail", "attention", "combined", "bounded", "gemma3-compiled"), default="tail")
    parser.add_argument("--suite", choices=SUITES, default="pilot")
    parser.add_argument("--mode", choices=("nll", "tasks"), default="nll")
    parser.add_argument("--name", required=True, help="New run directory name; existing evidence is never overwritten")
    parser.add_argument("--binary-dir", type=Path, default=ROOT / ".build/release")
    parser.add_argument("--single-pair", action="store_true", help="Only A/B; omits process repeatability check")
    parser.add_argument("--timeout-seconds", type=int, default=7200, help="Limit for each child; killed as a process group on timeout")
    parser.add_argument("--execute", action="store_true", help="Run GPU processes. Omit to inspect the plan without loading a model")
    args = parser.parse_args(argv)
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.name), "Invalid run directory name")
    require(args.timeout_seconds > 0, "Timeout must be positive")
    validate_candidate(args.model, args.candidate, args.mode)
    require(not (args.model != "gemma3-270m" and args.suite == "wrap"), "Use --suite window1024 to cross Gemma4's larger sliding window")
    require(args.mode != "tasks" or args.suite == "pilot", "Task mode uses the frozen 22-task pilot")
    model, implementation = MODELS[args.model]
    model = model.resolve()
    binary = (args.binary_dir / ("model-runner-quality-bench" if args.mode == "nll" else "model-runner-generation-bench")).resolve()
    library = (args.binary_dir / "mlx.metallib").resolve()
    corpus = DATA / ("reference-corpus/" + SUITES[args.suite][0] if args.mode == "nll" else "generated-pilot/tasks.jsonl")
    output = DATA / "runs" / args.name
    require(not output.exists(), f"Run already exists: {output}")
    model_record = model_provenance(model)
    artifacts = [info(binary), info(library), info(corpus), info(__file__), info(DATA / "corpus-selection.json")]
    if args.mode == "tasks":
        artifacts += [info(DATA / "generated-pilot/answers.jsonl"), info(ROOT / "Scripts/evaluate-generated.py")]
    else:
        artifacts += [info(ROOT / "Scripts/analyze-quality-nll.py"), info(ROOT / "Scripts/analyze-quality-token-diagnostics.py")]
    plan = {"format": 1, "created_at": datetime.now(timezone.utc).isoformat(), "mode": args.mode, "suite": args.suite,
            "candidate": args.candidate, "model": model_record, "expected_model_implementation": implementation,
            "environment_keys_forced_absent": ["MLX_DISABLE_COMPILE"],
            "artifacts": artifacts, "runs": [], "limitations": ["Quality timings are not speed measurements.",
            "A small corpus and a small NLL delta do not prove global model-quality equivalence.",
            "Kernel flags are startup settings. All arms use distinct serial processes with identical checkpoint files.",
            "Weight files are checked by size and mtime, not content hash; metadata and executable artifacts are SHA256 hashed."]}
    if args.mode == "nll":
        plan["maximum_scored_tokens_per_arm"] = sum(1 for line in corpus.read_text().splitlines() if line.strip()) * (SUITES[args.suite][1] - 1)
    for index, arm in enumerate("AB" if args.single_pair else "ABBA", 1):
        label = f"{index:02d}-{arm}"
        env = environment(args.candidate, arm)
        plan["runs"].append({"label": label, "arm": arm, "command": command(args, model, corpus, output / f"{label}.json", binary),
                             "environment": {k: v for k, v in env.items() if k in FIXED_ENV or k.startswith(ENV_PREFIXES)}})
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if args.mode == "nll":
        help_output = subprocess.run([str(binary), "--help"], capture_output=True, text=True, check=True, timeout=30).stdout
        require("--token-diagnostics" in help_output, "Quality binary is stale: build the diagnostic-capable executable first")
    output.mkdir(parents=True)
    write_new(output / "plan.json", plan)
    reports = {}
    try:
        for run in plan["runs"]:
            require([info(item["path"]) for item in artifacts] == artifacts, "A frozen artifact changed during the experiment")
            require(model_provenance(model) == model_record, "Checkpoint metadata or file identity changed")
            print(f"Starting {run['label']} ({args.model}, {args.mode}, {args.candidate})", flush=True)
            elapsed = run_process(run["command"], environment(args.candidate, run["arm"]), output / f"{run['label']}.log", args.timeout_seconds)
            report = output / f"{run['label']}.json"
            require(report.is_file(), "Native process produced no report")
            if args.candidate == "gemma3-compiled":
                validate_compiled_tail_activation(json.loads(report.read_text()), run["arm"])
            reports[run["label"]] = report
            write_new(output / f"{run['label']}-process.json", {"elapsed_seconds_diagnostic_only": elapsed, "report": info(report), "exit_code": 0})
        require([info(item["path"]) for item in artifacts] == artifacts, "A frozen artifact changed during the experiment")
        require(model_provenance(model) == model_record, "Checkpoint identity changed during the experiment")
        if args.mode == "nll":
            analyze_nll(reports, plan, output)
        else:
            scorer = module("evaluate-generated.py", "task_evaluation")
            scores = scorer.evaluate(corpus, DATA / "generated-pilot/answers.jsonl", reports, draws=10000, seed=20260929, retrieval_diagnostics=True)
            scores["kernel_experiment_provenance"] = info(output / "plan.json")
            scores["limitations"].append("New kernel flags are captured by the paired-process plan, not the native generation report's limited environment allowlist.")
            write_new(output / "task-analysis.json", scores)
        write_new(output / "completion.json", {"status": "analyzed", "quality_acceptance": "Not inferred automatically"})
    except BaseException as error:
        write_new(output / "failure.json", {"status": "failed", "error": str(error), "completed_reports": list(reports)})
        raise
    print(f"Saved paired evidence to {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
