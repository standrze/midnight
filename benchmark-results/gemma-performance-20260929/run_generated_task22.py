"""Prepare/run/score the frozen 22-task Gemma pilot. Preparation never loads a model."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = Path(__file__).resolve().parent
CORPUS = CAMPAIGN / "quality/generated-pilot"
BINARY_DIR = ROOT / "artifacts/gemma-runtime-stage4-20260929"
SCORER = ROOT / "Scripts/evaluate-generated.py"
MODELS = {
    "bf16": ROOT / "artifacts/gemma3-270m-it-bf16",
    "ordinary-g64": ROOT / "artifacts/gemma3-270m-four-arm-20260929-r2/models/ordinary-g64",
    "searched-g64": ROOT / "artifacts/gemma3-270m-four-arm-20260929-r2/models/searched-g64",
    "ordinary-selective-g128": ROOT / "artifacts/gemma3-270m-four-arm-20260929-r2/models/ordinary-selective-g128",
    "searched-g64-q8-query-key": ROOT / "artifacts/gemma3-270m-q8-arms-20260929/models/searched-g64-q8-query-key",
}
FROZEN_HASHES = {"tasks.jsonl": "bb827f9370101ce3391ac270234e5416accd991f5cc120bb05839dd43913e397",
                 "answers.jsonl": "6f9ce15c7a1d95431e01f80b3687697781718a53451ea6b409d8ef6562228945"}
SCRUB_PREFIXES = ("MLX_", "MIDNIGHT_", "MODEL_RUNNER_")
FLAGS = {
    "MLX_METAL_AFFINE_Q4_QMV_TAIL": "0", "MLX_METAL_AFFINE_Q4_QMV_TAIL_SCOPE": "all",
    "MIDNIGHT_METAL_SDPA_D512": "0", "MIDNIGHT_METAL_SDPA_D256_MASKED": "0",
    "MIDNIGHT_METAL_SDPA_D256_PRUNE": "0", "MIDNIGHT_METAL_GROUPED_EXPERT_VERIFY": "0",
    "MIDNIGHT_METAL_COMMAND_TIMING": "0", "MIDNIGHT_GEMMA4_BOUNDED_KV": "0",
    "MIDNIGHT_GEMMA4_EXPERT_GATE_UP": "0", "MIDNIGHT_GEMMA4_DENSE_FUSION": "0",
    "MIDNIGHT_GEMMA4_WINDOW_SLICING": "0", "MIDNIGHT_GEMMA_ASSISTANT_UNMASKED": "0",
    "MIDNIGHT_GEMMA3_COMPILED_TAIL": "0", "MIDNIGHT_MTP_ADAPTIVE_DRAFTS": "0",
    "MODEL_RUNNER_PREFIX_CACHE_ENTRIES": "0",
}
REFUSAL = re.compile(r"\b(?:i (?:cannot|can't|won't|am unable to)|i(?:'m| am) sorry|unable to (?:answer|assist|help)|cannot (?:answer|assist|help))\b", re.I)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path, expected_sha=None):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": expected_sha or sha(path),
            "hash_basis": "prior_frozen_manifest; recheck before execution" if expected_sha else "fresh_sha256"}


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify(identities):
    for entry in identities:
        path = Path(entry["path"])
        require(path.is_file() and path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"],
                f"Frozen artifact changed: {path}")


def bit_widths(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "bits":
                require(type(nested) in (int, float) and nested >= 4, "Quantization below four bits is forbidden")
                yield nested
            else:
                yield from bit_widths(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from bit_widths(nested)


def command(model, tasks, report):
    return [str(BINARY_DIR / "model-runner-generation-bench"), str(model), str(tasks), str(report),
            "--engine", "metal", "--tokens", "256", "--context-length", "8192",
            "--prefill-step-size", "512", "--kv-compression", "none"]


def validate_plan(plan, directory):
    require(plan["environment_overrides"] == FLAGS and plan["scrub_environment_prefixes"] == list(SCRUB_PREFIXES),
            "Plan environment differs from frozen runner")
    require([item["label"] for item in plan["runs"]] == list(MODELS), "Plan model roster changed")
    require(plan["inputs"] == {"tasks": str(directory / "inputs/tasks.jsonl"),
                                "answers": str(directory / "inputs/answers.jsonl")}, "Plan input paths changed")
    require(plan["score_policy"]["draws"] == 10000 and plan["score_policy"]["seed"] == 20260929,
            "Plan scoring settings changed")
    require(plan["score_policy"]["generated_code_execution"] is False, "Generated code execution is forbidden")
    for item in plan["runs"]:
        label = item["label"]
        report = directory / f"{label}.json"
        require(item["model"] == str(MODELS[label]) and item["report"] == str(report), "Plan model or report path changed")
        require(item["command"] == command(MODELS[label], plan["inputs"]["tasks"], report), "Plan generation command changed")


def prior_hashes():
    manifests = [CAMPAIGN / "gemma3-bf16-fixture-provenance.json",
                 ROOT / "artifacts/gemma3-270m-four-arm-20260929-r2/conversion-summary.json",
                 ROOT / "artifacts/gemma3-270m-q8-arms-20260929/conversion-summary.json",
                 BINARY_DIR / "identities.json"]
    bf16, four, q8, binaries = [json.loads(path.read_text()) for path in manifests]
    known = {str((ROOT / row["path"]).resolve()): row["sha256"] for row in bf16["files"]}
    for arm in four["arms"]:
        directory = Path(arm["model"])
        known[str(directory / "model.safetensors")] = arm["weights_sha256"]
        known[str(directory / "config.json")] = arm["config_sha256"]
    for label, arm in q8["arms"].items():
        directory = ROOT / "artifacts/gemma3-270m-q8-arms-20260929/models" / label
        known.update({str(directory / name): info["sha256"] for name, info in arm["files"].items()})
    known.update({str(BINARY_DIR / row["name"]): row["sha256"] for row in binaries})
    return known, manifests


def prepare(name):
    require(name and Path(name).name == name and name not in (".", ".."), "Name must be a new directory basename")
    evaluation = runpy.run_path(str(SCORER))
    for filename, expected in FROZEN_HASHES.items():
        require(sha(CORPUS / filename) == expected, f"Frozen {filename} identity mismatch")
    tasks, answers = evaluation["load_inputs"](CORPUS / "tasks.jsonl", CORPUS / "answers.jsonl")
    require(Counter(row["category"] for row in tasks) == {"math": 16, "retrieval": 6}, "Expected frozen 16 math + 6 retrieval tasks")
    require(all(row["kind"] in ("numeric", "exact") for row in answers.values()), "Generated code must never be executed")
    directory = CAMPAIGN / name
    directory.mkdir(exist_ok=False)
    inputs = directory / "inputs"
    inputs.mkdir()
    for filename in ("tasks.jsonl", "answers.jsonl", "provenance.json"):
        shutil.copyfile(CORPUS / filename, inputs / filename)
    (inputs / "answers.jsonl").chmod(0o600)
    known, manifests = prior_hashes()
    binary = BINARY_DIR / "model-runner-generation-bench"
    artifacts = [identity(path, known.get(str(path))) for path in (binary, BINARY_DIR / "mlx.metallib")]
    artifacts += [identity(path) for path in (Path(__file__), SCORER, *manifests, *sorted(inputs.iterdir()))]
    runs = []
    for label, model in MODELS.items():
        widths = list(bit_widths(json.loads((model / "config.json").read_text())))
        require(label == "bf16" or widths, f"No quantization metadata: {label}")
        model_files = [path for path in sorted(model.iterdir()) if path.is_file()]
        require(any(path.suffix == ".safetensors" for path in model_files), f"Missing checkpoint payload: {model}")
        entries = [identity(path, known.get(str(path))) for path in model_files]
        runs.append({"label": label, "model": str(model), "identities": entries,
            "report": str(directory / f"{label}.json"),
            "command": command(model, inputs / "tasks.jsonl", directory / f"{label}.json")})
    plan = {"format": 1, "created_at": time.time(), "purpose": "Frozen single-greedy generated math/retrieval pilot; timings are diagnostic only.",
        "generation_execution_started": False, "artifacts": artifacts, "runs": runs,
        "inputs": {"tasks": str(inputs / "tasks.jsonl"), "answers": str(inputs / "answers.jsonl")},
        "scrub_environment_prefixes": list(SCRUB_PREFIXES), "environment_overrides": FLAGS,
        "settings": {"tokens": 256, "context_length": 8192, "prefill_step_size": 512, "kv_compression": "none",
            "temperature": 0, "top_p": 1, "prompt_cache": False, "speculative_decoding": False,
            "prompt_format": "single_user_message_checkpoint_chat_template", "added_system_prompt": None},
        "identity_policy": "Prior verified manifest hashes freeze large artifacts without re-reading all payloads during preparation. Execution rehashes every artifact before any model process, after each model, and at completion; changed identities fail closed.",
        "score_policy": {"scorer": str(SCORER), "draws": 10000, "seed": 20260929,
            "math": "Exactly one #### final-number marker on the final nonempty line, exact rational/decimal equality; no intermediate-number guessing.",
            "retrieval": "Entire visible output must equal the expected value after outer whitespace trimming.",
            "failures": "In valid native reports, missing/failed generation remains unscored in all22 task slots; complete accuracy and paired inference are withheld. An absent or invalid native report produces an explicit scoring-readiness failure record and withholds campaign scoring. Completed refusals, format failures and wrong answers count as failures. Output-limit stops remain visible and are not retried or dropped; strict answer scoring is unchanged.",
            "refusal_audit": "Fixed lexical cues are diagnostic labels, not an alternative scoring criterion.",
            "generated_code_execution": False},
        "limitations": ["22 convenience tasks, including publicly available GSM8K, do not establish broad task quality or training-data exclusion.",
            "One greedy output per task and model; no retries, best-of-n selection or corpus changes are scored.",
            "The 512/4096 retrieval lengths are estimates; native reports retain actual rendered token counts and prohibit prompt truncation.",
            "Prompt-token IDs and execution settings gate paired comparisons; answer keys are never passed to the generator.",
            "The binary records visible content only; any separate reasoning events are not scored. All five models use the same interface."]}
    validate_plan(plan, directory)
    write_new(directory / "plan.json", plan)
    print(f"Prepared only; no model process started: {directory / 'plan.json'}")


def audit_outputs(result, reports):
    audit = {}
    for label, model in result["models"].items():
        raw = json.loads(Path(reports[label]).read_text())
        samples = {row["id"]: row for row in raw["samples"]}
        rows = []
        for score in model["samples"]:
            sample = samples.get(score["id"], {})
            text = sample.get("generated_text", "")
            rows.append({"id": score["id"], "category": score["category"], "scoring_status": score["status"],
                "passed": score["passed"], "scoring_reason": score["reason"], "generation_status": sample.get("status", "missing"),
                "output_limit_reached": sample.get("output_limit_reached"), "stop_reason": sample.get("stop_reason"),
                "generation_token_count": sample.get("generation_token_count"), "refusal_cue_detected": bool(REFUSAL.search(text)),
                "empty_visible_output": not text.strip(), "numeric_format_failure": score["reason"] in ("missing_or_ambiguous_final_number", "invalid_final_number"),
                "native_error": sample.get("error"), "native_failure_stage": sample.get("failure_stage"),
                "generated_text": text})
        audit[label] = {"reason_counts": dict(Counter(row["scoring_reason"] for row in rows)),
            "refusal_cue_count": sum(row["refusal_cue_detected"] for row in rows),
            "numeric_format_failure_count": sum(row["numeric_format_failure"] for row in rows),
            "output_limit_reached_count": sum(row["output_limit_reached"] is True for row in rows), "samples": rows}
    return {"purpose": "Supplemental audit. Refusal cues do not alter the frozen primary scorer; raw text is retained.", "models": audit}


def score(plan_path, suffix=""):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    validate_plan(plan, plan_path.parent)
    require(not suffix or re.fullmatch(r"[A-Za-z0-9_-]+", suffix), "Invalid score suffix")
    score_paths = {str(Path(__file__).resolve()), str(SCORER), *plan["inputs"].values()}
    score_ids = [item for item in plan["artifacts"] if item["path"] in score_paths]
    require({item["path"] for item in score_ids} == score_paths, "Missing frozen scorer or input identity")
    verify(score_ids)
    start = json.loads((plan_path.parent / "execution-start.json").read_text())
    completion = json.loads((plan_path.parent / "execution-complete.json").read_text())
    require(start["plan"]["sha256"] == sha(plan_path) and start["identities_verified"]
            and completion["artifacts_unchanged"], "Execution provenance does not match frozen plan")
    exits = {entry["label"]: entry for entry in completion["exits"]}
    reports = {run["label"]: Path(run["report"]) for run in plan["runs"]}
    readiness = []
    for item in plan["runs"]:
        label = item["label"]
        exit_record = exits[label]
        record = {"label": label, "planned_tasks": 22, "native_report_exists": reports[label].exists(),
                  "process_exit_status": exit_record["status"], "timed_out": exit_record["timed_out"]}
        if not reports[label].exists():
            record["failure"] = "missing_native_report; all22 task slots unscored; campaign scoring withheld"
        else:
            try:
                require(exit_record.get("native_report_identity") == identity(reports[label]), "Native report changed after execution")
                raw = json.loads(reports[label].read_text())
                require(raw["model_path"] == item["model"] and raw["invocation"] == item["command"], "Native model or invocation differs from planned arm")
                expected = {"requested_tokens": 256, "requested_context_length": 8192, "prefill_step_size": 512,
                            "kv_compression": "none", "temperature": 0, "top_p": 1, "prompt_cache": False,
                            "speculative_decoding": False, "requested_engine": "metal", "engine": "metal",
                            "context_length": 8192, "runtime_environment": {}}
                require(all(raw.get(key) == value for key, value in expected.items()), "Native execution settings differ from plan")
                config = next(entry for entry in item["identities"] if Path(entry["path"]).name == "config.json")
                require(raw.get("model_config_sha256") == config["sha256"], "Native model config identity is missing or different")
                require(raw.get("corpus_sha256") == FROZEN_HASHES["tasks.jsonl"] and raw.get("input_sample_count") == 22,
                        "Native report did not validate the frozen22 corpus")
                record.update(native_status=raw.get("status"), completed_samples=raw.get("completed_sample_count"),
                              native_error=raw.get("error"), native_failure_status=raw.get("failure_status"))
            except (ValueError, KeyError, StopIteration, TypeError) as error:
                record["failure"] = str(error)
        readiness.append(record)
    readiness_name = "scoring-readiness" + ("-" + suffix if suffix else "")
    write_new(plan_path.parent / f"{readiness_name}.json", {"models": readiness,
              "campaign_scoring_ready": all("failure" not in item for item in readiness)})
    require(all("failure" not in item for item in readiness), "Native report provenance or readiness failed; explicit failure inventory retained")
    evaluation = runpy.run_path(str(SCORER))
    result = evaluation["evaluate"](plan["inputs"]["tasks"], plan["inputs"]["answers"], reports,
        sandbox=None, draws=plan["score_policy"]["draws"], seed=plan["score_policy"]["seed"], retrieval_diagnostics=True)
    result["campaign_plan"] = identity(plan_path)
    result["campaign_purpose"] = "Frozen22 math/retrieval accuracy only; no code tasks or code execution."
    name = "scores" + ("-" + suffix if suffix else "")
    write_new(plan_path.parent / f"{name}.json", result)
    write_new(plan_path.parent / f"{name}-audit.json", audit_outputs(result, reports))
    print(json.dumps({label: model["overall"] for label, model in result["models"].items()}, indent=2))
    return result


def run(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    validate_plan(plan, plan_path.parent)
    all_ids = plan["artifacts"] + [item for run in plan["runs"] for item in run["identities"]]
    verify(all_ids)
    for item in plan["runs"]:
        require(not Path(item["report"]).exists(), "A model report already exists; no implicit resume or retry")
        require(plan["inputs"]["answers"] not in item["command"], "Answer key must not reach generator")
    write_new(plan_path.parent / "execution-start.json", {"started_at": time.time(), "plan": identity(plan_path), "identities_verified": True})
    env = {key: value for key, value in os.environ.items() if not key.startswith(SCRUB_PREFIXES)}
    env.update(FLAGS)
    exits = []
    for item in plan["runs"]:
        label = item["label"]
        start = time.monotonic()
        print(f"Start {label}: frozen22 target-only tasks", flush=True)
        timed_out = False
        with (plan_path.parent / f"{label}.log").open("x") as stream:
            process = subprocess.Popen(item["command"], cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
            try:
                status = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                status = process.wait()
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        exit_record = {"label": label, "status": status, "timed_out": timed_out,
                       "elapsed_seconds": time.monotonic() - start, "native_report_exists": Path(item["report"]).exists(),
                       "native_report_identity": identity(Path(item["report"])) if Path(item["report"]).exists() else None}
        write_new(plan_path.parent / f"{label}-exit.json", exit_record)
        exits.append(exit_record)
        verify(item["identities"])
        print(f"Finish {label}: exit {status}", flush=True)
    verify(all_ids)
    write_new(plan_path.parent / "execution-complete.json", {"finished_at": time.time(), "artifacts_unchanged": True, "exits": exits})
    score(plan_path)
    if any(item["status"] for item in exits):
        raise SystemExit("At least one model failed; partial reports and scoring retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--name", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--plan", type=Path, required=True)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--plan", type=Path, required=True)
    score_parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    if args.operation == "prepare": prepare(args.name)
    elif args.operation == "run": run(args.plan)
    else: score(args.plan, args.suffix)


if __name__ == "__main__":
    main()
