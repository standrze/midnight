#!/usr/bin/env python3
"""Score native generated answers, with paired category accuracy uncertainty.

Generate using model-runner-generation-bench MODEL tasks.jsonl report.json.
Only tasks.jsonl is supplied to the model. Then, for example:
  python3 Scripts/evaluate-generated.py --tasks tasks.jsonl --answers answers.jsonl \
    --report standard=standard.json --report awss=awss.json --output scores.json

Code is unscored unless --sandbox-image names an already installed, trusted
Linux Python image pinned by @sha256:<64 hex digits>. No images are pulled.
Generated code is NEVER executed by the host Python interpreter. Docker uses
no network, a read-only root/input, an unprivileged user, no capabilities, no
host directories beyond a temporary input, and bounded resources/output/time.
Containers are a containment layer, not a proof against kernel vulnerabilities
or adversarial benchmark-score spoofing. Use a disposable Docker VM for unknown
code. Only the provided MBPP tests define code success; this is not HumanEval.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

PINNED_IMAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}\Z")
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
NUMERIC = re.compile(r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:/\d+)?\Z")
CODE_LIMIT = 256 * 1024
COMPARISON_SETTINGS = ("requested_tokens", "temperature", "top_p", "context_length",
                       "prefill_step_size", "kv_compression", "engine", "prompt_format",
                       "prompt_cache", "speculative_decoding", "fused_gate_up_silu",
                       "compiled_laguna_block_tail", "fused_laguna_router_top_k",
                       "runtime_environment", "memory_limit_bytes")


def file_info(path):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def write_new(path, value):
    # Never clobber prior evidence. Each run gets a new output path.
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(r, dict) or not isinstance(r.get("id"), str) or not r["id"] for r in rows):
        raise ValueError(f"Nonempty records with string IDs required: {path}")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate IDs: {path}")
    return rows


def numeric_value(text):
    text = text.strip()
    if len(text) > 128 or not NUMERIC.fullmatch(text):
        raise ValueError("Expected a finite decimal, integer, or fraction without units")
    return Fraction(text.replace(",", ""))


def extract_code(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("missing_code")
    if len(text.encode()) > CODE_LIMIT:
        raise ValueError("code_size_limit")
    if "```" in text:
        # One complete Python/plain fence only; reject ambiguous multiple blocks
        # and prose outside it rather than guessing which program to execute.
        match = re.fullmatch(r"\s*```(?:python|py)?[ \t]*\n(.*?)\n```\s*", text, re.S | re.I)
        if not match or "```" in match.group(1):
            raise ValueError("ambiguous_or_incomplete_code_fence")
        text = match.group(1)
    if not text.strip():
        raise ValueError("missing_code")
    try:
        ast.parse(text)
    except (SyntaxError, ValueError, RecursionError) as error:
        raise ValueError("invalid_python_syntax") from error
    return text


def bounded_command(argv, timeout, output_limit=131072):
    """Capture bounded bytes and kill a process group on timeout/output overflow."""
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    captured = bytearray()
    failure = None
    start = time.monotonic()
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        while selector.get_map():
            if time.monotonic() - start > timeout:
                failure = "timeout"
                break
            for key, _ in selector.select(min(0.1, max(0.001, timeout - (time.monotonic() - start)))):
                chunk = os.read(key.fileobj.fileno(), 8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                remaining = max(0, output_limit - len(captured))
                captured.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    failure = "output_limit"
                    break
            if failure:
                break
    finally:
        selector.close()
        if failure:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=max(0.1, timeout - (time.monotonic() - start)))
        except subprocess.TimeoutExpired:
            failure = "timeout"
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        process.stdout.close()
    return {"returncode": process.returncode, "failure": failure,
            "output": captured.decode("utf-8", errors="replace"),
            "elapsed_seconds": time.monotonic() - start}


def docker_arguments(docker, image, work, name):
    return [docker, "run", "--rm", "--name", name, "--pull", "never",
            "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "64",
            "--memory", "256m", "--memory-swap", "256m", "--cpus", "1",
            "--user", "65534:65534", "--ulimit", "nofile=64:64",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m", "--workdir", "/work",
            "--mount", f"type=bind,src={work},dst=/work,readonly",
            "--entrypoint", "python3", image, "-I", "-S", "/work/check.py"]


class DockerSandbox:
    def __init__(self, image, timeout=15, docker=None):
        if not PINNED_IMAGE.fullmatch(image):
            raise ValueError("Sandbox image must use an immutable @sha256:<64 lowercase hex> digest")
        self.docker = docker or shutil.which("docker")
        if not self.docker:
            raise ValueError("Docker is unavailable; code cannot be scored safely")
        if not 1 <= timeout <= 120:
            raise ValueError("Code timeout must be in 1...120 seconds")
        self.image, self.timeout = image, timeout
        checked = subprocess.run([self.docker, "image", "inspect", image], capture_output=True, text=True, timeout=15)
        if checked.returncode:
            raise ValueError("Pinned sandbox image is not installed locally; no image was pulled")
        info = json.loads(checked.stdout)
        if len(info) != 1 or info[0].get("Os") != "linux" or image not in info[0].get("RepoDigests", []):
            raise ValueError("Sandbox image inspection must verify the exact Linux image digest")
        self.provenance = {"image": image, "image_id": info[0].get("Id"),
                           "architecture": info[0].get("Architecture"), "timeout_seconds": timeout,
                           "docker_executable": file_info(self.docker),
                           "network": "none", "rootfs": "read-only", "user": "65534:65534",
                           "memory_bytes": 268435456, "pids": 64, "cpus": 1,
                           "output_limit_bytes": 131072, "policy_version": 1}

    def score(self, code, answer):
        # No shell, host exec/eval or candidate-controlled command arguments.
        with tempfile.TemporaryDirectory(prefix="midnight-code-sandbox-") as directory:
            work = Path(directory)
            work.chmod(0o755)
            marker = "MIDNIGHT_TESTS_PASSED_" + uuid.uuid4().hex
            (work / "candidate.py").write_text(code)
            (work / "tests.json").write_text(json.dumps(answer))
            checker = '''import builtins, json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
with open('/work/tests.json') as f: test = json.load(f)
with open('/work/candidate.py') as f: code = f.read()
compile_code, execute, output = compile, exec, print
namespace = {'__name__': '__candidate__'}
execute(compile_code(test['setup'], '<setup>', 'exec'), namespace)
execute(compile_code(code, '<candidate>', 'exec'), namespace)
for case in test['tests']:
    execute(compile_code(case, '<test>', 'exec'), namespace)
output(MARKER, flush=True)
'''.replace("MARKER", repr(marker))
            (work / "check.py").write_text(checker)
            for path in work.iterdir():
                path.chmod(0o444)
            name = "midnight-eval-" + uuid.uuid4().hex
            command = docker_arguments(self.docker, self.image, str(work), name)
            try:
                result = bounded_command(command, self.timeout)
            finally:
                # Killing the CLI does not necessarily stop its daemon-side container.
                # Always remove our uniquely named container; never touch other containers.
                subprocess.run([self.docker, "rm", "--force", name], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=15)
            infrastructure_failure = result["returncode"] in (125, 126, 127)
            passed = (not result["failure"] and result["returncode"] == 0
                      and marker in result["output"].splitlines())
            return {"status": "unscored" if infrastructure_failure else "scored",
                    "passed": None if infrastructure_failure else passed,
                    "reason": "sandbox_infrastructure_failure" if infrastructure_failure else result["failure"] or ("tests_passed" if passed else "tests_failed"),
                    "sandbox": {**result, "output": result["output"].replace(marker, "MIDNIGHT_TESTS_PASSED")}}


def validate_security_key(key):
    verdict = key.get("verdict")
    cwes, lines = key.get("cwes"), key.get("evidence_lines")
    line_count = key.get("line_count")
    if not isinstance(verdict, str) or verdict not in {"safe", "vulnerable"} or type(line_count) is not int or line_count < 1:
        raise ValueError("Security keys require a safe/vulnerable verdict and positive line_count")
    if not isinstance(cwes, list) or any(not isinstance(v, str) or not re.fullmatch(r"CWE-[1-9][0-9]*", v) for v in cwes) or len(cwes) != len(set(cwes)):
        raise ValueError("Security keys require distinct canonical CWE labels")
    if not isinstance(lines, list) or any(type(v) is not int or not 1 <= v <= line_count for v in lines) or len(lines) != len(set(lines)):
        raise ValueError("Security keys require distinct in-range evidence lines")
    if (verdict == "safe" and (cwes or lines)) or (verdict == "vulnerable" and (not cwes or not lines)):
        raise ValueError("Security verdict disagrees with CWE/evidence labels")
    if "evidence_anchor_lines" in key:
        anchors = key["evidence_anchor_lines"]
        if (not isinstance(anchors, list)
                or any(type(v) is not int or v not in lines for v in anchors)
                or len(anchors) != len(set(anchors))
                or (verdict == "vulnerable" and not anchors)
                or (verdict == "safe" and anchors)):
            raise ValueError("Security anchors must be distinct accepted evidence lines and match the verdict")


def score_security_json(text, key):
    """Grade labeled code review without executing the supplied source or response."""
    validate_security_key(key)
    diagnostic = {"expected_verdict": key["verdict"], "predicted_verdict": None,
                  "format_valid": False, "classification_passed": False}
    result = {"status": "scored", "passed": False, "security_diagnostic": diagnostic}
    if not isinstance(text, str) or len(text.encode()) > CODE_LIMIT:
        return {**result, "reason": "invalid_or_oversized_security_response"}
    if "```" in text:
        match = re.fullmatch(r"\s*```json[ \t]*\n(.*?)\n```\s*", text, re.S | re.I)
        if not match or "```" in match.group(1):
            return {**result, "reason": "ambiguous_security_json_fence"}
        text = match.group(1)

    def unique_object(pairs):
        value = {}
        for name, item in pairs:
            if name in value:
                raise ValueError("duplicate JSON key")
            value[name] = item
        return value

    try:
        value = json.loads(text, object_pairs_hook=unique_object,
                           parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
        if not isinstance(value, dict) or set(value) != {"verdict", "cwes", "evidence_lines"}:
            raise ValueError("unexpected response fields")
        validate_security_key({**value, "line_count": key["line_count"]})
    except (ValueError, TypeError, RecursionError):
        return {**result, "reason": "invalid_security_json"}
    classification = value["verdict"] == key["verdict"]
    cwes = set(value["cwes"]) == set(key["cwes"])
    expected_lines, actual_lines = set(key["evidence_lines"]), set(value["evidence_lines"])
    evidence = (bool(actual_lines & expected_lines) and actual_lines <= expected_lines) if key["verdict"] == "vulnerable" else not actual_lines
    if key["verdict"] == "vulnerable" and "evidence_anchor_lines" in key:
        evidence = evidence and bool(actual_lines & set(key["evidence_anchor_lines"]))
    diagnostic.update(predicted_verdict=value["verdict"], format_valid=True,
                      classification_passed=classification, cwe_passed=cwes, evidence_passed=evidence)
    return {**result, "passed": classification and cwes and evidence, "reason": "security_verdict_cwe_evidence"}


def score_answer(text, key, sandbox=None, retrieval_diagnostics=False):
    kind = key["kind"]
    if kind == "security-json":
        return score_security_json(text, key)
    if kind == "exact":
        result = {"status": "scored", "passed": text.strip() == key["answer"].strip(), "reason": "exact_match"}
        if retrieval_diagnostics:
            expected = key["answer"].strip()
            values = re.findall(r"(?<![A-Za-z0-9_])V[0-9a-f]{12}(?![A-Za-z0-9_])", text)
            result["retrieval_diagnostic"] = {
                "purpose": "Post-generation formatting/content diagnostic, NOT retrieval accuracy or an alternative pass criterion.",
                "expected_value_present": expected in text,
                "emitted_value_count": len(values),
                "all_emitted_values_equal_expected": all(value == expected for value in values) if values else None,
                "tag_containing": bool(re.search(r"</?[A-Za-z][^<>\n]*>", text)),
                "value_pattern": "standalone uppercase V followed by exactly 12 lowercase hexadecimal digits",
            }
        return result
    if kind == "numeric":
        gold = numeric_value(key["answer"])
        matches = re.findall(r"^####[ \t]*([^\n]+)[ \t]*$", text, re.M)
        # A final answer marker must be the last nonempty line; don't guess from
        # intermediate reasoning or take the final number anywhere in the text.
        if len(matches) != 1 or not text.strip().splitlines()[-1].startswith("####"):
            return {"status": "scored", "passed": False, "reason": "missing_or_ambiguous_final_number"}
        try:
            value = numeric_value(matches[0])
        except (ValueError, ZeroDivisionError):
            return {"status": "scored", "passed": False, "reason": "invalid_final_number"}
        return {"status": "scored", "passed": value == gold, "reason": "numeric_exact_match"}
    if kind != "python-tests":
        raise ValueError(f"Unsupported answer kind: {kind}")
    try:
        code = extract_code(text)
    except ValueError as error:
        return {"status": "scored", "passed": False, "reason": str(error)}
    if sandbox is None:
        return {"status": "unscored", "passed": None, "reason": "sandbox_not_configured"}
    return sandbox.score(code, key)


def load_inputs(tasks_path, answers_path):
    tasks, keys = read_jsonl(tasks_path), read_jsonl(answers_path)
    if {t["id"] for t in tasks} != {a["id"] for a in keys}:
        raise ValueError("Public task IDs and private answer IDs must match exactly")
    keys = {a["id"]: a for a in keys}
    for task in tasks:
        if set(task) - {"id", "category", "prompt", "metadata"}:
            raise ValueError("Public tasks may contain only id/category/prompt/metadata; answers belong in the private key")
        if task.get("category") not in {"math", "code", "retrieval", "cybersecurity"} or not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
            raise ValueError("Each public task needs a supported category and nonblank prompt")
        answer = keys[task["id"]]
        expected = {"math": "numeric", "code": "python-tests", "retrieval": "exact", "cybersecurity": "security-json"}[task["category"]]
        if answer.get("kind") != expected:
            raise ValueError("Task category and answer kind disagree")
        if expected == "python-tests":
            if not isinstance(answer.get("setup"), str) or not isinstance(answer.get("tests"), list) or not answer["tests"] or not all(isinstance(t, str) and t.strip() for t in answer["tests"]):
                raise ValueError("Code keys require setup text and nonempty test strings")
            if len(json.dumps(answer).encode()) > 1024 * 1024:
                raise ValueError("Code test payload exceeds 1 MiB")
            ast.parse(answer["setup"])
            for case in answer["tests"]:
                ast.parse(case)
        elif expected == "security-json":
            validate_security_key(answer)
        elif not isinstance(answer.get("answer"), str) or not answer["answer"].strip():
            raise ValueError("Numeric/exact keys need nonblank answer text")
        elif expected == "numeric":
            numeric_value(answer["answer"])
    return tasks, keys


def summarize(rows):
    scored = [r for r in rows if r["status"] == "scored"]
    passed = sum(r["passed"] is True for r in scored)
    result = {"tasks": len(rows), "scored": len(scored), "passed": passed,
            "accuracy": passed / len(rows) if len(scored) == len(rows) and rows else None,
            "scored_subset_accuracy_diagnostic": passed / len(scored) if scored else None,
            "complete": len(scored) == len(rows),
            "output_limit_reached": sum(bool(r.get("output_limit_reached")) for r in rows)}

    security_rows = [r for r in rows if r.get("category") == "cybersecurity" or "security_diagnostic" in r]
    security = [r["security_diagnostic"] for r in security_rows if "security_diagnostic" in r]
    if security:
        tp = sum(r["expected_verdict"] == "vulnerable" and r["predicted_verdict"] == "vulnerable" for r in security)
        fp = sum(r["expected_verdict"] == "safe" and r["predicted_verdict"] == "vulnerable" for r in security)
        fn = sum(r["expected_verdict"] == "vulnerable" and r["predicted_verdict"] != "vulnerable" for r in security)
        tn = sum(r["expected_verdict"] == "safe" and r["predicted_verdict"] == "safe" for r in security)
        negatives = sum(r["expected_verdict"] == "safe" for r in security)
        result["security_detection"] = {
            "tasks": len(security), "true_positive": tp, "false_positive": fp,
            "false_negative_including_invalid": fn, "true_negative": tn,
            "invalid_response_count": sum(not r["format_valid"] for r in security),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall_including_invalid": tp / (tp + fn) if tp + fn else None,
            "false_positive_rate": fp / negatives if negatives else None,
            "classification_accuracy_including_invalid": (tp + tn) / len(security),
            "complete": len(security) == len(security_rows),
            "purpose": "Labeled vulnerability detection; primary task success also requires correct CWE and relevant evidence lines",
        }
    return result


def score_report(path, tasks, keys, tasks_hash, sandbox, retrieval_diagnostics=False):
    report = json.loads(Path(path).read_text())
    if report.get("corpus_sha256") != tasks_hash:
        raise ValueError(f"Report corpus SHA256 does not match public tasks: {path}")
    if report.get("format") != 1 or report.get("benchmark") != "native_greedy_generation":
        raise ValueError("Unsupported native generation report schema")
    if report.get("input_sample_count") != len(tasks):
        raise ValueError("Native input sample count disagrees with public tasks")
    samples = report.get("samples")
    if not isinstance(samples, list) or any(not isinstance(s, dict) for s in samples):
        raise ValueError("Native report needs a samples array")
    ids = [s.get("id") for s in samples]
    if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids) or set(ids) - {t["id"] for t in tasks}:
        raise ValueError("Native report contains duplicate, invalid or unexpected sample IDs")
    completed = sum(s.get("status") == "completed" for s in samples)
    if report.get("completed_sample_count") != completed:
        raise ValueError("Native completed sample counter is inconsistent")
    if report.get("status") == "completed" and completed != len(tasks):
        raise ValueError("Native report claims completion but samples are incomplete")
    if report.get("output_limit_reached_count") != sum(s.get("output_limit_reached") is True for s in samples):
        raise ValueError("Native output-limit counter is inconsistent")
    by_id, rows = {s["id"]: s for s in samples}, []
    for task in tasks:
        identifier = task["id"]
        sample = by_id.get(identifier)
        row = {"id": identifier, "category": task["category"], "metadata": task.get("metadata", {})}
        if sample is None or sample.get("status") != "completed":
            row.update(status="unscored", passed=None, reason="missing_or_failed_generation",
                       generation_status=sample.get("status") if sample else "missing")
        else:
            if sample.get("category") != task["category"] or not isinstance(sample.get("generated_text"), str):
                raise ValueError(f"Invalid native category/text for {identifier}")
            if sample.get("prompt_sha256") != hashlib.sha256(task["prompt"].encode()).hexdigest():
                raise ValueError(f"Prompt SHA256 disagrees with task {identifier}")
            if sample.get("prompt_truncated") is not False:
                raise ValueError(f"Full prompt coverage is not verified for {identifier}")
            generated = sample.get("generation_token_count")
            requested = report.get("requested_tokens")
            if type(requested) is not int or requested < 1 or type(generated) is not int or not 0 <= generated <= requested:
                raise ValueError(f"Invalid generated token count for {identifier}")
            if sample.get("stop_reason") not in {"stop", "length"}:
                raise ValueError(f"Invalid stop reason for {identifier}")
            if sample["stop_reason"] == "length" and generated != requested:
                raise ValueError(f"Length stop does not match output budget for {identifier}")
            expected_limit = sample["stop_reason"] == "length" or generated == requested
            if sample.get("output_limit_reached") is not expected_limit:
                raise ValueError(f"Output-limit flag disagrees with generation for {identifier}")
            row.update(score_answer(sample["generated_text"], keys[identifier], sandbox, retrieval_diagnostics))
        if sample:
            for field in ("prompt_token_id_fingerprint", "prompt_token_count", "generation_token_count", "stop_reason", "output_limit_reached"):
                row[field] = sample.get(field)
        rows.append(row)
    settings = {k: report[k] for k in ("model_path", *COMPARISON_SETTINGS) if k in report}
    return {"input": file_info(path), "native_status": report.get("status"), "settings": settings,
            "overall": summarize(rows),
            "categories": {category: summarize([r for r in rows if r["category"] == category])
                           for category in sorted({r["category"] for r in rows})}, "samples": rows}


def conservative_paired_bound(baseline_rows, candidate_rows, ids, confidence=0.95):
    """Weighted Hoeffding bound for paired deltas in [-1, 1].

    Group correlated tasks by declared category/family. Unit weights preserve
    record-weighted accuracy. Independence across units is an assumption, not
    something this bound can establish; unknown source correlation still matters.
    Unlike an empirical bootstrap, zero observed discordance never gives [0, 0].
    """
    if not ids or not 0 < confidence < 1:
        raise ValueError("A paired bound needs records and confidence in (0, 1)")
    units = {}
    for identifier in ids:
        row = baseline_rows[identifier]
        metadata = row.get("metadata", {})
        family = metadata.get("family") if isinstance(metadata, dict) else None
        key = (row["category"], "family", family) if isinstance(family, str) and family else (row["category"], "task", identifier)
        units.setdefault(key, []).append(int(candidate_rows[identifier]["passed"]) - int(row["passed"]))
    count = len(ids)
    mean = sum(sum(values) for values in units.values()) / count
    weights_squared = sum((len(values) / count) ** 2 for values in units.values())
    radius = math.sqrt(2 * math.log(2 / (1 - confidence)) * weights_squared)
    return {"confidence": confidence, "interval": [max(-1.0, mean - radius), min(1.0, mean + radius)],
            "independent_unit_count": len(units), "record_count": count,
            "method": "weighted Hoeffding, paired delta range [-1,1]",
            "unit": "category/family where declared; otherwise task ID",
            "assumptions": "Independent sampled units; arbitrary dependence within a family. Authored convenience tasks do not establish representative sampling or independence.",
            "promotion_gate": False}


def paired_comparison(baseline, candidate, draws, seed, allow_setting_differences=()):
    a = {r["id"]: r for r in baseline["samples"]}
    b = {r["id"]: r for r in candidate["samples"]}
    comparisons = {}
    for category in ["overall", *sorted({r["category"] for r in a.values()})]:
        ids = [i for i in a if category == "overall" or a[i]["category"] == category]
        errors = []
        for identifier in ids:
            x, y = a[identifier], b[identifier]
            if x["status"] != "scored" or y["status"] != "scored":
                errors.append("incomplete_scoring")
            fp = x.get("prompt_token_id_fingerprint")
            if not isinstance(fp, str) or not fp or fp != y.get("prompt_token_id_fingerprint"):
                errors.append("unverified_or_different_prompt_tokens")
            count = x.get("prompt_token_count")
            if type(count) is not int or count <= 0 or count != y.get("prompt_token_count"):
                errors.append("unverified_or_different_prompt_lengths")
        differences = {}
        for setting in COMPARISON_SETTINGS:
            left, right = baseline["settings"], candidate["settings"]
            if setting not in left or setting not in right:
                errors.append(f"unverified_or_different_{setting}")
            elif left[setting] != right[setting]:
                differences[setting] = {"baseline": left[setting], "candidate": right[setting]}
                if setting not in allow_setting_differences:
                    errors.append(f"unverified_or_different_{setting}")
        result = {"tasks": len(ids), "qualified": not errors, "reasons": sorted(set(errors)),
                  "setting_differences": differences,
                  "comparison_kind": "runtime_setting_experiment" if differences and not errors else "matched_generation_settings"}
        if not errors:
            groups = [[i for i in ids if a[i]["category"] == group]
                      for group in sorted({a[i]["category"] for i in ids})]
            delta = {i: int(b[i]["passed"]) - int(a[i]["passed"]) for i in ids}
            rng = random.Random(seed)
            bootstrap = sorted(sum(delta[rng.choice(group)] for group in groups for _ in group) / len(ids) for _ in range(draws))
            result.update(accuracy_delta=sum(delta.values()) / len(ids),
                          paired_95_percent_interval=[bootstrap[int(draws * .025)], bootstrap[min(draws - 1, int(draws * .975))]],
                          candidate_only_correct=sum(v == 1 for v in delta.values()),
                          baseline_only_correct=sum(v == -1 for v in delta.values()),
                          conservative_finite_sample_bound=conservative_paired_bound(a, b, ids))
        comparisons[category] = result
    return comparisons


def evaluate(tasks_path, answers_path, reports, sandbox=None, draws=10000, seed=20260904, allow_setting_differences=(), retrieval_diagnostics=False):
    if set(allow_setting_differences) - {"kv_compression"}:
        raise ValueError("Only kv_compression may be explicitly allowed to differ")
    tasks, keys = load_inputs(tasks_path, answers_path)
    if not reports or any(not LABEL.fullmatch(label) for label in reports):
        raise ValueError("Provide at least one valid labeled native report")
    if type(draws) is not int or not 100 <= draws <= 100000:
        raise ValueError("Bootstrap draws must be in 100...100000")
    tasks_info = file_info(tasks_path)
    models = {label: score_report(path, tasks, keys, tasks_info["sha256"], sandbox, retrieval_diagnostics) for label, path in reports.items()}
    return {"format": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "Single greedy generation task accuracy; code success uses supplied executable tests; cybersecurity success requires labeled verdict, CWE and source evidence. NLL/KL are separate diagnostics.",
            "provenance": {"tasks": tasks_info, "private_answers": file_info(answers_path),
                           "scorer": file_info(__file__), "python": sys.version,
                           "sandbox": sandbox.provenance if sandbox else None},
            "allowed_setting_differences": list(allow_setting_differences),
            "retrieval_diagnostics_enabled": retrieval_diagnostics,
            "bootstrap": {"draws": draws, "seed": seed, "unit": "paired task; overall stratified by category, record-weighted"},
            "models": models,
            "comparisons": {f"{b}_minus_{a}": paired_comparison(models[a], models[b], draws, seed, allow_setting_differences)
                            for a, b in itertools.combinations(models, 2)},
            "limitations": ["Public benchmark contamination, prior reference-evaluation exposure, source correlation, finite test coverage and multiple comparisons are not corrected by the bootstrap.",
                            "Overall accuracy weights each task equally; it is withheld if any task is unscored. Subset accuracy is diagnostic only.",
                            "One greedy answer per task is used. Output-limit failures remain in denominators; no best-of-n selection or retries are scored.",
                            "Matching token identities and all recorded execution settings gate paired comparisons. Only explicitly allowed kv_compression differences qualify as runtime-setting experiments, with both values retained.",
                            "Retrieval target lengths are estimates; sample prompt_token_count is the measured length. Synthetic key lookup is not a broad long-context reasoning test.",
                            "Code never runs on the host. The Docker sandbox is a containment boundary, not a formally secure or adversarially tamper-proof grader.",
                            "Security-json scores only the labeled threat scope and source lines; it does not establish broad security expertise, remediation quality or exploit validity.",
                            "Timing is excluded from all accuracy comparisons. No speed or universal-quality promotion follows automatically."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--report", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retrieval-diagnostics", action="store_true", help="Add expected-value/tag diagnostics; strict exact-match retrieval accuracy stays unchanged")
    parser.add_argument("--sandbox-image", help="Already installed trusted Python Linux image pinned by digest; no pull or unsandboxed fallback")
    parser.add_argument("--code-timeout", type=float, default=15)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--allow-setting-difference", action="append", choices=["kv_compression"], default=[], help="Explicit KV experiment; all other execution settings must still match")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("Output already exists")
        reports = {}
        for value in args.report:
            label, separator, path = value.partition("=")
            if not separator or label in reports:
                raise ValueError("Reports must use unique LABEL=PATH arguments")
            reports[label] = path
        sandbox = DockerSandbox(args.sandbox_image, args.code_timeout) if args.sandbox_image else None
        result = evaluate(args.tasks, args.answers, reports, sandbox, args.bootstrap_draws, args.seed, args.allow_setting_difference, args.retrieval_diagnostics)
        write_new(args.output, result)
        print(json.dumps({label: model["overall"] for label, model in result["models"].items()}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, SyntaxError, subprocess.SubprocessError) as error:
        print(f"Generated evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
