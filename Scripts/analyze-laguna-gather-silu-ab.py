#!/usr/bin/env python3
"""Analyze native same-loaded gather/SILU A/B reports without running inference."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

CAMPAIGN_PATH = Path(__file__).with_name("benchmark-campaign.py")
spec = importlib.util.spec_from_file_location("benchmark_campaign", CAMPAIGN_PATH)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)

STOCK = "laguna_gather_silu_stock"
FUSED = "laguna_gather_silu_fused"
METRICS = {
    "decode": ("tokens_per_second", False),
    "prefill": ("prompt_tokens_per_second", False),
    "ttft": ("time_to_first_token_milliseconds", True),
    "total": ("total_milliseconds", True),
}


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def analyze(report, maximum_drift_fraction=0.1, seed=20260904):
    """Keep diagnostic estimates, but never accept evidence that fails a gate."""
    if not isinstance(report, dict) or report.get("format") != 1 or report.get("status") != "measured":
        raise ValueError("Expected a format-1 measured native runtime report")
    if not 0 <= maximum_drift_fraction < 1:
        raise ValueError("maximum_drift_fraction must be between zero (inclusive) and one")
    trials = report.get("trials")
    requested = report.get("requested_tokens")
    if not isinstance(trials, list) or not trials or not integer(requested) or requested <= 0:
        raise ValueError("Expected nonempty trials and positive integer requested_tokens")
    errors = []
    trace_recorded = "laguna_gather_silu_trace_count" in report
    trace_count = report.get("laguna_gather_silu_trace_count")
    trace_positive = integer(trace_count) and trace_count > 0
    if trace_recorded and not trace_positive:
        errors.append("laguna_gather_silu_trace_count must be a positive integer when present; fused graph construction is unverified")
    trace_status = "observed" if trace_positive else ("invalid" if trace_recorded else "not_recorded")
    if report.get("measured_trials") != len(trials):
        errors.append("measured_trials does not match retained trial count")
    if len(trials) % 2:
        errors.append("incomplete final pair")
    sequences = [t.get("sequence") if isinstance(t, dict) else None for t in trials]
    if not all(integer(s) for s in sequences) or any(b != a + 1 for a, b in zip(sequences, sequences[1:])):
        errors.append("trial sequences are not consecutive chronological records")
    valid, rejected, identities, outputs, lengths = [], [], set(), set(), set()
    for offset in range(0, len(trials), 2):
        pair = trials[offset:offset + 2]
        reasons = []
        if len(pair) != 2 or any(not isinstance(t, dict) for t in pair):
            rejected.append({"pair": offset // 2, "reasons": ["missing or malformed arm"]})
            continue
        if {t.get("mode") for t in pair} != {STOCK, FUSED}:
            rejected.append({"pair": offset // 2, "reasons": ["each adjacent pair must contain exactly one stock and one fused trial"]})
            continue
        measurements = {}
        for trial in pair:
            metrics = trial.get("metrics", {})
            if not isinstance(metrics, dict):
                reasons.append("missing metrics object")
                continue
            fingerprint = trial.get("prompt_token_id_fingerprint")
            content = trial.get("content")
            if not isinstance(fingerprint, str) or not fingerprint.strip():
                reasons.append("missing verified prompt token fingerprint")
            else:
                identities.add(fingerprint)
            if not isinstance(content, str) or not content:
                reasons.append("missing nonempty output text")
            else:
                outputs.add(content)
            counts = tuple(metrics.get(k) for k in (
                "prompt_token_count", "generation_token_count", "prefilled_prompt_token_count", "cached_prompt_token_count"))
            if not all(integer(v) and v >= 0 for v in counts) or not counts[0]:
                reasons.append("missing or invalid prompt/generation/cache token lengths")
            else:
                lengths.add(counts)
                if counts[1] != requested:
                    reasons.append("generation length differs from requested_tokens")
                if counts[2] + counts[3] != counts[0]:
                    reasons.append("prefilled plus cached tokens differ from prompt length")
            measurement = {}
            for key, _ in METRICS.values():
                value = metrics.get(key) if key.endswith("per_second") else trial.get(key)
                if not positive(value):
                    reasons.append(f"invalid {key}")
                else:
                    measurement[key] = value
            measurements[trial["mode"]] = {"measurement": measurement, "baseline_first": pair[0]["mode"] == STOCK}
        if pair[0].get("prompt_token_id_fingerprint") != pair[1].get("prompt_token_id_fingerprint"):
            reasons.append("paired prompt token fingerprints differ")
        if pair[0].get("content") != pair[1].get("content"):
            reasons.append("paired output text differs")
        if reasons:
            rejected.append({"pair": offset // 2, "reasons": sorted(set(reasons))})
        else:
            valid.append((measurements[STOCK], measurements[FUSED]))
    if len(identities) != 1:
        errors.append("prompt token identities are not identical across all trials")
    if len(outputs) != 1:
        errors.append("output text is not identical across all trials")
    if len(lengths) != 1:
        errors.append("prompt/generation/cache token lengths are not identical across all trials")
    comparable = not errors and not rejected and len(valid) * 2 == len(trials)
    common_reasons = [] if comparable else ["failed or incomparable trials"]
    if len(valid) < 4:
        common_reasons.append("fewer than four complete pairs; exploratory only")
    ab_count = sum(a["baseline_first"] for a, _ in valid)
    ba_count = len(valid) - ab_count
    if ab_count != ba_count or not ab_count:
        common_reasons.append("AB and BA pair counts must be equal and nonzero")
    result = {
        "format": 1, "status": "analyzed" if comparable else "incomparable",
        "baseline_mode": STOCK, "candidate_mode": FUSED,
        "report_settings": {k: report.get(k) for k in (
            "created_at", "model_path", "served_model_name", "engine", "prompt", "requested_tokens",
            "context_length", "prefill_step_size", "kv_compression", "memory_limit_bytes", "allow_early_stop", "warmup_count")},
        "retained_trials": len(trials), "valid_pairs": len(valid),
        "orders": {"AB": ab_count, "BA": ba_count}, "comparable": comparable,
        "validation_errors": errors, "rejected_pairs": rejected,
        "fused_kernel_trace_verification": {
            "status": trace_status, "reported_count": trace_count,
            "interpretation": (
                "Positive count records fused graph construction on the loaded model, not every GPU dispatch or measured token."
                if trace_positive else
                "Invalid recorded counter; timing estimates are withheld."
                if trace_recorded else
                "Historical report has no trace counter. Retained statistical estimates are diagnostic and do not independently verify fused-kernel execution."
            ),
        },
        "prompt_token_identity_verified": comparable and len(identities) == 1,
        "output_text_matches_exactly": comparable and len(outputs) == 1,
        "output_text_sha256": hashlib.sha256(next(iter(outputs)).encode()).hexdigest() if len(outputs) == 1 else None,
        "thresholds": {"maximum_drift_fraction": maximum_drift_fraction, "minimum_pairs": 4, "seed": seed, "bootstrap_draws": 2000},
        "metrics": {},
        "limitations": [
            "Same-loaded native modes isolate the kernel toggle, subject to build/provenance verification outside this report.",
            "Exact decoded output text and generation length do not independently prove identical generated token IDs, expert routes or all internal activations.",
            "Accepted ratios are eligible measurements, not automatic promotion decisions. A single session and few pairs do not establish repeatability.",
            "Percentile intervals resample paired observations; thermal autocorrelation and multiple comparisons limit inference. First-half versus second-half drift checks cannot detect every nonlinear drift pattern.",
            "Native rates/latencies exclude HTTP serving overhead. Warmups are excluded from estimates and their count is recorded.",
        ],
    }
    for label, (key, lower_better) in METRICS.items():
        if valid:
            result["metrics"][label] = campaign.metric_summary(valid, key, lower_better,
                {"maximum_drift_fraction": maximum_drift_fraction, "seed": seed}, common_reasons)
        else:
            result["metrics"][label] = {"accepted_paired_median_ratio": None, "acceptance_reasons": common_reasons + ["no valid pairs"]}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, help="Write analysis JSON; default stdout")
    parser.add_argument("--maximum-drift-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args(argv)
    try:
        raw = args.report.read_bytes()
        result = analyze(json.loads(raw), args.maximum_drift_fraction, args.seed)
        result["provenance"] = {
            "native_report": str(args.report.resolve()), "native_report_sha256": hashlib.sha256(raw).hexdigest(),
            "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "statistics_helper_sha256": hashlib.sha256(CAMPAIGN_PATH.read_bytes()).hexdigest(),
            "build_attribution": "Native JSON does not contain executable/metallib hashes; retain the launch/build provenance separately.",
        }
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            if args.output.resolve() == args.report.resolve():
                raise ValueError("output must not overwrite the native report")
            if args.output.exists() and args.output.read_text() != payload:
                raise ValueError("output exists with differing content; choose a new path")
            args.output.write_text(payload)
        else:
            print(payload, end="")
        return 0 if result["comparable"] else 1
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
