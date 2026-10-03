#!/usr/bin/env python3
"""Compare compact greedy-choice diagnostics at identical teacher-forced prefixes."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

SOURCE = Path(__file__).with_name("analyze-quality-nll.py")
SPEC = importlib.util.spec_from_file_location("quality_nll", SOURCE)
nll = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nll)


def diagnostics(report):
    samples = nll.validate_report(report)
    nll.require(report["prefill_step_size"] == 1, "Diagnostics must use one-token model calls")
    blocks = report.get("token_diagnostics")
    nll.require(isinstance(blocks, list) and len(blocks) == len(samples), "Missing sample diagnostics")
    result = []
    for sample, block in zip(samples, blocks):
        nll.require(block.get("id") == sample["id"], "Diagnostic sample identity differs")
        rows = block.get("rows")
        nll.require(isinstance(rows, list) and len(rows) == sample["scored_token_count"], "Incomplete token diagnostics")
        for position, row in enumerate(rows, 1):
            nll.require(row.get("position") == position, "Token positions must be complete and ordered")
            for key in ("reference_token_id", "winner_token_id", "runner_up_token_id"):
                nll.require(type(row.get(key)) is int and row[key] >= 0, f"Invalid {key}")
            nll.require(row["winner_token_id"] != row["runner_up_token_id"], "Winner and runner-up must differ")
            for key in ("winner_logit", "runner_up_logit", "winner_margin", "reference_nll"):
                nll.require(type(row.get(key)) in (int, float) and math.isfinite(row[key]), f"Nonfinite {key}")
            nll.require(row["winner_margin"] >= 0 and row["reference_nll"] >= 0, "Negative margin or NLL")
            nll.require(math.isclose(row["winner_margin"], row["winner_logit"] - row["runner_up_logit"],
                                     rel_tol=1e-5, abs_tol=2e-5), "Margin does not reconcile")
            result.append({"id": sample["id"], "category": sample["category"], **row})
        nll.require(math.isclose(sum(r["reference_nll"] for r in rows), sample["nll_sum"],
                                 rel_tol=1e-10, abs_tol=1e-8), "Per-token NLL does not reconcile")
    return result


def analyze(baseline, candidate):
    for key in (*nll.SETTING_FIELDS, "model_implementation", "tokenization", "model_path"):
        nll.require(baseline.get(key) is not None and baseline[key] == candidate.get(key), f"Different/missing {key}")
    a, b = diagnostics(baseline), diagnostics(candidate)
    nll.require(len(a) == len(b), "Different diagnostic lengths")
    flips, baseline_correct, candidate_correct = [], 0, 0
    same_winner_logit_changes = []
    for left, right in zip(a, b):
        for key in ("id", "category", "position", "reference_token_id"):
            nll.require(left[key] == right[key], f"Different fixed-prefix identity: {key}")
        baseline_correct += left["winner_token_id"] == left["reference_token_id"]
        candidate_correct += right["winner_token_id"] == right["reference_token_id"]
        if left["winner_token_id"] != right["winner_token_id"]:
            flips.append({"id": left["id"], "category": left["category"], "position": left["position"],
                          "reference_token_id": left["reference_token_id"], "baseline": left, "candidate": right,
                          "candidate_minus_baseline_reference_nll": right["reference_nll"] - left["reference_nll"]})
        else:
            same_winner_logit_changes.append(abs(right["winner_logit"] - left["winner_logit"]))
    return {"format": 1, "status": "analyzed", "fixed_prefix_count": len(a),
            "winner_agreement_count": len(a) - len(flips), "winner_flip_count": len(flips),
            "winner_agreement_fraction": (len(a) - len(flips)) / len(a),
            "baseline_reference_top1_correct": baseline_correct, "candidate_reference_top1_correct": candidate_correct,
            "flip_baseline_margins_at_most": {str(t): sum(r["baseline"]["winner_margin"] <= t for r in flips)
                                               for t in (0, 0.0001, 0.001, 0.01, 0.1)},
            "maximum_same_winner_logit_change": max(same_winner_logit_changes, default=None),
            "flips": flips,
            "limitations": ["All comparisons use the same reference prefixes; no free-generation trajectories are compared.",
                            "Margin bins describe observed near ties; none is a quality acceptance threshold.",
                            "Only winner and runner-up logits are retained. This is not full-vocabulary max error or teacher KL.",
                            "Reference top-1 agreement and NLL cannot establish generated-task quality or global equivalence."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{result['winner_flip_count']} winner changes across {result['fixed_prefix_count']} identical prefixes")


if __name__ == "__main__":
    main()
