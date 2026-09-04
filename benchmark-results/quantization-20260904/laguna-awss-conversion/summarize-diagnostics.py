#!/usr/bin/env python3
"""Summarize the archived AWSS conversion objectives without models or inference."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "measurement-summary.json").read_text())
    archive = manifest["full_report_archive"]
    compressed = (root / archive["path"]).read_bytes()
    assert len(compressed) == archive["compressed_bytes"]
    assert hashlib.sha256(compressed).hexdigest() == archive["compressed_sha256"]
    payload = gzip.decompress(compressed)
    assert len(payload) == archive["original_bytes"]
    assert hashlib.sha256(payload).hexdigest() == archive["original_sha256"]
    report = json.loads(payload)
    diagnostics = report["activation_diagnostics"]
    retained = report["retained_template_experts"]
    values = list(diagnostics.values())
    for value in values:
        assert value["element_count"] == value["group_count"] * report["group_size"]
        assert 0 <= value["changed_group_count"] <= value["group_count"]
        assert 0 <= value["validation_rejected_group_count"] <= value["group_count"]
    total_elements = sum(value["element_count"] for value in values)
    aggregates = {}
    for objective in ["raw_mse", "calibration_weighted_mse", "validation_weighted_mse"]:
        scores = {}
        for side in ["template", "candidate"]:
            entries = [value[f"{side}_{objective}"] for value in values]
            assert all(math.isfinite(x) and x >= 0 for x in entries)
            scores[side] = math.fsum(
                value[f"{side}_{objective}"] * value["element_count"]
                for value in values
            ) / total_elements
        scores["reduction_percent"] = 100 * (1 - scores["candidate"] / scores["template"])
        aggregates[objective] = scores
    distributions = {}
    for key in ["raw_mse_reduction_percent", "calibration_weighted_mse_reduction_percent",
                "validation_weighted_mse_reduction_percent"]:
        entries = [value[key] for value in values]
        assert all(math.isfinite(x) for x in entries)
        distributions[key] = {
            "minimum": min(entries), "median": statistics.median(entries),
            "maximum": max(entries), "negative_count": sum(x < 0 for x in entries),
            "zero_count": sum(x == 0 for x in entries),
        }
    retained_entries = sum(len(experts) for experts in retained.values())
    retained_layer_experts = {(module.split(".mlp.")[0], expert)
                              for module, experts in retained.items() for expert in experts}
    routed_diagnostics = sum(".expert." in key for key in diagnostics)
    assert routed_diagnostics + retained_entries == 39 * 256 * 3
    summary = {
        "format": 1,
        "source_report_sha256": archive["original_sha256"],
        "source_report_bytes": archive["original_bytes"],
        "comparison_template": report["template_model"],
        "minimum_expert_positions_per_corpus": report["minimum_expert_positions"],
        "retention": {
            "routed_projection_modules": len(retained),
            "routed_expert_projection_entries": 39 * 256 * 3,
            "retained_expert_projection_entries": retained_entries,
            "retained_unique_layer_expert_pairs": len(retained_layer_experts),
            "refined_expert_projection_entries": routed_diagnostics,
        },
        "diagnostic_entries": {
            "total": len(diagnostics), "routed": routed_diagnostics,
            "dense": len(diagnostics) - routed_diagnostics,
        },
        "diagnostic_group_counts": {
            key: sum(value[key] for value in values)
            for key in ["group_count", "changed_group_count", "validation_rejected_group_count"]
        },
        "diagnostic_element_count": total_elements,
        "per_matrix_reduction_percent_distribution": distributions,
        "element_weighted_local_objectives": aggregates,
        "aggregation_formula": "sum(matrix_local_mse * matrix_element_count) / sum(matrix_element_count)",
        "qualifications": [
            "Only matrices/expert projections with diagnostics are included in objective aggregates; retained experts, Q8 routers and preserved embedding are excluded.",
            "Retained entries use the LS2 template. Coverage and usable-moment guards run before per-expert refinement; these entries have no candidate objective record.",
            "validation_rejected_group_count counts groups where at least one calibration-improving search candidate failed the development gate. It is accumulated across factors, not the number of groups finally retained; it can overlap changed_group_count.",
            "Weighted objectives use local, group-normalized channel moments. Combining these matrix errors does not define full-model activation error or token-level loss.",
            "Calibration and development corpora participate in fitting/selection; these are optimization diagnostics, not an independent model-quality comparison.",
            "Counts of worsening raw-MSE matrices weight each diagnostic entry equally; the element-weighted aggregate has a different weighting.",
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
