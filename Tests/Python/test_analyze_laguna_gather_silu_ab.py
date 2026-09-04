"""CPU-only validation of native same-loaded kernel A/B evidence gates."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "Scripts/analyze-laguna-gather-silu-ab.py"
spec = importlib.util.spec_from_file_location("analyze_gather_silu", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture(pairs=4):
    trials = []
    for pair in range(pairs):
        modes = [analysis.STOCK, analysis.FUSED]
        if pair % 2:
            modes.reverse()
        for mode in modes:
            fused = mode == analysis.FUSED
            trials.append({
                "sequence": len(trials) + 2, "mode": mode,
                "prompt_token_id_fingerprint": "fnv1a64:64:abcdef", "content": "A fixed greedy answer.",
                "time_to_first_token_milliseconds": 90 if fused else 100,
                "total_milliseconds": 900 if fused else 1000,
                "metrics": {"prompt_token_count": 64, "generation_token_count": 32,
                    "prefilled_prompt_token_count": 64, "cached_prompt_token_count": 0,
                    "tokens_per_second": 22 if fused else 20,
                    "prompt_tokens_per_second": 110 if fused else 100}})
    return {"format": 1, "status": "measured", "requested_tokens": 32,
            "measured_trials": len(trials), "warmup_count": 2, "trials": trials}


class NativeKernelABAnalysisTests(unittest.TestCase):
    def assert_withheld(self, report):
        result = analysis.analyze(report)
        for metric in result["metrics"].values():
            self.assertIsNone(metric["accepted_paired_median_ratio"])
        return result

    def test_balanced_identical_output_pairs_and_latency_direction(self):
        result = analysis.analyze(fixture())
        self.assertTrue(result["comparable"])
        self.assertEqual(result["orders"], {"AB": 2, "BA": 2})
        self.assertEqual(result["metrics"]["decode"]["accepted_paired_median_ratio"], 1.1)
        self.assertEqual(result["metrics"]["decode"]["accepted_paired_bootstrap_95_percent_interval"], [1.1, 1.1])
        self.assertAlmostEqual(result["metrics"]["ttft"]["accepted_paired_median_ratio"], 100 / 90)

    def test_positive_fused_trace_counter_is_retained(self):
        report = fixture()
        report["laguna_gather_silu_trace_count"] = 39
        result = analysis.analyze(report)
        self.assertTrue(result["comparable"])
        self.assertEqual(result["fused_kernel_trace_verification"]["status"], "observed")
        self.assertEqual(result["fused_kernel_trace_verification"]["reported_count"], 39)
        self.assertEqual(result["metrics"]["decode"]["accepted_paired_median_ratio"], 1.1)

    def test_present_invalid_fused_trace_counter_withholds_all_metrics(self):
        for value in (0, -1, True, False, None, "39", 39.0):
            with self.subTest(value=value):
                report = fixture()
                report["laguna_gather_silu_trace_count"] = value
                result = self.assert_withheld(report)
                self.assertFalse(result["comparable"])
                self.assertEqual(result["fused_kernel_trace_verification"]["status"], "invalid")
                self.assertTrue(any("trace_count" in error for error in result["validation_errors"]))

    def test_older_report_retains_statistics_with_explicit_missing_trace_provenance(self):
        result = analysis.analyze(fixture())
        self.assertTrue(result["comparable"])
        self.assertEqual(result["fused_kernel_trace_verification"]["status"], "not_recorded")
        self.assertIsNone(result["fused_kernel_trace_verification"]["reported_count"])
        self.assertIn("diagnostic", result["fused_kernel_trace_verification"]["interpretation"])
        self.assertEqual(result["metrics"]["decode"]["accepted_paired_median_ratio"], 1.1)

    def test_missing_or_different_fingerprints_withholds_all_metrics(self):
        for fingerprint in (None, "", "fnv1a64:different"):
            report = fixture()
            report["trials"][0]["prompt_token_id_fingerprint"] = fingerprint
            self.assertFalse(self.assert_withheld(report)["comparable"])

    def test_same_length_different_text_and_cross_pair_changes_rejected(self):
        report = fixture()
        report["trials"][0]["content"] = "A changed greedy text."
        self.assertFalse(self.assert_withheld(report)["comparable"])
        report = fixture()
        for trial in report["trials"][:2]:
            trial["content"] = "Both arms changed only in pair zero."
        self.assertFalse(self.assert_withheld(report)["comparable"])

    def test_early_stop_and_different_prompt_or_cache_lengths_rejected(self):
        for key, value in (("generation_token_count", 31), ("prompt_token_count", 65), ("cached_prompt_token_count", 1)):
            report = fixture()
            report["trials"][0]["metrics"][key] = value
            self.assertFalse(self.assert_withheld(report)["comparable"])

    def test_pairing_modes_count_and_sequence_corruption_rejected(self):
        for change in ("mode", "count", "sequence", "unpaired"):
            report = fixture()
            if change == "mode":
                report["trials"][0]["mode"] = analysis.FUSED
            elif change == "count":
                report["measured_trials"] += 1
            elif change == "sequence":
                report["trials"][1]["sequence"] = report["trials"][0]["sequence"]
            else:
                report["trials"].pop()
                report["measured_trials"] -= 1
            self.assertFalse(self.assert_withheld(report)["comparable"])

    def test_candidate_only_decode_drift_withholds_decode(self):
        report = fixture()
        for trial in report["trials"][4:]:
            if trial["mode"] == analysis.FUSED:
                trial["metrics"]["tokens_per_second"] *= 1.3
        result = analysis.analyze(report)
        self.assertIsNone(result["metrics"]["decode"]["accepted_paired_median_ratio"])
        self.assertAlmostEqual(result["metrics"]["decode"]["candidate_drift_fraction"], 0.3)
        self.assertIsNotNone(result["metrics"]["ttft"]["accepted_paired_median_ratio"])

    def test_ttft_drift_does_not_hide_behind_stable_decode(self):
        report = fixture()
        for trial in report["trials"][4:]:
            if trial["mode"] == analysis.STOCK:
                trial["time_to_first_token_milliseconds"] *= 1.3
        result = analysis.analyze(report)
        self.assertIsNone(result["metrics"]["ttft"]["accepted_paired_median_ratio"])
        self.assertIsNotNone(result["metrics"]["decode"]["accepted_paired_median_ratio"])

    def test_order_effect_and_unequal_order_counts_withhold(self):
        report = fixture()
        for trial in report["trials"]:
            if trial["mode"] == analysis.FUSED and trial["sequence"] in (3, 7):
                trial["metrics"]["tokens_per_second"] *= 1.3
        result = analysis.analyze(report)
        self.assertIsNone(result["metrics"]["decode"]["accepted_paired_median_ratio"])
        self.assertGreater(result["metrics"]["decode"]["order_effect_fraction"], 0.1)
        report = fixture()
        for offset in (2, 6):
            report["trials"][offset:offset + 2] = list(reversed(report["trials"][offset:offset + 2]))
        for i, trial in enumerate(report["trials"]):
            trial["sequence"] = i + 2
        self.assertTrue(self.assert_withheld(report)["comparable"])

    def test_small_sample_and_nonfinite_metrics_cannot_be_accepted(self):
        self.assertTrue(self.assert_withheld(fixture(2))["comparable"])
        for value in (0, -1, float("nan"), float("inf"), True):
            report = fixture()
            report["trials"][0]["metrics"]["tokens_per_second"] = value
            self.assertFalse(self.assert_withheld(report)["comparable"])

    def test_cli_hashes_retained_report_and_never_overwrites_it(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "native.json"
            out = Path(directory) / "analysis.json"
            payload = json.dumps(fixture()).encode()
            raw.write_bytes(payload)
            self.assertEqual(analysis.main([str(raw), "--output", str(out)]), 0)
            result = json.loads(out.read_text())
            self.assertEqual(result["provenance"]["native_report_sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(analysis.main([str(raw), "--output", str(out)]), 0)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(analysis.main([str(raw), "--output", str(raw)]), 2)
                out.write_text("retain me")
                self.assertEqual(analysis.main([str(raw), "--output", str(out)]), 2)
            self.assertEqual(raw.read_bytes(), payload)
            self.assertEqual(out.read_text(), "retain me")


if __name__ == "__main__":
    unittest.main()
