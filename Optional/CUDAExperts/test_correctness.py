"""CPU checks for the standalone binary's strict evidence reader."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import run_correctness
from probe import cases, validate_trace


class StandaloneEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directories = [Path(self.temporary.name) / label for label in ("stock", "candidate")]
        for directory, flag in zip(self.directories, ("0", "1")):
            directory.mkdir()
            records = []
            for fixture in cases():
                size = fixture["selected"] * fixture["m"] * fixture["n"] * 4
                (directory / (fixture["name"] + ".f32")).write_bytes(bytes(size))
                records.append(dict(fixture, output_bytes=size, max_reference_error=0.1, rms_reference_error=0.01))
            (directory / "results.json").write_text(json.dumps(dict(runtime_flag=flag, timing="not_measured", cases=records)))

    def test_exact_fixture_pass_keeps_production_gate_closed(self):
        result = run_correctness.compare(*self.directories)
        self.assertEqual(result["status"], "exact_fixtures_passed")
        self.assertEqual(result["promotion"], "blocked_pending_full_model_validation")

    def test_signed_zero_difference_fails_exact_gate(self):
        path = self.directories[1] / "gate-float16.f32"
        data = path.read_bytes()
        path.write_bytes(struct.pack("<f", -0.0) + data[4:])
        result = run_correctness.compare(*self.directories)
        self.assertEqual(result["status"], "failed_output_parity")
        self.assertEqual(result["promotion"], "blocked_by_output_parity")
        self.assertEqual(result["cases"][0]["differing_values"], 1)

    def test_truncated_or_nonfinite_outputs_rejected(self):
        path = self.directories[1] / "group128.f32"
        original = path.read_bytes()
        for data in (original[:-4], struct.pack("<f", float("nan")) + original[4:]):
            path.write_bytes(data)
            with self.assertRaises(ValueError):
                run_correctness.compare(*self.directories)

    def test_invalid_reference_diagnostic_rejected(self):
        path = self.directories[1] / "results.json"
        report = json.loads(path.read_text())
        report["cases"][0]["max_reference_error"] = float("nan")
        path.write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            run_correctness.compare(*self.directories)

    def test_stock_repeat_requires_explicit_stock_flags(self):
        path = self.directories[1] / "results.json"
        report = json.loads(path.read_text())
        report["runtime_flag"] = "0"
        path.write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            run_correctness.compare(*self.directories)
        result = run_correctness.compare(*self.directories, expected_flags=("0", "0"))
        self.assertEqual(result["status"], "exact_fixtures_passed")

    def test_baseline_trace_rejects_any_candidate_dispatch(self):
        log = "\n".join(line for fixture in cases() for line in (
            f"midnight_expert_fixture_begin {fixture['name']}",
            f"midnight_expert_fixture_end {fixture['name']}"))
        self.assertEqual(len(validate_trace(log, candidate=False)), 14)
        log = log.replace("midnight_expert_fixture_end gate-float16", "midnight_cuda_expert_qmv m=1 b=8 n=512 k=2048\nmidnight_expert_fixture_end gate-float16")
        with self.assertRaises(ValueError):
            validate_trace(log, candidate=False)


if __name__ == "__main__":
    unittest.main()
