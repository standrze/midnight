"""CPU checks for evidence gates; these do not execute or validate CUDA kernels."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("expert_probe", Path(__file__).with_name("probe.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def reports():
    return [dict(runtime_flag=flag, trace=False, mlx_extension_sha256="c" * 64,
                 mlx_extension="/tmp/mlx/core.so", python="/tmp/python", mlx_version="0.32.0",
                 cases=[dict(case, median_ms=1.0,
                    samples_ms=[1.0] * 5, output_sha256="a" * 64,
                    max_reference_error=0.01) for case in probe.cases()])
            for flag in ("0", "1", "1", "0")]


class EvidenceGateTests(unittest.TestCase):
    def test_exact_output_is_still_not_production_promotion(self):
        result = probe.analyze(reports())
        self.assertEqual(result["status"], "exact_fixtures_passed")
        self.assertEqual(result["promotion"], "blocked_pending_full_model_validation")

    def test_better_reference_error_does_not_excuse_changed_output(self):
        data = reports()
        data[1]["cases"][0].update(output_sha256="b" * 64, max_reference_error=0.0)
        self.assertEqual(probe.analyze(data)["status"], "failed_output_parity")

    def test_fallback_parity_is_required(self):
        data = reports()
        data[2]["cases"][-1]["output_sha256"] = "b" * 64
        self.assertEqual(probe.analyze(data)["status"], "failed_output_parity")

    def test_instrumented_measurements_are_rejected(self):
        data = reports()
        data[0]["trace"] = True
        with self.assertRaises(ValueError):
            probe.analyze(data)

    def test_missing_case_is_rejected(self):
        data = reports()
        data[1]["cases"].pop()
        with self.assertRaises(ValueError):
            probe.analyze(data)

    def test_different_binaries_are_rejected(self):
        data = reports()
        data[1]["mlx_extension_sha256"] = "d" * 64
        with self.assertRaises(ValueError):
            probe.analyze(data)

    def test_different_runtime_paths_are_rejected(self):
        for key in ("mlx_extension", "python", "mlx_version"):
            data = reports()
            data[1][key] = "different"
            with self.assertRaises(ValueError):
                probe.analyze(data)

    def test_median_and_sample_count_are_checked(self):
        for samples in ([1.0] * 4, [2.0] * 5):
            data = reports()
            data[1]["cases"][0]["samples_ms"] = samples
            with self.assertRaises(ValueError):
                probe.analyze(data)

    def test_empty_or_nonpositive_evidence_is_rejected(self):
        for field, value in (("output_sha256", ""), ("median_ms", 0), ("median_ms", float("nan")),
                             ("samples_ms", []), ("samples_ms", [float("inf")])):
            with self.subTest(field=field):
                data = reports()
                data[0]["cases"][0][field] = value
                with self.assertRaises(ValueError):
                    probe.analyze(data)

    def test_trace_requires_per_fixture_eligibility(self):
        lines = []
        for case in probe.cases():
            lines.append(f"midnight_expert_fixture_begin {case['name']}")
            if case["eligible"]:
                lines.append(f"midnight_cuda_expert_qmv m=1 b=8 n={case['n']} k={case['k']}")
            lines.append(f"midnight_expert_fixture_end {case['name']}")
        log = "\n".join(lines)
        self.assertEqual(len(probe.validate_trace(log)), len(probe.cases()))
        # Preserve aggregate six events, but move one to a fallback fixture.
        faulty = log.replace("midnight_cuda_expert_qmv m=1 b=8 n=512 k=2048\n", "", 1)
        faulty = faulty.replace("midnight_expert_fixture_end top7", "midnight_cuda_expert_qmv m=1 b=8 n=512 k=2048\nmidnight_expert_fixture_end top7")
        with self.assertRaises(ValueError):
            probe.validate_trace(faulty)

    def test_unscoped_or_incomplete_trace_is_rejected(self):
        for log in ("", "midnight_cuda_expert_qmv m=1 b=8 n=512 k=2048", "midnight_expert_fixture_begin gate-float16"):
            with self.assertRaises(ValueError):
                probe.validate_trace(log)


if __name__ == "__main__":
    unittest.main()
