"""CPU-only campaign integration tests using executable native-report fixtures."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[2] / "Scripts/benchmark-campaign.py"
spec = importlib.util.spec_from_file_location("benchmark_campaign", SCRIPT)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)

FAKE_BINARY = r'''#!/usr/bin/env python3
import json, os, re, sys
from pathlib import Path
args = sys.argv[1:]
quality = 'quality' in Path(sys.argv[0]).name
model = Path(args[0])
output = Path(args[2] if quality else args[1])
config = json.loads((model / 'config.json').read_text())
def option(key, default):
    return args[args.index(key) + 1] if key in args else default
print('fixture stdout', flush=True)
print('fixture stderr', file=sys.stderr, flush=True)
if config.get('failure') == 'nonzero':
    output.write_text('{"partial":true}')
    sys.exit(7)
if config.get('failure') == 'malformed':
    output.write_text('not json')
    sys.exit(0)
if config.get('failure') == 'timeout':
    import time
    time.sleep(5)
report = {'format':1, 'status':'measured', 'model_path':str(model)}
fingerprint = config.get('fingerprint', 'fnv1a64:fixed-input')
if quality:
    nll = config.get('nll', 2.0)
    report.update(metric='teacher_forced_next_token_nll', corpus_path=args[1],
        corpus_fingerprint='fnv1a64:fixed-corpus', token_id_fingerprint=fingerprint,
        maximum_tokens_per_sample=int(option('--max-tokens-per-sample', 512)),
        prefill_step_size=config.get('reported_prefill_step_size', int(option('--prefill-step-size', 0))),
        token_weighted_nll=nll, nll_sum=nll*20, perplexity=7.0,
        scored_token_count=20, sample_count=2, device='cpu', add_special_tokens=True)
else:
    tokens = int(option('--tokens', 256))
    repetition = int(re.search(r'-(\d{3})-', str(output)).group(1))
    rate = float(os.environ.get('MODEL_RUNNER_TEST_RATE', config.get('rate', 10)))
    if config.get('drift') and repetition >= 2:
        rate *= 2
    report.update(prompt=option('--prompt', ''), engine=option('--engine', 'cpu'),
        requested_tokens=tokens, measured_trials=1,
        prefill_step_size=int(option('--prefill-step-size', 512)),
        context_length=int(option('--context-length', 2048)),
        kv_compression=option('--kv-compression', 'none'), memory_limit_bytes=100000,
        trials=[{'mode':'target_only', 'content':config.get('content','text'),
        'prompt_token_id_fingerprint':fingerprint, 'time_to_first_token_milliseconds':(8 if config.get('ttft_drift') and repetition >= 2 else 2),
        'total_milliseconds':100, 'peak_active_memory_bytes':10000,
        'metrics':{'generation_token_count':tokens-1 if config.get('early_stop') else tokens,
        'prompt_token_count':12, 'tokens_per_second':rate, 'prompt_tokens_per_second':30}}])
output.write_text(json.dumps(report))
'''


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="midnight-campaign-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        release = self.root / "release"
        release.mkdir()
        self.binaries = {}
        for mode in ("runtime", "quality"):
            path = release / f"model-runner-{mode}-bench"
            path.write_text(FAKE_BINARY)
            path.chmod(0o755)
            self.binaries[mode] = str(path)
        (release / "mlx.metallib").write_bytes(b"fixture metal")
        self.models = []
        for label in ("standard", "candidate"):
            path = self.root / label
            path.mkdir()
            (path / "config.json").write_text(json.dumps({"model_type": "fixture", "rate": 10 if label == "standard" else 12}))
            (path / "tokenizer.json").write_text('{"fixture":true}')
            (path / "model.safetensors").write_bytes(b"test weights")
            self.models.append({"label": label, "path": str(path)})
        corpus = self.root / "corpus.jsonl"
        corpus.write_text('{"id":"fixture","text":"fixed corpus"}\n')
        self.manifest = {"version": 1, "seed": 41, "pairs": 4, "baseline": "standard",
            "binaries": self.binaries, "models": self.models,
            "runtime": {"native_args": ["--engine", "cpu", "--tokens", "8", "--warmups", "0"],
                        "prompts": [{"label": "fixed", "text": "Prompt with spaces, $literal and `backticks`."}]},
            "quality": {"native_args": ["--cpu", "--max-tokens-per-sample", "64"],
                        "corpora": [{"label": "fixed", "path": str(corpus)}]}}

    def set_config(self, label, **values):
        path = self.root / label / "config.json"
        config = json.loads(path.read_text())
        path.write_text(json.dumps({**config, **values}))

    def run_campaign(self, mode="runtime", extra=None):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(self.manifest))
        output = self.root / "results"
        args = ["run", str(manifest), "--output", str(output), "--mode", mode, *(extra or [])]
        # Keep the test CPU-only and avoid collecting unrelated host hardware state.
        with mock.patch.object(campaign, "machine_state", return_value={"fixture": True}), \
             mock.patch.object(campaign, "command_output", return_value={"exit_code": 0, "stdout": "fixture", "stderr": ""}), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = campaign.main(args)
        return code, output

    def test_seeded_order_is_reproducible_and_counterbalanced(self):
        manifest, modes = campaign.validate_manifest(self.manifest, "all")
        first = campaign.schedule(manifest, modes)
        self.assertEqual(first, campaign.schedule(manifest, modes))
        for mode in modes:
            runs = [r for r in first if r["mode"] == mode and r["model"] == "standard"]
            self.assertEqual(sum(r["baseline_first"] for r in runs), 2)
        self.assertEqual(len(first), 16)

    def test_real_subprocess_reports_are_aggregated_and_hashed(self):
        code, output = self.run_campaign("all", ["--full-weight-hash"])
        self.assertEqual(code, 0)
        summary = json.loads((output / "summary.json").read_text())
        runtime = next(c for c in summary["comparisons"] if c["mode"] == "runtime")
        self.assertAlmostEqual(runtime["accepted_paired_median_decode_ratio"], 1.2)
        self.assertTrue(runtime["prompt_token_identity_verified"])
        self.assertEqual(runtime["candidate_median_peak_active_memory_bytes"], 10000)
        prov = json.loads((output / "provenance.json").read_text())
        self.assertEqual(prov["binaries"]["runtime"]["executable"]["sha256"], campaign.sha256_file(self.binaries["runtime"]))
        self.assertIn("sha256", prov["models"][0]["weights"][0])
        self.assertTrue(prov["binaries"]["runtime"]["metallib"]["sha256"])
        self.assertEqual(len(list((output / "runs").glob("*/native.json"))), 16)
        self.assertEqual(len(list((output / "runs").glob("*/stderr.log"))), 16)

    def test_nonzero_process_keeps_partial_report_and_does_not_promote_ratio(self):
        self.set_config("candidate", failure="nonzero")
        code, output = self.run_campaign()
        self.assertEqual(code, 1)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["failed_runs"], 4)
        self.assertFalse(summary["comparisons"][0]["comparable"])
        partial = list((output / "runs").glob("*candidate/native.json"))
        self.assertEqual(len(partial), 4)
        self.assertEqual(json.loads(partial[0].read_text()), {"partial": True})

    def test_malformed_json_is_retained_as_failure(self):
        self.set_config("candidate", failure="malformed")
        code, output = self.run_campaign()
        self.assertEqual(code, 1)
        records = json.loads((output / "results.json").read_text())
        self.assertEqual(sum(r["status"] == "failed" for r in records), 4)

    def test_runtime_token_identity_mismatch_rejects_all_pairs(self):
        self.set_config("candidate", fingerprint="different")
        code, output = self.run_campaign()
        self.assertEqual(code, 1)
        comparison = json.loads((output / "summary.json").read_text())["comparisons"][0]
        self.assertEqual(comparison["valid_pairs"], 0)
        self.assertIn("prompt_token_id_fingerprint mismatch", comparison["rejected_pairs"][0]["reasons"])

    def test_quality_token_identity_mismatch_rejects_nll_comparison(self):
        self.set_config("candidate", fingerprint="different")
        code, output = self.run_campaign("quality")
        self.assertEqual(code, 1)
        comparison = json.loads((output / "summary.json").read_text())["comparisons"][0]
        self.assertFalse(comparison["comparable"])
        self.assertNotIn("accepted_paired_median_nll_delta", comparison)

    def test_early_stop_is_a_failure_even_when_allowed_by_native(self):
        self.manifest["runtime"]["native_args"].append("--allow-early-stop")
        self.set_config("candidate", early_stop=True)
        code, output = self.run_campaign()
        self.assertEqual(code, 1)
        self.assertIn("Incomplete generation", (output / "results.json").read_text())

    def test_drift_withholds_acceptance(self):
        self.set_config("standard", drift=True)
        code, output = self.run_campaign()
        self.assertEqual(code, 0)  # Execution completed, but no accepted performance ratio.
        comparison = json.loads((output / "summary.json").read_text())["comparisons"][0]
        self.assertIsNone(comparison["accepted_paired_median_decode_ratio"])
        self.assertGreater(comparison["baseline_drift_fraction"], 0.1)

    def test_missing_prompt_fingerprints_are_diagnostic_only(self):
        self.set_config("standard", fingerprint=None)
        self.set_config("candidate", fingerprint=None)
        code, output = self.run_campaign()
        self.assertEqual(code, 0)
        comparison = json.loads((output / "summary.json").read_text())["comparisons"][0]
        self.assertFalse(comparison["prompt_token_identity_verified"])
        self.assertIsNone(comparison["accepted_paired_median_decode_ratio"])
        self.assertIn("exact prompt token identity unavailable", comparison["acceptance_reasons"])

    def test_candidate_only_drift_withholds_acceptance(self):
        self.set_config("candidate", drift=True)
        code, output = self.run_campaign()
        self.assertEqual(code, 0)
        comparison = json.loads((output / "summary.json").read_text())["comparisons"][0]
        self.assertEqual(comparison["baseline_drift_fraction"], 0)
        self.assertGreater(comparison["candidate_drift_fraction"], 0.1)
        self.assertIsNone(comparison["accepted_paired_median_decode_ratio"])

    def test_ttft_drift_is_gated_independently_from_decode(self):
        self.set_config("candidate", ttft_drift=True)
        code, output = self.run_campaign()
        self.assertEqual(code, 0)
        comparison = json.loads((output / "summary.json").read_text())["comparisons"][0]
        self.assertAlmostEqual(comparison["accepted_paired_median_decode_ratio"], 1.2)
        self.assertIsNone(comparison["accepted_paired_median_ttft_speed_ratio"])
        self.assertEqual(comparison["metrics"]["decode"]["accepted_paired_bootstrap_95_percent_interval"], [1.2, 1.2])

    def test_per_arm_policy_and_allowlisted_environment_reach_native_process(self):
        self.manifest["environment_allowlist"] = ["MODEL_RUNNER_TEST_RATE"]
        self.models[1]["environment"] = {"MODEL_RUNNER_TEST_RATE": "15"}
        self.models[1]["runtime_native_args"] = ["--prefill-step-size", "128", "--context-length", "4096"]
        code, output = self.run_campaign()
        self.assertEqual(code, 0)
        records = json.loads((output / "results.json").read_text())
        candidate = next(r for r in records if r["model"] == "candidate")
        self.assertEqual(candidate["measurement"]["prefill_step_size"], 128)
        self.assertEqual(candidate["measurement"]["decode_tokens_per_second"], 15)
        self.assertEqual(candidate["environment"]["MODEL_RUNNER_TEST_RATE"], "15")

    def test_dry_run_does_not_launch_and_default_weight_hash_is_labeled_sampled(self):
        code, output = self.run_campaign(extra=["--dry-run"])
        self.assertEqual(code, 0)
        self.assertFalse(list((output / "runs").iterdir()))
        prov = json.loads((output / "provenance.json").read_text())
        self.assertIn("sample_sha256", prov["models"][0]["weights"][0])
        self.assertEqual(len(json.loads((output / "commands.json").read_text())), 8)

    def test_unsupported_flags_and_mismatched_workloads_fail_validation(self):
        self.manifest["runtime"]["native_args"].extend(["--seed", "1"])
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            campaign.validate_manifest(self.manifest, "runtime")
        self.manifest["runtime"]["native_args"] = []
        self.models[1]["runtime_native_args"] = ["--tokens", "99"]
        with self.assertRaisesRegex(ValueError, "must match"):
            campaign.validate_manifest(self.manifest, "runtime")

    def test_debug_binary_and_unallowlisted_environment_are_rejected(self):
        debug = self.root / "debug"
        debug.mkdir()
        binary = debug / "fixture"
        binary.write_text(FAKE_BINARY)
        binary.chmod(0o755)
        self.manifest["binaries"]["runtime"] = str(binary)
        with self.assertRaisesRegex(ValueError, "release directory"):
            campaign.validate_manifest(self.manifest, "runtime")
        self.manifest["environment"] = {"MODEL_RUNNER_OPT_IN": "1"}
        with self.assertRaisesRegex(ValueError, "allowlisted"):
            campaign.validate_manifest(self.manifest, "runtime")

    def test_quality_chunk_sizes_are_recorded_and_differences_remain_comparable(self):
        self.manifest["quality"]["native_args"] = ["--cpu", "--max-tokens-per-sample", "8192", "--prefill-step-size", "512"]
        self.models[1]["quality_native_args"] = ["--prefill-step-size", "2048"]
        code, output = self.run_campaign("quality")
        self.assertEqual(code, 0)
        comparison = json.loads((output / "summary.json").read_text())["comparisons"][0]
        self.assertTrue(comparison["comparable"])
        self.assertEqual(comparison["scoring_prefill_step_size_by_model"], {"standard": 512, "candidate": 2048})
        self.assertEqual(comparison["native_settings_by_model"]["candidate"]["--prefill-step-size"], "2048")

    def test_quality_report_must_confirm_explicit_chunk_size(self):
        self.manifest["quality"]["native_args"].extend(["--prefill-step-size", "512"])
        self.set_config("candidate", reported_prefill_step_size=0)
        code, output = self.run_campaign("quality")
        self.assertEqual(code, 1)
        records = json.loads((output / "results.json").read_text())
        self.assertEqual(sum(r["status"] == "failed" for r in records), 4)
        self.assertTrue(all("prefill_step_size" in r.get("error", "") for r in records if r["status"] == "failed"))

    def test_long_quality_samples_require_chunking_on_every_arm(self):
        self.manifest["quality"]["native_args"] = ["--max-tokens-per-sample", "8192"]
        with self.assertRaisesRegex(ValueError, "require positive"):
            campaign.validate_manifest(self.manifest, "quality")
        self.models[0]["quality_native_args"] = ["--prefill-step-size", "512"]
        self.models[1]["quality_native_args"] = ["--prefill-step-size", "2048"]
        campaign.validate_manifest(self.manifest, "quality")
        self.models[1]["quality_native_args"] = ["--prefill-step-size", "8193"]
        with self.assertRaisesRegex(ValueError, "8192"):
            campaign.validate_manifest(self.manifest, "quality")
        self.models[1]["quality_native_args"] = ["--prefill-step-size", "2048"]
        self.manifest["quality"]["native_args"] = ["--max-tokens-per-sample", "32769"]
        with self.assertRaisesRegex(ValueError, "32768"):
            campaign.validate_manifest(self.manifest, "quality")

    def test_timeout_keeps_logs_and_continues_other_arms(self):
        self.set_config("candidate", failure="timeout")
        code, output = self.run_campaign(extra=["--timeout", "0.15"])
        self.assertEqual(code, 1)
        results = json.loads((output / "results.json").read_text())
        self.assertEqual(len(results), 8)
        self.assertEqual(sum(r.get("error") == "timeout" for r in results), 4)


if __name__ == "__main__":
    unittest.main()
