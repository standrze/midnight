"""CPU-only validation of paired, fixed-prefix greedy-choice diagnostics."""
import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("token_diagnostics", ROOT / "Scripts/analyze-quality-token-diagnostics.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def fixture():
    rows = [{"position": i + 1, "reference_token_id": 2, "winner_token_id": 2,
             "runner_up_token_id": 3, "winner_logit": 2.0, "runner_up_logit": 1.9999,
             "winner_margin": .0001, "reference_nll": 1.0} for i in range(3)]
    sample = {"id": "sample", "category": "prose", "token_id_fingerprint": "tokens",
              "original_token_count": 4, "evaluated_token_count": 4, "scored_token_count": 3,
              "truncated": False, "nll_sum": 3.0, "nll": 1.0}
    return {"format": 1, "status": "measured", "metric": "teacher_forced_next_token_nll",
            "corpus_fingerprint": "corpus", "token_id_fingerprint": "combined", "backend": "metal",
            "device": "gpu", "model_type": "gemma3_text", "model_path": "/fixture/model",
            "model_implementation": "MLXLLM.Gemma3TextModel", "tokenization": "raw_text_add_special_tokens_no_chat_template",
            "add_special_tokens": True, "maximum_tokens_per_sample": 4, "prefill_step_size": 1,
            "samples": [sample], "sample_count": 1, "scored_token_count": 3,
            "nll_sum": 3.0, "token_weighted_nll": 1.0, "token_diagnostics": [{"id": "sample", "rows": rows}]}


class TokenDiagnosticsTests(unittest.TestCase):
    def test_identical_reports_and_near_tie_flip(self):
        baseline = fixture()
        self.assertEqual(analysis.analyze(baseline, baseline)["winner_agreement_fraction"], 1)
        candidate = copy.deepcopy(baseline)
        row = candidate["token_diagnostics"][0]["rows"][1]
        row["winner_token_id"], row["runner_up_token_id"] = 3, 2
        result = analysis.analyze(baseline, candidate)
        self.assertEqual(result["winner_flip_count"], 1)
        self.assertEqual(result["candidate_reference_top1_correct"], 2)
        self.assertEqual(result["flip_baseline_margins_at_most"]["0.0001"], 1)
        self.assertEqual(result["flips"][0]["position"], 2)

    def test_incomplete_or_corrupt_diagnostics_fail(self):
        for corruption in ("missing", "position", "reference", "margin", "nll", "nan", "winner", "step"):
            candidate = fixture()
            row = candidate["token_diagnostics"][0]["rows"][0]
            if corruption == "missing":
                candidate["token_diagnostics"][0]["rows"].pop()
            elif corruption == "position":
                row["position"] = 9
            elif corruption == "reference":
                row["reference_token_id"] = 1
            elif corruption == "margin":
                row["winner_margin"] = 1.0
            elif corruption == "nll":
                row["reference_nll"] = 2.0
            elif corruption == "nan":
                row["winner_logit"] = float("nan")
            elif corruption == "winner":
                row["winner_token_id"] = row["runner_up_token_id"]
            else:
                candidate["prefill_step_size"] = 512
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                analysis.analyze(fixture(), candidate)

    def test_changed_native_implementation_and_tokenization_fail(self):
        for field in ("model_implementation", "tokenization", "model_path", "token_id_fingerprint"):
            candidate = fixture()
            candidate[field] = "changed"
            with self.subTest(field=field), self.assertRaises(ValueError):
                analysis.analyze(fixture(), candidate)


if __name__ == "__main__":
    unittest.main()
