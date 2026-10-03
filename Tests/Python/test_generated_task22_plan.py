"""CPU-only campaign orchestration checks; no model process is started."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("generated_task22", ROOT / "benchmark-results/gemma-performance-20260929/run_generated_task22.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class GeneratedTask22PlanTests(unittest.TestCase):
    def plan(self, directory):
        inputs = {"tasks": str(directory / "inputs/tasks.jsonl"), "answers": str(directory / "inputs/answers.jsonl")}
        return {"environment_overrides": dict(pilot.FLAGS), "scrub_environment_prefixes": list(pilot.SCRUB_PREFIXES),
                "inputs": inputs, "score_policy": {"draws": 10000, "seed": 20260929, "generated_code_execution": False},
                "runs": [{"label": label, "model": str(model), "report": str(directory / f"{label}.json"),
                          "command": pilot.command(model, inputs["tasks"], directory / f"{label}.json")}
                         for label, model in pilot.MODELS.items()]}

    def test_answer_key_cannot_replace_public_prompt_input(self):
        directory = Path("/tmp/frozen-pilot")
        plan = self.plan(directory)
        pilot.validate_plan(plan, directory)
        plan["runs"][0]["command"][2] = plan["inputs"]["answers"]
        with self.assertRaisesRegex(ValueError, "command changed"):
            pilot.validate_plan(plan, directory)

    def test_changed_model_or_enabled_experiment_fails_closed(self):
        directory = Path("/tmp/frozen-pilot")
        plan = self.plan(directory)
        model_changed = copy.deepcopy(plan)
        model_changed["runs"][0]["model"] = "/tmp/unplanned-model"
        with self.assertRaisesRegex(ValueError, "model or report path"):
            pilot.validate_plan(model_changed, directory)
        plan["environment_overrides"]["MLX_METAL_AFFINE_Q4_QMV_TAIL"] = "1"
        with self.assertRaisesRegex(ValueError, "environment"):
            pilot.validate_plan(plan, directory)

    def test_nested_quantization_metadata_cannot_hide_subfour_bits(self):
        self.assertEqual(list(pilot.bit_widths({"quantization": {"bits": 4, "query": {"bits": 8}}})), [4, 8])
        with self.assertRaisesRegex(ValueError, "below four bits"):
            list(pilot.bit_widths({"quantization": {"bits": 4, "query": {"bits": 3}}}))

    def test_audit_retains_refusal_truncation_missing_and_numeric_failure(self):
        scores = [
            {"id": "refusal", "category": "math", "status": "scored", "passed": False, "reason": "missing_or_ambiguous_final_number"},
            {"id": "truncated", "category": "math", "status": "scored", "passed": True, "reason": "numeric_exact_match"},
            {"id": "missing", "category": "retrieval", "status": "unscored", "passed": None, "reason": "missing_or_failed_generation"},
        ]
        raw = {"samples": [
            {"id": "refusal", "status": "completed", "generated_text": "I cannot answer this question.", "output_limit_reached": False},
            {"id": "truncated", "status": "completed", "generated_text": "#### 42", "output_limit_reached": True, "stop_reason": "length", "generation_token_count": 256},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.json"
            path.write_text(json.dumps(raw))
            audit = pilot.audit_outputs({"models": {"model": {"samples": scores}}}, {"model": path})["models"]["model"]
        self.assertEqual(len(audit["samples"]), 3)
        self.assertEqual(audit["refusal_cue_count"], 1)
        self.assertEqual(audit["numeric_format_failure_count"], 1)
        self.assertEqual(audit["output_limit_reached_count"], 1)
        self.assertTrue(audit["samples"][1]["passed"])
        self.assertIsNone(audit["samples"][2]["passed"])
        self.assertEqual(audit["samples"][2]["generation_status"], "missing")


if __name__ == "__main__":
    unittest.main()
