"""CPU-only fixtures for lossless BF16 projection comparison; requires NumPy."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "Projection capture analyzer requires NumPy")
class GemmaProjectionCaptureTests(unittest.TestCase):
    def setUp(self):
        script = Path(__file__).resolve().parents[2] / "Scripts/compare-gemma-projection-captures.py"
        spec = importlib.util.spec_from_file_location("gemma_projection_compare", script)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temporary = tempfile.TemporaryDirectory(prefix="midnight-capture-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.left, self.right = [Path(self.temporary.name) / name for name in ("A", "B")]
        for directory in (self.left, self.right):
            directory.mkdir()
            (directory / "manifest.json").write_text(json.dumps({"token_ids": [2], "checkpoint": "fixture"}))

    def write(self, directory, outputs, inputs=None):
        header, payload = {}, b""
        for name, output in outputs.items():
            for suffix, values in (("input", inputs or [1, 2]), ("output", output)):
                array = np.array(values, dtype=np.float32)
                # Fixture values are exactly representable in BF16.
                data = (array.view(np.uint32) >> 16).astype("<u2").tobytes()
                header[f"model.layers.{name}.{suffix}"] = {
                    "dtype": "BF16", "shape": list(array.shape), "data_offsets": [len(payload), len(payload) + len(data)]}
                payload += data
        encoded = json.dumps(header).encode()
        (directory / "000.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)

    def test_local_difference_precedes_amplification_and_layers_sort_numerically(self):
        projections = {"10.mlp.up_proj": [2, 3], "2.self_attn.q_proj": [2, 3]}
        self.write(self.left, projections)
        self.write(self.right, {key: [2, 4] for key in projections})
        report = self.module.compare(self.left, self.right)
        self.assertEqual(report["identical_input_different_output_calls"], 2)
        self.assertEqual(report["first_divergence"]["projection"], "model.layers.2.self_attn.q_proj")
        self.assertEqual(report["first_divergence"]["output_max_change"], 1.0)
        self.assertTrue(report["first_divergence"]["identical_input"])

    def test_changed_input_is_not_reported_as_an_isolated_primitive_difference(self):
        self.write(self.left, {"0.mlp.up_proj": [2, 3]})
        self.write(self.right, {"0.mlp.up_proj": [2, 4]}, inputs=[1, 3])
        report = self.module.compare(self.left, self.right)
        self.assertEqual(report["identical_input_different_output_calls"], 0)
        self.assertIsNone(report["first_divergence_with_identical_input"])

    def test_mismatched_prefix_or_missing_capture_is_rejected(self):
        self.write(self.left, {"0.self_attn.q_proj": [2, 3]})
        with self.assertRaises(AssertionError):
            self.module.compare(self.left, self.right)
        self.write(self.right, {"0.self_attn.q_proj": [2, 3]})
        (self.right / "manifest.json").write_text(json.dumps({"token_ids": [3], "checkpoint": "fixture"}))
        with self.assertRaises(AssertionError):
            self.module.compare(self.left, self.right)


if __name__ == "__main__":
    unittest.main()
