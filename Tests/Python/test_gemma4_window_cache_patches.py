"""CPU-only overlap/preflight tests using tiny source and Git fixtures."""
import difflib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "Scripts/gemma4-window-cache-patches.py"
SPEC = importlib.util.spec_from_file_location("gemma4_window_cache_patches", SCRIPT)
patches = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patches)


class GemmaWindowPatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="midnight-gemma-patch-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.patch_directory = self.root / "Patches"
        self.patch_directory.mkdir()
        self.source = self.checkout / "Libraries/Model.swift"
        self.source.parent.mkdir()
        self.source.write_text("start\nbase\nend\n")
        subprocess.run(["git", "init", "-q", str(self.checkout)], check=True)
        subprocess.run(["git", "-C", str(self.checkout), "add", "."], check=True)
        subprocess.run([
            "git", "-C", str(self.checkout), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "-qm", "fixture",
        ], check=True)
        self.revision = patches.git(self.checkout, "rev-parse", "HEAD").stdout.strip()
        self.names = ("base.patch", "overlay.patch")
        self.write_patch("base.patch", "start\nbase\nend\n", "start\nfirst\nend\n")
        self.write_patch("overlay.patch", "start\nfirst\nend\n", "start\nfinal\nend\n")

    def write_patch(self, name, before, after):
        content = difflib.unified_diff(
            before.splitlines(True), after.splitlines(True),
            fromfile="a/Libraries/Model.swift", tofile="b/Libraries/Model.swift",
        )
        (self.patch_directory / name).write_text("".join(content))

    def prepare(self):
        return patches.prepare(self.root, self.checkout, self.names, self.revision)

    def test_clean_pin_applies_then_replay_is_idempotent(self):
        self.assertTrue(self.prepare())
        self.assertEqual(self.source.read_text(), "start\nfinal\nend\n")
        self.assertFalse(self.prepare())

    def test_prior_overlay_is_upgraded_and_unrelated_edits_survive(self):
        self.source.write_text("start\nfirst\nend\n\nunrelated edit\n")
        self.assertTrue(self.prepare())
        self.assertEqual(self.source.read_text(), "start\nfinal\nend\n\nunrelated edit\n")

    def test_conflict_leaves_checkout_byte_identical(self):
        self.source.write_text("start\nconflicting edit\nend\n")
        before = self.source.read_bytes()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(self.source.read_bytes(), before)

    def test_later_conflict_does_not_apply_earlier_patch(self):
        self.write_patch("overlay.patch", "start\nwrong predecessor\nend\n", "start\nfinal\nend\n")
        before = self.source.read_bytes()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(self.source.read_bytes(), before)

    def test_revision_mismatch_never_mutates_source(self):
        before = self.source.read_bytes()
        with self.assertRaises(ValueError):
            patches.prepare(self.root, self.checkout, self.names, "wrong revision")
        self.assertEqual(self.source.read_bytes(), before)

    def test_partial_final_overlay_is_rejected(self):
        self.source.write_text("start\nfinal with drift\nend\n")
        before = self.source.read_bytes()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(self.source.read_bytes(), before)

    def test_new_source_file_is_created_and_replayed_exactly(self):
        created = "import MLX\n// cache implementation\n"
        addition = "".join(difflib.unified_diff(
            [], created.splitlines(True), fromfile="/dev/null", tofile="b/Libraries/NewCache.swift"))
        patch = self.patch_directory / "overlay.patch"
        patch.write_text(patch.read_text() + addition)
        self.assertTrue(self.prepare())
        self.assertEqual((self.checkout / "Libraries/NewCache.swift").read_text(), created)
        self.assertFalse(self.prepare())


if __name__ == "__main__":
    unittest.main()
