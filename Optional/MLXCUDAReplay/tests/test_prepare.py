import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[1]
BASELINE = REPOSITORY / "benchmark-results/mlx-replay-integration-20260913/raw"
SPEC = importlib.util.spec_from_file_location("prepare_replay", ROOT / "prepare.py")
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="mlx-replay-overlay-test-")
        self.root = Path(self.directory.name)
        self.source = self.root / "source"
        backend = self.source / PREPARE.CUDA
        backend.mkdir(parents=True)
        for name in PREPARE.BASELINES:
            shutil.copyfile(BASELINE / (name + ".before"), backend / name)

    def tearDown(self):
        self.directory.cleanup()

    def test_round_trip_patch_and_source_preservation(self):
        patch = PREPARE.prepare(self.source, self.root / "overlay")
        for name, digest in PREPARE.BASELINES.items():
            self.assertEqual(PREPARE.sha256((self.source / PREPARE.CUDA / name).read_bytes()), digest)
        candidate = self.root / "candidate"
        shutil.copytree(self.source, candidate)
        subprocess.run(["patch", "-p1", "--batch", "--forward", "-i", str(patch)],
                       cwd=candidate, check=True, capture_output=True)
        for name in (*PREPARE.BASELINES, *PREPARE.HEADERS):
            self.assertEqual((candidate / PREPARE.CUDA / name).read_bytes(),
                             (self.root / "overlay" / PREPARE.CUDA / name).read_bytes())
        subprocess.run(["patch", "-p1", "--batch", "-R", "-i", str(patch)],
                       cwd=candidate, check=True, capture_output=True)
        for name, digest in PREPARE.BASELINES.items():
            self.assertEqual(PREPARE.sha256((candidate / PREPARE.CUDA / name).read_bytes()), digest)
        for name in PREPARE.HEADERS:
            self.assertFalse((candidate / PREPARE.CUDA / name).exists())

    def test_unreviewed_source_rejected_before_output(self):
        target = self.source / PREPARE.CUDA / "device.cpp"
        target.write_text(target.read_text() + "// changed\n")
        with self.assertRaises(ValueError):
            PREPARE.prepare(self.source, self.root / "overlay")
        self.assertFalse((self.root / "overlay").exists())

    def test_existing_output_and_source_subdirectory_rejected(self):
        (self.root / "overlay").mkdir()
        with self.assertRaises(FileExistsError):
            PREPARE.prepare(self.source, self.root / "overlay")
        with self.assertRaises(ValueError):
            PREPARE.prepare(self.source, self.source / "overlay")

    def test_existing_overlay_header_rejected(self):
        (self.source / PREPARE.CUDA / PREPARE.HEADERS[0]).write_text("user file\n")
        with self.assertRaises(ValueError):
            PREPARE.prepare(self.source, self.root / "overlay")


if __name__ == "__main__":
    unittest.main()
