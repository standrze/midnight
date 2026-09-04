"""Offline CPU tests for source pinning, byte stability and non-destructive reuse."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[2] / "Scripts/prepare-quantization-corpus.py"
spec = importlib.util.spec_from_file_location("prepare_quantization_corpus", SCRIPT)
corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus)


class CorpusPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="midnight-corpus-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.output = self.root / "output"
        self.data = {
            "mbpp": "".join(json.dumps({"task_id": i, "text": f"Return café {i}", "code": f"def result():\n    return {i}"}) + "\n" for i in range(10, 76)).encode(),
            "gsm8k": "".join(json.dumps({"question": f"How many? {i}", "answer": f"Work\n#### {i}"}) + "\n" for i in range(66)).encode(),
        }
        self.sources = copy.deepcopy(corpus.SOURCES)
        payloads = []
        for source in self.sources:
            data = self.data[source["name"]]
            payload = corpus.reference_payload(source["name"], data)
            source["source_sha256"] = corpus.sha256(data)
            source["corpus_sha256"] = corpus.sha256(payload)
            (self.cache / f"{source['name']}-source.jsonl").write_bytes(data)
            payloads.append(payload)
        self.combined_hash = corpus.sha256(b"".join(payloads))

    def prepare(self, **kwargs):
        return corpus.prepare(self.output, sources=self.sources, combined_sha256=self.combined_hash, **kwargs)

    def test_exact_selection_unicode_and_concatenation(self):
        with mock.patch.object(corpus.urllib.request, "urlopen", side_effect=AssertionError("Network forbidden")):
            result = self.prepare(cache=self.cache, offline=True)
        rows = [json.loads(line) for line in (self.output / "code-math-128.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 128)
        self.assertEqual([r["id"] for r in rows[:64]], [f"mbpp-test-{i}" for i in range(11, 75)])
        self.assertEqual(rows[0]["text"], "Task: Return café 11\nSolution:\ndef result():\n    return 11")
        self.assertEqual(rows[64]["text"], "Question: How many? 0\nAnswer: Work\n#### 0")
        self.assertEqual(rows[-1]["id"], "gsm8k-test-63")
        self.assertEqual(result["sha256"], self.combined_hash)
        self.assertEqual((self.output / "code-math-128.jsonl").read_bytes(),
                         (self.output / "mbpp-64.jsonl").read_bytes() + (self.output / "gsm8k-64.jsonl").read_bytes())

    def test_matching_outputs_are_reused_without_rewriting(self):
        self.prepare(cache=self.cache, offline=True)
        before = {p.name: p.stat().st_mtime_ns for p in self.output.iterdir()}
        self.prepare(offline=True)
        self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in self.output.iterdir()})

    def test_corrupt_cache_fails_without_output_or_network(self):
        (self.cache / "gsm8k-source.jsonl").write_bytes(b"tampered")
        with mock.patch.object(corpus.urllib.request, "urlopen", side_effect=AssertionError("Network forbidden")):
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self.prepare(cache=self.cache, offline=False)
        self.assertFalse(self.output.exists())

    def test_differing_output_preflight_preserves_everything(self):
        self.output.mkdir()
        conflict = self.output / "code-math-128.jsonl"
        conflict.write_bytes(b"existing different work")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            self.prepare(cache=self.cache, offline=True)
        self.assertEqual(list(self.output.iterdir()), [conflict])
        self.assertEqual(conflict.read_bytes(), b"existing different work")

    def test_missing_offline_cache_fails_without_network(self):
        with mock.patch.object(corpus.urllib.request, "urlopen", side_effect=AssertionError("Network forbidden")):
            with self.assertRaisesRegex(ValueError, "Offline source missing"):
                self.prepare(offline=True)

    def test_download_uses_exact_revision_and_verifies_bytes(self):
        urls = []
        def response(request, timeout):
            urls.append(request.full_url)
            name = "mbpp" if "mbpp/" in request.full_url else "gsm8k"
            return io.BytesIO(self.data[name])
        with mock.patch.object(corpus.urllib.request, "urlopen", side_effect=response):
            self.prepare()
        self.assertEqual(urls, [corpus.source_url(source) for source in self.sources])
        self.assertTrue(all("/commits" not in url and "/main/" not in url for url in urls))
        entries = json.loads((self.output / "provenance.json").read_text())
        self.assertEqual(entries[0]["revision"], corpus.SOURCES[0]["revision"])

    def test_corrupt_download_is_rejected_before_writing(self):
        with mock.patch.object(corpus.urllib.request, "urlopen", return_value=io.BytesIO(b"bad")):
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self.prepare()
        self.assertFalse(self.output.exists())

    def test_transformation_and_combined_payload_hashes_are_checked(self):
        self.sources[0]["corpus_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "mbpp corpus"):
            self.prepare(cache=self.cache, offline=True)
        self.sources[0]["corpus_sha256"] = corpus.sha256(corpus.reference_payload("mbpp", self.data["mbpp"]))
        self.combined_hash = "0" * 64
        with self.assertRaisesRegex(ValueError, "code-math-128"):
            self.prepare(cache=self.cache, offline=True)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
