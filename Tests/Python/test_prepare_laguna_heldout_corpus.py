"""CPU-only corpus byte-identity, pinning and safe reuse checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'benchmark-results/quantization-20260904/prepare-laguna-heldout-corpus.py'
spec = importlib.util.spec_from_file_location('prepare_laguna_heldout', SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class LagunaHeldoutCorpusTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sources = [self.root / 'wiki.jsonl', self.root / 'code-math.jsonl']
        self.records = [{'id': str(i), 'category': 'fixture', 'text': f'café {i}\nline'} for i in range(192)]
        for path, records in zip(self.sources, (self.records[:64], self.records[64:])):
            path.write_text(''.join(json.dumps(r) + '\n' for r in records))
        self.payload = ''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in self.records).encode()
        self.provenance = {'source_files': [{'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in self.sources],
                           'record_count': 192, 'output_sha256': hashlib.sha256(self.payload).hexdigest()}
        self.output = self.root / 'out.jsonl'

    def prepare(self):
        return builder.prepare(*self.sources, self.output, provenance=self.provenance)

    def test_exact_utf8_order_and_matching_reuse(self):
        self.assertEqual(self.prepare(), self.provenance['output_sha256'])
        self.assertEqual(self.output.read_bytes(), self.payload)
        modified = self.output.stat().st_mtime_ns
        self.prepare()
        self.assertEqual(self.output.stat().st_mtime_ns, modified)

    def test_wrong_source_or_transformation_hash_creates_no_output(self):
        self.sources[0].write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Source SHA256'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_differing_output_is_never_overwritten(self):
        self.output.write_text('preserve')
        with self.assertRaisesRegex(ValueError, 'differing'):
            self.prepare()
        self.assertEqual(self.output.read_text(), 'preserve')

    def test_output_hash_mismatch_fails_before_write(self):
        self.provenance['output_sha256'] = 'incorrect'
        with self.assertRaisesRegex(ValueError, 'payload SHA256'):
            self.prepare()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
