"""Frozen-source integrity and the unreviewed-key scoring barrier."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'Scripts' / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


builder = module('owasp_builder', 'prepare-owasp-review-proposals.py')
evaluation = module('owasp_scorer', 'evaluate-generated.py')
reviewed = module('owasp_reviewed', 'prepare-reviewed-owasp-development.py')


class OWASPPreparationTests(unittest.TestCase):
    def source(self, root):
        root.mkdir()
        row = ['BenchmarkTest00008', 'sqli', 'true', '89']
        labels = (','.join(row) + '\n').encode()
        license_data = b'Fixture license notice'
        java = b'class BenchmarkTest00008 {\nvoid run(String input) {\nString sql = input;\nstatement.executeQuery(sql);\n}\n}\n'
        (root / 'expectedresults-1.2.csv').write_bytes(labels)
        (root / 'LICENSE').write_bytes(license_data)
        (root / 'BenchmarkTest00008.java').write_bytes(java)
        identity = {'file': 'BenchmarkTest00008.java', 'bytes': len(java), 'sha256': builder.digest(java)}
        manifest = {'repository': 'fixture', 'revision': 'abc', 'source_cases': 1,
                    'expected_labels_sha256': builder.digest(labels), 'license_sha256': builder.digest(license_data),
                    'files': [identity]}
        (root / 'source-manifest.json').write_text(json.dumps(manifest))
        (root / 'selection.json').write_text(json.dumps({'repository': 'fixture', 'revision': 'abc', 'rows': [row]}))
        return root

    def test_proposals_preserve_logic_but_cannot_be_scored_or_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root / 'source')
            output = root / 'proposals'
            builder.prepare(source, output)
            task = json.loads((output / 'tasks.jsonl').read_text())
            self.assertNotIn('BenchmarkTest00008', task['prompt'])
            self.assertIn('4: statement.executeQuery(sql);', task['prompt'])
            self.assertNotIn('verdict', task['metadata'])
            with self.assertRaisesRegex(ValueError, 'category and answer kind disagree'):
                evaluation.load_inputs(output / 'tasks.jsonl', output / 'proposed-answers.jsonl')
            before = (output / 'proposed-answers.jsonl').read_bytes()
            with self.assertRaisesRegex(ValueError, 'never overwritten'):
                builder.prepare(source, output)
            self.assertEqual((output / 'proposed-answers.jsonl').read_bytes(), before)

    def test_changed_source_or_publisher_labels_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root / 'source')
            original = (source / 'BenchmarkTest00008.java').read_bytes()
            (source / 'BenchmarkTest00008.java').write_bytes(original + b'// changed\n')
            with self.assertRaisesRegex(ValueError, 'Java source changed'):
                builder.prepare(source, root / 'bad-source')
            self.assertFalse((root / 'bad-source').exists())
            (source / 'BenchmarkTest00008.java').write_bytes(original)
            (source / 'expectedresults-1.2.csv').write_text('BenchmarkTest00008,sqli,false,89\n')
            with self.assertRaisesRegex(ValueError, 'labels or license changed'):
                builder.prepare(source, root / 'bad-labels')
            self.assertFalse((root / 'bad-labels').exists())

    def test_reviewed_keys_require_complete_active_anchors_and_verified_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root / 'source')
            helpers = source / 'helpers'
            helpers.mkdir()
            (helpers / 'helper.java').write_bytes(b'fixture')
            record = {'file': 'helper.java', 'bytes': 7, 'sha256': builder.digest(b'fixture')}
            for name in ['manifest.json', 'resources-manifest.json']:
                (helpers / name).write_text(json.dumps({'revision': 'abc', 'files': [record]}))
            review = root / 'review'
            review.mkdir()
            annotation = {'anchors': [4], 'evidence': [3, 4], 'reason': 'Input reaches query execution'}
            document = {'review_status': 'reviewed', 'cases': {'BenchmarkTest00008': annotation}}
            (review / 'path-and-command-reviewed.json').write_text(json.dumps({'review_status': 'reviewed', 'cases': {}}))
            (review / 'sql-reviewed.json').write_text(json.dumps(document))
            (review / 'randomness-excluded.json').write_text(json.dumps({
                'status': 'excluded_before_model_generation', 'generation_started': False, 'excluded_cases': []}))
            output = root / 'approved'
            reviewed.prepare(source, review, output)
            tasks, keys = evaluation.load_inputs(output / 'tasks.jsonl', output / 'answers.jsonl')
            self.assertEqual(len(tasks), 1)
            key = next(iter(keys.values()))
            self.assertEqual(key['evidence_anchor_lines'], [4])
            self.assertEqual((output / 'answers.jsonl').stat().st_mode & 0o777, 0o600)
            with self.assertRaisesRegex(ValueError, 'Frozen output exists'):
                reviewed.prepare(source, review, output)
            annotation['anchors'] = []
            (review / 'sql-reviewed.json').write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'disagrees with publisher verdict'):
                reviewed.prepare(source, review, root / 'bad-anchor')
            self.assertFalse((root / 'bad-anchor').exists())
            annotation['anchors'] = [4]
            (review / 'sql-reviewed.json').write_text(json.dumps(document))
            (helpers / 'helper.java').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Helper changed'):
                reviewed.prepare(source, review, root / 'bad-helper')
            self.assertFalse((root / 'bad-helper').exists())


if __name__ == '__main__':
    unittest.main()
