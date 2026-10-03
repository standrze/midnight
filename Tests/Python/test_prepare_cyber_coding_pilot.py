"""Public/private dataset separation and immutable preparation output."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('cyber_coding_builder', ROOT / 'Scripts/prepare-cyber-coding-pilot.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class CyberCodingPreparationTests(unittest.TestCase):
    def blueprint(self, directory, **updates):
        case = {'family': 'fixture', 'prompt': 'Implement f().',
                'reference': 'def f(): return 7', 'fault': 'def f(): return 8',
                'extra_faults': ['def f(): return 9'], 'tests': ['assert f() == 7']}
        case.update(updates)
        path = directory / 'private.json'
        path.write_text(json.dumps([case]))
        return path

    def test_external_cases_keep_private_solutions_separate_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = self.blueprint(root)
            output = root / 'prepared'
            result = builder.prepare(output, private)
            tasks = (output / 'tasks.jsonl').read_text()
            self.assertNotIn('return 7', tasks)
            self.assertNotIn('assert f()', tasks)
            self.assertEqual(result['faulty_control_count'], 2)
            self.assertEqual(result['counts'], {'code': 1, 'cybersecurity': 0})
            for name in ['answers.jsonl', 'references.jsonl']:
                self.assertEqual((output / name).stat().st_mode & 0o777, 0o600)
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaisesRegex(ValueError, 'never replaced'):
                builder.prepare(output, private)
            self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, before)

    def test_invalid_blueprint_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = self.blueprint(root, extra_faults='not a list')
            output = root / 'prepared'
            with self.assertRaisesRegex(ValueError, 'faulty control list'):
                builder.prepare(output, private)
            self.assertFalse(output.exists())

    def test_legacy_task_payloads_remain_exact(self):
        frozen = ROOT / 'Tests/Fixtures/CyberCodingPilot'
        for name, rows in zip(['tasks.jsonl', 'answers.jsonl', 'references.jsonl'], builder.records()):
            expected = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
            self.assertEqual(expected, (frozen / name).read_bytes(), name)
