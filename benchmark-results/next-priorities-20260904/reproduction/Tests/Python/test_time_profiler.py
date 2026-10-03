import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('profiler', Path(__file__).resolve().parents[2] / 'Scripts/analyze-time-profiler.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TimeProfilerTests(unittest.TestCase):
    def analyze(self, rows, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.xml'
            path.write_text('<root>' + rows + '</root>')
            return module.analyze(path, **kwargs)

    def test_weights_refs_recursion_and_missing(self):
        report = self.analyze('''
        <row><sample-time>1000000000</sample-time><weight id="w">2000000</weight>
        <tagged-backtrace id="t"><backtrace><frame id="f" name="leaf"/><frame ref="f"/><frame name="caller"/></backtrace></tagged-backtrace></row>
        <row><sample-time>2000000000</sample-time><weight ref="w"/><tagged-backtrace ref="t"/></row>
        <row><sample-time>3000000000</sample-time><weight>1000000</weight></row>''')
        self.assertEqual(report['symbolized_cpu_ms'], 4)
        self.assertEqual(report['missing_stack_cpu_ms'], 1)
        self.assertEqual(report['leaf'][0]['percent_of_symbolized_weight'], 100)
        self.assertEqual(report['inclusive'][0]['sampled_cpu_ms'], 4)

    def test_interval(self):
        rows = '<row><sample-time>1000000000</sample-time><weight>1000000</weight><tagged-backtrace><backtrace><frame name="a"/></backtrace></tagged-backtrace></row>'
        with self.assertRaises(ValueError):
            self.analyze(rows, start=2)
        with self.assertRaises(ValueError):
            self.analyze(rows, start=float('nan'))

    def test_bad_reference(self):
        with self.assertRaises(ValueError):
            self.analyze('<row><sample-time ref="missing"/><weight>1</weight></row>')
