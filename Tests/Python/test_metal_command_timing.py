"""CPU-only patch replay, semantic preservation, and timing analysis tests."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
MLX = ROOT / '.build/checkouts/mlx-swift/Source/Cmlx/mlx'


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'Scripts' / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


patcher = load('timing_patch', 'metal-command-timing-patch.py')
analysis = load('timing_analysis', 'analyze-metal-command-timing.py')


def fixture():
    rows = []
    for index, start in enumerate([10.0, 10.6, 21.0], 1):
        rows.append({'sequence': index, 'queue_key': 99, 'operations': 10, 'referenced_bytes': 4096,
                     'commit_start': start, 'commit_return': start + .02, 'gpu_start': start + .1,
                     'gpu_end': start + .4, 'callback_observed': start + .41, 'gpu_timing_valid': True, 'status': 4})
    return {'format': 1, 'kind': 'midnight_metal_command_timing', 'clock': 'system_mach_seconds',
            'snapshot': 'atexit_without_gpu_wait', 'capacity': 8, 'skip_command_buffers': 0,
            'submitted_command_buffers': 3, 'incomplete_sample_count': 0, 'unsampled_due_to_capacity': 0,
            'waits_seen': 1, 'queues_created': 2, 'queue_metadata_overflow': 0,
            'queues': [{'generation': 1, 'queue_key': 99, 'stream_index': 7, 'created': 9.0},
                       {'generation': 2, 'queue_key': 99, 'stream_index': 8, 'created': 20.0}],
            'commands': rows, 'waits': [{'queue_key': 99, 'start': 10.0, 'end': 10.42}]}


class TimingAnalysisTests(unittest.TestCase):
    def test_numeric_metrics_and_queue_address_reuse(self):
        result = analysis.analyze(fixture())
        self.assertAlmostEqual(result['metrics']['gpu_execution_interval']['sum_seconds'], .9)
        self.assertAlmostEqual(result['metrics']['cpu_commit_call']['sum_seconds'], .06)
        self.assertAlmostEqual(result['metrics']['existing_synchronize_wait_all_retained']['sum_seconds'], .42)
        self.assertEqual([q['sample_count'] for q in result['queues']], [2, 1])
        self.assertAlmostEqual(result['queues'][0]['queue_gpu_gap']['sum_seconds'], .3)
        self.assertAlmostEqual(result['queues'][0]['submit_after_previous_gpu_end']['sum_seconds'], .2)
        self.assertEqual(result['queues'][1]['queue_gpu_gap']['count'], 0)

    def test_missing_completion_disables_adjacent_gap_claims(self):
        report = fixture()
        report['commands'].pop(1)
        report['incomplete_sample_count'] = 1
        result = analysis.analyze(report)
        self.assertFalse(result['adjacent_queue_gap_metrics_available'])
        self.assertNotIn('queue_gpu_gap', result['queues'][0])

    def test_invalid_gpu_time_is_not_fabricated_and_sequence_filter_is_explicit(self):
        report = fixture()
        report['commands'][0].update(gpu_start=0, gpu_end=0, gpu_timing_valid=False)
        result = analysis.analyze(report, sequence_from=1, sequence_to=2)
        self.assertEqual(result['selected_commands'], 2)
        self.assertEqual(result['invalid_gpu_samples'], 1)
        self.assertEqual(result['metrics']['gpu_execution_interval']['count'], 1)

    def test_corrupt_reports_fail_closed(self):
        for kind in ['duplicate', 'nan', 'time', 'counts', 'status', 'clock']:
            report = fixture()
            if kind == 'duplicate': report['commands'][1]['sequence'] = 1
            elif kind == 'nan': report['commands'][0]['gpu_start'] = float('nan')
            elif kind == 'time': report['commands'][0]['commit_return'] = 1
            elif kind == 'counts': report['incomplete_sample_count'] = 2
            elif kind == 'status': report['commands'][0]['status'] = 2
            else: report['clock'] = 'wall'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                analysis.analyze(report)


class TelemetryPatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='midnight-timing-test-')
        self.addCleanup(self.temp.cleanup)
        self.checkout = Path(self.temp.name)
        self.source = self.checkout / patcher.TARGETS[0]
        self.source.parent.mkdir(parents=True)
        pinned = patcher.require(patcher.git(MLX, 'show', f'{patcher.PIN}:{patcher.TARGETS[0]}'), 'Read pinned device source')
        self.source.write_text(pinned)
        subprocess.run(['git', 'init', '-q', str(self.checkout)], check=True)
        subprocess.run(['git', '-C', str(self.checkout), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(self.checkout), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                        '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null', 'commit', '-qm', 'fixture'], check=True)
        self.revision = patcher.git(self.checkout, 'rev-parse', 'HEAD').stdout.strip()
        self.patch = ROOT / 'Patches/mlx-metal-command-timing.patch'

    def apply(self):
        return patcher.prepare(self.checkout, self.patch, self.revision)

    def test_exact_replay_is_idempotent_and_matches_stage(self):
        self.assertTrue(self.apply())
        applied = {target: (self.checkout / target).read_bytes() for target in patcher.TARGETS}
        self.assertFalse(self.apply())
        self.assertEqual(applied, {target: (self.checkout / target).read_bytes() for target in patcher.TARGETS})
        self.assertEqual(patcher.git(self.checkout, 'apply', '--reverse', '--check', str(self.patch)).returncode, 0)

    def test_conflict_cannot_partially_create_new_header(self):
        self.source.write_text(self.source.read_text().replace('  buffer_->commit();', '  conflicting_commit();'))
        before = self.source.read_bytes()
        with self.assertRaises(ValueError): self.apply()
        self.assertEqual(self.source.read_bytes(), before)
        self.assertFalse((self.checkout / patcher.TARGETS[1]).exists())

    def test_wrong_pin_and_drift_preserve_evidence(self):
        before = self.source.read_bytes()
        with self.assertRaises(ValueError): patcher.prepare(self.checkout, self.patch, 'wrong')
        self.assertEqual(self.source.read_bytes(), before)
        self.apply()
        header = self.checkout / patcher.TARGETS[1]
        header.write_text(header.read_text().replace('8192, 65536', '8192, 32768'))
        before = {p: (self.checkout / p).read_bytes() for p in patcher.TARGETS}
        with self.assertRaises(ValueError): self.apply()
        self.assertEqual(before, {p: (self.checkout / p).read_bytes() for p in patcher.TARGETS})

    def test_native_completion_and_error_handler_remains_byte_identical(self):
        before = self.source.read_text()
        self.apply()
        after = self.source.read_text()
        start = '  buffer_->addCompletedHandler(\n      [&error_ = error_,'
        end = '      });'
        original = before.split(start, 1)[1].split(end, 1)[0]
        staged = after.split(start, 1)[1].split(end, 1)[0]
        self.assertEqual(original, staged)
        self.assertEqual(before.count('synchronize();'), after.count('synchronize();'))
        self.assertEqual(before.count('error_.check();'), after.count('error_.check();'))
        self.assertEqual(before.count('signal(value);'), after.count('signal(value);'))
        self.assertIn('if (command_timing::enabled())', after)
        callback = after.split('buffer->addCompletedHandler([sample]', 1)[1].split('  });', 1)[0]
        self.assertNotIn('waitUntil', callback)
        self.assertNotIn('fprintf', callback)
        self.assertNotIn('mutex', callback)
        self.assertNotIn('new ', callback)


if __name__ == '__main__':
    unittest.main()
