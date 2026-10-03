"""CPU-only checks for kernel-arm isolation, provenance validation and cleanup."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('kernel_quality_runner', ROOT / 'Scripts/run-gemma-kernel-quality.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
FIXTURE_SPEC = importlib.util.spec_from_file_location('choice_fixture', Path(__file__).with_name('test_quality_token_diagnostics.py'))
fixture_module = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(fixture_module)


class KernelQualityRunnerTests(unittest.TestCase):
    def test_inherited_experiments_cannot_leak_into_baseline(self):
        with patch.dict(os.environ, {'MIDNIGHT_METAL_UNKNOWN': '1', 'MLX_METAL_AFFINE_Q4_QMV_TAIL': '1',
                                     'MIDNIGHT_GEMMA3_COMPILED_TAIL': '1', 'MIDNIGHT_GEMMA3_UNKNOWN': '1',
                                     'MLX_DISABLE_COMPILE': '0'}):
            a, b = runner.environment('tail', 'A'), runner.environment('tail', 'B')
        self.assertNotIn('MIDNIGHT_METAL_UNKNOWN', a)
        self.assertNotIn('MIDNIGHT_GEMMA3_UNKNOWN', a)
        self.assertNotIn('MIDNIGHT_GEMMA3_COMPILED_TAIL', a)
        self.assertNotIn('MIDNIGHT_GEMMA3_COMPILED_TAIL', b)
        self.assertNotIn('MLX_DISABLE_COMPILE', a)
        self.assertNotIn('MLX_DISABLE_COMPILE', b)
        self.assertEqual({k for k in a if a[k] != b[k]}, {'MLX_METAL_AFFINE_Q4_QMV_TAIL'})
        self.assertEqual(a['MIDNIGHT_METAL_SDPA_D512'], '0')
        self.assertEqual(a['MODEL_RUNNER_PREFIX_CACHE_ENTRIES'], '0')

    def test_nested_quantization_below_four_bits_fails(self):
        self.assertEqual(runner.quantization_bits({'bits': 4, 'layer': {'bits': 8}}), [4, 8])
        for value in [3, True, 2.0, '4']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.quantization_bits({'bits': 4, 'layer': {'bits': value}})

    def test_bounded_cache_arm_changes_only_cache_selection(self):
        a, b = runner.environment('bounded', 'A'), runner.environment('bounded', 'B')
        self.assertEqual({k for k in a if a[k] != b[k]}, {'MIDNIGHT_GEMMA4_BOUNDED_KV'})

    def test_claimed_bounded_cache_requires_actual_cache_implementation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = fixture_module.fixture()
            env = {k: v for k, v in runner.environment('bounded', 'B').items() if k.startswith(runner.ENV_PREFIXES)}
            record['runtime_environment'] = env
            record['cache_implementations'] = ['MLXLMCommon.KVCacheSimple']
            report = root / 'B.json'
            report.write_text(json.dumps(record))
            plan = {'candidate': 'bounded', 'expected_model_implementation': 'MLXLLM.Gemma3TextModel',
                    'model': {'path': '/fixture/model'},
                    'runs': [{'label': 'B', 'arm': 'B', 'environment': env}]}
            with self.assertRaisesRegex(ValueError, 'Bounded cache selection'):
                runner.analyze_nll({'B': report}, plan, root)

    def test_gemma3_compiled_arm_is_isolated_and_model_restricted(self):
        with patch.dict(os.environ, {'MIDNIGHT_GEMMA3_COMPILED_TAIL': '1', 'MIDNIGHT_GEMMA3_UNKNOWN': '1'}):
            a, b = runner.environment('gemma3-compiled', 'A'), runner.environment('gemma3-compiled', 'B')
        self.assertEqual({k for k in a if a[k] != b[k]}, {'MIDNIGHT_GEMMA3_COMPILED_TAIL'})
        self.assertEqual(a['MIDNIGHT_GEMMA3_COMPILED_TAIL'], '0')
        self.assertEqual(b['MLX_METAL_AFFINE_Q4_QMV_TAIL'], '0')
        self.assertNotIn('MIDNIGHT_GEMMA3_UNKNOWN', b)
        runner.validate_candidate('gemma3-270m', 'gemma3-compiled', 'nll')
        for model, candidate, mode in [('gemma4-26b', 'gemma3-compiled', 'nll'),
                                       ('gemma4-31b', 'gemma3-compiled', 'nll'),
                                       ('gemma3-270m', 'gemma3-compiled', 'tasks'),
                                       ('gemma3-270m', 'attention', 'nll'),
                                       ('gemma3-270m', 'bounded', 'nll')]:
            with self.subTest(model=model, candidate=candidate, mode=mode), self.assertRaises(ValueError):
                runner.validate_candidate(model, candidate, mode)

    def test_gemma3_compiled_activation_requires_valid_nonzero_candidate_and_zero_baseline(self):
        valid = {'gemma3_compiled_tail_trace_counts': [1] * 18, 'gemma3_compiled_tail_trace_count': 18}
        baseline = {'gemma3_compiled_tail_trace_counts': [0] * 18, 'gemma3_compiled_tail_trace_count': 0}
        runner.validate_compiled_tail_activation(valid, 'B')
        runner.validate_compiled_tail_activation(baseline, 'A')
        bad = [({}, 'B'), (baseline, 'B'), (valid, 'A'), (valid, 'unknown')]
        for counts, total in [([], 0), ([1, 0], 1), ([True], 1), ([1.0], 1), ([-1], -1),
                              ([1], True), ([1], 2), ([1], None)]:
            bad.append(({'gemma3_compiled_tail_trace_counts': counts,
                         'gemma3_compiled_tail_trace_count': total}, 'B'))
        for record, arm in bad:
            with self.subTest(record=record, arm=arm), self.assertRaises(ValueError):
                runner.validate_compiled_tail_activation(record, arm)

    def test_gemma3_compiled_reports_gate_analysis_before_writing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports, records, runs = {}, {}, []
            for label, arm in [('01-A', 'A'), ('02-B', 'B')]:
                record = fixture_module.fixture()
                env = {k: v for k, v in runner.environment('gemma3-compiled', arm).items()
                       if k.startswith(runner.ENV_PREFIXES)}
                record['runtime_environment'] = env
                record['gemma3_compiled_tail_trace_counts'] = [0 if arm == 'A' else 1] * 18
                record['gemma3_compiled_tail_trace_count'] = sum(record['gemma3_compiled_tail_trace_counts'])
                records[label] = record
                reports[label] = root / f'{label}.json'
                reports[label].write_text(json.dumps(record))
                runs.append({'label': label, 'arm': arm, 'environment': env})
            plan = {'candidate': 'gemma3-compiled', 'expected_model_implementation': 'MLXLLM.Gemma3TextModel',
                    'model': {'path': '/fixture/model'}, 'runs': runs}
            (root / 'plan.json').write_text(json.dumps(plan))
            stale = copy.deepcopy(records['02-B'])
            stale.pop('gemma3_compiled_tail_trace_counts')
            reports['02-B'].write_text(json.dumps(stale))
            with self.assertRaisesRegex(ValueError, 'trace counts'):
                runner.analyze_nll(reports, plan, root)
            self.assertFalse((root / 'nll-analysis.json').exists())
            reports['02-B'].write_text(json.dumps(records['02-B']))
            runner.analyze_nll(reports, plan, root)
            self.assertEqual(json.loads((root / 'fixed-prefix-analysis.json').read_text())['winner_flip_count'], 0)

    def test_failed_process_stops_and_preserves_its_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'failed.log'
            with self.assertRaisesRegex(ValueError, 'Process exited 7'):
                runner.run_process([sys.executable, '-c', 'print("failure context");raise SystemExit(7)'], dict(os.environ), log, 5)
            self.assertIn('failure context', log.read_text())

    def test_timed_out_process_is_reaped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pidfile = root / 'pid'
            source = 'import os,time,pathlib;pathlib.Path(__import__("sys").argv[1]).write_text(str(os.getpid()));time.sleep(20)'
            with self.assertRaises(subprocess.TimeoutExpired):
                runner.run_process([sys.executable, '-c', source, str(pidfile)], dict(os.environ), root / 'timeout.log', .3)
            pid = int(pidfile.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_native_flags_gate_analysis_and_outputs_reconcile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, runs = {}, []
            for label, arm in [('01-A', 'A'), ('02-B', 'B')]:
                record = fixture_module.fixture()
                env = {k: v for k, v in runner.environment('tail', arm).items() if k.startswith(runner.ENV_PREFIXES)}
                record['runtime_environment'] = env
                records[label] = root / f'{label}.json'
                records[label].write_text(json.dumps(record))
                runs.append({'label': label, 'environment': env})
            plan = {'expected_model_implementation': 'MLXLLM.Gemma3TextModel', 'model': {'path': '/fixture/model'}, 'runs': runs}
            (root / 'plan.json').write_text(json.dumps(plan))
            runner.analyze_nll(records, plan, root)
            result = json.loads((root / 'fixed-prefix-analysis.json').read_text())
            self.assertEqual(result['winner_flip_count'], 0)
            record['runtime_environment'] = {}
            records['02-B'].write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'Native kernel flags'):
                runner.analyze_nll(records, plan, root)


if __name__ == '__main__':
    unittest.main()
