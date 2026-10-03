"""CPU-only guards for two-resident runtime plans and statistical adapters."""
import argparse
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('gemma_paired_runner', ROOT / 'Scripts/run-gemma-paired-runtime.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def plan_fixture(root):
    binary = root / 'binary'
    binary.mkdir()
    (binary / 'model-runner-paired-runtime-bench').write_text('fixture executable, never launched')
    (binary / 'model-runner-paired-runtime-bench').chmod(0o755)
    (binary / 'mlx.metallib').write_bytes(b'fixture metal bytes')
    models = []
    for name in ('baseline', 'candidate'):
        directory = root / name
        directory.mkdir()
        (directory / 'config.json').write_text('{"model_type":"gemma3_text"}')
        (directory / 'model.safetensors').write_bytes(b'a' * 4096)
        (directory / 'tokenizer.json').write_text('{}')
        models.append(directory)
    args = argparse.Namespace(name='fixture', pairs=8, maximum_drift_fraction=.1, maximum_null_bias_fraction=.03,
                              context_length=8192, binary_dir=binary, baseline=models[0], candidate=models[1],
                              output_root=root / 'results', prompt_file=None)
    path = runner.make_plan(args)
    return path, runner.read_json(path)


def native_fixture(plan, case):
    tokens = [11, 22, 33]
    models = []
    for arm, path in zip(('A', 'B'), case['models']):
        models.append({'arm': arm, 'path': path,
                       'configuration_sha256': plan['input_files'][str(Path(path) / 'config.json')]['sha256'],
                       'stored_weight_bytes': 4096, 'shards': [{'name': 'model.safetensors', 'bytes': 4096}]})
    report = {**runner.SETTINGS, 'status': 'completed_raw_trials', 'pairs': 8, 'context_length': 8192,
              'runtime_environment': plan['runtime_environment'], 'prompt': plan['prompt'],
              'rendered_prompt_tokens': tokens, 'models': models,
              'loads': [{'arm': arm, 'order': i + 1, 'implementation': 'MLXLLM.Gemma3TextModel'}
                        for i, arm in enumerate(('A', 'B'))], 'trials': []}
    for phase, count in [('warmup', 3), ('measured', 8)]:
        for pair in range(1, count + 1):
            order = 'AB' if pair % 2 else 'BA'
            for position, arm in enumerate(order, 1):
                text = 'same text' if case['kind'] == 'control' else f'stable quantized output {arm}'
                report['trials'].append({
                    'sequence': len(report['trials']) + 1, 'phase': phase, 'pair': pair, 'order': order,
                    'position_in_pair': position, 'arm': arm, 'validation_issues': [], 'tool_call_count': 0,
                    'content': text, 'reasoning': '', 'prompt_token_count': 3,
                    'prompt_token_id_fingerprint': runner.fingerprint(tokens),
                    'metrics': {'prompt_token_count': 3, 'prefilled_prompt_token_count': 3,
                                'cached_prompt_token_count': 0, 'generation_token_count': 512,
                                'stop_reason': 'length', 'tokens_per_second': 500,
                                'prompt_tokens_per_second': 1000},
                    'time_to_first_token_milliseconds': 10, 'total_milliseconds': 1050,
                    'peak_active_memory_bytes': 500000000})
    return report


class PairedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.plan_path, self.plan = plan_fixture(self.root)
        self.case = self.plan['cases'][0]
        self.report = native_fixture(self.plan, self.case)

    def tearDown(self):
        self.temporary.cleanup()

    def analyze(self, report=None, case=None, **kwargs):
        return runner.analyze_report(report or self.report, case or self.case, self.plan, True, **kwargs)

    def test_fingerprint_matches_swift_little_endian_vectors(self):
        for tokens, expected in [([], 'a8c7f832281a39c5'), ([0], '392209f14dea4c24'),
                                 ([11, 22, 33], '7401ecc7c0b5641a'), ([0, 255, 256, 262143], '9961baddd7e741b4')]:
            self.assertEqual(runner.fingerprint(tokens), 'fnv1a64:' + expected)
        for values in [[True], [1.0], [-1], [262144]]:
            with self.assertRaises(ValueError):
                runner.fingerprint(values)

    def test_valid_control_and_quantized_text_differences(self):
        control = self.analyze()
        self.assertTrue(control['control_eligible'])
        self.assertEqual(control['valid_pairs'], 8)
        case = self.plan['cases'][1]
        result = self.analyze(native_fixture(self.plan, case), case)
        self.assertFalse(result['cross_arm_outputs_match_exactly'])
        self.assertEqual(result['metrics']['decode']['accepted_paired_median_ratio'], 1)
        self.assertTrue(result['measured_repeatability']['A']['exactly_repeatable'])

    def test_each_schedule_and_work_guard_withholds_all_acceptance(self):
        mutations = [
            lambda t: t.update(sequence=99), lambda t: t.update(position_in_pair=2),
            lambda t: t.update(phase='warmup'), lambda t: t.update(prompt_token_count=True),
            lambda t: t.update(prompt_token_id_fingerprint='stale'),
            lambda t: t['metrics'].update(generation_token_count=511),
            lambda t: t['metrics'].update(cached_prompt_token_count=1),
            lambda t: t['metrics'].update(stop_reason='stop'),
            lambda t: t['metrics'].update(tokens_per_second=float('nan')),
            lambda t: t['metrics'].update(tokens_per_second=True),
            lambda t: t.update(time_to_first_token_milliseconds=0),
            lambda t: t.update(validation_issues=['early stop']),
        ]
        for mutate in mutations:
            report = copy.deepcopy(self.report)
            mutate(report['trials'][6])
            result = self.analyze(report)
            self.assertFalse(result['control_eligible'])
            self.assertTrue(result['rejected_trials'])
            for metric in result['metrics'].values():
                self.assertIsNone(metric['accepted_paired_median_ratio'])

    def test_malformed_types_and_stale_native_metadata_fail_closed(self):
        reports = [[], copy.deepcopy(self.report), copy.deepcopy(self.report), copy.deepcopy(self.report),
                   copy.deepcopy(self.report), copy.deepcopy(self.report), copy.deepcopy(self.report)]
        reports[1]['trials'][0]['metrics'] = []
        reports[2]['trials'][0]['content'] = 1
        reports[3]['models'][0]['configuration_sha256'] = 'stale'
        reports[4]['models'][0]['stored_weight_bytes'] = 4095
        reports[5]['loads'][0]['order'] = True
        reports[6]['trials'].pop()
        for report in reports:
            with self.subTest(report_type=type(report)), self.assertRaises(ValueError):
                runner.analyze_report(report, self.case, self.plan, True)

    def test_reasoning_one_off_is_retained_and_rejects_control(self):
        self.report['trials'][6]['reasoning'] = 'one off'
        result = self.analyze()
        self.assertFalse(result['control_eligible'])
        self.assertIn(7, result['measured_repeatability']['A']['one_off_sequences'])
        self.assertEqual(len(self.report['trials']), 22)

    def test_null_bias_drift_and_metric_specific_control(self):
        for trial in self.report['trials']:
            if trial['arm'] == 'B':
                trial['metrics']['tokens_per_second'] = 600
        result = self.analyze()
        self.assertFalse(result['control_eligible'])
        self.assertGreater(result['control_null_bias_fraction'], .19)
        case = self.plan['cases'][1]
        result = self.analyze(native_fixture(self.plan, case), case,
                              control_metrics={'decode': True, 'prefill': True, 'ttft': False, 'memory': True})
        self.assertIsNotNone(result['metrics']['decode']['accepted_paired_median_ratio'])
        self.assertIsNone(result['metrics']['ttft']['accepted_paired_median_ratio'])
        report = native_fixture(self.plan, case)
        for trial in report['trials']:
            if trial['phase'] == 'measured' and trial['pair'] > 4:
                trial['metrics']['tokens_per_second'] *= .7
        self.assertIsNone(self.analyze(report, case)['metrics']['decode']['accepted_paired_median_ratio'])

    def test_edited_cases_cannot_escape_full_hashes(self):
        changed = copy.deepcopy(self.plan)
        changed['cases'][0]['models'][1] = self.plan['models']['candidate']
        with self.assertRaisesRegex(ValueError, 'Plan cases'):
            runner.validate_plan(changed)
        changed = copy.deepcopy(self.plan)
        changed['cases'][1]['models'][1] = '/unhashed/model'
        with self.assertRaises(ValueError):
            runner.validate_plan(changed)

    def test_full_hash_detects_middle_byte_and_symlink_inventory(self):
        before = runner.snapshot(self.plan)
        weight = Path(self.plan['models']['baseline']) / 'model.safetensors'
        data = bytearray(weight.read_bytes())
        data[2048] ^= 1
        weight.write_bytes(data)
        self.assertIn(str(weight), runner.identity_changes(before, runner.snapshot(self.plan)))
        data[2048] ^= 1
        weight.write_bytes(data)
        alias = weight.with_name('extra.safetensors')
        alias.symlink_to(weight)
        self.assertIn(str(alias), runner.identity_changes(before, runner.snapshot(self.plan)))

    def test_post_run_missing_weights_retains_invocation_and_blocks_candidate(self):
        calls = []
        def fake_run(command, environment, directory, timeout):
            calls.append(command)
            (directory / 'native.json').write_text(json.dumps(self.report))
            (Path(self.plan['models']['baseline']) / 'model.safetensors').unlink()
            return {'exit_code': 0, 'elapsed_seconds': 1}
        with patch.object(runner, 'run_process', side_effect=fake_run):
            result = runner.execute_plan(self.plan_path, 3)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['status'], 'stopped')
        invocation = runner.read_json(self.plan_path.parent / '01-aa-control/invocation.json')
        self.assertEqual(invocation['exit_code'], 0)
        self.assertIn('provenance_error', invocation)
        self.assertTrue((self.plan_path.parent / '01-aa-control/native.json').exists())

    def test_failed_control_never_launches_candidate(self):
        def fake_run(command, environment, directory, timeout):
            report = copy.deepcopy(self.report)
            report['trials'][0]['content'] = 'changed'
            (directory / 'native.json').write_text(json.dumps(report))
            return {'exit_code': 0}
        with patch.object(runner, 'run_process', side_effect=fake_run) as process:
            result = runner.execute_plan(self.plan_path, 3)
        self.assertEqual(process.call_count, 1)
        self.assertFalse(result['control_passed'])

    def test_default_planning_never_executes_or_changes_environment(self):
        binary_dir = self.root / 'binary'
        with patch.object(runner.subprocess, 'Popen', side_effect=AssertionError('Must not execute')), \
                contextlib.redirect_stdout(io.StringIO()):
            result = runner.main(['--binary-dir', str(binary_dir), '--name', 'planned',
                                  '--baseline', self.plan['models']['baseline'],
                                  '--candidate', self.plan['models']['candidate'],
                                  '--output-root', str(self.root / 'planned')])
        self.assertEqual(result, 0)
        environment = {'MIDNIGHT_GEMMA3_COMPILED_TAIL': '1', 'MODEL_RUNNER_MLX_WIRED_TUNE_TOKENS': '513',
                       'MODEL_RUNNER_API_KEY': 'secret'}
        self.assertEqual(runner.selected_environment(environment),
                         {'MIDNIGHT_GEMMA3_COMPILED_TAIL': '1', 'MODEL_RUNNER_MLX_WIRED_TUNE_TOKENS': '513'})


if __name__ == '__main__':
    unittest.main()
