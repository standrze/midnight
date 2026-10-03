#!/usr/bin/env python3
"""Plan an A/A control then Q4/Q8-QK comparison using the two-resident native CLI.

Default is CPU-only planning. --execute or --execute-plan explicitly launches the
native executable; this script never builds, downloads, or changes feature flags.
Full file hashing happens outside native timing. Raw reports are never rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'artifacts/gemma3-270m-four-arm-20260929-r2/models/searched-g64'
CANDIDATE = ROOT / 'artifacts/gemma3-270m-q8-arms-20260929/models/searched-g64-q8-query-key'
OUTPUT = ROOT / 'benchmark-results/gemma-performance-20260929/paired-resident'
PROMPT = ('Write a detailed practical tutorial on maintaining a bicycle, covering tires, brakes, '
          'the chain, gears, and a regular maintenance schedule.')
SETTINGS = {'format': 1, 'experiment': 'two_resident_gemma3_270m_models', 'requested_tokens': 512,
            'warmups_per_arm': 3, 'maximum_resident_models': 2, 'temperature': 0, 'top_p': 1,
            'enable_prompt_cache': False, 'enable_speculative_decoding': False,
            'engine': 'metal', 'prefill_step_size': 512}
MODEL_SUFFIXES = {'.safetensors', '.json', '.jinja', '.model', '.tiktoken'}
LIMITATIONS = [
    'Each case is one process holding two models; paired trials within a process are not independent processes.',
    'Bootstrap resamples adjacent pairs. Serial correlation, small samples and shared hardware load limit inference.',
    'Accepted metrics mean eligible measurements, not proven improvement. An A/A control does not remove later external load.',
    'Greedy quantizer outputs can differ. Timing compares natural trajectories, not fixed activations or equal quality.',
    'Generated raw token IDs are not available; repeatability checks decoded content, reasoning, counts and stop reason.',
    'Peak memory includes both resident models. TTFT is the first content event, not a raw-token timestamp.',
    'Load and wired-memory tuning are excluded from trial timing; OS file caches are not flushed.',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def file_info(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), f'Expected a regular file: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        after = os.fstat(stream.fileno())
    current = path.stat()
    signature = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    require(signature(before) == signature(after) == signature(current), f'File changed while hashing: {path}')
    return {'path': str(path), 'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns, 'sha256': digest.hexdigest()}


def selected_environment(environment):
    result = {}
    for key, value in environment.items():
        selected = key.startswith(('MODEL_RUNNER_', 'MIDNIGHT_GEMMA', 'MIDNIGHT_METAL_', 'MIDNIGHT_MTP_', 'MLX_'))
        upper = key.upper()
        secret = upper.endswith('_TOKEN') or any(s in upper for s in ('SECRET', 'PASSWORD', 'API_KEY', 'ACCESS_KEY'))
        if selected and not secret:
            result[key] = value
    return result


def fingerprint(tokens):
    require(isinstance(tokens, list) and all(type(n) is int and 0 <= n < 262144 for n in tokens),
            'Prompt IDs must be integral Gemma vocabulary IDs, not booleans or floats')
    value = 0xcbf29ce484222325
    for number in [len(tokens), *tokens]:
        for byte in struct.pack('<Q', number):
            value = ((value ^ byte) * 0x100000001b3) & 0xffffffffffffffff
    return f'fnv1a64:{value:016x}'


def finite_positive(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0, f'{label} must be finite and positive')
    return float(value)


def model_files(directory):
    directory = Path(directory).resolve(strict=True)
    require(directory.is_dir(), f'Missing checkpoint directory: {directory}')
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix in MODEL_SUFFIXES)
    require(any(p.suffix == '.safetensors' for p in files), f'No weights: {directory}')
    require((directory / 'config.json') in files, f'No configuration: {directory}')
    return files


def snapshot(plan):
    files = {Path(__file__).resolve(), ROOT / 'Scripts/benchmark-campaign.py', Path(plan['prompt_file'])}
    runtime = Path(plan['binary'])
    require(runtime.is_file(), f'Missing paired runtime: {runtime}')
    require((runtime.parent / 'mlx.metallib').is_file(), 'The paired runtime must have an adjacent mlx.metallib')
    files.update(p for p in runtime.parent.iterdir()
                 if p.is_file() and (p == runtime or p.suffix in {'.metallib', '.dylib', '.so'} or p.name == 'identities.json'))
    for path in set(plan['models'].values()):
        files.update(model_files(path))
    return {str(p.absolute()): file_info(p) for p in sorted(files)}


def identity_changes(before, after):
    keys = set(before) | set(after)
    return sorted(key for key in keys if key not in before or key not in after
                  or any(before[key][field] != after[key][field] for field in ('path', 'sha256', 'bytes')))


def campaign_module():
    spec = importlib.util.spec_from_file_location('paired_campaign_metrics', ROOT / 'Scripts/benchmark-campaign.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def output_identity(trial):
    metrics = trial.get('metrics') or {}
    return (trial.get('content'), trial.get('reasoning'), metrics.get('generation_token_count'), metrics.get('stop_reason'))


def repeatability(trials):
    result = {}
    for arm in ('A', 'B'):
        selected = [t for t in trials if t.get('arm') == arm]
        counts = Counter(output_identity(t) for t in selected)
        modal = counts.most_common(1)[0][0] if counts else None
        result[arm] = {
            'trials': len(selected), 'distinct_outputs': len(counts), 'exactly_repeatable': len(counts) == 1,
            'one_off_sequences': [t.get('sequence') for t in selected if counts[output_identity(t)] == 1],
            'nonmodal_sequences': [t.get('sequence') for t in selected if output_identity(t) != modal],
            'output_groups': [
                {'content_sha256': hashlib.sha256((key[0] or '').encode()).hexdigest(),
                 'reasoning_sha256': hashlib.sha256((key[1] or '').encode()).hexdigest(),
                 'generated_tokens': key[2], 'stop_reason': key[3], 'occurrences': count}
                for key, count in counts.most_common()],
        }
    return result


def validate_trial(trial, expected, prompt_ids, prompt_fingerprint):
    errors = []
    for key, value in expected.items():
        if type(trial.get(key)) is not type(value) or trial.get(key) != value:
            errors.append(f'{key} does not match the frozen schedule')
    try:
        require(not trial.get('error'), 'Native trial error')
        require(trial.get('validation_issues') == [], 'Native validation issues or missing issue list')
        require(type(trial.get('tool_call_count')) is int and trial['tool_call_count'] == 0, 'Unexpected tool calls')
        require(isinstance(trial.get('content'), str) and isinstance(trial.get('reasoning'), str), 'Missing output strings')
        require(type(trial.get('prompt_token_count')) is int and trial['prompt_token_count'] == len(prompt_ids),
                'Prompt count mismatch')
        require(trial.get('prompt_token_id_fingerprint') == prompt_fingerprint, 'Exact prompt fingerprint mismatch')
        metrics = trial['metrics']
        for key, value in [('prompt_token_count', len(prompt_ids)), ('prefilled_prompt_token_count', len(prompt_ids)),
                           ('cached_prompt_token_count', 0), ('generation_token_count', 512)]:
            require(type(metrics.get(key)) is int and metrics[key] == value, f'{key} mismatch')
        require(metrics.get('stop_reason') == 'length', 'Stop reason is not the fixed-length limit')
        for key in ('proposed_draft_tokens', 'accepted_draft_tokens'):
            require(metrics.get(key) is None or (type(metrics[key]) is int and metrics[key] == 0), 'Speculative tokens appeared')
        require(metrics.get('speculative_passthrough_reason') is None, 'Speculative passthrough appeared')
        for value, name in [(metrics.get('tokens_per_second'), 'decode'),
                            (metrics.get('prompt_tokens_per_second'), 'prefill'),
                            (trial.get('time_to_first_token_milliseconds'), 'TTFT'),
                            (trial.get('total_milliseconds'), 'total duration'),
                            (trial.get('peak_active_memory_bytes'), 'peak active memory')]:
            finite_positive(value, name)
    except (ValueError, KeyError, TypeError) as error:
        errors.append(str(error))
    return errors


def analyze_report(report, case, plan, provenance_unchanged, control_passed=True, control_metrics=None):
    require(isinstance(report, dict), 'Native report must be an object')
    require(not report.get('error'), 'Native report carries an error')
    require(report.get('status') == 'completed_raw_trials', 'Native report did not complete every trial')
    for key, value in {**SETTINGS, 'pairs': plan['pairs'], 'context_length': plan['context_length']}.items():
        require(type(report.get(key)) is type(value) and report.get(key) == value, f'Native setting mismatch: {key}')
    require(report.get('runtime_environment') == plan['runtime_environment'], 'Runtime environment differs from plan')
    require(report.get('prompt') == plan['prompt'], 'Native prompt text differs from plan')
    require(isinstance(report.get('models'), list) and all(isinstance(m, dict) for m in report['models']),
            'Invalid native model metadata')
    require([m.get('path') for m in report['models']] == case['models'], 'Native model paths differ from plan')
    for model in report['models']:
        directory = Path(model['path'])
        config = plan['input_files'][str(directory / 'config.json')]
        require(model.get('configuration_sha256') == config['sha256'], 'Native configuration hash differs from plan')
        shards = {Path(path).name: info['bytes'] for path, info in plan['input_files'].items()
                  if Path(path).parent == directory and Path(path).suffix == '.safetensors'}
        require(isinstance(model.get('shards'), list) and all(isinstance(v, dict) for v in model['shards']),
                'Invalid native shard metadata')
        require(len(model['shards']) == len(shards) and
                all(v.get('name') in shards and type(v.get('bytes')) is int and v['bytes'] == shards[v['name']]
                    for v in model['shards']) and len({v.get('name') for v in model['shards']}) == len(shards),
                'Native shard inventory or sizes differ from plan')
        require(type(model.get('stored_weight_bytes')) is int and model['stored_weight_bytes'] == sum(shards.values()),
                'Native stored weight size differs from plan')
    require([m.get('arm') for m in report.get('models', [])] == ['A', 'B'], 'Native arm order differs from plan')
    require(isinstance(report.get('loads'), list) and len(report['loads']) == 2
            and all(isinstance(v, dict) for v in report['loads']), 'Exactly two model loads required')
    for position, load in enumerate(report['loads']):
        require(load.get('arm') == ('A', 'B')[position] and type(load.get('order')) is int and load['order'] == position + 1,
                'Model load order mismatch')
        require(load.get('implementation') == 'MLXLLM.Gemma3TextModel', 'Wrong native model implementation')
    prompt_ids = report.get('rendered_prompt_tokens')
    prompt_fingerprint = fingerprint(prompt_ids)
    require(prompt_ids, 'Empty rendered prompt')
    trials = report.get('trials')
    require(isinstance(trials, list) and all(isinstance(t, dict) for t in trials), 'Missing or malformed raw trials')
    require(all(isinstance(t.get('metrics'), dict) and isinstance(t.get('content'), str)
                and isinstance(t.get('reasoning'), str) for t in trials), 'Malformed metrics or output strings')
    require(len(trials) == 2 * (3 + plan['pairs']), 'Missing or extra scheduled raw trials')
    schedule = []
    for phase, count in [('warmup', 3), ('measured', plan['pairs'])]:
        for pair in range(1, count + 1):
            order = 'AB' if pair % 2 else 'BA'
            for position, arm in enumerate(order, 1):
                schedule.append({'sequence': len(schedule) + 1, 'phase': phase, 'pair': pair,
                                 'order': order, 'position_in_pair': position, 'arm': arm})
    failures, valid_trials = [], {}
    for trial, expected in zip(trials, schedule):
        errors = validate_trial(trial, expected, prompt_ids, prompt_fingerprint)
        if errors:
            failures.append({'sequence': expected['sequence'], 'phase': expected['phase'], 'reasons': errors})
        else:
            valid_trials[(expected['phase'], expected['pair'], expected['arm'])] = trial
    measured = [t for t in trials if t.get('phase') == 'measured']
    repeat = repeatability(measured)
    all_repeat = repeatability(trials)
    exact_cross_arm = all(output_identity(trials[2 * i]) == output_identity(trials[2 * i + 1])
                          for i in range(len(trials) // 2))
    common_reasons = []
    if failures:
        common_reasons.append('invalid scheduled trials, including warmups; no accepted fixed-work metric')
    if not provenance_unchanged:
        common_reasons.append('full model/runtime/input identities changed or were not verified')
    if not control_passed:
        common_reasons.append('preceding A/A control was not eligible')
    if not all(value['exactly_repeatable'] for value in repeat.values()):
        common_reasons.append('per-arm greedy measured outputs were not exactly repeatable')
    if case['kind'] == 'control' and (not exact_cross_arm or not all(v['exactly_repeatable'] for v in all_repeat.values())):
        common_reasons.append('A/A content, reasoning, counts, or stop reasons differ')
    valid = []
    for pair in range(1, plan['pairs'] + 1):
        a, b = (valid_trials.get(('measured', pair, arm)) for arm in ('A', 'B'))
        if a is None or b is None:
            continue
        def adapt(trial):
            return {'baseline_first': pair % 2 == 1, 'measurement': {
                'decode_tokens_per_second': trial['metrics']['tokens_per_second'],
                'prefill_tokens_per_second': trial['metrics']['prompt_tokens_per_second'],
                'ttft_ms': trial['time_to_first_token_milliseconds'],
                'peak_active_memory_bytes': trial['peak_active_memory_bytes'],
            }}
        valid.append((adapt(a), adapt(b)))
    if len(valid) < 4:
        common_reasons.append('fewer than four valid adjacent pairs; exploratory only')
    metrics = {}
    if valid:
        module = campaign_module()
        for name, key, lower in [('decode', 'decode_tokens_per_second', False),
                                 ('prefill', 'prefill_tokens_per_second', False), ('ttft', 'ttft_ms', True),
                                 ('memory', 'peak_active_memory_bytes', True)]:
            reasons = list(common_reasons)
            if case['kind'] != 'control' and control_metrics is not None and not control_metrics.get(name, False):
                reasons.append(f'preceding A/A {name} control was not eligible')
            metrics[name] = module.metric_summary(valid, key, lower, plan, reasons)
    null_bias = None
    if case['kind'] == 'control' and 'decode' in metrics:
        null_bias = abs(metrics['decode']['diagnostic_paired_median_ratio'] - 1)
        if null_bias > plan['maximum_null_bias_fraction']:
            reason = 'A/A median decode ratio exceeds the frozen null-bias tolerance'
            for metric in metrics.values():
                metric['acceptance_reasons'].append(reason)
                metric['accepted_paired_median_ratio'] = None
                metric['accepted_paired_bootstrap_95_percent_interval'] = None
    control_eligible = (case['kind'] == 'control' and exact_cross_arm and not common_reasons
                        and metrics.get('decode', {}).get('accepted_paired_median_ratio') is not None)
    return {'case': case['name'], 'kind': case['kind'], 'valid_pairs': len(valid), 'expected_pairs': plan['pairs'],
            'prompt_token_id_fingerprint': prompt_fingerprint, 'rejected_trials': failures,
            'cross_arm_outputs_match_exactly': exact_cross_arm, 'measured_repeatability': repeat,
            'including_warmup_repeatability': all_repeat, 'provenance_unchanged': provenance_unchanged,
            'control_eligible': control_eligible, 'control_null_bias_fraction': null_bias,
            'metrics': metrics, 'common_acceptance_reasons': common_reasons, 'limitations': LIMITATIONS}


def run_process(command, environment, directory, timeout):
    started = time.monotonic()
    record = {'command': command, 'started_at': now()}
    process = None
    with (directory / 'stdout.log').open('xb') as stdout, (directory / 'stderr.log').open('xb') as stderr:
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                       env=environment, start_new_session=True)
            record['exit_code'] = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            record['error'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'timeout'
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            record['exit_code'] = process.returncode if process is not None else None
        except OSError as error:
            record['error'] = str(error)
    record['elapsed_seconds'] = time.monotonic() - started
    record['finished_at'] = now()
    return record


def validate_plan(plan):
    require(isinstance(plan, dict) and plan.get('format') == 1
            and plan.get('status') == 'prepared_no_gpu_started', 'Unknown plan format/state')
    require(isinstance(plan.get('models'), dict) and set(plan['models']) == {'baseline', 'candidate'}, 'Invalid model map')
    baseline, candidate = plan['models']['baseline'], plan['models']['candidate']
    expected = [{'name': '01-aa-control', 'kind': 'control', 'models': [baseline, baseline]},
                {'name': '02-searched-vs-q8-qk', 'kind': 'quantization', 'models': [baseline, candidate]}]
    require(plan.get('cases') == expected, 'Plan cases must be A/A baseline first, then baseline/candidate')
    require(type(plan.get('pairs')) is int and 4 <= plan['pairs'] <= 64 and plan['pairs'] % 2 == 0, 'Invalid pair count')
    require(plan.get('settings') == SETTINGS, 'Plan settings changed')
    for key in ('maximum_drift_fraction', 'maximum_null_bias_fraction'):
        require(0 < finite_positive(plan.get(key), key) <= .5, 'Invalid frozen threshold')
    require(type(plan.get('context_length')) is int and 513 <= plan['context_length'] <= 32768, 'Invalid context length')
    require(isinstance(plan.get('input_files'), dict), 'Missing input hashes')
    for directory in set(plan['models'].values()):
        require(str(Path(directory).resolve()) == directory and str(Path(directory) / 'config.json') in plan['input_files'],
                'Model path is not canonical or not covered by frozen hashes')
    require(plan['binary'] in plan['input_files'] and plan['prompt_file'] in plan['input_files'], 'Unhashed executable or prompt')


def execute_plan(plan_path, timeout):
    plan_path = Path(plan_path).resolve(strict=True)
    plan = read_json(plan_path)
    validate_plan(plan)
    require(selected_environment(os.environ) == plan['runtime_environment'], 'Runtime environment changed since plan')
    directory = plan_path.parent
    results, control_metrics = [], {}
    control_passed = False
    for case in plan['cases']:
        run_dir = directory / case['name']
        run_dir.mkdir()
        record = {'status': 'preflight'}
        raw = run_dir / 'native.json'
        try:
            before = snapshot(plan)
            write_new(run_dir / 'before-provenance.json', before)
            changes = identity_changes(plan['input_files'], before)
            require(not changes, 'Inputs changed since plan: ' + ', '.join(changes))
            command = [plan['binary'], *case['models'], str(raw),
                       '--prompt-file', plan['prompt_file'], '--pairs', str(plan['pairs']),
                       '--context-length', str(plan['context_length'])]
            record = run_process(command, dict(os.environ), run_dir, timeout)
            try:
                after = snapshot(plan)
                write_new(run_dir / 'after-provenance.json', after)
                record['identity_changes'] = identity_changes(before, after)
            except (OSError, ValueError, KeyError, TypeError) as error:
                record['provenance_error'] = str(error)
                write_new(run_dir / 'after-provenance.json', {'error': str(error)})
            if raw.is_file():
                record['native_report'] = file_info(raw)
            require(record.get('exit_code') == 0 and not record.get('error'), 'Native process failed')
            require(not record.get('provenance_error'), 'Post-run full provenance could not be verified')
            result = analyze_report(read_json(raw), case, plan, not record['identity_changes'],
                                    control_passed=case['kind'] == 'control' or control_passed,
                                    control_metrics=control_metrics)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            record['validation_error'] = str(error)
            result = {'case': case['name'], 'status': 'invalid', 'error': str(error), 'control_eligible': False}
        write_new(run_dir / 'invocation.json', record)
        write_new(run_dir / 'analysis.json', result)
        results.append(result)
        if case['kind'] == 'control':
            control_passed = result.get('control_eligible', False)
            control_metrics = {name: value.get('accepted_paired_median_ratio') is not None
                               for name, value in result.get('metrics', {}).items()}
            if not control_passed:
                break
        elif result.get('status') == 'invalid':
            break
    summary = {'format': 1, 'finished_at': now(), 'control_passed': control_passed,
               'status': 'complete' if len(results) == 2 and all('metrics' in r for r in results) else 'stopped',
               'cases': results, 'limitations': LIMITATIONS}
    write_new(directory / 'analysis.json', summary)
    return summary


def make_plan(args):
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', args.name or ''), 'Provide a safe --name')
    require(type(args.pairs) is int and 4 <= args.pairs <= 64 and args.pairs % 2 == 0,
            '--pairs must be an even number from 4 to 64')
    require(0 < args.maximum_drift_fraction <= .5 and 0 < args.maximum_null_bias_fraction <= .5,
            'Drift and null-bias thresholds must be positive and at most 0.5')
    require(513 <= args.context_length <= 32768, 'Invalid context length')
    require(args.binary_dir is not None, '--binary-dir is required to prepare a plan')
    binary = (Path(args.binary_dir).resolve(strict=True) / 'model-runner-paired-runtime-bench')
    require(binary.is_file() and os.access(binary, os.X_OK), 'Paired runtime executable is missing or not executable')
    models = {'baseline': str(Path(args.baseline).resolve(strict=True)), 'candidate': str(Path(args.candidate).resolve(strict=True))}
    directory = Path(args.output_root).resolve() / args.name
    directory.mkdir(parents=True)
    prompt = Path(args.prompt_file).read_text() if args.prompt_file else PROMPT
    require(prompt.strip() and len(prompt.encode()) <= 1048576, 'Invalid prompt')
    prompt_file = directory / 'prompt.txt'
    with prompt_file.open('x') as stream:
        stream.write(prompt)
    plan = {'format': 1, 'status': 'prepared_no_gpu_started', 'created_at': now(), 'binary': str(binary),
            'prompt': prompt, 'prompt_file': str(prompt_file), 'models': models, 'pairs': args.pairs,
            'context_length': args.context_length, 'seed': 20260929,
            'maximum_drift_fraction': args.maximum_drift_fraction,
            'maximum_null_bias_fraction': args.maximum_null_bias_fraction,
            'runtime_environment': selected_environment(os.environ), 'settings': SETTINGS,
            'measurement_unit': 'Adjacent A/B trial pair in one two-model process; never relabel pairs as separate processes.',
            'cases': [{'name': '01-aa-control', 'kind': 'control', 'models': [models['baseline'], models['baseline']]},
                      {'name': '02-searched-vs-q8-qk', 'kind': 'quantization', 'models': [models['baseline'], models['candidate']]}],
            'limitations': LIMITATIONS}
    plan['input_files'] = snapshot(plan)
    write_new(directory / 'plan.json', plan)
    return directory / 'plan.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary-dir', type=Path)
    parser.add_argument('--name')
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--candidate', type=Path, default=CANDIDATE)
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    parser.add_argument('--prompt-file', type=Path)
    parser.add_argument('--pairs', type=int, default=8)
    parser.add_argument('--context-length', type=int, default=8192)
    parser.add_argument('--maximum-drift-fraction', type=float, default=.1)
    parser.add_argument('--maximum-null-bias-fraction', type=float, default=.03)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--execute-plan', type=Path)
    args = parser.parse_args(argv)
    try:
        require(args.timeout > 0 and math.isfinite(args.timeout), '--timeout must be finite and positive')
        require(not (args.execute and args.execute_plan), 'Choose --execute or --execute-plan')
        plan_path = args.execute_plan or make_plan(args)
        if args.execute or args.execute_plan:
            summary = execute_plan(plan_path, args.timeout)
            print(json.dumps({'status': summary['status'], 'control_passed': summary['control_passed'],
                              'analysis': str(Path(plan_path).parent / 'analysis.json')}, indent=2))
            return 0 if summary['status'] == 'complete' else 1
        print(json.dumps({'status': 'planned_no_gpu', 'plan': str(plan_path),
                          'execute_command': [sys.executable, str(Path(__file__).resolve()), '--execute-plan', str(plan_path)]}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Paired runtime campaign failed: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
