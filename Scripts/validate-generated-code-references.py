#!/usr/bin/env python3
"""Run the pinned MBPP reference solutions against private tests in Docker only.

This is a grader/source compatibility check, never a model evaluation. Public
prompts remain solution-free. All failures are retained; no record is silently
excluded from the frozen suite. Requires an already installed pinned image.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def load_script(filename):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(filename.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load_script('evaluate-generated.py')
builder = load_script('prepare-generated-eval-corpus.py')


def validate(tasks_path, answers_path, cache, sandbox):
    tasks, keys = evaluation.load_inputs(tasks_path, answers_path)
    source = next(s for s in builder.pinned.SOURCES if s['name'] == 'mbpp')
    source_path = Path(cache) / 'mbpp-source.jsonl'
    data = builder.pinned.verify(source_path.read_bytes(), source['source_sha256'], str(source_path))
    references = {f"mbpp-test-{row['task_id']}": row for row in
                  (json.loads(s) for s in data.decode().splitlines() if s.strip())}
    results = []
    for task in tasks:
        if task['category'] != 'code':
            continue
        row = references.get(task['id'])
        if row is None:
            raise ValueError(f"Missing pinned reference for {task['id']}")
        key = keys[task['id']]
        if key['setup'] != row.get('test_setup_code', '') or key['tests'] != row['test_list']:
            raise ValueError(f"Private test key differs from pinned source: {task['id']}")
        if not all(signature in task['prompt'] for signature in builder.signatures(row)):
            raise ValueError(f"Public prompt omits a required tested API: {task['id']}")
        result = sandbox.score(row['code'], key)
        results.append({'id': task['id'], 'reference_code_sha256': hashlib.sha256(row['code'].encode()).hexdigest(), **result})
        print(f"{task['id']}: {result['reason']}", file=sys.stderr, flush=True)
    return {'format': 1, 'purpose': __doc__.strip(),
            'status': 'valid' if results and all(r['passed'] is True for r in results) else 'invalid',
            'count': len(results), 'passed': sum(r['passed'] is True for r in results),
            'provenance': {'tasks': evaluation.file_info(tasks_path), 'answers': evaluation.file_info(answers_path),
                           'source': evaluation.file_info(source_path), 'source_revision': source['revision'],
                           'validator': evaluation.file_info(__file__), 'scorer': evaluation.file_info(evaluation.__file__),
                           'sandbox': sandbox.provenance}, 'results': results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--answers', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--sandbox-image', required=True)
    parser.add_argument('--code-timeout', type=float, default=15)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError('Output already exists')
        sandbox = evaluation.DockerSandbox(args.sandbox_image, args.code_timeout)
        result = validate(args.tasks, args.answers, args.cache_dir, sandbox)
        evaluation.write_new(args.output, result)
        print(json.dumps({'status': result['status'], 'count': result['count'], 'passed': result['passed']}))
        return 0 if result['status'] == 'valid' else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Reference validation failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
