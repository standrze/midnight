#!/usr/bin/env python3
"""Check authored pilot references and faulty controls in the pinned Docker sandbox.

This verifies the coding grader, not model quality or independent cyber labels.
"""
import argparse
import importlib.util
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-dir', type=Path, required=True)
    parser.add_argument('--sandbox-image', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = Path(__file__).with_name('evaluate-generated.py')
    spec = importlib.util.spec_from_file_location('pilot_evaluation', path)
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    try:
        if args.output.exists():
            raise ValueError('Validation output must be new')
        tasks_path, answers_path, refs_path = (args.corpus_dir / name for name in
                                               ['tasks.jsonl', 'answers.jsonl', 'references.jsonl'])
        tasks, keys = evaluation.load_inputs(tasks_path, answers_path)
        references = evaluation.read_jsonl(refs_path)
        code_ids = {t['id'] for t in tasks if t['category'] == 'code'}
        if {r['id'] for r in references} != code_ids:
            raise ValueError('Reference IDs must match every coding task exactly')
        sandbox = evaluation.DockerSandbox(args.sandbox_image, timeout=15)
        results = []
        for reference in references:
            required = {'id', 'code', 'faulty_code'}
            if not required <= set(reference) or set(reference) - required - {'extra_faulty_controls'}:
                raise ValueError('Unexpected reference schema')
            extras = reference.get('extra_faulty_controls', [])
            if not isinstance(extras, list) or any(not isinstance(code, str) or not code.strip() for code in extras):
                raise ValueError('Extra faulty controls must be source strings')
            key = keys[reference['id']]
            correct = sandbox.score(evaluation.extract_code(reference['code']), key)
            controls = [sandbox.score(evaluation.extract_code(code), key)
                        for code in [reference['faulty_code']] + extras]
            passed = (correct['status'] == 'scored' and correct['passed'] is True
                      and all(control['status'] == 'scored' and control['passed'] is False
                              and control['reason'] == 'tests_failed' for control in controls))
            results.append({'id': reference['id'], 'passed': passed,
                            'reference': correct, 'faulty_control': controls[0],
                            'extra_faulty_controls': controls[1:]})
            print(f"{reference['id']}: {'passed' if passed else 'failed'}", file=sys.stderr, flush=True)
        valid = bool(results) and all(r['passed'] for r in results)
        report = {'format': 1, 'status': 'valid' if valid else 'invalid', 'count': len(results),
                  'passed': sum(r['passed'] for r in results), 'results': results,
                  'faulty_control_count': sum(1 + len(r['extra_faulty_controls']) for r in results),
                  'provenance': {'tasks': evaluation.file_info(tasks_path),
                                 'answers': evaluation.file_info(answers_path),
                                 'references': evaluation.file_info(refs_path),
                                 'validator': evaluation.file_info(__file__),
                                 'scorer': evaluation.file_info(path), 'sandbox': sandbox.provenance},
                  'limitations': ['Checks coding references and faulty implementations only.',
                                  'Cyber threat-scope labels have not had independent review.',
                                  'Finite faulty implementations do not establish exhaustive test coverage.']}
        evaluation.write_new(args.output, report)
        return 0 if valid else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Pilot validation failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
