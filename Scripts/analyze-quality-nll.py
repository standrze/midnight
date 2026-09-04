#!/usr/bin/env python3
"""Validate and compare paired native NLL reports with category-stratified uncertainty."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import sys


IDENTITY_FIELDS = ('id', 'category', 'token_id_fingerprint', 'original_token_count',
                   'evaluated_token_count', 'scored_token_count', 'truncated')
SETTING_FIELDS = ('corpus_fingerprint', 'token_id_fingerprint', 'backend', 'device',
                  'model_type', 'add_special_tokens', 'maximum_tokens_per_sample', 'prefill_step_size')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def finite_nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def validate_report(report):
    require(isinstance(report, dict) and report.get('format') == 1 and report.get('status') == 'measured',
            'Expected a measured format-1 native report')
    require(report.get('metric') == 'teacher_forced_next_token_nll', 'Wrong report metric')
    samples = report.get('samples')
    require(isinstance(samples, list) and bool(samples), 'Missing sample records')
    require(report.get('sample_count') == len(samples), 'Sample count differs from retained samples')
    seen = set()
    for sample in samples:
        require(isinstance(sample, dict), 'Malformed sample')
        for key in ('id', 'category', 'token_id_fingerprint'):
            require(isinstance(sample.get(key), str) and bool(sample[key].strip()), f'Missing sample {key}')
        require(sample['id'] not in seen, 'Duplicate sample IDs')
        seen.add(sample['id'])
        original, evaluated, scored = (sample.get(k) for k in ('original_token_count', 'evaluated_token_count', 'scored_token_count'))
        require(all(positive_integer(v) for v in (original, evaluated, scored)), 'Invalid sample token lengths')
        require(original >= evaluated == scored + 1, 'Inconsistent sample token lengths')
        require(sample.get('truncated') is (evaluated < original), 'Inconsistent truncation indicator')
        nll_sum = sample.get('nll_sum')
        require(finite_nonnegative(nll_sum) and finite_nonnegative(sample.get('nll')), 'Nonfinite/negative sample NLL')
        require(math.isclose(nll_sum / scored, sample['nll'], rel_tol=1e-10, abs_tol=1e-10), 'Sample NLL does not reconcile')
    tokens = sum(s['scored_token_count'] for s in samples)
    total = sum(s['nll_sum'] for s in samples)
    require(report.get('scored_token_count') == tokens, 'Report token total does not reconcile')
    require(finite_nonnegative(report.get('nll_sum')) and math.isclose(report['nll_sum'], total, rel_tol=1e-10, abs_tol=1e-8),
            'Report NLL sum does not reconcile')
    require(finite_nonnegative(report.get('token_weighted_nll')) and math.isclose(report['token_weighted_nll'], total / tokens, rel_tol=1e-10, abs_tol=1e-10),
            'Report weighted NLL does not reconcile')
    for key in SETTING_FIELDS:
        require(report.get(key) is not None, f'Missing report setting: {key}')
    require(positive_integer(report['maximum_tokens_per_sample']), 'Invalid maximum sample length')
    require(all(s['evaluated_token_count'] <= report['maximum_tokens_per_sample'] for s in samples), 'Evaluated sample exceeds report limit')
    require(isinstance(report['prefill_step_size'], int) and not isinstance(report['prefill_step_size'], bool)
            and 0 <= report['prefill_step_size'] <= 8192, 'Invalid prefill setting')
    return samples


def bootstrap_interval(groups, draws, seed):
    """Sample pairs inside each category, preserving observed category counts."""
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sampled = [pair for group in groups for pair in rng.choices(group, k=len(group))]
        count = sum(a['scored_token_count'] for a, _ in sampled)
        estimates.append(sum(b['nll_sum'] - a['nll_sum'] for a, b in sampled) / count)
    estimates.sort()
    return [estimates[max(0, math.ceil(draws * .025) - 1)], estimates[math.ceil(draws * .975) - 1]]


def summarize_pair(groups, draws, seed):
    pairs = [pair for group in groups for pair in group]
    tokens = sum(a['scored_token_count'] for a, _ in pairs)
    baseline = sum(a['nll_sum'] for a, _ in pairs) / tokens
    candidate = sum(b['nll_sum'] for _, b in pairs) / tokens
    return {'sample_count': len(pairs), 'scored_token_count': tokens,
            'baseline_token_weighted_nll': baseline, 'candidate_token_weighted_nll': candidate,
            'candidate_minus_baseline_nll': candidate - baseline,
            'paired_bootstrap_95_percent_interval': bootstrap_interval(groups, draws, seed),
            'candidate_lower_nll_samples': sum(b['nll'] < a['nll'] for a, b in pairs),
            'candidate_higher_nll_samples': sum(b['nll'] > a['nll'] for a, b in pairs),
            'tied_samples': sum(b['nll'] == a['nll'] for a, b in pairs)}


def analyze(reports, draws=10000, seed=20260904):
    require(2 <= len(reports) <= 8, 'Supply two to eight labeled native reports')
    require(isinstance(draws, int) and 100 <= draws <= 100000, 'Bootstrap draws must be in 100...100000')
    labels = list(reports)
    sample_sets = {label: validate_report(report) for label, report in reports.items()}
    reference = reports[labels[0]]
    for label in labels[1:]:
        for key in SETTING_FIELDS:
            require(reports[label][key] == reference[key], f'{label}: report setting differs: {key}')
        require(len(sample_sets[label]) == len(sample_sets[labels[0]]), f'{label}: sample count differs')
        for a, b in zip(sample_sets[labels[0]], sample_sets[label]):
            for key in IDENTITY_FIELDS:
                require(a[key] == b[key], f'{label}: paired sample {a["id"]} differs: {key}')
    categories = sorted({s['category'] for s in sample_sets[labels[0]]})
    model_summaries = {}
    for label in labels:
        samples = sample_sets[label]
        model_summaries[label] = {
            'model_path': reports[label].get('model_path'),
            'sample_count': len(samples), 'scored_token_count': reports[label]['scored_token_count'],
            'token_weighted_nll': reports[label]['token_weighted_nll'],
            'truncated_samples': sum(s['truncated'] for s in samples),
            'categories': {category: {
                'sample_count': sum(s['category'] == category for s in samples),
                'scored_token_count': sum(s['scored_token_count'] for s in samples if s['category'] == category),
                'token_weighted_nll': sum(s['nll_sum'] for s in samples if s['category'] == category) / sum(s['scored_token_count'] for s in samples if s['category'] == category),
            } for category in categories},
        }
    comparisons = []
    for baseline, candidate in itertools.combinations(labels, 2):
        pairs = list(zip(sample_sets[baseline], sample_sets[candidate]))
        groups = [[pair for pair in pairs if pair[0]['category'] == category] for category in categories]
        comparisons.append({'baseline': baseline, 'candidate': candidate,
            'overall': summarize_pair(groups, draws, seed),
            'categories': {category: summarize_pair([group], draws, seed) for category, group in zip(categories, groups)}})
    return {'format': 1, 'status': 'analyzed', 'shared_settings': {key: reference[key] for key in SETTING_FIELDS},
            'all_sample_identities_lengths_and_categories_match': True, 'models': model_summaries,
            'comparisons': comparisons,
            'bootstrap': {'draws': draws, 'seed': seed, 'method': 'Paired percentile bootstrap of records within category; overall preserves observed category sample counts and recomputes token-weighted NLL.'},
            'interpretation': 'Negative candidate-minus-baseline NLL favors the candidate on these references. A confidence interval spanning zero does not prove equivalence or noninferiority.',
            'limitations': ['Teacher-forced reference NLL is not generated-task accuracy, teacher KL, or exact greedy-output preservation.',
                'Intervals describe the observed convenience corpus composition; they do not establish population representativeness or training-data exclusion.',
                'Resampling records does not capture correlations within a source, numeric/run variation or measurement choices; multiple comparisons are not corrected.',
                'No runtime or efficiency comparison is computed from quality report elapsed times. Native memory metrics are not process RSS.',
                'Native model paths do not independently fingerprint all weights or attest executable/Metal-library build provenance; retain separate artifact provenance.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', action='append', required=True, metavar='LABEL=PATH')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260904)
    args = parser.parse_args(argv)
    try:
        reports, provenance = {}, {}
        for value in args.report:
            label, separator, path = value.partition('=')
            require(separator and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', label), 'Report syntax is LABEL=PATH')
            require(label not in reports, 'Duplicate report label')
            source = Path(path).resolve()
            require(source != args.output.resolve(), 'Output must not replace a native report')
            payload = source.read_bytes()
            reports[label] = json.loads(payload)
            provenance[label] = {'path': str(source), 'sha256': hashlib.sha256(payload).hexdigest()}
        result = analyze(reports, args.draws, args.seed)
        result['provenance'] = {'native_reports': provenance, 'analyzer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
        require(not args.output.exists() or args.output.read_text() == payload, 'Output exists with differing content')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f'Wrote {len(result["comparisons"])} paired comparisons to {args.output}')
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
