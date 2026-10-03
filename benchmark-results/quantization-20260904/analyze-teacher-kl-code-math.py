#!/usr/bin/env python3
"""Paired sample bootstrap for this fixed teacher-forced code/math experiment."""
from pathlib import Path
import hashlib
import json
import math
import random
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'teacher-kl-code-math.json'
DRAWS = 10000
SEED = 20260904
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == '44aee3f9b6265f1098c814370b5274115b1fe2e82beaab473bbc9f0299af8280', 'This analyzer is pinned to the archived native report'
report = json.loads(SOURCE.read_text())
assert report['status'] == 'measured'
assert report['metric'] == 'teacher_forced_full_vocab_kl_teacher_to_student'
assert report['kl_direction'] == 'KL(teacher || student)'
teacher = {row['id']: row for row in report['teacher']['samples']}
assert len(teacher) == 128 == report['sample_count']
assert sum(row['scored_token_count'] for row in teacher.values()) == report['scored_token_count']
students = {}
for entry in report['students']:
    path = entry['model_path']
    label = 'awss' if '-AWSS1-' in path else ('ls2' if '-ScaleSearch-LS2-' in path else 'standard')
    assert label not in students
    assert entry['token_id_fingerprint'] == report['token_id_fingerprint']
    rows = {row['id']: row for row in entry['samples']}
    assert rows.keys() == teacher.keys()
    for ident, row in rows.items():
        ref = teacher[ident]
        assert all(row[key] == ref[key] for key in ('category', 'token_id_fingerprint', 'scored_token_count'))
        assert math.isclose(row['teacher_nll'], ref['nll'], rel_tol=1e-9, abs_tol=1e-9)
        for key in ('teacher_kl_sum', 'student_nll'):
            assert math.isfinite(row[key]) and row[key] >= 0
    n = sum(row['scored_token_count'] for row in rows.values())
    assert math.isclose(sum(row['teacher_kl_sum'] for row in rows.values()) / n, entry['token_weighted_teacher_kl'], rel_tol=1e-9)
    assert math.isclose(sum(row['student_nll'] * row['scored_token_count'] for row in rows.values()) / n, entry['summary']['token_weighted_nll'], rel_tol=1e-9)
    students[label] = {'entry': entry, 'rows': rows}
assert set(students) == {'standard', 'ls2', 'awss'}


def quantile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def weighted_summary(ids, label):
    rows = [students[label]['rows'][i] for i in ids]
    count = sum(row['scored_token_count'] for row in rows)
    nll = sum(row['student_nll'] * row['scored_token_count'] for row in rows) / count
    return {'teacher_kl': sum(row['teacher_kl_sum'] for row in rows) / count,
            'reference_nll': nll, 'reference_perplexity': math.exp(nll),
            'reference_top1_accuracy': sum(row['student_ground_truth_top1_correct_count'] for row in rows) / count,
            'teacher_top1_agreement': sum(row['teacher_student_top1_agreement_count'] for row in rows) / count}


def comparison(ids, baseline, candidate, category):
    a = [students[baseline]['rows'][i] for i in ids]
    b = [students[candidate]['rows'][i] for i in ids]
    counts = [row['scored_token_count'] for row in a]
    kl_a = [row['teacher_kl_sum'] for row in a]
    kl_b = [row['teacher_kl_sum'] for row in b]
    nll_delta_sums = [(cb['student_nll'] - ca['student_nll']) * n for ca, cb, n in zip(a, b, counts)]
    top1_delta_counts = [cb['student_ground_truth_top1_correct_count'] - ca['student_ground_truth_top1_correct_count'] for ca, cb in zip(a, b)]
    def estimate(indices):
        n = sum(counts[i] for i in indices)
        ka, kb = sum(kl_a[i] for i in indices), sum(kl_b[i] for i in indices)
        return ((kb-ka)/n, 100*(1-kb/ka), sum(nll_delta_sums[i] for i in indices)/n,
                100*sum(top1_delta_counts[i] for i in indices)/n)
    names = ('teacher_kl_delta', 'teacher_kl_reduction_percent', 'reference_nll_delta', 'reference_top1_delta_percentage_points')
    point = estimate(range(len(ids)))
    # Retain the observed 64/64 category composition in the overall bootstrap.
    groups = [[i for i, ident in enumerate(ids) if teacher[ident]['category'] == group]
              for group in sorted({teacher[ident]['category'] for ident in ids})]
    rng = random.Random(SEED)
    distributions = [[] for _ in names]
    for _ in range(DRAWS):
        indices = [i for group in groups for i in rng.choices(group, k=len(group))]
        for values, value in zip(distributions, estimate(indices)):
            values.append(value)
    return {'baseline': baseline, 'candidate': candidate,
            'metrics': {key: {'estimate': value, 'paired_sample_bootstrap_95_percent_interval': [quantile(dist, .025), quantile(dist, .975)]}
                        for key, value, dist in zip(names, point, distributions)},
            'samples_with_lower_teacher_kl': sum(cb['teacher_kl'] < ca['teacher_kl'] for ca, cb in zip(a,b)),
            'samples_with_lower_reference_nll': sum(cb['student_nll'] < ca['student_nll'] for ca, cb in zip(a,b))}

results = {}
for category in ('all', 'code-reference', 'math-reference'):
    ids = [ident for ident, row in teacher.items() if category == 'all' or row['category'] == category]
    count = sum(teacher[ident]['scored_token_count'] for ident in ids)
    results[category] = {'sample_count': len(ids), 'scored_token_count': count,
        'truncated_samples': sum(teacher[ident]['truncated'] for ident in ids),
        'teacher_reference_nll': sum(teacher[ident]['nll_sum'] for ident in ids) / count,
        'students': {label: weighted_summary(ids, label) for label in students},
        'comparisons': [comparison(ids, baseline, candidate, category)
                        for baseline, candidate in (('standard','ls2'),('standard','awss'),('ls2','awss'))]}
output = {'version':1, 'created_at':datetime.now(timezone.utc).isoformat(),
    'source_report':str(SOURCE), 'source_report_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'corpus_sha256':'5fc84a9794338a138e7284f415018323aa6bfec54227d09db8a95ad562e532f6',
    'corpus_hash_provenance':'Exact payload verified with Scripts/prepare-quantization-corpus.py; source corpora are not stored in this result archive.',
    'token_id_fingerprint':report['token_id_fingerprint'],
    'validation':'All 128 sample IDs, categories, token fingerprints and scored token counts match across teacher and three students; report totals reconcile.',
    'method':{'bootstrap_draws':DRAWS, 'seed':SEED, 'resampling_unit':'paired reference sample, with replacement; overall resampling stratified by category',
              'weighting':'Every point and bootstrap estimate uses summed token metrics / summed scored tokens.',
              'interval':'Percentile 95%; no multiplicity correction.'},
    'limitations':['Fixed convenience subsets (64 code and 64 math references), not a random sample of all tasks; bootstrap intervals describe observed sample variation and are not population accuracy guarantees.',
                   'One deterministic teacher-forced evaluation; intervals do not measure run-to-run hardware/numerical variability.',
                   'Scores cover full Task/Question plus Solution/Answer strings, not answer spans alone.',
                   'Lower teacher KL means closer BF16 token distributions. Lower reference NLL means higher likelihood of supplied reference text; neither establishes generated-code pass@1 or generated math accuracy.',
                   'Tokenwise reference top-1 accuracy is teacher-forced next-token accuracy, not task completion accuracy.'],
    'results':results}
path = ROOT/'teacher-kl-code-math-analysis.json'
path.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
for category, result in results.items():
    print(category, result['sample_count'], 'samples,', result['scored_token_count'],'tokens')
    for comparison in result['comparisons']:
        print(comparison['candidate'],'vs',comparison['baseline'],json.dumps(comparison['metrics']))
print(path)
