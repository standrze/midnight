#!/usr/bin/env python3
"""Reproduce the paired long-context prefill-NLL comparison without inference."""
import hashlib
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
INPUTS = {
    'laguna-long-nll-512.json': '7ca1d57269b06e866939e9211b9bac88d90e1511b931da19c65c2f3b9b283fcc',
    'laguna-long-nll-2048.json': 'eb2bb68453143cb368e7b3830d45f9f552b2ce1ff638c8ee5f0fe6ca7e67455b',
}


def checked_reports():
    reports = []
    for name, expected in INPUTS.items():
        raw = (ROOT / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f'Native report content hash mismatch: {name}')
        report = json.loads(raw)
        if report.get('status') != 'measured' or report.get('metric') != 'teacher_forced_next_token_nll':
            raise ValueError(f'Unexpected native report: {name}')
        samples = report['samples']
        if len(samples) != 8 or report['sample_count'] != len(samples):
            raise ValueError('Expected all eight retained samples')
        if len({s['id'] for s in samples}) != len(samples):
            raise ValueError('Duplicate sample IDs')
        for sample in samples:
            n = sample['scored_token_count']
            if n <= 0 or sample['evaluated_token_count'] != n + 1 or sample['truncated']:
                raise ValueError('Incomplete or invalid sample token counts')
            if sample['evaluated_token_count'] != sample['original_token_count']:
                raise ValueError('Sample was truncated')
            if not math.isfinite(sample['nll_sum']) or sample['nll_sum'] < 0:
                raise ValueError('Invalid sample NLL')
            if not math.isclose(sample['nll_sum'] / n, sample['nll'], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Sample NLL does not reconcile')
        total_tokens = sum(s['scored_token_count'] for s in samples)
        total_nll = sum(s['nll_sum'] for s in samples)
        if total_tokens != report['scored_token_count'] or not math.isclose(total_nll, report['nll_sum'], rel_tol=1e-12):
            raise ValueError('Report totals do not reconcile with samples')
        if not math.isclose(total_nll / total_tokens, report['token_weighted_nll'], rel_tol=1e-12):
            raise ValueError('Report weighted NLL does not reconcile')
        reports.append(report)
    base, candidate = reports
    for field in ('model_path', 'model_type', 'corpus_fingerprint', 'token_id_fingerprint', 'device',
                  'backend', 'add_special_tokens', 'maximum_tokens_per_sample', 'scored_token_count'):
        if base.get(field) is None or base[field] != candidate[field]:
            raise ValueError(f'Incomparable report field: {field}')
    if (base['prefill_step_size'], candidate['prefill_step_size']) != (512, 2048):
        raise ValueError('Unexpected chunk sizes')
    for a, b in zip(base['samples'], candidate['samples']):
        for field in ('id', 'category', 'token_id_fingerprint', 'original_token_count',
                      'evaluated_token_count', 'scored_token_count', 'truncated'):
            if a.get(field) is None or a[field] != b[field]:
                raise ValueError(f'Incomparable paired sample field: {field}')
    return base, candidate


def analyze():
    base, candidate = checked_reports()
    pairs = list(zip(base['samples'], candidate['samples']))
    rng = random.Random(20260904)
    draws = []
    for _ in range(10000):
        sample = rng.choices(pairs, k=len(pairs))
        tokens = sum(a['scored_token_count'] for a, _ in sample)
        draws.append(sum(b['nll_sum'] - a['nll_sum'] for a, b in sample) / tokens)
    draws.sort()
    delta = (candidate['nll_sum'] - base['nll_sum']) / base['scored_token_count']
    return {
        'format': 1, 'status': 'analyzed', 'native_reports_sha256': INPUTS,
        'sample_count': len(pairs), 'scored_token_count': base['scored_token_count'],
        'corpus_fingerprint': base['corpus_fingerprint'], 'token_id_fingerprint': base['token_id_fingerprint'],
        'all_samples_untruncated': True, 'all_sample_identities_and_lengths_match': True,
        'baseline_prefill_step_size': 512, 'candidate_prefill_step_size': 2048,
        'baseline_token_weighted_nll': base['token_weighted_nll'],
        'candidate_token_weighted_nll': candidate['token_weighted_nll'],
        'candidate_minus_baseline_nll': delta,
        'paired_bootstrap_95_percent_interval': [draws[249], draws[9749]],
        'bootstrap': {'seed': 20260904, 'draws': 10000, 'method': 'Resample eight paired records with replacement, then recompute token-weighted NLL difference; percentile order statistics 249 and 9749 (zero-based).'},
        'candidate_lower_nll_samples': sum(b['nll'] < a['nll'] for a, b in pairs),
        'candidate_higher_nll_samples': sum(b['nll'] > a['nll'] for a, b in pairs),
        'baseline_mlx_peak_memory_bytes': base['mlx_peak_memory_bytes'],
        'candidate_mlx_peak_memory_bytes': candidate['mlx_peak_memory_bytes'],
        'candidate_minus_baseline_mlx_peak_memory_bytes': candidate['mlx_peak_memory_bytes'] - base['mlx_peak_memory_bytes'],
        'samples': [{'id': a['id'], 'scored_token_count': a['scored_token_count'],
                     'baseline_nll': a['nll'], 'candidate_nll': b['nll'], 'nll_delta': b['nll'] - a['nll']}
                    for a, b in pairs],
        'conclusion': 'The paired interval spans zero. This convenience subset does not establish improved reference NLL, noninferiority, generated-task accuracy or a production-default choice.',
        'limitations': [
            'One run per setting, eight paired convenience records from a fixed prose source; correlated material and arithmetic/run variation are not captured by this bootstrap.',
            'No formal noninferiority margin was specified, and absence of a significant difference is not proof of equivalence.',
            'Reference NLL is teacher-forced likelihood, not generated-task accuracy or exact greedy-output preservation.',
            'No inference-time comparison: CPU/build activity overlapped the runs. Native elapsed times are retained only in raw reports.',
            'Reported MLX peak memory is neither process maximum RSS nor whole-system footprint.',
            'Report model_path and token fingerprints do not independently hash all model weights; consult the retained runtime campaign model provenance and post-run artifact snapshot.',
        ],
    }


if __name__ == '__main__':
    result = analyze()
    (ROOT / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('candidate_minus_baseline_nll', 'paired_bootstrap_95_percent_interval', 'conclusion')}, indent=2))
