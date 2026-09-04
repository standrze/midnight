"""CPU-only checks for paired native-quality comparisons and provenance gates."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'Scripts/analyze-quality-nll.py'
spec = importlib.util.spec_from_file_location('analyze_quality_nll', SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture(nlls=(1.0, 2.0, 3.0, 5.0), counts=(10, 20, 2, 4)):
    samples = [{'id': f'sample-{i}', 'category': 'prose' if i < 2 else 'code',
                'token_id_fingerprint': f'tokens-{i}', 'original_token_count': n + 1,
                'evaluated_token_count': n + 1, 'scored_token_count': n, 'truncated': False,
                'nll': nll, 'nll_sum': nll * n} for i, (nll, n) in enumerate(zip(nlls, counts))]
    nll_sum = sum(s['nll_sum'] for s in samples)
    return {'format': 1, 'status': 'measured', 'metric': 'teacher_forced_next_token_nll',
            'corpus_fingerprint': 'corpus', 'token_id_fingerprint': 'combined-tokens',
            'backend': 'metal', 'device': 'gpu', 'model_type': 'laguna', 'model_path': '/fixture/model',
            'add_special_tokens': True, 'maximum_tokens_per_sample': 2048, 'prefill_step_size': 512,
            'samples': samples, 'sample_count': len(samples), 'scored_token_count': sum(counts),
            'nll_sum': nll_sum, 'token_weighted_nll': nll_sum / sum(counts)}


class PairedQualityAnalysisTests(unittest.TestCase):
    def test_three_reports_produce_all_pairs_and_weighted_totals(self):
        result = analysis.analyze({'standard': fixture(), 'ls2': fixture((.9, 1.9, 2.9, 4.9)),
                                   'awss': fixture((1.05, 2.05, 2.5, 4.5))}, draws=100)
        self.assertEqual(len(result['comparisons']), 3)
        self.assertAlmostEqual(result['models']['standard']['token_weighted_nll'], 76 / 36)
        comparison = result['comparisons'][0]
        self.assertEqual((comparison['baseline'], comparison['candidate']), ('standard', 'ls2'))
        self.assertAlmostEqual(comparison['overall']['candidate_minus_baseline_nll'], -.1)
        self.assertEqual(comparison['overall']['candidate_lower_nll_samples'], 4)
        self.assertAlmostEqual(comparison['categories']['code']['candidate_minus_baseline_nll'], -.1)

    def test_overall_bootstrap_preserves_category_counts(self):
        result = analysis.analyze({'a': fixture((2, 2, 2, 2), (1, 1, 1, 1)),
                                   'b': fixture((1, 1, 3, 3), (1, 1, 1, 1))}, draws=100)
        comparison = result['comparisons'][0]
        self.assertEqual(comparison['overall']['paired_bootstrap_95_percent_interval'], [0, 0])
        self.assertEqual(comparison['categories']['prose']['paired_bootstrap_95_percent_interval'], [-1, -1])
        self.assertEqual(comparison['categories']['code']['paired_bootstrap_95_percent_interval'], [1, 1])

    def test_identity_category_setting_and_length_changes_fail_closed(self):
        for field, value in [('token_id_fingerprint', 'changed'), ('category', 'changed'), ('id', 'changed')]:
            candidate = fixture()
            candidate['samples'][0][field] = value
            with self.assertRaisesRegex(ValueError, 'paired sample'):
                analysis.analyze({'a': fixture(), 'b': candidate}, draws=100)
        for field, value in [('prefill_step_size', 2048), ('device', 'cpu'), ('corpus_fingerprint', 'changed')]:
            candidate = fixture()
            candidate[field] = value
            with self.assertRaisesRegex(ValueError, 'setting differs'):
                analysis.analyze({'a': fixture(), 'b': candidate}, draws=100)
        with self.assertRaises(ValueError):
            analysis.analyze({'a': fixture(), 'b': fixture(counts=(11, 20, 2, 4))}, draws=100)

    def test_nonfinite_inconsistent_or_duplicate_samples_are_rejected(self):
        for corruption in ('nan', 'total', 'duplicate', 'truncation', 'missing_setting'):
            report = fixture()
            if corruption == 'nan':
                report['samples'][0]['nll_sum'] = float('nan')
            elif corruption == 'total':
                report['nll_sum'] += 1
            elif corruption == 'duplicate':
                report['samples'][0]['id'] = report['samples'][1]['id']
            elif corruption == 'truncation':
                report['samples'][0]['truncated'] = True
            else:
                del report['prefill_step_size']
            with self.assertRaises(ValueError):
                analysis.analyze({'a': fixture(), 'b': report}, draws=100)

    def test_cli_hashes_inputs_and_preserves_differing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = []
            for label in ('standard', 'ls2', 'awss'):
                path = root / f'{label}.json'
                path.write_text(json.dumps(fixture()))
                reports.extend(['--report', f'{label}={path}'])
            output = root / 'analysis.json'
            args = [*reports, '--output', str(output), '--draws', '100']
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(analysis.main(args), 0)
                result = json.loads(output.read_text())
                self.assertEqual(result['provenance']['native_reports']['standard']['sha256'],
                                 hashlib.sha256((root / 'standard.json').read_bytes()).hexdigest())
                self.assertEqual(analysis.main(args), 0)
                output.write_text('preserve')
                self.assertEqual(analysis.main(args), 2)
            self.assertEqual(output.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
