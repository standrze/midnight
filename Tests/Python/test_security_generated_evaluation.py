"""Structured cyber grading tests; never execute task source or model output."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('security_generated_evaluation', ROOT / 'Scripts/evaluate-generated.py')
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class SecurityGeneratedEvaluationTests(unittest.TestCase):
    vulnerable = {'kind': 'security-json', 'verdict': 'vulnerable', 'cwes': ['CWE-89'],
                  'evidence_lines': [3, 4], 'line_count': 5}
    safe = {'kind': 'security-json', 'verdict': 'safe', 'cwes': [], 'evidence_lines': [], 'line_count': 5}

    def response(self, verdict='vulnerable', cwes=None, lines=None):
        return json.dumps({'verdict': verdict, 'cwes': ['CWE-89'] if cwes is None else cwes,
                           'evidence_lines': [4] if lines is None else lines})

    def test_verdict_cwe_and_evidence_all_matter(self):
        self.assertTrue(evaluation.score_answer(self.response(), self.vulnerable)['passed'])
        self.assertTrue(evaluation.score_answer('```json\n' + self.response() + '\n```', self.vulnerable)['passed'])
        for text in [self.response(cwes=['CWE-78']), self.response(lines=[1]),
                     self.response(lines=[1, 4]), self.response('safe', [], [])]:
            self.assertFalse(evaluation.score_answer(text, self.vulnerable)['passed'], text)
        self.assertTrue(evaluation.score_answer(self.response('safe', [], []), self.safe)['passed'])
        false_positive = evaluation.score_answer(self.response(), self.safe)
        self.assertFalse(false_positive['passed'])
        self.assertEqual(false_positive['security_diagnostic']['predicted_verdict'], 'vulnerable')

    def test_ambiguous_invalid_and_fabricated_responses_fail(self):
        for text in ['', '{}', '[]', self.response() + ' explanation',
                     self.response(lines=[True]), self.response(lines=[0]),
                     self.response(lines=[6]), self.response(lines=[4, 4]),
                     self.response(cwes=['CWE-89', 'CWE-89']),
                     '{"verdict":"safe","verdict":"vulnerable","cwes":["CWE-89"],"evidence_lines":[4]}',
                     '{"verdict":"vulnerable","cwes":["CWE-89"],"evidence_lines":[NaN]}',
                     '```json\n' + self.response() + '\n```\n```json\n{}\n```',
                     'Explanation\n```json\n' + self.response() + '\n```']:
            result = evaluation.score_answer(text, self.vulnerable)
            self.assertFalse(result['passed'], text)
            self.assertFalse(result['security_diagnostic']['format_valid'], text)

    def test_operation_anchor_prevents_source_only_evidence_credit(self):
        key = {**self.vulnerable, 'evidence_anchor_lines': [4]}
        self.assertFalse(evaluation.score_answer(self.response(lines=[3]), key)['passed'])
        self.assertTrue(evaluation.score_answer(self.response(lines=[3, 4]), key)['passed'])
        self.assertTrue(evaluation.score_answer(self.response(lines=[3]), self.vulnerable)['passed'])
        for anchors in [[], [True], [1], [4, 4], '4']:
            with self.assertRaises(ValueError):
                evaluation.validate_security_key({**key, 'evidence_anchor_lines': anchors})
        evaluation.validate_security_key({**self.safe, 'evidence_anchor_lines': []})
        with self.assertRaises(ValueError):
            evaluation.validate_security_key({**self.safe, 'evidence_anchor_lines': [4]})

    def test_no_source_or_response_execution_and_malformed_key_rejected(self):
        with mock.patch.object(subprocess, 'Popen', side_effect=AssertionError('Execution forbidden')):
            self.assertTrue(evaluation.score_answer(self.response(), self.vulnerable)['passed'])
        for key in [{**self.vulnerable, 'line_count': False}, {**self.vulnerable, 'evidence_lines': []},
                    {**self.vulnerable, 'cwes': []}, {**self.safe, 'cwes': ['CWE-89']}]:
            with self.assertRaises(ValueError):
                evaluation.validate_security_key(key)

    def test_precision_recall_and_false_positives_retain_invalid_outputs(self):
        records = []
        for text, key in [(self.response(), self.vulnerable), (self.response(), self.safe),
                          (self.response('safe', [], []), self.safe), ('{}', self.vulnerable)]:
            records.append(evaluation.score_answer(text, key))
        result = evaluation.summarize(records)
        self.assertEqual(result['accuracy'], 0.5)
        metrics = result['security_detection']
        self.assertEqual(metrics['precision'], 0.5)
        self.assertEqual(metrics['recall_including_invalid'], 0.5)
        self.assertEqual(metrics['false_positive_rate'], 0.5)
        self.assertEqual(metrics['invalid_response_count'], 1)
        self.assertTrue(metrics['complete'])
        records.append({'status': 'unscored', 'passed': None, 'category': 'cybersecurity'})
        incomplete = evaluation.summarize(records)
        self.assertIsNone(incomplete['accuracy'])
        self.assertFalse(incomplete['security_detection']['complete'])

    def test_mixed_category_security_completeness_uses_its_own_denominator(self):
        cyber = evaluation.score_answer(self.response(), self.vulnerable)
        cyber['category'] = 'cybersecurity'
        mixed = evaluation.summarize([cyber, {'category': 'code', 'status': 'scored', 'passed': True}])
        self.assertTrue(mixed['security_detection']['complete'])
        self.assertEqual(mixed['security_detection']['tasks'], 1)

    def test_zero_discordance_retains_finite_sample_uncertainty_and_family_dependence(self):
        a = {str(i): {'category': 'cybersecurity', 'passed': True,
                      'metadata': {'family': str(i // 2)}} for i in range(12)}
        bound = evaluation.conservative_paired_bound(a, a, list(a))
        self.assertEqual(bound['independent_unit_count'], 6)
        self.assertLess(bound['interval'][0], 0)
        self.assertGreater(bound['interval'][1], 0)
        self.assertFalse(bound['promotion_gate'])
        independent = {i: {**r, 'metadata': {}} for i, r in a.items()}
        self.assertLess(evaluation.conservative_paired_bound(independent, independent, list(a))['interval'][1],
                        bound['interval'][1])
        larger = {str(i): {'category': 'code', 'passed': True} for i in range(1000)}
        self.assertLess(evaluation.conservative_paired_bound(larger, larger, list(larger))['interval'][1], 0.1)

    def test_public_private_schema_admits_cyber_keys_and_rejects_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {'id': 'cyber1', 'category': 'cybersecurity', 'prompt': 'Review numbered source.'}
            key = {'id': 'cyber1', **self.vulnerable}
            tasks, answers = root / 'tasks.jsonl', root / 'answers.jsonl'
            tasks.write_text(json.dumps(task) + '\n')
            answers.write_text(json.dumps(key) + '\n')
            actual, keys = evaluation.load_inputs(tasks, answers)
            self.assertEqual(actual[0]['category'], 'cybersecurity')
            self.assertEqual(keys['cyber1']['kind'], 'security-json')
            tasks.write_text(json.dumps({**task, 'answer': key}) + '\n')
            with self.assertRaisesRegex(ValueError, 'private key'):
                evaluation.load_inputs(tasks, answers)


if __name__ == '__main__':
    unittest.main()
