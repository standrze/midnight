"""CPU scoring/provenance tests; optional Docker tests never load MLX."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]

def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), ROOT / 'Scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

evaluation = load('evaluate-generated')
builder = load('prepare-generated-eval-corpus')

class PreparationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='midnight-generated-test-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cache = self.root / 'cache'
        self.cache.mkdir()
        self.sources = copy.deepcopy(builder.pinned.SOURCES)
        self.inputs = {
            'mbpp': builder.payload([{'task_id': 11, 'text': 'Return the larger argument.', 'code': 'def greater(a, b):\n    return SECRET_REFERENCE_BODY', 'test_setup_code': '', 'test_list': ['assert greater(2, 5) == 5']}]),
            'gsm8k': builder.payload([{'question': 'How many pears remain?', 'answer': 'SECRET_REASONING\n#### 424242'}])}
        for source in self.sources:
            data = self.inputs[source['name']]
            source['source_sha256'] = hashlib.sha256(data).hexdigest()
            (self.cache / (source['name'] + '-source.jsonl')).write_bytes(data)

    def prepare(self, output=None):
        return builder.prepare(output or self.root / 'out', self.cache, 1, 1, (512,), sources=self.sources)

    def test_public_private_separation_and_deterministic_retrieval(self):
        self.prepare()
        raw = (self.root / 'out/tasks.jsonl').read_text()
        for secret in ('SECRET_', '424242', 'assert greater'):
            self.assertNotIn(secret, raw)
        self.assertIn('def greater(a, b):', raw)
        tasks = evaluation.read_jsonl(self.root / 'out/tasks.jsonl')
        answers = {a['id']: a for a in evaluation.read_jsonl(self.root / 'out/answers.jsonl')}
        retrieval = [t for t in tasks if t['category'] == 'retrieval']
        self.assertEqual(len(retrieval), 3)
        for task in retrieval:
            self.assertEqual(task['prompt'].count(answers[task['id']]['answer']), 1)
        result = self.prepare(self.root / 'other')
        self.assertEqual((self.root / 'other/tasks.jsonl').read_bytes(), (self.root / 'out/tasks.jsonl').read_bytes())
        self.assertEqual(result['task_count'], 5)
        self.assertEqual((self.root / 'out/answers.jsonl').stat().st_mode & 0o777, 0o600)

    def test_corrupt_source_conflict_and_matching_cache(self):
        self.prepare()
        before = {p.name: p.stat().st_mtime_ns for p in (self.root / 'out').iterdir()}
        self.prepare()
        self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in (self.root / 'out').iterdir()})
        protected = self.root / 'out/tasks.jsonl'
        protected.write_text('preserve existing work')
        with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
            self.prepare()
        self.assertEqual(protected.read_text(), 'preserve existing work')
        (self.cache / 'gsm8k-source.jsonl').write_text('bad source')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.prepare(self.root / 'different')
        self.assertFalse((self.root / 'different').exists())

class ScoringTests(unittest.TestCase):
    def test_math_requires_one_final_marker_and_exact_value(self):
        gold = {'kind': 'numeric', 'answer': '1,250'}
        for text in ('Work.\n#### 1250.0', '#### 2500/2'):
            self.assertTrue(evaluation.score_answer(text, gold)['passed'])
        for text in ('1250', '#### 1,25', '#### 1250\nMore work', '#### 1250\n#### 1250', '#### NaN', '#### 1250 dollars', '#### 1/0'):
            self.assertFalse(evaluation.score_answer(text, gold)['passed'], text)

    def test_retrieval_is_strict_and_host_never_executes_code(self):
        self.assertTrue(evaluation.score_answer(' Vabc\n', {'kind': 'exact', 'answer': 'Vabc'})['passed'])
        self.assertFalse(evaluation.score_answer('The value is Vabc.', {'kind': 'exact', 'answer': 'Vabc'})['passed'])
        with mock.patch.object(subprocess, 'Popen', side_effect=AssertionError('Host execution forbidden')):
            result = evaluation.score_answer("raise RuntimeError('must never execute')", {'kind': 'python-tests'})
        self.assertEqual(result['reason'], 'sandbox_not_configured')
        self.assertIsNone(result['passed'])

    def test_retrieval_diagnostics_do_not_change_primary_accuracy(self):
        key={'kind':'exact','answer':'Vba2951d8449b'}
        text='Vba2951d8449b\n</think>Vba2951d8449b'
        default=evaluation.score_answer(text,key)
        result=evaluation.score_answer(text,key,retrieval_diagnostics=True)
        self.assertFalse(default['passed']); self.assertFalse(result['passed'])
        self.assertNotIn('retrieval_diagnostic',default)
        diagnostic=result['retrieval_diagnostic']
        self.assertTrue(diagnostic['expected_value_present'])
        self.assertTrue(diagnostic['all_emitted_values_equal_expected'])
        self.assertEqual(diagnostic['emitted_value_count'],2)
        self.assertTrue(diagnostic['tag_containing'])
        conflict=evaluation.score_answer(text+' V000000000000',key,retrieval_diagnostics=True)
        self.assertFalse(conflict['retrieval_diagnostic']['all_emitted_values_equal_expected'])
        empty=evaluation.score_answer('No matching record',key,retrieval_diagnostics=True)
        self.assertFalse(empty['retrieval_diagnostic']['expected_value_present'])
        self.assertIsNone(empty['retrieval_diagnostic']['all_emitted_values_equal_expected'])
        self.assertFalse(empty['retrieval_diagnostic']['tag_containing'])

    def test_code_fences_empty_and_syntax(self):
        self.assertEqual(evaluation.extract_code('```python\ndef add(a,b): return a+b\n```'), 'def add(a,b): return a+b')
        for text in ('', '```python\n\n```', '```python\ndef f(): pass', '```python\na=1\n```\n```python\na=2\n```', 'Explanation\n```python\na=1\n```', 'def bad(:'):
            with self.assertRaises(ValueError, msg=text):
                evaluation.extract_code(text)

    def test_docker_isolation_policy_and_digest(self):
        args = evaluation.docker_arguments('/usr/bin/docker', 'python@sha256:' + 'a'*64, '/tmp/single-input', 'unique-name')
        for option, value in (('--pull', 'never'), ('--network', 'none'), ('--cap-drop', 'ALL'), ('--user', '65534:65534'), ('--memory', '256m'), ('--pids-limit', '64')):
            self.assertEqual(args[args.index(option)+1], value)
        self.assertIn('--read-only', args)
        self.assertIn('no-new-privileges', args)
        self.assertEqual(args[args.index('--mount')+1], 'type=bind,src=/tmp/single-input,dst=/work,readonly')
        self.assertNotIn('--privileged', args)
        self.assertNotIn('/var/run/docker.sock', ' '.join(args))
        with self.assertRaisesRegex(ValueError, 'immutable'):
            evaluation.DockerSandbox('python:latest')

    def test_trusted_command_timeout_and_output_limit(self):
        timeout = evaluation.bounded_command([sys.executable, '-c', 'import time; time.sleep(5)'], .1)
        self.assertEqual(timeout['failure'], 'timeout')
        flood = evaluation.bounded_command([sys.executable, '-c', "print('x'*100000)"], 5, output_limit=1024)
        self.assertEqual(flood['failure'], 'output_limit')
        self.assertEqual(len(flood['output']), 1024)

class ReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='midnight-report-test-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.tasks = [{'id': 'm1', 'category': 'math', 'prompt': 'Question one'}, {'id': 'r1', 'category': 'retrieval', 'prompt': 'Find key.'}]
        self.keys = [{'id': 'm1', 'kind': 'numeric', 'answer': '2'}, {'id': 'r1', 'kind': 'exact', 'answer': 'Vabcd'}]
        self.taskfile, self.keyfile = self.root/'tasks.jsonl', self.root/'answers.jsonl'
        self.taskfile.write_bytes(builder.payload(self.tasks))
        self.keyfile.write_bytes(builder.payload(self.keys))
        self.report = {'format': 1, 'benchmark': 'native_greedy_generation', 'input_sample_count': 2, 'completed_sample_count': 2, 'output_limit_reached_count': 0, 'engine': 'metal', 'prompt_format': 'single_user_message_checkpoint_chat_template', 'prompt_cache': False, 'speculative_decoding': False, 'fused_gate_up_silu': False, 'compiled_laguna_block_tail': True, 'fused_laguna_router_top_k': True, 'runtime_environment': {}, 'memory_limit_bytes': 10000000, 'status': 'completed', 'model_path': '/models/example', 'requested_tokens': 128, 'context_length': 4096, 'prefill_step_size': 512, 'kv_compression': 'none', 'temperature': 0, 'top_p': 1, 'corpus_sha256': evaluation.file_info(self.taskfile)['sha256'],
                       'samples': [{'id': t['id'], 'category': t['category'], 'status': 'completed', 'generated_text': text, 'prompt_token_id_fingerprint': 'fnv1a64:'+str(i), 'prompt_sha256': hashlib.sha256(t['prompt'].encode()).hexdigest(), 'prompt_truncated': False, 'prompt_token_count': 10+i, 'generation_token_count': 3, 'stop_reason': 'stop', 'output_limit_reached': False} for i,(t,text) in enumerate(zip(self.tasks,['#### 2','Vabcd']))]}

    def evaluate(self, modified=None):
        a,b = self.root/'a.json',self.root/'b.json'
        a.write_text(json.dumps(self.report)); b.write_text(json.dumps(modified or self.report))
        return evaluation.evaluate(self.taskfile,self.keyfile,{'standard':a,'candidate':b},draws=100)

    def test_complete_accuracy_paired_delta_and_provenance(self):
        other = copy.deepcopy(self.report); other['samples'][0]['generated_text']='#### 3'
        result = self.evaluate(other)
        self.assertEqual(result['models']['candidate']['overall']['accuracy'],.5)
        paired = result['comparisons']['candidate_minus_standard']['overall']
        self.assertTrue(paired['qualified']); self.assertEqual(paired['accuracy_delta'],-.5)
        self.assertEqual(paired['baseline_only_correct'],1)
        self.assertEqual(result['provenance']['private_answers']['sha256'],evaluation.file_info(self.keyfile)['sha256'])

    def test_partial_generation_withholds_complete_accuracy(self):
        other=copy.deepcopy(self.report); other['samples'].pop(); other.update(status='running',completed_sample_count=1)
        result=self.evaluate(other)
        self.assertIsNone(result['models']['candidate']['overall']['accuracy'])
        self.assertEqual(result['models']['candidate']['overall']['tasks'],2)
        self.assertFalse(result['comparisons']['candidate_minus_standard']['overall']['qualified'])

    def test_token_settings_and_length_stop_gates(self):
        other=copy.deepcopy(self.report); other['samples'][0].pop('prompt_token_id_fingerprint')
        self.assertFalse(self.evaluate(other)['comparisons']['candidate_minus_standard']['overall']['qualified'])
        other=copy.deepcopy(self.report); other['requested_tokens']=64
        self.assertIn('unverified_or_different_requested_tokens',self.evaluate(other)['comparisons']['candidate_minus_standard']['overall']['reasons'])
        other=copy.deepcopy(self.report); other['samples'][0].update(generated_text='unfinished',output_limit_reached=True,generation_token_count=128,stop_reason='length'); other['output_limit_reached_count']=1
        result=self.evaluate(other)
        self.assertEqual(result['models']['candidate']['categories']['math']['accuracy'],0)
        self.assertEqual(result['models']['candidate']['overall']['output_limit_reached'],1)

    def test_corpus_duplicates_and_public_answers_rejected(self):
        other=copy.deepcopy(self.report); other['corpus_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'corpus SHA256'): self.evaluate(other)
        other=copy.deepcopy(self.report); other['samples'].append(other['samples'][0])
        with self.assertRaisesRegex(ValueError,'duplicate'): self.evaluate(other)
        self.tasks[0]['answer']='2'; self.taskfile.write_bytes(builder.payload(self.tasks))
        with self.assertRaisesRegex(ValueError,'private key'): evaluation.load_inputs(self.taskfile,self.keyfile)


    def test_execution_differences_strict_by_default_explicit_kv_experiment(self):
        other=copy.deepcopy(self.report); other['kv_compression']='affine8'
        result=self.evaluate(other)
        self.assertFalse(result['comparisons']['candidate_minus_standard']['overall']['qualified'])
        left,right=result['models']['standard'],result['models']['candidate']
        comparison=evaluation.paired_comparison(left,right,100,1,('kv_compression',))['overall']
        self.assertTrue(comparison['qualified'])
        self.assertEqual(comparison['comparison_kind'],'runtime_setting_experiment')
        self.assertEqual(comparison['setting_differences']['kv_compression'],{'baseline':'none','candidate':'affine8'})
        right['settings']['prefill_step_size']=2048
        self.assertFalse(evaluation.paired_comparison(left,right,100,1,('kv_compression',))['overall']['qualified'])

    def test_counter_and_prompt_hash_mismatches_rejected(self):
        other=copy.deepcopy(self.report); other['completed_sample_count']=1
        with self.assertRaisesRegex(ValueError,'counter'): self.evaluate(other)
        other=copy.deepcopy(self.report); other['samples'][0]['prompt_sha256']='a'*64
        with self.assertRaisesRegex(ValueError,'Prompt SHA256'): self.evaluate(other)
        other=copy.deepcopy(self.report); other['samples'][0]['prompt_truncated']=True
        with self.assertRaisesRegex(ValueError,'Full prompt'): self.evaluate(other)

@unittest.skipUnless(os.environ.get('MIDNIGHT_EVAL_SANDBOX_IMAGE'),'Set pinned image to run real Docker isolation checks')
class DockerIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sandbox=evaluation.DockerSandbox(os.environ['MIDNIGHT_EVAL_SANDBOX_IMAGE'],timeout=6)

    def score(self,code):
        return self.sandbox.score(code,{'kind':'python-tests','setup':'','tests':['assert f() is True']})

    def test_correct_incorrect_and_exit_spoof(self):
        self.assertTrue(self.score('def f(): return True')['passed'])
        self.assertFalse(self.score('def f(): return False')['passed'])
        self.assertFalse(self.score('import os; os._exit(0)')['passed'])

    def test_host_files_network_and_read_only_paths(self):
        code='''import os, socket
def f():
    assert os.geteuid()==65534
    assert not os.path.exists('/Users')
    assert not os.path.exists('/var/run/docker.sock')
    for path in ['/root/escape','/work/candidate.py','/etc/midnight-escape']:
        try: open(path,'w').write('escape')
        except OSError: pass
        else: return False
    try: socket.create_connection(('1.1.1.1',80),timeout=.2)
    except OSError: return True
    return False
'''
        self.assertTrue(self.score(code)['passed'])

    def test_hang_and_output_flood(self):
        hang=self.score('while True: pass')
        self.assertFalse(hang['passed']); self.assertLess(hang['sandbox']['elapsed_seconds'],10)
        flood=self.score("while True: print('x'*8192,flush=True)")
        self.assertFalse(flood['passed']); self.assertEqual(flood['reason'],'output_limit')

if __name__=='__main__': unittest.main()
