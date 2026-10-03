import copy
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest

import compare
import fixture_server


class AnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {'blocks':4, 'variants':[{'name':'baseline'},{'name':'candidate'}], 'workloads':['short'],
                         'gate':{'latency_tolerance':.03,'minimum_pairs':4,'drift_tolerance':.1}}
        self.rows = []
        for block in range(4):
            for variant in ('baseline','candidate'):
                for workload, metric in [('startup','process_cold_ready_ms'),('memory','sampled_peak_rss_kib'),('short','ttft_ms')]:
                    row = {'block':block,'variant':variant,'workload':workload,'stage':'measurement','ok':True, metric:100,
                           'content':'ORBIT731','quality_ok':True,'input_sha256':'same','input_tokens':10,'output_tokens':2,'finish':'stop'}
                    if workload=='short': row['total_ms']=150
                    self.rows.append(row)

    def result(self):
        return compare.analyze(self.rows,self.manifest)

    def test_stable_neutral_and_explicit_gate(self):
        self.assertEqual(self.result()['verdict'],'nonregression')

    def test_regression(self):
        for row in self.rows:
            if row['variant']=='candidate' and row['workload']=='short':row['ttft_ms']*=1.1
        self.assertEqual(self.result()['verdict'],'regression')

    def test_noisy_is_not_pass(self):
        factors = [.8,1.2,1.2,.8]
        for row in self.rows:
            if row['variant']=='candidate' and row['workload']=='short':row['ttft_ms']*=factors[row['block']]
        self.assertEqual(self.result()['verdict'],'inconclusive')

    def test_drift_is_not_pass(self):
        for row in self.rows:
            if row['workload']=='short' and row['block']>=2:row['ttft_ms']*=1.3
        self.assertEqual(self.result()['verdict'],'inconclusive')

    def test_missing_entire_workload_cannot_pass(self):
        self.manifest['workloads'].append('decode')
        self.assertEqual(self.result()['verdict'],'inconclusive')

    def test_duplicate_cannot_pass(self):
        self.rows.append(copy.deepcopy(self.rows[-1]))
        self.assertEqual(self.result()['verdict'],'inconclusive')

    def test_insufficient_blocks(self):
        self.rows = [r for r in self.rows if r['block']<3]
        self.assertEqual(self.result()['verdict'],'inconclusive')

    def test_nonfinite_cannot_pass(self):
        self.rows[-1]['ttft_ms']=float('nan')
        self.assertEqual(self.result()['verdict'],'inconclusive')

    def test_cache_usage_is_not_output_parity(self):
        for row in self.rows:row['cached_tokens']=100 if row['variant']=='candidate' else 0
        self.assertEqual(self.result()['verdict'],'nonregression')

    def test_reasoning_tool_finish_tokens_affect_parity(self):
        for key,value in [('reasoning','different'),('tool_calls',{'0':{'function':{'name':'read','arguments':'{}'}}}),('finish','length'),('output_tokens',4)]:
            changed=copy.deepcopy(self.rows)
            changed[-1][key]=value
            self.assertEqual(compare.analyze(changed,self.manifest)['verdict'],'inconclusive',key)

    def test_quality_regression_and_baseline_failure_differ(self):
        self.rows[-1]['quality_ok']=False
        self.assertEqual(self.result()['verdict'],'quality_regression')
        self.rows[-4]['quality_ok']=False
        self.assertEqual(self.result()['verdict'],'inconclusive')

    def test_identity_drift_cannot_pass(self):
        self.assertEqual(compare.analyze(self.rows,self.manifest,False)['verdict'],'inconclusive')

    def test_missing_startup_metric_cannot_pass(self):
        self.rows[0].pop('process_cold_ready_ms')
        self.assertEqual(self.result()['verdict'],'inconclusive')

    def test_runtime_mismatch_requires_declaration(self):
        for row in self.rows:
            row['runtime']={'loadedModel':{'context_length':8192,'prefill_step_size':512 if row['variant']=='baseline' else 256,'created':row['block']}}
        self.assertEqual(self.result()['verdict'],'inconclusive')
        self.manifest['runtime_allowed_differences']=['loadedModel.prefill_step_size']
        self.assertEqual(self.result()['verdict'],'nonregression')

    def test_runtime_changes_within_variant_cannot_pass(self):
        for row in self.rows:row['runtime']={'loadedModel':{'context_length':8192}}
        self.rows[-1]['runtime']['loadedModel']['context_length']=4096
        self.assertEqual(self.result()['verdict'],'inconclusive')

    def test_bad_cache_state_is_excluded_from_metric_pairs(self):
        self.manifest['workloads']=['fresh_long']
        for row in self.rows:
            if row['workload']=='short':
                row['workload']='fresh_long'
                row['cached_tokens']=0
        self.rows[-1]['cached_tokens']=8
        result=self.result()
        self.assertEqual(result['verdict'],'inconclusive')
        self.assertTrue(all(m['pairs']==3 for m in result['metrics'] if m['workload']=='fresh_long'))

    def test_prompts_are_identical_across_blocks(self):
        self.assertEqual(compare.workload_plan(0),compare.workload_plan(100))


class StreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=fixture_server.make_server()
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.url='http://127.0.0.1:'+str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()

    def request(self,text,api='chat',**kwargs):
        return compare.stream_request(self.url,compare.request_payload({'model':{'served_name':'bench'}},[{'role':'user','content':text}],api,32),api,**kwargs)

    def test_role_event_is_not_visible_ttft(self):
        row=self.request('ORBIT731')
        self.assertTrue(row['ok'])
        self.assertEqual(row['content'],'ORBIT731')
        self.assertGreater(row['ttft_ms'],10)
        self.assertGreater(row['total_ms'],row['ttft_ms']+row['visible_span_ms'])

    def test_partial_stream_missing_usage_fails_and_keeps_content(self):
        row=self.request('MISSING_USAGE')
        self.assertFalse(row['ok'])
        self.assertEqual(row['content'],'ORBIT731')
        self.assertTrue(row['events'])

    def test_reasoning_retained_rate_unavailable(self):
        row=self.request('REASONING')
        self.assertEqual(row['reasoning'],'Think.')
        self.assertIsNone(row['visible_decode_tps'])

    def test_invalid_numeric_usage_fails(self):
        row=self.request('INVALID_TOKENS')
        self.assertFalse(row['ok'])
        self.assertIn('numeric usage',row['error'])

    def test_responses_completed_output_retained(self):
        row=self.request('ORBIT731',api='responses')
        self.assertTrue(row['ok'])
        self.assertTrue(row['response_id'])
        self.assertEqual(row['final_output'][0]['content'][0]['text'],'ORBIT731')

    def test_cancellation_timestamp_before_next_request(self):
        row=self.request('Cancellation request',cancel=True)
        self.assertTrue(row['cancelled'])
        nextrow=self.request('NOVA284')
        self.assertEqual(nextrow['http_status'],429)
        self.assertLess(row['completed_perf_counter'],nextrow['completed_perf_counter'])
        # Let the explicitly cancelled synthetic producer drain before later tests.
        import time
        time.sleep(.2)

    def test_identity_detects_byte_change(self):
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/'model';file.write_text('one')
            before=compare.identity(folder)
            file.write_text('two')
            self.assertNotEqual(before['sha256'],compare.identity(folder)['sha256'])


class OwnedProcessTests(unittest.TestCase):
    def manifest(self, folder, blocks=4):
        model=Path(folder)/'model.txt';model.write_text('synthetic identity\n')
        fixture=str(Path(fixture_server.__file__).resolve())
        return {'schema_version':1,'model':{'path':str(model),'served_name':'owned-fixture'},'blocks':blocks,
                'config':{},'variants':[{'name':name,'command':[sys.executable,fixture,'--config','{config}','--model','{model}',
                    '--name','{served_model}','--port','{port}'],'identity_paths':[fixture]} for name in ('baseline','candidate')],
                'workloads':['short'],'long_repeats':1,'limits':{'startup_seconds':5,'request_seconds':5}}

    def test_rotation_identity_and_cleanup_without_touching_existing_listener(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest=self.manifest(folder)
            output=Path(folder)/'output'
            with socket.socket() as unrelated:
                unrelated.bind(('127.0.0.1',0));unrelated.listen(1)
                unrelated_port=unrelated.getsockname()[1]
                with contextlib.redirect_stdout(io.StringIO()):
                    result=compare.run(manifest,output)
                self.assertTrue(result['identity_unchanged'])
                rows=[json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]
                starts=[row for row in rows if row['workload']=='startup']
                self.assertEqual([row['variant'] for row in starts],['baseline','candidate','candidate','baseline']*2)
                self.assertEqual(len([r for r in rows if r['stage']=='warmup']),16)
                self.assertFalse(result['quality_failures'])
                self.assertFalse(result['exclusions'])
                self.assertTrue((output/'harness-source.py').exists())
                for launch in output.glob('block-*/launch.json'):
                    metadata=json.loads(launch.read_text())
                    self.assertNotEqual(metadata['port'],unrelated_port)
                    with socket.socket() as probe:
                        self.assertNotEqual(probe.connect_ex(('127.0.0.1',metadata['port'])),0)
                    with self.assertRaises(ProcessLookupError):os.kill(metadata['pid'],0)
                with socket.create_connection(('127.0.0.1',unrelated_port),timeout=1):pass

    def test_startup_failure_is_preserved_and_cannot_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest=self.manifest(folder,blocks=1)
            for variant in manifest['variants']:variant['command'].append('--intentional-invalid-argument')
            output=Path(folder)/'output'
            with contextlib.redirect_stdout(io.StringIO()):result=compare.run(manifest,output)
            self.assertEqual(result['verdict'],'inconclusive')
            rows=[json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]
            self.assertEqual(len([row for row in rows if row['stage']=='failure']),2)
            self.assertTrue(all('unrecognized arguments' in path.read_text() for path in output.glob('block-*/server.log')))


if __name__=='__main__':unittest.main()
