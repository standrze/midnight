#!/usr/bin/env python3
"""Verify model-aware output limits on an isolated local Midnight listener.
No downloads. Optionally verifies Lowlight's headless client negotiation too.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

from midnight_api_auth import json_headers

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', required=True, type=Path)
parser.add_argument('--model', required=True, type=Path)
parser.add_argument('--lowlight', action='append', default=[], type=Path)
parser.add_argument('--port', type=int, default=18849)
args = parser.parse_args()
base = f'http://127.0.0.1:{args.port}'
root = Path(tempfile.mkdtemp(prefix='midnight-output-limits-'))
(root / 'empty.json').write_text('{}')
with socket.socket() as probe:
    probe.bind(('127.0.0.1', args.port))
log = (root / 'server.log').open('w')
server = subprocess.Popen([str(args.binary.resolve()), '--model', str(args.model.resolve()),
    '--name', 'output-limits', '--context-length', '2048', '--host', '127.0.0.1',
    '--port', str(args.port), '--config', str(root / 'empty.json')], stdout=log,
    stderr=subprocess.STDOUT, cwd=root)

def api(path, payload=None, expected=200):
    request = urllib.request.Request(base + path, data=None if payload is None else json.dumps(payload).encode(),
        headers=json_headers())
    try: response = urllib.request.urlopen(request, timeout=90)
    except urllib.error.HTTPError as error: response = error
    with response:
        body = json.load(response)
        assert response.status == expected, (path, response.status, body)
        return body

def ready():
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        assert server.poll() is None, (root / 'server.log').read_text()[-3000:]
        try:
            models = api('/v1/models')['data']
            if models: return models[0]
        except (OSError, urllib.error.URLError): pass
        time.sleep(.2)
    raise AssertionError('Model startup timed out')

schema = {'type':'object','properties':{'answer':{'type':'string','enum':['ok']}},
          'required':['answer'],'additionalProperties':False}
format = {'type':'json_schema','name':'answer','strict':True,'schema':schema}
results = []
try:
    model = ready()
    assert model['context_length'] == 2048 and model['default_output_tokens'] == 512
    assert model['max_output_tokens'] == 2047, model
    print('PASS small-context default and separate advertised ceiling', flush=True)
    long_prompt = ' a' * 1800 + '\nReturn ok.'
    for requested in [1800, None]:
        body = {'model':'output-limits','input':long_prompt,'text':{'format':format},'temperature':0}
        if requested is not None: body['max_output_tokens'] = requested
        response = api('/v1/responses', body)
        remaining = 2048 - response['usage']['input_tokens'] - 1
        assert 0 < remaining < 512, response['usage']
        assert response['max_output_tokens'] == remaining, response
        assert response['status'] == 'completed', response
        results.append({'requested':requested,'effective':remaining,'usage':response['usage']})
    print('PASS Responses explicit and omitted limits fit exact remaining context', flush=True)
    chat = api('/v1/chat/completions', {'model':'output-limits','messages':[{'role':'user','content':long_prompt}],
        'max_tokens':1800,'temperature':0,'response_format':{'type':'json_schema','json_schema':
        {'name':'answer','strict':True,'schema':schema}}})
    assert chat['usage']['prompt_tokens'] + 1800 > 2048
    assert json.loads(chat['choices'][0]['message']['content']) == {'answer':'ok'}, chat
    print('PASS Chat Completions bounds output by exact prompt length', flush=True)
    api('/v1/responses', {'model':'output-limits','input':'hi','max_output_tokens':2048}, expected=400)
    api('/v1/responses', {'model':'output-limits','input':' a'*2300,'max_output_tokens':1}, expected=400)
    print('PASS invalid ceiling and full-context prompts are rejected before generation', flush=True)
    for cap in [None, 512]:
        load = {'model':str(args.model.resolve()),'name':'output-limits','contextLength':16384}
        if cap: load['maxTokens'] = cap
        api('/v1/runtime/load', load, expected=202)
        model = ready()
        assert model['default_output_tokens'] == (cap or 4096), model
        assert model['max_output_tokens'] == (cap or 16383), model
        for binary in args.lowlight:
            for protocol in ['responses','chat-completions']:
                command = [str(binary.resolve()),'run','--model','output-limits','--endpoint',base+'/v1','--api',protocol]
                if cap: command += ['--max-tokens','4096']
                result = subprocess.run(command,input='Say hello briefly.',text=True,capture_output=True,timeout=90)
                assert result.returncode == 0, (binary, protocol, result.stderr)
                assert json.loads(result.stdout)['answer'].strip(), result.stdout
                if cap: assert '4096 to 512' in result.stderr, result.stderr
        print('PASS large-context defaults and client negotiation, operator cap=' + str(cap), flush=True)
    (root/'results.json').write_text(json.dumps(results,indent=2))
finally:
    server.terminate()
    try: server.wait(timeout=20)
    except subprocess.TimeoutExpired:
        server.kill(); server.wait()
    log.close()
    print('Artifacts:',root,flush=True)
