#!/usr/bin/env python3
"""Real-model release smoke test; writes reports only to the requested output directory."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--schemes', default='none,affine8,affine4,turbo8v4')
    parser.add_argument('--context-length', type=int, default=4096)
    parser.add_argument('--repetitions', type=int, default=180)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for scheme in args.schemes.split(','):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        base = f'http://127.0.0.1:{port}/v1'
        def request(route, body=None):
            data = None if body is None else json.dumps(body).encode()
            req = urllib.request.Request(base + route, data=data,
                headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=180) as response:
                    return response.status, response.headers.get('Content-Type', ''), response.read()
            except urllib.error.HTTPError as error:
                return error.code, error.headers.get('Content-Type', ''), error.read()
        with (output / f'{scheme}.log').open('w') as log:
            process = subprocess.Popen([str(Path(args.binary).resolve()), '--model', args.model,
                '--name', 'memory-test', '--port', str(port), '--max-tokens', '64',
                '--context-length', str(args.context_length), '--prefill-step-size', '256',
                '--kv-compression', scheme], stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'NSUnbufferedIO': 'YES'})
            try:
                deadline = time.monotonic() + 180
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'{scheme}: server exited; see {log.name}')
                    try:
                        status, _, data = request('/models')
                        if status == 200:
                            break
                    except (OSError, urllib.error.URLError):
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError(f'{scheme}: server startup timed out')
                    time.sleep(0.2)
                model = json.loads(data)['data'][0]
                assert model['context_length'] == args.context_length, model
                assert model['prefill_step_size'] == 256, model
                assert model['kv_compression'] == scheme, model
                assert model['memory_limit_bytes'] > 0, model
                # Both transport modes must reject before committing SSE headers.
                for stream in (False, True):
                    status, content_type, data = request('/chat/completions', {
                        'model': 'memory-test', 'messages': [{'role': 'user', 'content': ' hello' * (args.context_length + 1000)}],
                        'max_tokens': 8, 'temperature': 0, 'stream': stream})
                    assert status == 400 and 'application/json' in content_type, (status, content_type, data)
                    assert json.loads(data)['error']['code'] == 'request_exceeds_limits', data
                messages = [{'role': 'user', 'content':
                    'Read the following repeated notes.\n' + 'The project color is blue.\n' * args.repetitions
                    + 'Reply with the project color only.'}]
                start = time.monotonic()
                status, _, data = request('/chat/completions', {'model': 'memory-test',
                    'messages': messages, 'max_tokens': 8, 'temperature': 0, 'stream': False})
                response = json.loads(data)
                assert status == 200, response
                assert response['usage']['prompt_tokens'] > 512, response
                assert response['usage']['completion_tokens'] > 0, response
                assert response['choices'][0]['message']['content'], response
                duration = time.monotonic() - start
                messages += [response['choices'][0]['message'], {'role': 'user', 'content': 'Repeat that color.'}]
                status, content_type, data = request('/chat/completions', {'model': 'memory-test',
                    'messages': messages, 'max_tokens': 8, 'temperature': 0, 'stream': True})
                assert status == 200 and 'text/event-stream' in content_type, data
                assert b'data: [DONE]' in data and b'"error"' not in data, data
                records.append({'scheme': scheme, 'model': Path(args.model).name,
                    'descriptor': model, 'usage': response['usage'],
                    'content': response['choices'][0]['message']['content'],
                    'request_seconds': duration, 'overflow_rejected_json_both_modes': True,
                    'cached_continuation_stream_passed': True})
                (output / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
                print(f'{scheme}: passed ({response["usage"]["prompt_tokens"]} prompt tokens)', flush=True)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    print('All requested HTTP smoke checks passed.', flush=True)


if __name__ == '__main__':
    main()
