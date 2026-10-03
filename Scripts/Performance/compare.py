#!/usr/bin/env python3
"""Owned-process, paired Midnight performance comparisons. Python standard library only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from midnight_api_auth import authorization_headers, json_headers

VERSION = 1
WORKLOADS = ('short', 'fresh_long', 'repeat_long', 'shared_prefix', 'decode', 'multiturn', 'cancel_next')
PERF_PREFIXES = ('MODEL_RUNNER_', 'MIDNIGHT_BENCH_', 'MLX_', 'CUDA_', 'OMP_', 'OPENBLAS_', 'VECLIB_')
BASE_ENV = ('PATH', 'HOME', 'TMPDIR', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH', 'HF_HUB_OFFLINE', 'MIDNIGHT_API_KEY')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def identity(path):
    """Hash bytes, relative paths and symlink targets, including linked model weights."""
    source = Path(path).absolute()
    if not source.exists():
        raise ValueError('Missing identity path: ' + str(source))
    entries = []
    paths = sorted(source.rglob('*')) if source.is_dir() else [source]
    for child in paths:
        if not child.is_file():
            continue
        h = hashlib.sha256()
        with child.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        entries.append({'path': str(child.relative_to(source)) if source.is_dir() else child.name,
                        'bytes': child.stat().st_size, 'sha256': h.hexdigest(),
                        'symlink': os.readlink(child) if child.is_symlink() else None})
    return {'path': str(source), 'resolved_path': str(source.resolve()),
            'sha256': digest(entries), 'files': entries}


def identities(manifest):
    paths = [manifest['model']['path']]
    for variant in manifest['variants']:
        paths += [variant['command'][0]] + variant.get('identity_paths', [])
    return {str(Path(p).absolute()): identity(p) for p in dict.fromkeys(paths)}


def environment(overrides):
    result = {k: v for k, v in os.environ.items()
              if k in BASE_ENV or k.startswith(PERF_PREFIXES)}
    result.update({k: str(v) for k, v in overrides.items()})
    return result


def get_json(url, timeout=2):
    with urllib.request.urlopen(urllib.request.Request(url, headers=authorization_headers()), timeout=timeout) as response:
        return json.load(response)


def runtime_snapshot(url):
    try:
        return get_json(url + '/v1/runtime')
    except (OSError, ValueError) as error:
        return {'unavailable': str(error)}


def sse_events(response):
    data = []
    for line in response:
        line = line.decode('utf-8').rstrip('\r\n')
        if not line:
            if data:
                yield '\n'.join(data)
                data = []
        elif line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
    if data:
        yield '\n'.join(data)


def stream_request(url, payload, api='chat', timeout=180, cancel=False):
    """Keep every wire event. TTFT starts at first non-whitespace visible text."""
    started = time.perf_counter()
    result = {'ok': False, 'api': api, 'request': payload, 'events': [], 'content': '',
              'reasoning': '', 'tool_calls': {}, 'finish': None, 'usage': None,
              'response_id': None, 'ttft_ms': None, 'visible_span_ms': None,
              'visible_decode_tps': None, 'http_status': None}
    first = last = None
    terminal = False
    try:
        request = urllib.request.Request(url + ('/v1/responses' if api == 'responses' else '/v1/chat/completions'),
                                         data=canonical(payload).encode(), headers=json_headers())
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result['http_status'] = response.status
            for raw in sse_events(response):
                now = time.perf_counter()
                if now - started > timeout:
                    raise TimeoutError('Overall request deadline exceeded')
                if raw == '[DONE]':
                    terminal = True
                    break
                event = json.loads(raw)
                result['events'].append({'elapsed_ms': (now-started)*1000, 'data': event})
                if event.get('error'):
                    raise ValueError('Server error: ' + canonical(event['error']))
                text = ''
                if api == 'responses':
                    kind = event.get('type', '')
                    if kind == 'response.created':
                        result['response_id'] = event['response']['id']
                    if kind == 'response.output_text.delta':
                        text = event.get('delta', '')
                    if kind in ('response.reasoning_text.delta', 'response.reasoning_summary_text.delta'):
                        result['reasoning'] += event.get('delta', '')
                    if kind in ('response.completed', 'response.incomplete'):
                        final = event['response']
                        result['finish'] = {'status': final.get('status', kind.split('.')[1]),
                                            'incomplete_details': final.get('incomplete_details')}
                        result['final_output'] = final.get('output', [])
                        result['usage'] = final.get('usage')
                        result['response_id'] = final.get('id', result['response_id'])
                        terminal = True
                        break
                    if kind in ('response.failed', 'error'):
                        raise ValueError('Responses failed: ' + canonical(event))
                else:
                    if event.get('usage') is not None:
                        result['usage'] = event['usage']
                    for choice in event.get('choices', []):
                        if choice.get('index', 0) != 0:
                            raise ValueError('Only one output choice is supported')
                        if choice.get('finish_reason') is not None:
                            result['finish'] = choice['finish_reason']
                        delta = choice.get('delta', {})
                        text += delta.get('content') or ''
                        result['reasoning'] += delta.get('reasoning_content', delta.get('reasoning', '')) or ''
                        for tool in delta.get('tool_calls', []):
                            index = str(tool.get('index', 0))
                            target = result['tool_calls'].setdefault(index, {'type': None, 'function': {'name': '', 'arguments': ''}})
                            target['type'] = tool.get('type', target['type'])
                            for key in ('name', 'arguments'):
                                target['function'][key] += tool.get('function', {}).get(key, '')
                result['content'] += text
                if text.strip():
                    if first is None:
                        first = now
                    last = now
                    if cancel:
                        result['cancelled'] = True
                        break
            if cancel and result.get('cancelled'):
                result['ok'] = True
            else:
                if not terminal or result['finish'] is None or result['usage'] is None:
                    raise ValueError('Incomplete stream: terminal, finish and usage are required')
                result['ok'] = True
    except urllib.error.HTTPError as error:
        result['http_status'] = error.code
        result['error'] = error.read().decode('utf-8', errors='replace')
        error.close()
    except (OSError, ValueError, KeyError, TypeError) as error:
        result['error'] = str(error)
    result['total_ms'] = (time.perf_counter() - started) * 1000
    result['completed_perf_counter'] = time.perf_counter()
    result['ttft_ms'] = (first-started)*1000 if first is not None else None
    result['visible_span_ms'] = (last-first)*1000 if first is not None and last is not None else None
    usage = result['usage'] or {}
    result['input_tokens'] = usage.get('input_tokens', usage.get('prompt_tokens'))
    result['output_tokens'] = usage.get('output_tokens', usage.get('completion_tokens'))
    details = usage.get('input_tokens_details', usage.get('prompt_tokens_details', {})) or {}
    result['cached_tokens'] = details.get('cached_tokens')
    output_details = usage.get('output_tokens_details', usage.get('completion_tokens_details', {})) or {}
    reasoning_tokens = output_details.get('reasoning_tokens', 0)
    if result['ok'] and not result.get('cancelled'):
        invalid = []
        for key in ('input_tokens', 'output_tokens'):
            if type(result[key]) is not int or result[key] < 0:
                invalid.append(key)
        if 'total_tokens' in usage and (type(usage['total_tokens']) is not int or
                invalid or usage['total_tokens'] != result['input_tokens']+result['output_tokens']):
            invalid.append('total_tokens')
        if result['cached_tokens'] is not None and (type(result['cached_tokens']) is not int or
                result['cached_tokens'] < 0 or not invalid and result['cached_tokens'] > result['input_tokens']):
            invalid.append('cached_tokens')
        if type(reasoning_tokens) is not int or reasoning_tokens < 0 or not invalid and reasoning_tokens > result['output_tokens']:
            invalid.append('reasoning_tokens')
        if result['content'].strip() and result['output_tokens'] == 0:
            invalid.append('visible output with zero output tokens')
        if invalid:
            result['ok'] = False
            result['error'] = 'Invalid or missing numeric usage: '+', '.join(invalid)
    nontext_output = any(item.get('type') in ('reasoning', 'function_call') for item in result.get('final_output', []) if isinstance(item,dict))
    if (last is not None and first is not None and last > first and
            result['ok'] and type(result['output_tokens']) is int and result['output_tokens'] > 1 and
            not result['reasoning'] and not reasoning_tokens and not result['tool_calls'] and not nontext_output):
        result['visible_decode_tps'] = (result['output_tokens']-1)/(last-first)
    return result


def normalized_output(value):
    """Discard transport-generated IDs only; keep content/reasoning/tool/finish fields."""
    if isinstance(value, list):
        return [normalized_output(item) for item in value]
    if isinstance(value, dict):
        return {k: normalized_output(v) for k, v in value.items() if k not in ('id', 'call_id')}
    return value


def parity(row):
    result = {key: normalized_output(row.get(key)) for key in
              ('content', 'reasoning', 'tool_calls', 'finish', 'final_output', 'input_tokens', 'output_tokens')}
    def token_details(value):
        if not isinstance(value, dict):
            return value
        return {key: token_details(item) for key,item in value.items()
                if key != 'cached_tokens' and token_details(item) != {}}
    result['usage'] = token_details(row.get('usage'))
    return result


def runtime_configuration(runtime):
    """Stable inference configuration only, excluding IDs, uptime and live memory."""
    if not runtime or runtime.get('unavailable'):
        return None
    model = runtime.get('loadedModel')
    if isinstance(model, dict):
        return {'loadedModel.'+key:value for key,value in model.items()
                if key not in ('created', 'object', 'owned_by')}
    # Compatible synthetic/custom servers may publish these directly.
    return {key:runtime[key] for key in ('contextLength','prefillStepSize','kvCompression','model') if key in runtime}


class OwnedServer:
    def __init__(self, manifest, variant, folder):
        self.manifest, self.variant, self.folder = manifest, variant, Path(folder)
        self.process = None
        self.memory = []
        self.stop_sampling = threading.Event()

    def __enter__(self):
        self.folder.mkdir(parents=True)
        config = self.variant.get('config', self.manifest.get('config', {}))
        write_json(self.folder/'config.json', config)
        with socket.socket() as reserved:
            reserved.bind(('127.0.0.1', 0))
            port = reserved.getsockname()[1]
        self.url = 'http://127.0.0.1:' + str(port)
        replacements = {'config': str(self.folder/'config.json'), 'port': str(port),
                        'model': self.manifest['model']['path'], 'served_model': self.manifest['model']['served_name']}
        self.command = [word.format_map(replacements) for word in self.variant['command']]
        self.env = environment(self.variant.get('env', {}))
        recorded_env = {k: ('[redacted]' if k == 'MIDNIGHT_API_KEY' else v) for k, v in self.env.items()}
        self.metadata = {'command': self.command, 'environment': recorded_env, 'config_sha256': digest(config), 'port': port,
                         'startup_poll_seconds': self.manifest.get('limits', {}).get('startup_poll_seconds', .02)}
        write_json(self.folder/'launch.json', self.metadata)
        self.log = (self.folder/'server.log').open('w')
        start = time.perf_counter()
        self.process = subprocess.Popen(self.command, stdout=self.log, stderr=subprocess.STDOUT,
                                        env=self.env, cwd=self.folder, start_new_session=True)
        self.metadata['pid'] = self.process.pid
        self.sampler = threading.Thread(target=self.sample, daemon=True)
        self.sampler.start()
        deadline = time.monotonic() + self.manifest.get('limits', {}).get('startup_seconds', 180)
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('Owned server exited during startup; inspect ' + str(self.folder/'server.log'))
                try:
                    models = get_json(self.url+'/v1/models')
                    if any(item.get('id') == self.manifest['model']['served_name'] for item in models.get('data', [])):
                        self.metadata['process_cold_ready_ms'] = (time.perf_counter()-start)*1000
                        self.metadata['ready_runtime'] = runtime_snapshot(self.url)
                        write_json(self.folder/'launch.json', self.metadata)
                        return self
                except (OSError, ValueError):
                    pass
                time.sleep(self.metadata['startup_poll_seconds'])
            raise TimeoutError('Owned server startup timeout')
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def sample(self):
        while not self.stop_sampling.is_set():
            try:
                if sys.platform.startswith('linux'):
                    status = Path('/proc/%d/status' % self.process.pid).read_text()
                    rss = next(int(line.split()[1]) for line in status.splitlines() if line.startswith('VmRSS:'))
                else:
                    rss = int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(self.process.pid)], text=True).strip())
                self.memory.append({'monotonic': time.monotonic(), 'rss_kib': rss})
            except (OSError, ValueError, StopIteration, subprocess.SubprocessError):
                pass
            self.stop_sampling.wait(.5)

    def __exit__(self, *_):
        self.stop_sampling.set()
        if hasattr(self, 'sampler'):
            self.sampler.join(timeout=2)
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        if hasattr(self, 'log'):
            self.log.close()
        write_json(self.folder/'memory.json', self.memory)
        self.metadata['exit_code'] = self.process.returncode if self.process is not None else None
        self.metadata['sampled_peak_rss_kib'] = max((row['rss_kib'] for row in self.memory), default=None)
        write_json(self.folder/'launch.json', self.metadata)


def request_payload(manifest, messages, api, tokens, previous=None):
    payload = dict(manifest.get('request', {}))
    payload.update(model=manifest['model']['served_name'], stream=True)
    if api == 'responses':
        payload.update(input=messages if previous is None else messages[-1:], max_output_tokens=tokens, store=True)
        if previous:
            payload['previous_response_id'] = previous
    else:
        payload.update(messages=messages, max_tokens=tokens, stream_options={'include_usage': True})
    return payload


def workload_plan(block, long_repeats=70):
    paragraph = 'The workshop guide recommends inspecting the chain, tyres, brakes, gears and frame. Clean components carefully and follow the manufacturer instructions. '
    marker = 'ORBIT731'
    # Each arm/block gets a new process. Block-dependent prompts would confound
    # temporal drift with different tokenization/answers, so prompts stay fixed.
    system = 'Reference document. ' + paragraph*long_repeats + '\nThe verification marker is ' + marker + '.'
    ask = 'Return only the verification marker, with no other text.'
    long = [{'role': 'system', 'content': system}, {'role': 'user', 'content': ask}]
    return {
        'short': ([{'role': 'user', 'content': 'Return exactly the code ORBIT731 and nothing else.'}], marker, 32),
        'fresh_long': (long, marker, 32),
        'repeat_long': (long, marker, 32),
        'shared_prefix': ([long[0], {'role': 'user', 'content': 'Extract the verification marker from the document. Reply with the marker only.'}], marker, 32),
        'decode': ([{'role': 'user', 'content': 'Decode request. Explain bicycle maintenance in detail using at least 400 words.'}], None, 256),
        'multiturn': ([{'role': 'system', 'content': 'Recall conversation. Remember marker ORBIT731. When asked, reply with only this marker.'}, {'role': 'user', 'content': 'What is the marker?'}], marker, 32),
        'cancel_next': ([{'role': 'user', 'content': 'Cancellation request. Explain bicycle maintenance in detail using at least 400 words.'}], None, 256)
    }


def validate_manifest(manifest):
    if manifest.get('schema_version') != VERSION or len(manifest.get('variants', [])) != 2:
        raise ValueError('schema_version:1 and exactly two ordered variants (baseline, candidate) required')
    if len({v['name'] for v in manifest['variants']}) != 2:
        raise ValueError('Variant names must differ')
    if manifest.get('blocks', 4) < 1:
        raise ValueError('At least one block required')
    if any(w not in WORKLOADS for w in manifest.get('workloads', WORKLOADS)):
        raise ValueError('Unknown workload')
    for variant in manifest['variants']:
        if not Path(variant['command'][0]).is_absolute() or '{port}' not in variant['command']:
            raise ValueError('Use absolute executable and a separate {port} argument for an owned loopback server')
        for policy in variant.get('cache_expectations', manifest.get('cache_expectations', {})).values():
            if policy not in ('zero', 'positive', 'observe'):
                raise ValueError('Cache policy must be zero, positive or observe')


def run(manifest, output):
    validate_manifest(manifest)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'harness-source.py').write_bytes(Path(__file__).read_bytes())
    write_json(output/'manifest.json', manifest)
    before = identities(manifest)
    write_json(output/'identity-before.json', before)
    write_json(output/'host.json', {'platform': platform.platform(), 'machine': platform.machine(),
                                    'python': sys.version, 'cpu_count': os.cpu_count(), 'harness': identity(__file__),
                                    'memory_sampling': 'owned server RSS every 500 ms; excludes children/GPU allocations'})
    rows = []
    def save(row):
        rows.append(row)
        with (output/'samples.jsonl').open('a') as file:
            file.write(canonical(row)+'\n')
        print(canonical({key: row.get(key) for key in ('block', 'variant', 'workload', 'stage', 'ok', 'ttft_ms', 'total_ms', 'error')}), flush=True)
    try:
        for block in range(manifest.get('blocks', 4)):
            variants = manifest['variants'] if block % 2 == 0 else list(reversed(manifest['variants']))
            for order, variant in enumerate(variants):
                base = {'block': block, 'variant': variant['name'], 'order': order}
                folder = output/('block-%02d-%s' % (block, variant['name']))
                try:
                    with OwnedServer(manifest, variant, folder) as server:
                        save(dict(base, workload='startup', stage='measurement', ok=True,
                                  process_cold_ready_ms=server.metadata['process_cold_ready_ms'], runtime=server.metadata['ready_runtime']))
                        plan = workload_plan(block, manifest.get('long_repeats', 70))
                        timeout = manifest.get('limits', {}).get('request_seconds', 180)
                        def perform(name, messages, expected, tokens, stage='measurement', api='chat', previous=None, cancel=False):
                            payload = request_payload(manifest, messages, api, tokens, previous)
                            row = stream_request(server.url, payload, api, timeout, cancel)
                            row.update(base, workload=name, stage=stage, monotonic=time.monotonic(),
                                       input_sha256=digest({'messages': messages, 'api': api, 'options': manifest.get('request', {}), 'tokens': tokens}),
                                       expected=expected)
                            row['quality_ok'] = (row['content'].strip() == expected if expected is not None else bool(row['content'].strip())) if row['ok'] else False
                            if not cancel:
                                row['runtime'] = runtime_snapshot(server.url)
                            save(row)
                            return row
                        # Disjoint namespaces avoid warming the measured prefix. Both timings remain in raw data.
                        perform('warmup_short', [{'role':'user', 'content':'Warmup only. Return only READY.'}], 'READY', 8, 'warmup')
                        perform('warmup_long', [{'role':'system', 'content':'Kernel warmup only. '+('Keep bicycle parts clean. '*250)}, {'role':'user', 'content':'Return only READY.'}], 'READY', 8, 'warmup')
                        selected = manifest.get('workloads', list(WORKLOADS))
                        long_started = False
                        for name in WORKLOADS:
                            if name not in selected:
                                continue
                            messages, expected, tokens = plan[name]
                            if name in ('repeat_long', 'shared_prefix') and not long_started:
                                perform('fresh_long', *plan['fresh_long'], stage='cache_setup')
                                long_started = True
                            if name == 'fresh_long':
                                long_started = True
                            if name == 'multiturn':
                                conversation = json.loads(canonical(messages))
                                previous = None
                                for turn in range(3):
                                    row = perform('multiturn_%d' % (turn+1), conversation, expected, tokens,
                                                  api=manifest.get('conversation_api', 'responses'), previous=previous)
                                    if not row['ok']:
                                        break
                                    if manifest.get('conversation_api', 'responses') == 'responses' and not row['response_id']:
                                        save(dict(base, workload='session', stage='failure', ok=False, error='Responses continuation requires response_id'))
                                        break
                                    previous = row['response_id']
                                    conversation += [{'role':'assistant', 'content':row['content']}, {'role':'user', 'content':'Repeat the remembered marker only.'}]
                            elif name == 'cancel_next':
                                row = perform('cancel', messages, None, manifest.get('cancel_tokens', tokens), stage='cancellation', cancel=True)
                                cancel_end = row['completed_perf_counter']
                                attempts = []
                                next_messages = [{'role':'user', 'content':'Return exactly the code NOVA284 and nothing else.'}]
                                while time.perf_counter()-cancel_end < timeout:
                                    next_row = stream_request(server.url, request_payload(manifest, next_messages, 'chat', 32), timeout=timeout)
                                    attempts.append(next_row)
                                    if next_row['ok'] or next_row['http_status'] not in (409, 429, 503):
                                        break
                                    time.sleep(.05)
                                next_row.update(base, workload=name, stage='measurement', expected='NOVA284',
                                                quality_ok=next_row['ok'] and next_row['content'].strip() == 'NOVA284',
                                                input_sha256=digest(next_messages), cancellation_ok=row.get('cancelled', False),
                                                cancel_next_ready_ms=(time.perf_counter()-cancel_end)*1000,
                                                readiness_attempts=attempts[:-1], runtime=runtime_snapshot(server.url))
                                save(next_row)
                            else:
                                perform(name, messages, expected, tokens)
                        save(dict(base, workload='memory', stage='measurement', ok=True,
                                  sampled_peak_rss_kib=max((s['rss_kib'] for s in server.memory), default=None)))
                except (OSError, ValueError, RuntimeError) as error:
                    save(dict(base, workload='session', stage='failure', ok=False, error=str(error)))
    finally:
        try:
            after = identities(manifest)
            write_json(output/'identity-after.json', after)
            stable = before == after
        except (OSError, ValueError) as error:
            stable = False
            write_json(output/'identity-after.json', {'error': str(error)})
        write_json(output/'integrity.json', {'identity_unchanged': stable})
        result = analyze(rows, manifest, stable)
        write_json(output/'analysis.json', result)
        (output/'report.md').write_text(report(result))
    return result


def interval(values, draws=5000):
    rng = random.Random(731)
    samples = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(draws))
    return [samples[int(.025*(draws-1))], samples[int(.975*(draws-1))]]


def analyze(rows, manifest, identity_stable=True):
    gate = {'latency_tolerance':.03, 'minimum_pairs':4, 'drift_tolerance':.10, **manifest.get('gate', {})}
    names = [v['name'] for v in manifest['variants']]
    metrics = ('ttft_ms', 'total_ms', 'visible_decode_tps', 'process_cold_ready_ms', 'sampled_peak_rss_kib', 'cancel_next_ready_ms')
    groups = {}
    exclusions = []
    for row in rows:
        if row.get('stage') != 'measurement':
            if row.get('stage') == 'failure' or (row.get('stage') in ('warmup', 'cache_setup') and not row.get('ok')):
                exclusions.append({'block': row['block'], 'variant': row['variant'], 'reason': row.get('error', 'session failed')})
            continue
        slot = groups.setdefault(row['workload'], {}).setdefault(row['block'], {})
        if row['variant'] in slot:
            exclusions.append({'workload':row['workload'], 'block':row['block'], 'variant':row['variant'], 'reason':'duplicate sample'})
        slot[row['variant']] = row
    required = ['startup', 'memory']
    for workload in manifest.get('workloads', WORKLOADS):
        required += ['multiturn_1', 'multiturn_2', 'multiturn_3'] if workload == 'multiturn' else [workload]
    for workload in required:
        groups.setdefault(workload, {})
    summaries = []
    quality_failures = []
    runtime_differences = []
    stable_runtime_by_variant = {}
    for row in rows:
        state = runtime_configuration(row.get('runtime'))
        if state is None:
            continue
        previous = stable_runtime_by_variant.setdefault(row['variant'], state)
        if state != previous:
            exclusions.append({'block':row['block'], 'variant':row['variant'], 'workload':row['workload'], 'reason':'resolved runtime configuration changed within variant', 'before':previous, 'after':state})
    expected_blocks = manifest.get('blocks', 4)
    for workload, blocks in sorted(groups.items()):
        valid = []
        for block in range(expected_blocks):
            pair = blocks.get(block, {})
            if any(name not in pair for name in names):
                exclusions.append({'workload':workload, 'block':block, 'reason':'missing pair'})
                continue
            a, b = (pair[n] for n in names)
            if not a.get('ok') or not b.get('ok'):
                exclusions.append({'workload':workload, 'block':block, 'reason':'request failed'})
                continue
            left, right = runtime_configuration(a.get('runtime')), runtime_configuration(b.get('runtime'))
            if left != right:
                allowed = set(manifest.get('runtime_allowed_differences', []))
                differences = {key:{names[0]:(left or {}).get(key),names[1]:(right or {}).get(key)}
                               for key in set(left or {}) | set(right or {}) if (left or {}).get(key)!=(right or {}).get(key)}
                runtime_differences.append({'workload':workload,'block':block,'differences':differences,'declared':all(key in allowed for key in differences)})
                if left is None or right is None or any(key not in allowed for key in differences):
                    exclusions.append({'workload':workload,'block':block,'reason':'resolved runtime mismatch not declared','differences':differences})
                    continue
            if any(isinstance(row.get(metric), (int,float)) and not math.isfinite(row[metric]) for row in (a,b) for metric in metrics):
                exclusions.append({'workload':workload, 'block':block, 'reason':'nonfinite metric'})
                continue
            setup_metric = {'startup':'process_cold_ready_ms','memory':'sampled_peak_rss_kib'}.get(workload)
            if setup_metric and any(type(row.get(setup_metric)) not in (int,float) or row[setup_metric]<=0 for row in (a,b)):
                exclusions.append({'workload':workload,'block':block,'reason':'required setup metric missing or invalid'})
                continue
            if workload not in ('startup', 'memory'):
                if not a.get('quality_ok') or not b.get('quality_ok'):
                    quality_failures.append({'workload':workload, 'block':block, 'baseline_ok':a.get('quality_ok'), 'candidate_ok':b.get('quality_ok')})
                    continue
                if workload == 'cancel_next' and (not a.get('cancellation_ok') or not b.get('cancellation_ok')):
                    exclusions.append({'workload':workload, 'block':block, 'reason':'cancellation did not reach visible generation'})
                    continue
                if a.get('input_sha256') != b.get('input_sha256'):
                    exclusions.append({'workload':workload, 'block':block, 'reason':'input diverged; observational only'})
                    continue
                if parity(a) != parity(b):
                    exclusions.append({'workload':workload, 'block':block, 'reason':'output/token/finish parity differs; timing not controlled',
                                       'baseline_output':parity(a), 'candidate_output':parity(b)})
                    continue
                cache_valid = True
                for row in (a,b):
                    variant = next(v for v in manifest['variants'] if v['name']==row['variant'])
                    policy = variant.get('cache_expectations', manifest.get('cache_expectations', {})).get(workload, 'zero' if workload=='fresh_long' else 'observe')
                    cached = row.get('cached_tokens')
                    if policy == 'zero' and cached != 0 or policy == 'positive' and (cached is None or cached<=0):
                        exclusions.append({'workload':workload, 'block':block, 'variant':row['variant'], 'reason':'cache state differs from declared '+policy, 'cached_tokens':cached})
                        cache_valid = False
                if not cache_valid:
                    continue
                needed = ('ttft_ms', 'total_ms') + (('cancel_next_ready_ms',) if workload=='cancel_next' else ())
                if any(not isinstance(row.get(metric), (int,float)) or row[metric]<=0 for row in (a,b) for metric in needed):
                    exclusions.append({'workload':workload, 'block':block, 'reason':'required latency metric missing or invalid'})
                    continue
            valid.append((block, a, b))
        for metric in metrics:
            available = [(block,a,b) for block,a,b in valid if isinstance(a.get(metric), (int,float)) and isinstance(b.get(metric), (int,float)) and a[metric] > 0 and b[metric] > 0]
            if not available:
                continue
            ratios = [(a[metric]/b[metric] if metric.endswith('_tps') else b[metric]/a[metric])-1 for _,a,b in available]
            ci = interval(ratios)
            drifts = {}
            for index, name in enumerate(names, start=1):
                values = [pair[index][metric] for pair in available]
                half = len(values)//2
                drifts[name] = statistics.median(values[-half:])/statistics.median(values[:half])-1 if half else None
            drift = any(v is not None and abs(v)>gate['drift_tolerance'] for v in drifts.values())
            advisory = metric in ('sampled_peak_rss_kib', 'visible_decode_tps')
            verdict = ('inconclusive' if len(available)<max(4,gate['minimum_pairs']) or len(available)!=expected_blocks or drift
                       else 'regression' if ci[0]>gate['latency_tolerance']
                       else 'improvement' if ci[1]<0
                       else 'nonregression' if ci[1]<=gate['latency_tolerance'] else 'inconclusive')
            summaries.append({'workload':workload, 'metric':metric, 'units':'tokens/s' if metric.endswith('_tps') else 'KiB' if metric.endswith('_kib') else 'ms',
                              'advisory':advisory, 'pairs':len(available), 'baseline_median':statistics.median(a[metric] for _,a,b in available),
                              'candidate_median':statistics.median(b[metric] for _,a,b in available),
                              'paired_change':statistics.median(ratios), 'ci95':ci, 'paired_changes':ratios,
                              'drift':drifts, 'verdict':verdict,
                              'cached_tokens':{name:[pair[i].get('cached_tokens') for pair in available] for i,name in enumerate(names,start=1)}})
    primary = [s for s in summaries if not s['advisory']]
    if any(q['baseline_ok'] and not q['candidate_ok'] for q in quality_failures):
        verdict = 'quality_regression'
    elif not identity_stable or exclusions or quality_failures or not primary:
        verdict = 'inconclusive'
    elif any(s['verdict']=='regression' for s in primary):
        verdict = 'regression'
    elif any(s['verdict']=='inconclusive' for s in primary):
        verdict = 'inconclusive'
    else:
        verdict = 'improvement' if any(s['verdict']=='improvement' for s in primary) else 'nonregression'
    return {'schema_version':VERSION, 'verdict':verdict, 'variant_order':names, 'identity_unchanged':identity_stable,
            'gate':gate, 'metrics':summaries, 'quality_failures':quality_failures, 'exclusions':exclusions,
            'runtime_differences':runtime_differences, 'runtime_configurations':stable_runtime_by_variant,
            'method':'Median of candidate/baseline latency ratios (inverse for rate); deterministic 95% percentile bootstrap over independent process pairs. Positive change is worse. Drift is late/early arm median minus one.'}


def report(result):
    lines = ['# Performance comparison', '', '**Verdict: %s**' % result['verdict'], '', result['method'], '',
             'Default nonregression tolerance: %.1f%%. At least %d independent process pairs; incomplete pairs and drift cannot pass.' % (100*result['gate']['latency_tolerance'], max(4,result['gate']['minimum_pairs'])), '',
             '| Workload | Metric | Baseline | Candidate | Change (95% CI) | Pairs | Result |',
             '|---|---|---:|---:|---|---:|---|']
    for s in result['metrics']:
        lines.append('| %s | %s (%s)%s | %.3f | %.3f | %+.2f%% [%+.2f%%, %+.2f%%] | %d | %s |' % (
            s['workload'],s['metric'],s['units'],' advisory' if s['advisory'] else '',s['baseline_median'],s['candidate_median'],
            s['paired_change']*100,s['ci95'][0]*100,s['ci95'][1]*100,s['pairs'],s['verdict']))
    lines += ['', '%d quality failures; %d exclusions. Identity unchanged: %s. See analysis.json and samples.jsonl for all failures, output parity, cache token counts and runtime snapshots.' % (len(result['quality_failures']),len(result['exclusions']),result['identity_unchanged']), '',
              'Process-cold readiness includes process/model startup, with OS file caches uncontrolled. RSS excludes GPU allocations and child processes. Visible decode rate is an approximation from visible chunk boundaries and reported output tokens; EOS/tail cleanup contributes only to total. Cold, repeated and shared-prefix requests remain separate workloads.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--manifest', required=True)
    run_parser.add_argument('--out', required=True)
    analyze_parser = sub.add_parser('analyze')
    analyze_parser.add_argument('directory')
    args = parser.parse_args()
    if args.command == 'run':
        result = run(json.loads(Path(args.manifest).read_text()), args.out)
    else:
        folder = Path(args.directory)
        manifest = json.loads((folder/'manifest.json').read_text())
        rows = [json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines() if line.strip()]
        stable = json.loads((folder/'integrity.json').read_text())['identity_unchanged']
        result = analyze(rows, manifest, stable)
        write_json(folder/'analysis.json', result)
        (folder/'report.md').write_text(report(result))
    print(result['verdict'])
    return 0 if result['verdict'] in ('nonregression', 'improvement') else 2


if __name__ == '__main__':
    sys.exit(main())
