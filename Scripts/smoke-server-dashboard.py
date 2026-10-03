#!/usr/bin/env python3
"""Verify live dashboard telemetry against an HTTP response using an existing text checkpoint."""
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import termios
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.request

MODEL = str(Path(sys.argv[1]).resolve())
BINARY = str(Path(sys.argv[2] if len(sys.argv) > 2 else '.build/debug/midnight').resolve())
KEY = 'midnight-dashboard-smoke-key-0000000000000000'
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
master, slave = pty.openpty()
fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 30, 100, 0, 0))
original = termios.tcgetattr(slave)
data = bytearray()


def collect(seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if select.select([master], [], [], 0.05)[0]:
            data.extend(os.read(master, 65536))


def request(path, body=None):
    value = urllib.request.Request(f'http://127.0.0.1:{port}/v1/{path}',
                                  data=None if body is None else json.dumps(body).encode(),
                                  headers={'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json'})
    with urllib.request.urlopen(value, timeout=120) as response:
        return json.load(response)


with tempfile.TemporaryDirectory(prefix='midnight-dashboard-smoke-') as directory:
    config = Path(directory) / 'settings.json'
    config.write_text('{}')
    environment = dict(os.environ, TERM='xterm-256color', MIDNIGHT_API_KEY=KEY,
                       MODEL_RUNNER_MODELS_DIR=directory,
                       MIDNIGHT_MODEL_AVAILABILITY_FILE=str(Path(directory) / 'availability.json'))
    process = subprocess.Popen([BINARY, '--config', str(config), '--model', MODEL, '--port', str(port)],
                               stdin=slave, stdout=slave, stderr=slave, env=environment)
    try:
        deadline = time.monotonic() + 120
        while process.poll() is None and time.monotonic() < deadline:
            collect(0.1)
            try:
                state = request('runtime')
                if state['phase'] == 'ready':
                    break
                assert not state.get('lastError'), state
            except (OSError, TimeoutError):
                pass
        else:
            raise AssertionError('Checkpoint did not become ready')
        model = state['loadedModel']['id']
        body = dict(model=model, messages=[dict(role='user', content='Write a long story about a lighthouse, with many details.')],
                    max_tokens=256, temperature=0, stream=False)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(request, 'chat/completions', body)
            deadline = time.monotonic() + 120
            while not future.done() and time.monotonic() < deadline:
                collect(0.1)
            assert future.done(), 'Generation timed out'
            response = future.result()
        collect(0.8)
        # Resizing forces a complete frame so final counters can be compared with HTTP usage.
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 30, 101, 0, 0))
        process.send_signal(signal.SIGWINCH)
        collect(0.5)
        completed = response['usage']['completion_tokens']
        expected = f'Completed · {completed} / 256 token limit'.encode()
        assert expected in data, 'Dashboard token count differs from HTTP usage'
        assert response['usage']['completion_tokens'] > 0, response
        assert b'Generating' in data, 'No live generation phase was painted'
        assert b'output limit' in data and b'tok/s' in data, 'Token progress or throughput missing'
        assert b'First token' in data, 'First-token timing missing'
        assert b'MLX budget' in data and b'Cache' in data and b'Peak' in data, 'Memory dashboard missing'
        os.write(master, b'q')
        deadline = time.monotonic() + 10
        while process.poll() is None and time.monotonic() < deadline:
            collect(0.1)
        assert process.poll() == 0, 'Console did not quit cleanly'
        assert termios.tcgetattr(slave) == original, 'Terminal modes were not restored'
        assert b'\x1b[?1049l' in data and b'\x1b[?25h' in data, 'Terminal screen or cursor was not restored'
        print(f'PASS live text dashboard: {response["usage"]["completion_tokens"]} completion tokens')
    finally:
        Path('/tmp/midnight-dashboard-pty.ansi').write_bytes(data)
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)
        os.close(slave)
