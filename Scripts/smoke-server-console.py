#!/usr/bin/env python3
"""Exercise Midnight's console and terminal restoration without loading weights."""
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
import urllib.request

BINARY = str(Path(sys.argv[1] if len(sys.argv) > 1 else '.build/debug/midnight').resolve())
KEY = 'midnight-console-smoke-key-0000000000000000'


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def exercise(exit_key=None, exit_signal=None, plain=False, missing_model=False, bind_failure=False):
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 24, 80, 0, 0))
    original = termios.tcgetattr(slave)
    data = bytearray()
    port = free_port()
    blocker = None
    if bind_failure:
        blocker = socket.socket()
        blocker.bind(('127.0.0.1', port))
        blocker.listen()
    with tempfile.TemporaryDirectory(prefix='midnight-console-smoke-') as directory:
        environment = dict(os.environ, TERM='xterm-256color', MIDNIGHT_API_KEY=KEY,
                           MODEL_RUNNER_MODELS_DIR=directory,
                           MIDNIGHT_MODEL_AVAILABILITY_FILE=str(Path(directory) / 'availability.json'))
        fixture = Path(directory) / '000-console-smoke'
        fixture.mkdir()
        fixture_source = Path(__file__).resolve().parent.parent / 'Tests/Fixtures/MistralTiny/config.json'
        (fixture / 'config.json').write_bytes(fixture_source.read_bytes())
        (fixture / 'weights.safetensors').write_bytes(b'not real weights')
        config = Path(directory) / 'settings.json'
        config.write_text('{}')
        arguments = [BINARY, '--config', str(config), '--port', str(port), '--verbose']
        if plain:
            arguments += ['--idle', '--no-ui']
        if missing_model:
            arguments += ['--model', str(Path(directory) / 'missing')]
        process = subprocess.Popen(arguments, stdin=slave, stdout=slave, stderr=slave, env=environment)

        def collect(seconds):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if select.select([master], [], [], 0.05)[0]:
                    data.extend(os.read(master, 65536))

        try:
            deadline = time.monotonic() + 15
            while process.poll() is None and time.monotonic() < deadline:
                collect(0.1)
                if bind_failure:
                    continue
                try:
                    request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/runtime',
                                                     headers={'Authorization': f'Bearer {KEY}'})
                    with urllib.request.urlopen(request, timeout=0.2) as response:
                        state = json.load(response)
                    assert state['phase'] == 'empty', state
                    break
                except (OSError, TimeoutError):
                    pass
            else:
                if not bind_failure:
                    raise AssertionError(f'Listener never became ready: {bytes(data)!r}')
            collect(0.7)
            if not bind_failure:
                if not plain:
                    assert b'\x1b[?1049h' in data, 'Alternate screen not entered'
                    assert b'Installed models' in data
                    assert b'tok/s' in data and b'Requests' in data
                    assert b'Cache' in data and b'Peak' in data
                    assert b'Enter' in data and b'load' in data
                    # Toggle availability without loading the synthetic checkpoint.
                    os.write(master, b'a')
                    collect(0.7)
                    models_request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/models',
                                                            headers={'Authorization': f'Bearer {KEY}'})
                    with urllib.request.urlopen(models_request) as response:
                        assert '000-console-smoke' not in [item['id'] for item in json.load(response)['data']]
                    saved = json.loads((Path(directory) / 'availability.json').read_text())
                    assert fixture.name in [Path(name).name for name in saved['unavailableModels']]
                    os.write(master, b'\r')
                    collect(0.3)
                    assert b'Press' in data, 'Unavailable model was not blocked in the console'
                    os.write(master, b'a')
                    collect(0.7)
                    with urllib.request.urlopen(models_request) as response:
                        assert '000-console-smoke' in [item['id'] for item in json.load(response)['data']]
                    # Trigger refresh, unload and narrow-terminal rendering.
                    os.write(master, b'ru')
                    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 12, 40, 0, 0))
                    process.send_signal(signal.SIGWINCH)
                    collect(0.6)
                    assert b"\x1b[12;2H" in data, "Console did not repaint at the resized height"
                    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 24, 80, 0, 0))
                    process.send_signal(signal.SIGWINCH)
                    collect(0.3)
                if exit_signal:
                    process.send_signal(exit_signal)
                else:
                    os.write(master, exit_key)
            deadline = time.monotonic() + 10
            while process.poll() is None and time.monotonic() < deadline:
                collect(0.1)
            assert process.poll() is not None, 'Process did not exit'
            collect(0.1)
            assert termios.tcgetattr(slave) == original, 'Terminal modes were not restored'
            if not plain:
                assert b'\x1b[?1049l' in data, 'Alternate screen not restored'
                assert b'\x1b[?25h' in data, 'Cursor not restored'
                assert process.returncode == (1 if bind_failure else 0), (process.returncode, bytes(data))
            else:
                assert b'\x1b[' not in data, 'Plain output contains terminal controls'
                assert b'Listening:' in data
            if missing_model:
                assert b'Initial model load failed' in data
        finally:
            Path('/tmp/midnight-console-pty.ansi').write_bytes(data)
            if process.poll() is None:
                process.kill()
                process.wait()
            if blocker:
                blocker.close()
            os.close(master)
            os.close(slave)


for options in [dict(exit_key=b'q'), dict(exit_key=b'\x03'), dict(exit_signal=signal.SIGTERM),
                dict(exit_key=b'q', missing_model=True), dict(bind_failure=True),
                dict(exit_signal=signal.SIGTERM, plain=True)]:
    exercise(**options)
    print(f'PASS {options}')
print('Server console PTY smoke passed.')
