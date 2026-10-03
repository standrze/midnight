#!/usr/bin/env python3
"""Exercise OpenAI response_format against a real locally installed model.

Build Midnight first, then run from any directory:
  python3 Scripts/SmokeStructuredOutput.py --model /path/to/model
  python3 Scripts/SmokeStructuredOutput.py --model /path/to/model \
      --binary /path/to/midnight --port 18843

Uses only Python's standard library. Starts its own loopback server, stops it
on completion, and retains the temporary server log when a check fails.
"""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from midnight_api_auth import authorization_headers, json_headers


def require(condition, detail):
    if not condition:
        raise RuntimeError(str(detail))


def payload(fmt=None, stream=False, prompt="Say hello briefly.", **extras):
    result = {
        "model": "structured-smoke",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 128,
        "stream": stream,
    }
    if fmt is not None:
        result["response_format"] = fmt
    result.update(extras)
    return result


def schema(properties):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "test",
            "strict": True,
            "schema": {
                "type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False,
            },
        },
    }


def run_checks(base, server):
    def request(body):
        req = urllib.request.Request(
            base + "/v1/chat/completions", json.dumps(body).encode(),
            json_headers())
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                return response.status, response.read().decode(), response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode(), error.headers.get("Content-Type", "")

    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        require(server.poll() is None, "Server exited before the model became ready.")
        try:
            with urllib.request.urlopen(urllib.request.Request(base + "/v1/models", headers=authorization_headers()), timeout=1) as response:
                if json.load(response).get("data"):
                    break
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.5)
    else:
        raise RuntimeError("Model did not become ready within 180 seconds.")

    enum_format = schema({"answer": {"type": "string", "enum": ["hello", "goodbye"]}})
    for label, fmt, prompt in [
        ("json_object", {"type": "json_object"},
         "Return a JSON object with the key answer and value hello."),
        ("schema_enum", enum_format, "Say hello."),
        ("schema_unicode", schema({"answer": {"type": "string"}}),
         "Put the Japanese greeting こんにちは in answer."),
        ("schema_nested", schema({
            "items": {"type": "array", "items": {"type": "integer"}},
            "ok": {"type": "boolean"}, "note": {"type": ["string", "null"]},
        }), "Use items [1,2], ok true and note null."),
    ]:
        started = time.monotonic()
        status, raw, _ = request(payload(fmt, prompt=prompt))
        require(status == 200, (label, status, raw))
        result = json.loads(raw)
        choice = result["choices"][0]
        output = json.loads(choice["message"]["content"])
        require(isinstance(output, dict) and choice["finish_reason"] == "stop", result)
        if label == "schema_enum":
            require(output["answer"] in ["hello", "goodbye"], output)
        if label == "schema_unicode":
            require(set(output) == {"answer"} and isinstance(output["answer"], str), output)
        if label == "schema_nested":
            require(set(output) == {"items", "ok", "note"}, output)
            require(isinstance(output["items"], list)
                    and all(type(item) is int for item in output["items"]), output)
            require(type(output["ok"]) is bool
                    and (output["note"] is None or isinstance(output["note"], str)), output)
        print(label, json.dumps(output, ensure_ascii=False),
              f"{time.monotonic() - started:.2f}s", flush=True)

    status, raw, content_type = request(payload(enum_format, stream=True))
    require(status == 200 and "text/event-stream" in content_type, (status, raw))
    require("data: [DONE]" in raw, "Streaming response did not end with [DONE].")
    events = [json.loads(line[6:]) for line in raw.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    require(all("error" not in event for event in events), events)
    choices = [choice for event in events for choice in event.get("choices", [])]
    output = "".join(choice.get("delta", {}).get("content", "") or "" for choice in choices)
    require(json.loads(output)["answer"] in ["hello", "goodbye"], output)
    require(any(choice.get("finish_reason") == "stop" for choice in choices), events)
    print("stream_schema", output, flush=True)

    status, raw, _ = request(payload(enum_format, max_tokens=1))
    require(status == 200 and json.loads(raw)["choices"][0]["finish_reason"] == "length", raw)
    print("token_limit_preserved", flush=True)

    tools = [{"type": "function", "function": {
        "name": "test", "parameters": {"type": "object"},
    }}]
    for label, bad in [
        ("unsupported_schema", payload(schema({"answer": {"type": "string", "pattern": "x"}}))),
        ("unknown_format", payload({"type": "xml"})),
        ("custom_stop", payload(enum_format, stop=["}"])),
        ("enabled_tools", payload(enum_format, tools=tools)),
    ]:
        bad["stream"] = True
        status, raw, content_type = request(bad)
        require(status == 400 and "application/json" in content_type, (label, status, raw))
        error = json.loads(raw)["error"]
        require("response_format" in str(error.get("param", "")), (label, error))
        print(label, "rejected before streaming", flush=True)

    status, raw, _ = request(payload(enum_format, tools=tools, tool_choice="none"))
    require(status == 200, (status, raw))
    choice = json.loads(raw)["choices"][0]
    require(choice["finish_reason"] == "stop", choice)
    require(not choice["message"].get("tool_calls"), choice)
    require(json.loads(choice["message"]["content"])["answer"] in ["hello", "goodbye"], choice)
    print("disabled_tools_schema", choice["message"]["content"], flush=True)

    for label, body in [("text_regression", payload({"type": "text"})),
                        ("omitted_format_regression", payload())]:
        status, raw, _ = request(body)
        require(status == 200 and json.loads(raw)["choices"][0]["message"]["content"], (label, raw))
        print(label, "passed", flush=True)


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path, help="Directory containing a local MLX model")
    parser.add_argument("--binary", type=Path, default=root / ".build/debug/midnight",
                        help="Midnight executable (default: repository .build/debug/midnight)")
    parser.add_argument("--port", type=int, default=18843, help="Unused loopback port (default: 18843)")
    args = parser.parse_args()
    model = args.model.expanduser().resolve()
    binary = args.binary.expanduser().resolve()
    if not model.is_dir():
        parser.error(f"Model directory does not exist: {model}")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f"Midnight executable is missing or not executable: {binary}")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
    except OSError as error:
        parser.error(f"Port {args.port} is unavailable: {error}")

    log = tempfile.NamedTemporaryFile(mode="w", prefix="midnight-structured-", suffix=".log", delete=False)
    log_path = Path(log.name)
    server = None
    exit_code = 0
    try:
        server = subprocess.Popen([
            str(binary), "--model", str(model), "--host", "127.0.0.1",
            "--port", str(args.port), "--name", "structured-smoke", "--max-tokens", "256",
        ], cwd=root, stdout=log, stderr=subprocess.STDOUT)
        run_checks(f"http://127.0.0.1:{args.port}", server)
    except KeyboardInterrupt:
        print("Smoke test interrupted.", file=sys.stderr)
        exit_code = 130
    except Exception as error:
        print(f"Smoke test failed: {error}", file=sys.stderr)
        exit_code = 1
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        log.close()
    if exit_code:
        print(f"Server log: {log_path}\n{log_path.read_text(errors='replace')[-8000:]}", file=sys.stderr)
    else:
        log_path.unlink()
        print("All structured-output smoke checks passed.", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
