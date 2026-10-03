#!/usr/bin/env python3
"""Check Talkie HTTP integration on an owned, isolated loopback listener.

Uses an existing local checkpoint and release binary; never downloads or changes
models. Timing fields are diagnostics, not performance measurements.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
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


ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("midnight_performance", ROOT / "Scripts/Performance/compare.py")
performance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(performance)
ROLE_MARKERS = ("<|user|>", "<|assistant|>", "<|system|>", "<|end|>", "<|endoftext|>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / ".build/release/midnight")
    parser.add_argument("--model", type=Path, default=Path("/private/tmp/midnight-talkie-model"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fuse-projections", choices=("0", "1"), default="1")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or ROOT / "benchmark-results/talkie-20260913" / ("http-" + stamp)).resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary, model = args.binary.resolve(), args.model.resolve()
    evidence = {
        "created_at": stamp, "binary": str(binary), "model": str(model),
        "scope": "HTTP correctness only; timings are diagnostics under uncontrolled machine load",
        "fusion": args.fuse_projections == "1", "checks": [], "ok": False,
    }
    process = None
    log = (output / "server.log").open("w")

    def save():
        (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")

    def record(name, result, passed):
        evidence["checks"].append({"name": name, "ok": bool(passed), "result": result})
        save()
        print(("PASS " if passed else "FAIL ") + name, flush=True)

    def clean_text(value):
        return isinstance(value, str) and bool(value.strip()) and not any(marker in value for marker in ROLE_MARKERS)

    def payload(messages, *, tokens=64, stream=False):
        result = {"model": "talkie-smoke", "messages": messages, "temperature": 0,
                  "max_tokens": tokens, "stream": stream}
        if stream:
            result["stream_options"] = {"include_usage": True}
        return result

    def api(path, body=None, timeout=None):
        start = time.perf_counter()
        request = urllib.request.Request(
            base + path, data=None if body is None else json.dumps(body).encode(),
            headers=json_headers())
        try:
            response = urllib.request.urlopen(request, timeout=timeout or args.timeout)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read().decode("utf-8")
            return {"http_status": response.status, "request": body,
                    "response": json.loads(raw), "total_ms": (time.perf_counter() - start) * 1000}

    def completed_chat(row):
        if row["http_status"] != 200:
            return False
        body = row["response"]
        choices = body.get("choices", [])
        usage = body.get("usage", {})
        return (len(choices) == 1 and choices[0].get("message", {}).get("role") == "assistant"
                and clean_text(choices[0]["message"].get("content"))
                and choices[0].get("finish_reason") in ("stop", "length")
                and type(usage.get("prompt_tokens")) is int and usage["prompt_tokens"] > 0
                and type(usage.get("completion_tokens")) is int and usage["completion_tokens"] > 0
                and usage.get("total_tokens") == usage["prompt_tokens"] + usage["completion_tokens"])

    def completed_stream(row):
        # stream_request verifies [DONE], finish reason, complete numeric usage,
        # and prompt/output/total arithmetic, retaining all parsed SSE events.
        return row["ok"] and clean_text(row["content"]) and row["finish"] in ("stop", "length")

    try:
        if not binary.is_file() or not (model / "config.json").is_file():
            raise ValueError("An existing release binary and checkpoint config.json are required")
        with tempfile.TemporaryDirectory(prefix="midnight-talkie-http-") as temporary:
            config = Path(temporary) / "config.json"
            config.write_text('{"mlxRunner": {}}\n')
            (output / "config.json").write_text(config.read_text())
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            base = f"http://127.0.0.1:{port}"
            command = [str(binary), "--model", str(model), "--name", "talkie-smoke",
                       "--host", "127.0.0.1", "--port", str(port), "--config", str(config),
                       "--engine", "metal", "--context-length", "2048", "--max-tokens", "256",
                       "--prefill-step-size", "512", "--kv-compression", "none"]
            env = os.environ.copy()
            env["MIDNIGHT_TALKIE_FUSE_PROJECTIONS"] = args.fuse_projections
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       cwd=temporary, env=env)
            evidence.update(command=command, port=port, pid=process.pid)
            save()
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Owned server exited during startup; inspect server.log")
                try:
                    models = api("/v1/models", timeout=2)
                    if any(item.get("id") == "talkie-smoke" for item in models["response"].get("data", [])):
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError("Owned server startup timed out")
            record("model_ready", models, models["http_status"] == 200)
            evidence["runtime"] = api("/v1/runtime")["response"]

            messages = [{"role": "user", "content": "Write a short description of a quiet railway station at dawn."}]
            ordinary = api("/v1/chat/completions", payload(messages, tokens=96))
            record("ordinary_chat", ordinary, completed_chat(ordinary))

            streamed = performance.stream_request(base, payload([
                {"role": "user", "content": "Write two sentences about a lighthouse beside the sea."}
            ], stream=True), timeout=args.timeout)
            record("sse_content_terminal_finish_usage", streamed, completed_stream(streamed))

            if completed_chat(ordinary):
                messages += [ordinary["response"]["choices"][0]["message"],
                             {"role": "user", "content": "Rewrite that description in one sentence."}]
                multiturn = api("/v1/chat/completions", payload(messages))
                record("multiturn_chat", multiturn, completed_chat(multiturn))
            else:
                record("multiturn_chat", {"error": "First assistant response unavailable"}, False)

            too_long = api("/v1/chat/completions", payload([
                {"role": "user", "content": "The " * 3000}
            ], tokens=1))
            error = too_long["response"].get("error", {})
            record("over_context_rejected", too_long,
                   too_long["http_status"] == 400 and bool(error)
                   and any(word in json.dumps(error).lower() for word in ("context", "token")))

            canceled = performance.stream_request(base, payload([
                {"role": "user", "content": "Write a long account of a sea voyage, describing each day in detail."}
            ], tokens=256, stream=True), timeout=args.timeout, cancel=True)
            record("cancel_after_first_visible_content", canceled,
                   canceled["ok"] and canceled.get("cancelled") and clean_text(canceled["content"]))
            attempts = []
            deadline = time.monotonic() + 30
            while True:
                following = performance.stream_request(base, payload([
                    {"role": "user", "content": "Write one sentence about a spring morning."}
                ], tokens=48, stream=True), timeout=args.timeout)
                attempts.append(following)
                if following["http_status"] not in (409, 429, 503) or time.monotonic() >= deadline:
                    break
                time.sleep(0.2)
            record("response_after_cancel", {"attempts": attempts}, completed_stream(following))
            evidence["ok"] = all(check["ok"] for check in evidence["checks"])
    except Exception as error:
        evidence["error"] = f"{type(error).__name__}: {error}"
        print(evidence["error"], flush=True)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        evidence["owned_process_exit_code"] = None if process is None else process.returncode
        log.close()
        save()
        print("Evidence:", output, flush=True)
    return 0 if evidence["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
