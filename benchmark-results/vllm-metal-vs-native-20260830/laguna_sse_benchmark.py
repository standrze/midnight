#!/usr/bin/env python3

import argparse
import hashlib
import http.client
import json
import statistics
import time
from pathlib import Path


PROMPT = (
    "Write a long, detailed technical tutorial about implementing a lock-free "
    "work-stealing scheduler in Swift. Continue with implementation details and "
    "code examples until the output limit; do not conclude or summarize early."
)


def run_request(host, port, model, max_tokens, return_token_ids=False):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_completion_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if return_token_ids:
        body["return_token_ids"] = True

    connection = http.client.HTTPConnection(host, port, timeout=900)
    encoded = json.dumps(body).encode()
    started = time.perf_counter()
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=encoded,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Connection": "close",
        },
    )
    response = connection.getresponse()
    if response.status != 200:
        payload = response.read().decode(errors="replace")
        connection.close()
        raise RuntimeError(f"HTTP {response.status}: {payload}")

    first_content = None
    finish_time = None
    finish_reason = None
    usage = None
    prompt_token_ids = None
    pieces = []
    while True:
        raw = response.readline()
        if not raw:
            break
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        if prompt_token_ids is None and event.get("prompt_token_ids") is not None:
            prompt_token_ids = event["prompt_token_ids"]
        if event.get("usage") is not None:
            usage = event["usage"]
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                if first_content is None:
                    first_content = time.perf_counter()
                pieces.append(content)
            reason = choice.get("finish_reason")
            if reason is not None and finish_time is None:
                finish_time = time.perf_counter()
                finish_reason = reason

    ended = time.perf_counter()
    connection.close()
    if first_content is None or finish_time is None:
        raise RuntimeError("Streaming response lacked visible content or finish_reason")
    if usage is None:
        raise RuntimeError("Streaming response lacked the requested usage block")

    completion_tokens = int(usage["completion_tokens"])
    prompt_tokens = int(usage["prompt_tokens"])
    decode_seconds = finish_time - first_content
    content = "".join(pieces)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "ttft_seconds": first_content - started,
        "decode_seconds": decode_seconds,
        "e2e_seconds": finish_time - started,
        "stream_close_seconds": ended - started,
        "decode_tokens_per_second": (
            (completion_tokens - 1) / decode_seconds if completion_tokens > 1 else 0
        ),
        "e2e_tokens_per_second": completion_tokens / (finish_time - started),
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "content": content,
        "prompt_token_ids": prompt_token_ids,
    }


def median(records, field):
    return statistics.median(record[field] for record in records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--return-token-ids", action="store_true")
    args = parser.parse_args()

    warmups = []
    trials = []
    for index in range(args.warmups):
        record = run_request(
            args.host, args.port, args.model, args.tokens, args.return_token_ids
        )
        warmups.append(record)
        print(
            f"warmup {index + 1}/{args.warmups}: "
            f"{record['decode_tokens_per_second']:.2f} tok/s, "
            f"TTFT {record['ttft_seconds'] * 1000:.2f} ms"
        )
    for index in range(args.trials):
        record = run_request(
            args.host, args.port, args.model, args.tokens, args.return_token_ids
        )
        if record["completion_tokens"] != args.tokens:
            raise RuntimeError(
                f"Trial generated {record['completion_tokens']} tokens, expected {args.tokens}"
            )
        if record["finish_reason"] != "length":
            raise RuntimeError(f"Unexpected finish_reason: {record['finish_reason']}")
        trials.append(record)
        print(
            f"trial {index + 1}/{args.trials}: "
            f"{record['decode_tokens_per_second']:.2f} tok/s, "
            f"TTFT {record['ttft_seconds'] * 1000:.2f} ms, "
            f"E2E {record['e2e_seconds']:.3f} s"
        )

    report = {
        "host": args.host,
        "port": args.port,
        "model": args.model,
        "prompt": PROMPT,
        "requested_tokens": args.tokens,
        "warmups": warmups,
        "trials": trials,
        "median_decode_tokens_per_second": median(
            trials, "decode_tokens_per_second"
        ),
        "median_ttft_milliseconds": median(trials, "ttft_seconds") * 1000,
        "median_e2e_tokens_per_second": median(trials, "e2e_tokens_per_second"),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"median decode {report['median_decode_tokens_per_second']:.2f} tok/s; "
        f"median TTFT {report['median_ttft_milliseconds']:.2f} ms"
    )


if __name__ == "__main__":
    main()
