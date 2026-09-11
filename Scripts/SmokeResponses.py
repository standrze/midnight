#!/usr/bin/env python3
"""Exercise Midnight's Responses API against a locally installed model.

Build Midnight first, then run:
  python3 Scripts/SmokeResponses.py --model /path/to/model
  python3 Scripts/SmokeResponses.py --model /path/to/model --sdk --tools

The default checks use only the standard library. --sdk additionally requires
the already installed openai and pydantic packages; --tools tests model-dependent
automatic function calling. Starts a loopback server and stops it on completion.
Server logs are retained on failure. No packages or models are downloaded.
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
from typing import Literal
import urllib.error
import urllib.parse
import urllib.request


MODEL = "responses-smoke"
GREETING_FORMAT = {
    "type": "json_schema", "name": "greeting", "strict": True,
    "schema": {
        "type": "object", "properties": {"answer": {"type": "string", "enum": ["hello"]}},
        "required": ["answer"], "additionalProperties": False,
    },
}
WEATHER_TOOL = {
    "type": "function", "name": "get_weather", "description": "Get current weather for a city.",
    "strict": False, "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"], "additionalProperties": False,
    },
}


def require(condition, detail):
    if not condition:
        raise RuntimeError(str(detail))


def payload(**extras):
    body = {"model": MODEL, "input": "Say hello briefly.", "temperature": 0, "max_output_tokens": 96}
    body.update(extras)
    return body


def output_text(response):
    return "".join(part.get("text", "") for item in response["output"]
                   if item["type"] == "message" for part in item["content"]
                   if part["type"] == "output_text")


def check_greeting(response):
    require(response["status"] == "completed", response)
    require(json.loads(output_text(response)) == {"answer": "hello"}, response)


class API:
    def __init__(self, base):
        self.base = base

    def raw(self, path, body=None, method=None):
        request = urllib.request.Request(
            self.base + path, None if body is None else json.dumps(body).encode(),
            {"Content-Type": "application/json"}, method=method)
        try:
            response = urllib.request.urlopen(request, timeout=180)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, response.read().decode(), response.headers.get("Content-Type", "")

    def json(self, path, body=None, method=None, status=200):
        actual, raw, content_type = self.raw(path, body, method)
        require(actual == status and "application/json" in content_type, (path, actual, raw))
        return json.loads(raw)

    def create(self, **extras):
        return self.json("/v1/responses", payload(**extras))

    def input_items(self, response_id, **query):
        query = {"order": "asc", **query}
        result = self.json(f"/v1/responses/{response_id}/input_items?" + urllib.parse.urlencode(query))
        require(result["object"] == "list" and isinstance(result["data"], list), result)
        return result


def sse_events(raw):
    events = []
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        lines = block.splitlines()
        data = "\n".join(line[5:].lstrip(" ") for line in lines if line.startswith("data:"))
        if not data:
            continue
        require(data != "[DONE]", "Responses streams must terminate with a typed lifecycle event.")
        event = json.loads(data)
        names = [line[6:].strip() for line in lines if line.startswith("event:")]
        require(names == [event["type"]], ("SSE name and payload differ", block))
        events.append(event)
    require(events, "No SSE events received.")
    require([event["sequence_number"] for event in events] == list(range(len(events))), events)
    return events


def run_checks(api, tools=False):
    text = api.create(max_output_tokens=48)
    require(text["object"] == "response" and text["status"] == "completed" and output_text(text), text)
    usage = text["usage"]
    require(usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"], usage)
    require("cached_tokens" in usage["input_tokens_details"]
            and "reasoning_tokens" in usage["output_tokens_details"], usage)
    print("responses_text_and_usage passed", flush=True)

    first = api.create(text={"format": GREETING_FORMAT}, instructions="TOP_LEVEL_OLD_8f24. Say hello.")
    check_greeting(first)
    require(api.json(f"/v1/responses/{first['id']}") == first, "Stored response differs from create result.")
    original_inputs = api.input_items(first["id"])["data"]
    require(len(original_inputs) == 1 and original_inputs[0]["role"] == "user", original_inputs)
    print("responses_schema_and_stored_retrieval passed", flush=True)

    status, raw, content_type = api.raw("/v1/responses", payload(text={"format": GREETING_FORMAT}, stream=True))
    require(status == 200 and "text/event-stream" in content_type, (status, raw))
    events = sse_events(raw)
    names = [event["type"] for event in events]
    require(names[:2] == ["response.created", "response.in_progress"], names)
    require(names[-1] == "response.completed" and "response.failed" not in names, names)
    for name in ["response.output_item.added", "response.content_part.added",
                 "response.output_text.delta", "response.output_text.done",
                 "response.content_part.done", "response.output_item.done"]:
        require(name in names, ("Missing lifecycle event", name, names))
    streamed = events[-1]["response"]
    check_greeting(streamed)
    deltas = "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta")
    require(deltas == output_text(streamed), (deltas, streamed))
    done_items = [event["item"] for event in events if event["type"] == "response.output_item.done"]
    require(done_items == streamed["output"], (done_items, streamed))
    print("responses_sse_lifecycle passed", flush=True)

    limited = api.create(text={"format": GREETING_FORMAT}, max_output_tokens=1)
    require(limited["status"] == "incomplete" and limited["error"] is None, limited)
    require(limited["incomplete_details"] == {"reason": "max_output_tokens"}, limited)
    require(all(item["status"] == "incomplete" for item in limited["output"]), limited)
    print("responses_token_limit passed", flush=True)

    continued = api.create(previous_response_id=first["id"], input="Say hello again.",
                           instructions="TOP_LEVEL_NEW_f512. Give a greeting.", text={"format": GREETING_FORMAT})
    check_greeting(continued)
    require(continued["instructions"] == "TOP_LEVEL_NEW_f512. Give a greeting.", continued)
    history = api.input_items(continued["id"])["data"]
    prior_count = len(original_inputs) + len(first["output"])
    require(history[:prior_count] == original_inputs + first["output"], history)
    require(len(history) == prior_count + 1 and history[-1]["role"] == "user", history)
    require("TOP_LEVEL_OLD_8f24" not in json.dumps(history), history)
    page = api.input_items(continued["id"], limit=1)
    require(page["data"] == history[:1] and page["has_more"] is True, page)
    following = api.input_items(continued["id"], after=page["last_id"])
    require(following["data"] == history[1:] and following["has_more"] is False, following)
    print("responses_continuation_instructions_and_pagination passed", flush=True)

    unstored = api.create(store=False, text={"format": GREETING_FORMAT})
    check_greeting(unstored)
    api.json(f"/v1/responses/{unstored['id']}", status=404)
    deleted = api.json(f"/v1/responses/{first['id']}", method="DELETE")
    require(deleted["id"] == first["id"] and deleted["deleted"] is True, deleted)
    api.json(f"/v1/responses/{first['id']}", status=404)
    api.json("/v1/responses", payload(previous_response_id=first["id"]), status=404)
    print("responses_store_false_and_delete passed", flush=True)

    replay_items = [
        {"role": "user", "content": "Look up the weather and then greet me."},
        *first["output"],
        {"type": "function_call", "id": "fc_smoke_replay", "call_id": "call_smoke_replay",
         "name": "get_weather", "arguments": '{"city":"Boston"}', "status": "completed"},
        {"type": "function_call_output", "call_id": "call_smoke_replay", "output": '{"weather":"sunny"}'},
        {"role": "user", "content": "Now say hello."},
    ]
    replayed = api.create(input=replay_items, tools=[WEATHER_TOOL], tool_choice="none",
                          text={"format": GREETING_FORMAT})
    check_greeting(replayed)
    replay_history = api.input_items(replayed["id"])["data"]
    require(len(replay_history) == len(replay_items), replay_history)
    require(replay_history[1] == first["output"][0], replay_history)
    require(replay_history[-3]["call_id"] == replay_history[-2]["call_id"] == "call_smoke_replay", replay_history)
    print("responses_stateless_output_and_function_result_replay passed", flush=True)

    for label, extras in [
        ("unsupported_background", {"background": True}),
        ("unsupported_field", {"not_a_response_option": True}),
        ("unsupported_image", {"input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": "https://example.invalid/image.png"}]}]}),
        ("unsupported_strict_tools", {"tools": [{**WEATHER_TOOL, "strict": True}]}),
    ]:
        status, raw, content_type = api.raw("/v1/responses", payload(stream=True, **extras))
        require(status == 400 and "application/json" in content_type, (label, status, raw))
        error = json.loads(raw)["error"]
        require(error["type"] == "invalid_request_error" and error.get("param"), (label, error))
        print(label, "rejected before streaming", flush=True)

    chat = api.json("/v1/chat/completions", {
        "model": MODEL, "messages": [{"role": "user", "content": "Say hello briefly."}],
        "temperature": 0, "max_tokens": 48,
    })
    require(chat["choices"][0]["message"]["content"], chat)
    print("ordinary_chat_regression passed", flush=True)
    if tools:
        run_tool_checks(api)


def run_tool_checks(api):
    status, raw, content_type = api.raw("/v1/responses", payload(
        stream=True, tools=[WEATHER_TOOL], tool_choice="auto", input=(
        "Call get_weather with city Boston to obtain the current weather. "
        "You need the tool result; do not guess the weather.")))
    require(status == 200 and "text/event-stream" in content_type, (status, raw))
    events = sse_events(raw)
    require(events[-1]["type"] == "response.completed", events[-1])
    called = events[-1]["response"]
    calls = [item for item in called["output"] if item["type"] == "function_call"]
    require(called["status"] == "completed" and calls, ("Model did not make the requested tool call", called))
    for call in calls:
        require(call["name"] == "get_weather" and isinstance(json.loads(call["arguments"]), dict), call)
        deltas = [event["delta"] for event in events
                  if event["type"] == "response.function_call_arguments.delta" and event["item_id"] == call["id"]]
        require("".join(deltas) == call["arguments"], (call, deltas))
        require(any(event["type"] == "response.function_call_arguments.done"
                    and event["item_id"] == call["id"] for event in events), call)
    result = api.create(previous_response_id=called["id"], tools=[WEATHER_TOOL], tool_choice="none", input=[
        {"type": "function_call_output", "call_id": call["call_id"], "output": '{"weather":"sunny"}'}
        for call in calls
    ])
    require(result["status"] == "completed" and output_text(result), result)
    print("responses_streamed_function_call_and_continuation passed", flush=True)


def run_sdk_checks(base):
    from openai import OpenAI, NotFoundError
    from pydantic import BaseModel

    class Greeting(BaseModel):
        answer: Literal["hello"]

    with OpenAI(base_url=base + "/v1", api_key="local", timeout=180, max_retries=0,
                _strict_response_validation=True) as client:
        created = client.responses.create(**payload(text={"format": GREETING_FORMAT}))
        require(json.loads(created.output_text) == {"answer": "hello"}, created)
        print("openai_sdk_create passed", flush=True)
        fetched = client.responses.retrieve(created.id)
        require(fetched.model_dump() == created.model_dump(), fetched)
        inputs = list(client.responses.input_items.list(created.id, order="asc", limit=1))
        require(len(inputs) == 1, inputs)
        with client.responses.stream(**payload(text={"format": GREETING_FORMAT})) as stream:
            chunks = [event.delta for event in stream if event.type == "response.output_text.delta"]
            final = stream.get_final_response()
        require("".join(chunks) == final.output_text, final)
        require(json.loads(final.output_text) == {"answer": "hello"}, final)
        print("openai_sdk_stream passed", flush=True)
        parsed = client.responses.parse(**payload(), text_format=Greeting)
        require(isinstance(parsed.output_parsed, Greeting) and parsed.output_parsed.answer == "hello", parsed)
        print("openai_sdk_parse passed", flush=True)
        replayed = client.responses.create(**payload(
            input=[*(item.model_dump() for item in parsed.output),
                   {"role": "user", "content": "Say hello again."}],
            text={"format": GREETING_FORMAT}))
        require(json.loads(replayed.output_text) == {"answer": "hello"}, replayed)
        print("openai_sdk_parsed_output_replay passed", flush=True)
        # Some SDK versions discard the deletion envelope and return None.
        client.responses.delete(created.id)
        try:
            client.responses.retrieve(created.id)
        except NotFoundError:
            pass
        else:
            raise RuntimeError("SDK-deleted response is still retrievable.")
    print("openai_sdk_create_stream_parse_retrieve_input_items_delete passed", flush=True)


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path, help="Directory containing a local MLX model")
    parser.add_argument("--binary", type=Path, default=root / ".build/debug/midnight")
    parser.add_argument("--port", type=int, default=18844, help="Unused loopback port (default: 18844)")
    parser.add_argument("--sdk", action="store_true", help="Also test already installed OpenAI/Pydantic SDKs")
    parser.add_argument("--tools", action="store_true", help="Also test model-dependent automatic function calling")
    args = parser.parse_args()
    model, binary = args.model.expanduser().resolve(), args.binary.expanduser().resolve()
    if not model.is_dir():
        parser.error(f"Model directory does not exist: {model}")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f"Midnight executable is missing or not executable: {binary}")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.sdk:
        try:
            import openai  # noqa: F401
            import pydantic  # noqa: F401
        except ImportError as error:
            parser.error(f"--sdk requires an environment with openai and pydantic installed: {error}")
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
    except OSError as error:
        parser.error(f"Port {args.port} is unavailable: {error}")

    base = f"http://127.0.0.1:{args.port}"
    log = tempfile.NamedTemporaryFile(mode="w", prefix="midnight-responses-", suffix=".log", delete=False)
    log_path, server, exit_code = Path(log.name), None, 0
    try:
        server = subprocess.Popen([
            str(binary), "--model", str(model), "--host", "127.0.0.1", "--port", str(args.port),
            "--name", MODEL, "--max-tokens", "256",
        ], cwd=root, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            require(server.poll() is None, "Server exited before the model became ready.")
            try:
                with urllib.request.urlopen(base + "/v1/models", timeout=1) as response:
                    if json.load(response).get("data"):
                        break
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("Model did not become ready within 180 seconds.")
        run_checks(API(base), tools=args.tools)
        if args.sdk:
            run_sdk_checks(base)
    except KeyboardInterrupt:
        print("Smoke test interrupted.", file=sys.stderr)
        exit_code = 130
    except Exception as error:
        print(f"Smoke test failed: {error}", file=sys.stderr)
        if error.__cause__ is not None:
            print(f"Caused by: {error.__cause__}", file=sys.stderr)
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
        print("All Responses API smoke checks passed.", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
