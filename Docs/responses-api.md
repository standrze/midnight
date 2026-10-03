# OpenAI Responses API

Midnight serves OpenAI's Responses format alongside Chat Completions. It runs
the loaded local model and uses the same inference, admission, cancellation,
tool parsing, and structured-output engine.

## OpenAI SDK usage

Use the name returned by `GET /v1/models`:

```python
import os
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key=os.environ["MIDNIGHT_API_KEY"])
response = client.responses.create(
    model="my-local-model",
    input="Explain affine quantization in one paragraph.",
    max_output_tokens=128,
)
print(response.output_text)

followup = client.responses.create(
    model="my-local-model",
    previous_response_id=response.id,
    input="Give a short example.",
    max_output_tokens=128,
)
print(followup.output_text)
```

The SDK sends `MIDNIGHT_API_KEY` as a bearer credential, which Midnight checks
on every request. `base_url` selects the local server. Model capabilities still depend on the
loaded checkpoint and its supported tokenizer and tool parser.

## Routes and conversation history

| Method | Route | Behavior |
| --- | --- | --- |
| POST | `/v1/responses` | Create a response, optionally streamed |
| GET | `/v1/responses/{response_id}` | Retrieve a stored response |
| DELETE | `/v1/responses/{response_id}` | Delete one stored response |
| GET | `/v1/responses/{response_id}/input_items` | List normalized inputs, including inherited history |

`store` defaults to true. Midnight keeps up to 128 results and 64 MiB of encoded
history in process memory, for at most one hour. The oldest stored responses
are evicted when capacity is reached. Restarting the server clears history.
No response archive is written to disk. An individual response too large for
the store fails explicitly; use `store:false` or reduce its input.

`store:false` returns a result without making its ID retrievable or usable for
future continuation. It can still use an existing `previous_response_id` for
that request. For stateless conversations, send prior input and `response.output`
items explicitly in the next `input` array.

`previous_response_id` restores the prior inputs and outputs. Its top-level
`instructions` do not carry forward; specify instructions again when needed.
Explicit system/developer messages in `input` remain part of the history.
The previous response must belong to the same served model name. Retrieval
and deletion remain available while the model is unloaded. Stored branches
are independent snapshots; deleting a parent does not delete a saved child.

`input_items` accepts `order=asc|desc` (default desc), `limit=1..100` (default 20),
and an `after` item ID. It returns `data`, `first_id`, `last_id`, and `has_more`.
The list excludes top-level instructions. Missing, deleted, or expired response
IDs return HTTP 404, including when passed as `previous_response_id`.

## Streaming and structured output

```python
with client.responses.stream(
    model="my-local-model",
    input="Say hello briefly.",
    max_output_tokens=128,
) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
    response = stream.get_final_response()
```

Streams use named SSE events with monotonically increasing `sequence_number`
values: response created/in-progress, output item and content part lifecycle,
text or function-argument deltas, then completed/incomplete/failed. Items keep
stable IDs and output indices. The final event contains the response and usage.
Responses streams end with that typed event; Chat Completions retains its
`data: [DONE]` convention.

Use `text.format` for constrained JSON:

```python
response = client.responses.create(
    model="my-local-model",
    input="Give a short greeting.",
    max_output_tokens=128,
    text={"format": {
        "type": "json_schema",
        "name": "greeting",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    }},
)
```

`text` and `json_object` formats are supported too. `client.responses.parse()`
works with Pydantic schemas within Midnight's [supported schema subset](structured-output.md).
An exhausted budget returns `status:"incomplete"` and
`incomplete_details.reason:"max_output_tokens"`; JSON may be partial. Generation
errors after streaming starts produce `response.failed`, never a successful
completion. Disconnects cancel inference.

## Function tools

Responses uses flattened function definitions and separate call/result items:

```python
tool = {
    "type": "function", "name": "get_weather",
    "description": "Get weather for a city.", "strict": False,
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"], "additionalProperties": False,
    },
}
response = client.responses.create(
    model="my-local-model", input="What is the weather in Boston?", tools=[tool],
)
# For each function_call in response.output, execute the named function in your
# application and send its string result with the SAME call_id:
result_input = [{"type": "function_call_output", "call_id": call.call_id,
                 "output": '{"weather":"sunny"}'}
                for call in response.output if call.type == "function_call"]
```

Send those results as `input` with `previous_response_id=response.id` and the
same tools. Midnight emits function calls; the client executes them. A call's
arguments are currently streamed as one complete delta after the model parser
recognizes the call. `auto`, `none`, `required`, and named function choices map
to existing model support; required/named choices need a compatible backend.
`parallel_tool_calls:false` prevents multiple calls from being returned as a
successful response; a model that violates it fails the request.

Tool schemas are best effort when strictness is omitted or false, and the
response echoes `strict:false`. Explicit `strict:true` function tools are
rejected because strict argument generation is not implemented. Structured
`text.format` requires tools to be absent or disabled with `tool_choice:"none"`.

## Supported controls and limits

Supported controls include `model`, text `input`, `instructions`,
`max_output_tokens`, `temperature`, `top_p`, `stream`, `store`,
`previous_response_id`, `metadata`, `tools`, `tool_choice`,
`parallel_tool_calls`, `text.format`, and existing `reasoning.effort`
values low/medium/high on models that support them. Output-token limits follow
the loaded model's configured ceiling. `top_p` must be greater than zero.
Usage uses OpenAI token-count fields; the runtime does not separately measure
reasoning tokens, so `output_tokens_details.reasoning_tokens` is zero.
It also reports `input_tokens_details.cache_write_tokens:0`; separate cache
write-token accounting is not available from this runtime.

User/safety identifiers and `prompt_cache_key` are accepted as metadata; they
do not add authentication, safety filtering, or a new cache policy.
`background:false`, `truncation:"disabled"`, default service tier, empty
`include`, and `top_logprobs:0` are accepted. Stream obfuscation is not emitted;
an explicit `stream_options.include_obfuscation:true` is rejected.

Unsupported controls return explicit errors before streaming. These include
background jobs/cancellation by ID, Conversations API, automatic truncation,
reasoning summaries or encrypted reasoning items, verbosity control, logprobs,
images/audio/files, built-in hosted tools, MCP/custom tools, and strict function
arguments. WebSockets, embeddings, and the broader OpenAI hosted services are
separate capabilities. This endpoint implements the documented local subset.

## Verification

```sh
python3 Scripts/SmokeResponses.py --model /path/to/model
# In an environment with the OpenAI Python SDK and Pydantic installed:
python3 Scripts/SmokeResponses.py --model /path/to/model --sdk --tools
```

The script starts and stops a loopback server. It checks creation, schemas,
stream events, token limits, stored history, pagination, deletion, stateless
replay, validation, and existing chat. `--sdk` exercises create, stream, parse,
retrieve, input listing, and deletion through the official SDK with strict
response validation. `--tools` additionally checks actual model function calls
and result continuation, which requires a tool-capable model.

The wire contract follows OpenAI's [Responses migration guide](https://developers.openai.com/api/docs/guides/migrate-to-responses),
[streaming events](https://developers.openai.com/api/docs/guides/streaming-responses),
and [function-calling guide](https://developers.openai.com/api/docs/guides/function-calling),
subject to the explicit local limits above.

### Output limits

Model metadata advertises `max_output_tokens` and `default_output_tokens`. Midnight bounds each response by the exact remaining context after rendering its complete input history and instructions. The response object reports the effective `max_output_tokens`, which can be lower than the requested allowance when context is nearly full. Explicit operator caps remain enforced.
