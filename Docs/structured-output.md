# OpenAI-compatible structured output

`POST /v1/chat/completions` accepts OpenAI's `response_format` request field:

| Type | Behavior |
| --- | --- |
| Omitted or `{"type":"text"}` | Ordinary chat output |
| `{"type":"json_object"}` | Generate a JSON object |
| `{"type":"json_schema","json_schema":{...}}` | Generate an object constrained by a supported JSON schema |

For example:

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $MIDNIGHT_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Give a short greeting."}],
    "max_tokens": 128,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "greeting",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {"answer": {"type": "string"}},
          "required": ["answer"],
          "additionalProperties": false
        }
      }
    }
  }'
```

The JSON document is returned as a string in `choices[0].message.content`,
inside the usual Chat Completions response. With `stream: true`, concatenate
`choices[].delta.content` to obtain the document. The server filters tokens
before sampling to enforce syntax and the accepted schema; prompting alone is
not the enforcement mechanism.

## Supported schemas

The root must describe an object. Supported schema features are:

- Nested objects with `properties`, `required`, and `additionalProperties: false`.
- Arrays with one `items` schema.
- String, number, integer, boolean, and null types; nullable type arrays.
- `enum`, `const`, and nested `anyOf`.
- Nonrecursive local references through `$defs` and `$ref`.
- `title` and `description` annotations.

With `strict: true`, every object property must be required. Use a nullable type
for optional values. With strictness omitted or false, optional object properties
are allowed, while all accepted schema constraints are still enforced.

Unsupported constraints such as `pattern`, numeric ranges, array/string length
bounds, recursive references, and root `anyOf` produce an explicit HTTP 400
error. Unknown format names and malformed schemas also fail before streaming
begins. Properties are generated in lexicographic order; input schema property
order is not preserved. JSON object mode permits arbitrary property order.

Schemas are bounded to 512 parsed nodes, 32 levels of schema nesting, and 65,536
encoded bytes across enum/const values. Runtime grammar complexity is also
bounded; exceeding that bound fails generation instead of emitting an
unconstrained response.

## Request and runtime limits

- Structured output currently requires tools to be absent or disabled with
  `tool_choice: "none"`. Strict tool-argument generation is separate future work.
- Custom `stop` strings are rejected with structured output because they could
  interrupt a JSON value. The model's EOS token ends a completed object.
- An exhausted output-token budget returns `finish_reason: "length"`. The JSON
  can be incomplete in that case, including with `strict: true`. Only parse it
  as a complete document after a successful completion. Cancellation or an
  internal decoding failure also does not guarantee a complete document.
- Structured requests use ordinary target-model generation with fresh KV state.
  Conversation-prefix caching and DFlash remain available for ordinary text
  requests. Structured requests do not reuse those paths.
- Supported tokenizer decoders are ByteLevel (including a single-decoder
  Sequence), Metaspace, and the standard SentencePiece
  Replace/ByteFallback/Fuse/optional-Strip sequence in `tokenizer.json`.
  Unsupported tokenizer layouts return HTTP 422 before streaming starts.
- Schema enforcement guarantees structure, not the factual correctness of values.

The [Responses API](responses-api.md) supports the same constrained decoder at
`POST /v1/responses`. Use `text.format` there: for `json_schema`, put `name`,
`schema`, and `strict` directly inside `format`, without the Chat Completions
`json_schema` wrapper. Exhausted output budgets produce response status
`incomplete` with reason `max_output_tokens`.

## Verification

After building Midnight, run the real-model API checks with an installed model:

```sh
python3 Scripts/SmokeStructuredOutput.py --model /path/to/model
```

The script starts a temporary loopback server, checks JSON/schema output,
streaming, Unicode, token limits, request rejection, and ordinary chat, then
stops the server. Use `--binary /path/to/midnight` to check another build.

The wire format follows the
[OpenAI Structured Outputs documentation](https://developers.openai.com/api/docs/guides/structured-outputs),
subject to the explicit subset and limits above.
