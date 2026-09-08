# Pool CLI compatibility handoff

Observed 2026-09-07 with Pool v1.0.16 and midnight runner at http://127.0.0.1:8081/v1.

## Resolution — 2026-09-07

The confirmed runner bug was **text-part arrays in `messages[].content`**.
Pool sends system/user content as `[{"type":"text","text":"…",
"cache_control":{"type":"ephemeral"}}]`. Midnight previously accepted only
string/null content. Decoding the sanitized capture against the original wire
type failed at `messages[0].content`: “Expected to decode String but found an
array instead.” None of the captured requests included `reasoning_effort`.

The fix is confined to the existing wire decoder and error reporting:

- Accept text-part arrays and concatenate their text in order, preserving
  whitespace. Ignore optional part metadata such as `cache_control`.
- Preserve existing string/null messages, tool fields, model input strings,
  and response encoding. Reject non-text or malformed parts explicitly.
- Return the full decoding path in the HTTP error message and `param`, with
  the existing 400 status and `invalid_json` code.

The API key check belongs to Pool. For this local endpoint,
`POOLSIDE_API_KEY=EMPTY` works; Midnight does not validate API keys. The saved
Laguna model warning is unrelated.

### Verified limits and working launch

The original runner on port 8081 is configured for **4096 context / 512 output**.
These limits are also stored in this model's `midnight.json`. Pool estimates
its system prompt at roughly 9806 tokens and requests 8192 output tokens for
summarization (32000 in its last fallback), so fixing JSON alone is insufficient.
The model config declares `max_position_embeddings=131072`; this is a declared
ceiling, not a recommendation or a long-context quality test.

A temporary patched runner with **32768 context / 8192 output** successfully
handled a real Pool ACP `hi` prompt. It returned “Hello! How can I help you
today?”, `stopReason=end_turn`, and task success. Reported usage was 4687 input
tokens and 65 output tokens. Pool's normal chat request omitted an explicit
output limit; the server default applied. This verifies a short conversation,
not a long-session compaction or 131k-context workload. Pool's exceptional
32000-output fallback still exceeds the tested output ceiling and is rejected.

To use the patched debug build, run from this repository in one terminal
(choose a free port; 18081 was used for the isolated smoke):

```sh
.build/debug/midnight \
  --model gpt-oss-moe-7.77b-recovered-step24 \
  --host 127.0.0.1 --port 18081 \
  --context-length 32768 --max-tokens 8192 --prefill-step-size 256
```

Then start Pool in another terminal:

```sh
POOLSIDE_API_KEY=EMPTY \
POOLSIDE_STANDALONE_BASE_URL=http://127.0.0.1:18081/v1 \
POOLSIDE_STANDALONE_MODEL=gpt-oss-moe-7.77b-recovered-step24 \
POOLSIDE_STANDALONE_CONTEXT_LENGTH=32768 \
pool
```

The existing runner on 8081 and persistent settings were not changed. It needs
the patched build and matching limits on restart to benefit from this fix.

### Validation and other clients

- 34 focused Swift tests passed across text content, error diagnostics, existing
  tool messages/choices, reasoning effort, and HTTP compatibility wiring.
- The sanitized Pool summarizer fixture is retained under `Tests/Fixtures/Pool`.
  The captured normal streaming request and both summarizer variants also decode
  successfully with the patched wire type.
- The temporary live endpoint returned a clear error with
  `param=messages[0].content[0].type` for unsupported non-text content.
- The same text-array fix applies to other OpenAI Chat Completions clients.
  Client-side key requirements can also occur elsewhere: see
  [Aider's OpenAI-compatible setup](https://aider.chat/docs/llms/openai-compat.html).
  Choose a Chat Completions provider mode, as distinguished in
  [OpenCode's custom-provider docs](https://opencode.ai/docs/providers/#custom-provider).
  Midnight does not implement `/v1/responses` or Anthropic Messages; those are
  separate protocol requirements, not API-key problems. Other CLIs were not
  tested end to end here.

The original investigation below is retained as historical context.

## Reproduction reported by the user

```sh
POOLSIDE_API_KEY=EMPTY \
POOLSIDE_STANDALONE_BASE_URL=http://127.0.0.1:8081/v1 \
POOLSIDE_STANDALONE_MODEL=gpt-oss-moe-7.77b-recovered-step24 \
POOLSIDE_STANDALONE_CONTEXT_LENGTH=4096 \
pool
```

User sends `hi`.

Without POOLSIDE_API_KEY, session/new returned Authentication required. The placeholder resolves that error. Pool now connects, starts a session, and fails on session/prompt:

> summarizer fallback (no-reasoning) failed: 400 Bad Request: Invalid JSON request: The data couldn’t be read because it isn’t in the correct format.

Pool also reports that saved model laguna-xs-2.1:nvfp4 in /Users/stephen/.config/poolside/settings.yaml was not found, and correctly falls back to the requested model. This warning is separate from the failure.

## Verified evidence

Log location: /Users/stephen/Library/Application Support/poolside/pool/logs/-Users-stephen/latest-acp.log
Session directory at inspection: 01a07a4e-bba0-7d1f-9f08-af91ec3aee29. The latest log link can change on another launch.

At 2026-09-07T01:19:36-04:00:
- summarizableEntries: systemTokens=9806, maxSummarize=2867, nextEntryTokens=2.
- First entry exceeds MaxSummarizeTokenCount; Pool attempts compression immediately.
- Three summarizer requests fail HTTP 400, code invalid_json.
- no_tools fallback also fails (maxCompletionTokens=8192).
- no_reasoning fallback also fails (maxCompletionTokens=32000).

Two issues need investigation:
1. The configured 4096-token window is smaller than Pool's estimated system prompt alone (9806 tokens). An adequately supported context size is required on BOTH the runner/model and Pool; changing only the client declaration does not increase actual capacity.
2. Runner rejects the summarization request during JSON decoding. The exact rejected field has NOT been captured or identified. Do not assume which field is responsible.

## Source locations inspected, not changed

- Sources/ModelRunnerProtocol/OpenAIWire.swift: ChatCompletionRequest around line 226.
- Sources/ModelRunner/ModelHTTPServer.swift: Invalid JSON request errors around lines 875 and 976.
- ChatCompletionRequest.ReasoningEffort currently recognizes low, medium, high. Check the actual captured request before deciding whether effort values are involved.

## Suggested investigation

Capture a minimal local request from this harmless `hi` reproduction, with credentials redacted. Preserve a sanitized fixture. Report DecodingError codingPath and debugDescription to identify the exact mismatch, rather than relying on localizedDescription. Check request message content shapes, nullable fields, and reasoning effort against the actual payload. Add focused decoder compatibility tests for the confirmed mismatch. Verify the model's real supported context and server budget before recommending a larger Pool context.

No runner code, configuration, running process, or model was changed in this investigation. This file is a handoff only; the user requested that runner work happen separately.

## Separate midnight client work completed

Installed the client build in ~/.midnight/bin via its normal installer. Both midnight and legacy midnight-chat now launch the same current build. Client changes include thin composer rules, /effort low|medium|high|default, explicit reasoning_content/reasoning separation from answer content, Ctrl-T to toggle thinking, basic Markdown rendering, and double Ctrl-C confirmation. Actual terminal checks confirmed Up/Down changes slash-menu selection and the second Ctrl-C exits cleanly. 65 tests passed. These changes do not fix Pool's runner-side request failure.
