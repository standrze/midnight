# Native Muse and Laguna protocols

Midnight advertises `native_protocol: "muse"` or `"laguna"` on model discovery
records, based on checkpoint `model_type`, independently of the served name.

A client can send the same `native_protocol` field on `POST /v1/chat/completions`.
This is a **Midnight-specific extension**, not an OpenAI-compatible tool response.
Input messages and tool definitions keep their existing wire shapes. The output
`content` / streaming `delta.content` carries native framing, including the
assistant prefix supplied by the chat template and the terminating control token.
It is not translated to OpenAI `tool_calls` or split into `reasoning_content`.
Usage and the SSE envelope retain their existing shapes. Native finish reasons
are token-generation status (`stop` / `length`); clients derive tool actions from
the parsed native frames.

The mode must match the loaded text model. Structured output and custom stop
strings cannot be combined with native output. Native generation currently uses
the target model without DFlash or retained prompt caches. Existing compatible
requests continue using Midnight's server-side protocol parsers. No routes were
added; the current route inventory remains 24.

Lowlight selects native mode only when a Midnight discovery record advertises it.
Its separate Muse/Onyx-ATEM and Laguna parsers preserve reasoning, multiline string
arguments, typed JSON arguments, multiple calls, and call/result correlation.
Only complete turns with declared calls are eligible for execution. Truncation,
malformed frames, undeclared functions, recipient mismatches and duplicates fail
closed. Tool turns preserve reasoning and return results with matching call IDs.

The initial Lowlight tool set is read-only: `read_file` and `search_files`, confined
to its selected workspace. Reads and search output are bounded; search skips hidden
files and large files. No editing, shell, network, or server-side tool execution is
registered. Host integrations can supply a different explicit tool registry and
executor through `configureNativeTools`. Native tool loops are limited to 16
model turns. Mid-generation steering is disabled for native tool sessions;
cancellation remains available. Full JSON Schema validation is not claimed.

Lowlight Relay also selects the advertised native format and parses the same Muse
and Laguna frames. It connects completed calls to its existing registered tools,
execution permits, argument validation, call budget, and repeated-action guards.
Reasoning and call/result history survive subsequent model turns. Relay retains
its existing steering and cancellation flow; Lowlight's read-only default tool
set does not replace Relay's tools. Native mode is text-model support and does
not add vision capabilities to Muse or Laguna. Relay's macOS run passes 240 Swift Testing cases and 59 XCTest cases; five
opt-in native-browser checks are skipped by their existing test configuration.

Laguna uses `<tool_call>name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>`;
Muse uses recipient-qualified Onyx frames containing ATEM invoke/parameter blocks.
These protocols do not define a general escape for literal closing structural
delimiters inside raw string values. Such ambiguous output is not a supported
argument representation. Older Laguna templates omit prior reasoning when
`enable_thinking=false`; preserve thinking for multi-step reasoning workflows.

Sources: [Poolside template](https://huggingface.co/poolside/Laguna-XS-2.1/blob/main/chat_template.jinja)
and the installed Muse checkpoint's `chat_template.jinja`. The Laguna regression
fixture records its exact upstream revision in `Tests/Fixtures/LagunaToolProtocol/source.json`.
