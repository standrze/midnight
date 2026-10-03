# Muse Glimmer tool calling

Local `muse_glimmer` checkpoints use the token-aware ATEM/Onyx decoder, selected
from `config.json` rather than the checkpoint folder name. Muse's recipient
headers and ATEM markup are internal model syntax, not assistant message text.

The OpenAI-compatible `POST /v1/chat/completions` endpoint accepts ordinary
function declarations. Completed calls are returned in `message.tool_calls`
with an ID, function name, JSON-encoded `arguments`, and `finish_reason:
"tool_calls"`. Streaming uses `delta.tool_calls`; a complete call is emitted
after its ATEM frame closes rather than incrementally streaming argument bytes.
Private `to=self` text is exposed separately as the existing Midnight-specific
`reasoning_content` extension, not mixed into `content`.

`tool_choice` supports `auto`, `none`, `required`, and named functions for Muse.
Required selection constrains the generated recipient and invocation to the
declared functions while preserving the model's scores among valid choices.
Named selection restricts that set to one function. Arguments remain generated
by the model; this is not strict JSON Schema constrained argument generation.
Required/named selection uses target-only decoding rather than DFlash.

Return results as `role: "tool"` messages with the matching `tool_call_id`, and
retain the preceding assistant `tool_calls` in the conversation. Midnight does
not execute client tools. The same decoded calls feed the Responses endpoint
as `function_call` items; its existing storage and compatibility limits apply.

The local serving path remains text-only. This change does not add image input,
server-side tools, or a new API route.
