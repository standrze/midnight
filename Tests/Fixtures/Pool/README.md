# Pool chat compatibility fixture

`pool-1.0.16-summarizer.json` preserves the request shape captured on
2026-09-07 from Pool 1.0.16's ACP `session/prompt` with input `hi` and
`POOLSIDE_STANDALONE_CONTEXT_LENGTH=4096`. A temporary loopback HTTP
endpoint returned a controlled 400 response; no model generation or tool
actions were needed for capture. The API key was the placeholder `EMPTY`;
no HTTP headers were recorded. All message text was replaced with `hi`.

The first system and user messages use arrays of text parts, including
`cache_control` metadata, while the remaining messages use strings.
Previously, decoding failed at `messages[0].content` because a string was
required. The fixture intentionally retains `max_completion_tokens=8192`:
decoding this request does not imply that a server with a smaller configured
output limit or context window can run it.
