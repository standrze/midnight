# Concurrent clients and shared prompt caching

Midnight accepts overlapping text-generation requests through Chat Completions and Responses. A FIFO execution queue holds up to 64 waiting operations. Prompt preparation, generation and model inspection share the same gate. Cancelled waiters are removed; an active generation retains ownership until its producer and GPU cleanup finish. Full queues return the existing `model_busy` error.

This is concurrent HTTP admission with serialized model execution, not continuous batching or parallel token decoding. A long generation can delay other clients. Speech backends retain their existing admission behavior.

Independent requests can reuse exact rendered text-token prefixes from the same loaded model. Four immutable checkpoints, at 128, 256, 512 and 1,024 tokens, share a maximum 64 MiB budget (also capped by `MODEL_RUNNER_PREFIX_CACHE_MIB`). Lookup verifies every prefix token. Each request receives independent cache arrays, and generated answers are not stored in this cache. Memory pressure evicts or skips snapshots; model unload drops them. `MODEL_RUNNER_PREFIX_CACHE_MIB=0` or `MODEL_RUNNER_PREFIX_CACHE_ENTRIES=0` disables retention.

The shared path uses uncompressed text caches. Structured outputs, forced tool prefixes, speculative decoding and Gemma prompt normalization retain their existing generation paths. Models requiring auxiliary position state fall back to cold generation. Existing compatible hot conversation continuations remain available. Sharing is local to the loaded model; this does not add user authentication or tenant partitions.

Chat Completions reports `usage.prompt_tokens_details.cached_tokens`; Responses reports `usage.input_tokens_details.cached_tokens`. Input totals include both cached and freshly processed tokens. Lowlight consumes both forms and shows the latest cached input count in `/usage`; overlapping Lowlight clients require no new configuration.

## Verification

`swift test` covers queue handoff, cancellation, overflow and usage compatibility alongside existing lifecycle and API tests. Run `Scripts/SmokeSharedPromptCache.py WARM_URL COLD_URL` against two local servers using the same model, with `MODEL_RUNNER_PREFIX_CACHE_MIB=0` on the cold server. It checks simultaneous clients, independent cold-answer parity, prefix isolation, token accounting and both API usage fields.

Embeddings and retrieval are shelved under `artifacts/shelved/embeddings-retrieval-2026-09-10`. Their routes, flags and Lowlight commands are removed. Earlier OpenAI Responses and structured-output support remains. Saved conversations, downloaded checkpoints and user index data are preserved.
