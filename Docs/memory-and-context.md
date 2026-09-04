# Memory and long-context operation

Midnight 0.2.0-beta.1 adds request admission, device-aware Metal budgets,
shared conversation-cache accounting, configurable prefill, and opt-in KV
compression. These controls do not extend a model's trained context window.

## Configuration

```bash
./run.sh --model /absolute/path/to/model \
  --context-length 32768 --max-tokens 2048 --prefill-step-size 512
```

The macOS launcher now builds and launches release by default. Set
`MODEL_RUNNER_BUILD_CONFIGURATION=debug` for development. To launch without
building, run `.build/release/midnight` directly after `./build.sh`.

The settings-file equivalents are `mlxRunner.contextLength`,
`mlxRunner.prefillStepSize`, and `mlxRunner.kvCompression`. CLI values override
settings. Context defaults to the model's declared maximum, or a conservative
4096 when no recognized context field exists. An explicit value cannot exceed
a known model maximum. The limit covers **prompt plus requested output**; the
runner never silently truncates a transcript. Prefill defaults to 512 and
accepts 1–8192 tokens. Smaller chunks can reduce temporary allocations; larger
chunks can improve prompt throughput. Compare them on your own model because
chunk boundaries may change floating-point results.

`GET /v1/models` and model detail responses report `context_length`,
`prefill_step_size`, `kv_compression`, and `memory_limit_bytes`. Context length
is a configured ceiling, not a guarantee that every request of that length fits.

## Memory admission

Metal's allocator ceiling is the smaller of its recommended working set and
physical RAM minus host reserve. Host reserve defaults to the larger of 2 GiB
or 20% of physical RAM. Overrides remain strict plain decimal numbers:

- `MODEL_RUNNER_HOST_RESERVE_GIB`: reserve at least 2 GiB for the host.
- `MODEL_RUNNER_MLX_MEMORY_LIMIT_GIB`: optionally lower the resulting ceiling.
- `MODEL_RUNNER_MLX_CACHE_LIMIT_MIB`: allocator recycling cache; default at most
  256 MiB, maximum the smaller of 2 GiB or one-eighth of the allocator ceiling.

CPU and CUDA retain their existing conservative backend limits; the pinned
CUDA API does not expose a reliable device-memory capacity query. A CUDA
memory override cannot exceed that backend's existing cap.

Checkpoint file size is checked before loading. Before text generation, the
runner counts the actual rendered prompt, reserves requested output, estimates
KV growth and prefill workspace, and accounts for live allocations. It evicts
retained caches when needed. Context or estimated-memory rejection is reported
as `request_exceeds_limits`; HTTP preflight returns JSON HTTP 400 even for a
streaming request. Generation rechecks admission because available allocations
can change after preflight.

The estimate budgets FP32 K/V, cache growth slack, transient prefill rows,
and workspace. Known sliding-window layers stop growing at their native window.
Unknown geometry uses a conservative fallback and is identified at startup.
Compression receives no admission discount: temporary full-precision buffers
may exist during conversion. The estimator is not a proof of exact peak memory,
and OS pressure from other applications can still reduce available capacity.
The existing measured wired-residency policy remains active within the ceiling.

## Conversation retention

`MODEL_RUNNER_PREFIX_CACHE_MIB` (default 2048) now budgets the retained hot
conversation **and** saved branch snapshots together. Active generation still
uses the process allocation budget. Oversized hot sessions are released after
the response; subsequent turns are recomputed. Set the byte budget to zero to
disable retention. `MODEL_RUNNER_PREFIX_CACHE_ENTRIES=0` disables branch
snapshots while permitting hot-session reuse within the byte budget.

The runner measures existing KV arrays without copying them, skips disabled or
oversized snapshots, and evicts old entries before allocating a snapshot.
Mistral-family hot sessions use the same retained-memory budget. This remains
completed-transcript reuse; it is not paged attention or a global token-radix
cache.

## Experimental compression

```bash
./run.sh --model /absolute/path/to/mistral-or-laguna \
  --context-length 32768 --kv-compression affine8
```

Supported values: `none` (default), `affine8`, `affine4`, and `turbo8v4`.
The runner uses MLX Swift LM's typed cache configuration and requires at least
one compatible attention layer. Model-defined sliding windows are preserved;
unsupported layers retain their existing cache. Startup reports eligible and
skipped layer counts. Compression is restricted to native Laguna and
Mistral-family text models and cannot be combined with DFlash in this release.
Speech models reject text-cache options.

Compression changes numerical results and can cost latency. It remains opt-in
pending workload-specific retrieval, coding, tool-use, and quality evaluation.
No universal speedup or quality-equivalence claim accompanies these options.
Paged attention and continuous batching are separate future work.

## Validation

Run `Scripts/test-metal.sh` on macOS and run the shell tests. The script builds
the release and tests, placing the Metal library beside the test executable.
The release includes tests for overflow/context limits, nested model metadata,
hybrid KV estimates, memory policy, retention eviction, actual compressed
hybrid-model generation, and matching build/launch configurations. The optional
`Scripts/test-long-context-http.py` exercises a local model through real HTTP.
