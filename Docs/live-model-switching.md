# Live model switching

Midnight can unload a model and load another without restarting its process or
rebinding its HTTP port. Start with a model as usual, or start an empty server:

```sh
export MIDNIGHT_API_KEY="$(openssl rand -hex 32)"
midnight --idle --host 127.0.0.1 --port 8080
```

## Automatic selection and installed-model discovery

`GET /v1/models` now lists local checkpoints that pass loader preflight, not
just the resident model. `GET /v1/models/{id}` works for unloaded entries too.
The Midnight-specific `loaded` boolean identifies the ready resident entry.
Discovery only reads metadata; it does not load weights or download anything.
Preflight cannot guarantee sufficient memory or valid tensor contents, so a
later weight load may still fail explicitly.

The catalog scans immediate checkpoint directories under
`MODEL_RUNNER_MODELS_DIR` (default `~/.midnight/models`), the launch checkout's
`Models/`, and optional colon-separated `MIDNIGHT_MODEL_DIRS` roots. Incomplete
or preflight-incompatible entries are omitted. Give duplicate IDs unique
`servedModelName` values in their `midnight.json`; ambiguous IDs are omitted.
Successfully loaded explicit selections retain their aliases and load options
for the lifetime of the server. They are revalidated before being advertised.

Use an exact discovered ID in `model` on Chat Completions, Responses, or either
speech dialect. Midnight waits for ongoing work, switches if necessary, and
only then serves the request on that model. It never silently falls back to
another model. Unknown IDs return 404 `model_not_found`; arbitrary paths and
remote repository downloads are not accepted as inference selections. A load
failure returns 503 `model_load_failed` with the cause. Requests without a model
keep the current-model behavior of their existing dialect.

Automatic admissions are FIFO, with at most 64 waiting admissions (additional
requests receive 409 `model_transition_in_progress`). Same-model requests can
run concurrently after admission. A different-model admission waits for active
leases to finish before unloading. Canceled queued requests are removed;
cancellation during a load does not interrupt the shared model transition.
There is no force-cancel or multi-model warm pool. Manual load/unload operations
return 409 while automatic admission owns a transition. Local-only restrictions
still apply to manual controls; ordinary inference clients may select only
catalog IDs. First requests need a timeout long enough for loading plus inference.

The listener starts before the initial model loads. An initial load failure
leaves the server available for status and a corrected load request. `--idle`
overrides the default model from stack settings and cannot accompany `--model`.

## Load, inspect, and unload

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/runtime/load \
  -H "Authorization: Bearer $MIDNIGHT_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"LFM2.5-1.2B-Instruct-MLX-4bit","name":"local","maxTokens":1024}'

curl --fail-with-body -H "Authorization: Bearer $MIDNIGHT_API_KEY" http://127.0.0.1:8080/v1/runtime

curl --fail-with-body http://127.0.0.1:8080/v1/runtime/unload \
  -H "Authorization: Bearer $MIDNIGHT_API_KEY" \
  -H 'Content-Type: application/json' -d '{}'
```

Load and unload return HTTP 202 with an operation ID and current state. Poll
`GET /v1/runtime` until the same operation ID reaches `ready` (load) or `empty`
(unload). A load that reaches `empty` with `lastError` failed. The operation ID
remains in status after completion. Unloading an already empty server is
idempotent. A second mutation during a transition returns HTTP 409.

The load body uses these camelCase fields:

| Field | Meaning |
| --- | --- |
| `model` | Required local model name or checkpoint directory |
| `name` | Name exposed through the inference API |
| `adapter`, `adapterScale` | Optional LoRA directory and scale |
| `dflashModel`, `dflashBlockSize` | Optional Laguna drafter and verification size |
| `maxTokens`, `contextLength` | Output ceiling and total context limit |
| `prefillStepSize`, `kvCompression` | Text prefill and cache policy |
| `engine` | `auto`, `metal`, `cuda`, or `cpu`, subject to the installed build |
| `expectedGeneration`, `expectedInstanceID` | Optional guards against replacing a selection changed since the client last inspected it |

Unload accepts the same two optional guards. A stale guard returns HTTP 409
`stale_model_selection` before changing the model. `loadedModel.modality` is
`text`, `voice`, or `vision`; `loadedModel.loadRequest` contains the resolved
selection a client can replay, preserving automatic versus explicitly capped
output limits. Save both this request and the runtime identity when temporarily
switching models.

Each load resolves the selected model's `midnight.json` and matching stack
settings. Per-model CLI overrides apply to the initial load only. Host, port,
stack configuration, environment settings, and the CLI engine default persist
for the process. Speech models retain their existing configuration restrictions.

On Apple silicon, a compatible FastVLM selection can use the separately built
[vision worker](../Optional/Vision/README.md). Configure `MIDNIGHT_VISION_WORKER`
before starting the listener. Vision uses these same load/unload routes and the
same public HTTP port; text selection stops and reaps its worker first. Both
processes hold independent leases protecting managed model files. CUDA vision
is rejected during preflight, preserving an existing text model.

Mutation routes accept loopback native clients with `Content-Type:
application/json`; browser-origin and remote requests are rejected. Studio's
own web controls forward requests through its native server. The control body
limit is 32 KiB. No checkpoint downloads occur through this API.

## State and request behavior

Runtime status includes stable `instanceID` and `processID`, a `phase`, the most
recent `operationID`, `modelGeneration`, `loadedModel`, `targetModel`, and
`lastError`. `memory` reports MLX `activeBytes`, `cachedBytes`, and `peakBytes`;
the peak is diagnostic and may be reset by model load/tuning. Optional fields
are omitted when absent.

The runtime memory counters describe the parent process. With vision loaded,
`GET /v1/vision/status` exposes the separate worker's counters and labels their
scope `vision_worker`.

- `ready`: a model accepts requests.
- `draining`: new model requests are blocked while admitted requests finish.
- `unloading`: generation producers and GPU work finish; old backend references
  and cached allocations are released.
- `loading`: the new backend is being constructed.
- `empty`: the process is running without a model.

Switching closes admission before awaiting active requests. Every admitted
request retains one model and its limits through preparation, generation, and
response completion. A disconnected client cancels its generation, and unload
still waits for producer cleanup. The default switch waits for active work;
there is no force-cancel switch. Explicit control mutations are not queued;
automatic model-selection admissions use the FIFO described above.

`GET /v1/models` retains installed entries during transitions, with `loaded:false`.
Requests selecting a catalog model wait for readiness. Other model-dependent
requests return HTTP 503 with `model_unavailable` while no model is ready.
`GET /v1/inspector/runtime` also returns runtime status and
preserves the original process identity fields. Clients should use the operation
ID and model generation in addition to model name when checking readiness or
invalidating inspection results.

Preflight rejects invalid directories and settings before releasing a working
model. Runtime loading can still fail after the old model has been released;
in that case Midnight remains empty, reports the error, and accepts another
load. The old and new models are never deliberately resident together.

Unload releases model weights, tokenizer/backend state, drafter state, and
conversation caches. Client-side transcripts can be sent to the next model,
which computes a new prompt cache. The MLX worker and HTTP listener remain
alive. Process exit is still controlled separately by the launcher or shell.

## Verification

Focused lifecycle tests cover request draining, producer cancellation, release
ordering, model identity, invalid preflight, failed load recovery, repeated
replacement, and HTTP status/control behavior. Run these with:

```sh
MODEL_RUNNER_BUILD_CONFIGURATION=debug Scripts/test-metal.sh \
  --filter 'ModelLifecycleManagerTests|ModelHTTPLifecycleTests|ModelLoaderTests|StreamProducerLifetimeTests|MLXPinnedRuntimeTests|ModelHTTPRequestHandlerTests'
```

When measuring a real checkpoint, compare live MLX allocations after repeated
unloads. The process's RSS also contains the persistent runtime and does not
need to return to its original value.

The native smoke script launches its own temporary listener, checks A → B → A,
text generation, rejected selections, failed-load recovery, and memory release,
then stops that test process. A must be a local text checkpoint; B can be text
or speech. It writes logs and memory snapshots to a printed temporary directory.

```sh
swift Scripts/SmokeLiveModelSwitching.swift \
  .build/debug/midnight /absolute/path/to/text-model /absolute/path/to/second-model
```
