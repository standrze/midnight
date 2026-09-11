# Live model switching

Midnight can unload a model and load another without restarting its process or
rebinding its HTTP port. Start with a model as usual, or start an empty server:

```sh
midnight --idle --host 127.0.0.1 --port 8080
```

The listener starts before the initial model loads. An initial load failure
leaves the server available for status and a corrected load request. `--idle`
overrides the default model from stack settings and cannot accompany `--model`.

## Load, inspect, and unload

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/runtime/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"LFM2.5-1.2B-Instruct-MLX-4bit","name":"local","maxTokens":1024}'

curl --fail-with-body http://127.0.0.1:8080/v1/runtime

curl --fail-with-body http://127.0.0.1:8080/v1/runtime/unload \
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

Each load resolves the selected model's `midnight.json` and matching stack
settings. Per-model CLI overrides apply to the initial load only. Host, port,
stack configuration, environment settings, and the CLI engine default persist
for the process. Speech models retain their existing configuration restrictions.

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
there is no force-cancel switch or queue of pending model changes.

`GET /v1/models` reports only a ready model and returns an empty list in other
phases. Model-dependent requests return HTTP 503 with `model_unavailable` while
no model is ready. `GET /v1/inspector/runtime` also returns runtime status and
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
