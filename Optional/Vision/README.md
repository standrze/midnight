# Optional Midnight vision runtime

This is an independently built **macOS Apple silicon** vision worker. The ordinary Midnight text package does not depend on MLXVLM or load vision weights. Its normal model manager starts this separate process only when a compatible vision checkpoint is explicitly selected. Linux returns a clear unsupported-backend error; CUDA vision is not implemented in this slice.

Launching the executable explicitly selects and loads one local vision model. Its weights stay loaded across image requests until the process is stopped. This implements the user's per-model isolation requirement: switching to a vision workload can cost time and memory; an ordinary text model does no image preprocessing. Starting both servers manually can create GPU contention. The normal model manager drains and unloads the previous model before loading vision. Loading text again stops the vision worker before loading text weights; simultaneous residency is not required.

## Build and configure managed vision

```sh
./Optional/Vision/build.sh
MIDNIGHT_VISION_WORKER="$PWD/Optional/Vision/.build/release/midnight-vision-worker" \
  midnight --idle --host 127.0.0.1 --port 8080
```

This starts the normal listener without loading a model or starting vision.
Select FastVLM through the normal load API below, or use `/vision load MODEL`
in Lowlight. `/vision off` restores Lowlight's previous model configuration.

For an explicitly separate endpoint, the worker can also run standalone:

```sh
Optional/Vision/.build/release/midnight-vision-worker \
  --model /absolute/path/to/a/compatible-vision-model \
  --served-model-name my-vision-model \
  --memory-limit-gib 4 \
  --port 8001
```

The memory limit is required and applies only to this process. It does not reserve GPU memory against other applications. Models are never downloaded implicitly. Managed downloads are supported. The parent and worker acquire independent shared file leases through the same Foundation-only `Shared/ModelFiles` package before opening weights. Standalone workers also hold these leases, including when their launching parent exits. Removal remains blocked until every owning process releases the model. Stop the explicitly launched process to release its weights and caches; changing a Lowlight client toggle does not terminate a separately launched server.

The worker uses the same pinned upstream MLX/MLX-LM revisions as Midnight, but links MLXVLM only in this separate package. Its dependency checkouts are independent and do not inherit Midnight’s optional dependency patches. The build script also compiles the colocated Metal library required at runtime. The pinned MLXVLM processors import CoreImage and AVFoundation and therefore are not portable to Linux. The initial supported model layout is FastVLM with a fixed 1024-pixel image processor and 256 expanded image positions. Other MLXVLM architectures are rejected until their image-context accounting is implemented and verified. This does not execute arbitrary Hugging Face Python code. No repository Python code is executed.


## Normal model selection

Set `MIDNIGHT_VISION_WORKER` to the absolute executable path before starting the
normal Midnight listener. `MIDNIGHT_VISION_MEMORY_GIB` optionally sets the worker
budget (default 4 GiB; accepted range 1–256 GiB). These values are consulted only
when a FastVLM checkpoint is selected. Building the optional worker is separate;
Midnight does not download, build, or install it implicitly.

Send the existing model-management request to the normal listener:

```json
{"model":"/absolute/path/to/compatible/FastVLM","name":"my-vision-model"}
```

`POST /v1/runtime/load` preflights the model and worker, drains the previous
backend, starts the worker, and reports asynchronous readiness through
`GET /v1/runtime`. `loadedModel.modality` identifies `vision`; `loadRequest`
contains the resolved request needed to restore an earlier text selection,
including the distinction between automatic and explicitly capped output limits.
The optional `expectedGeneration` and `expectedInstanceID` guards prevent a stale
client from replacing a model that another action has selected.

Once ready, send the same image Chat Completions request to the **normal listener**
using the selected model name. `GET /v1/vision/status` reports worker readiness,
busy state, and current/cache/peak/limit memory under `memory`, in bytes.
Its `memory_scope` is `vision_worker`; `/v1/runtime.memory` describes the parent
process and must not be mistaken for combined parent/worker GPU usage.

`POST /v1/runtime/unload` waits for admitted requests and cancelled producers,
then stops and reaps the worker. Selecting another model performs that same
shutdown first. The worker is launched on a private random loopback port with an
authenticated control connection and a readiness handshake; clients use the main
listener. A managed worker also exits if its parent dies. A separately launched
standalone worker remains under its explicit owner's control.

No CUDA vision or new vision architecture is included. Vision supports one image
and nonstreaming Chat Completions; selecting it does not add image processing to
text generation or enable tools/Responses/structured output for vision.

## API and limits

- `GET /v1/models` returns the explicitly selected vision model.
- `GET /v1/vision/status` returns the model/runtime identity.
- `POST /v1/chat/completions` accepts the standard OpenAI content array with `text` and `image_url` parts and returns a Chat Completions response.
- The model must match explicitly. Exactly one inline PNG/JPEG base64 image in a user message is required. Remote URLs and local-file URLs are rejected.
- Image limit: 8 MiB compressed, 16 million pixels, and 8192 pixels per edge. Incoming HTTP body limit: 12 MiB. Message text limit: 64 KiB, at most 32 messages. Literal extra image control markers are rejected.
- Input decoding preserves aspect ratio and bounds its longest edge to 1024 for `detail:auto`/`high`, or 512 for `low`. The selected model's processor may then resize to its required resolution; FastVLM uses a fixed 1024 input. Lower detail therefore reduces source detail and is not a universal inference speed guarantee.
- The combined prepared prompt and requested output must fit the smaller of the model's configured context and 8192 tokens, including 256 image positions in place of the image placeholder; overflow returns an error instead of silently truncating. Output limit defaults to 512 and accepts 1–1024 through `max_tokens` or `max_completion_tokens`.
- Only nonstreaming generation is currently supported. Unsupported parameters, tools, structured output, and Responses endpoints return explicit errors rather than being silently ignored.
- One request generates at a time. An overlapping request receives `429 vision_busy`. Disconnecting a client cancels its generation; admission stays occupied until the MLX producer has drained. Each request owns fresh image/conversation state; weights are shared, KV state is not retained across requests.

```json
{
  "model": "my-vision-model",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Read the access code in this screenshot."},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,...", "detail": "auto"}}
    ]
  }],
  "stream": false,
  "max_tokens": 128
}
```

## Validation scope

The request boundary is covered by CPU tests, including unsupported URLs and parameters, invalid numeric controls, MIME mismatch, missing/multiple images, model mismatch, and size limits. `Artifacts/` records the pinned test checkpoint, an actual headless Chrome screenshot of a synthetic local page, and execution evidence as validation completes. Model files remain in a temporary directory and are not bundled.

This optional package is a first working image-input slice, not a claim of full multimodal OpenAI compatibility, CUDA vision support, or a completed absent-versus-inactive performance gate. Normal model selection now manages the optional worker lifecycle while the main text executable stays independent of MLXVLM.
