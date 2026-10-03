# Midnight Activation Capture

Midnight Activation Capture is a public, read-only feature of the Midnight runtime. Any HTTP client permitted by the listener’s configuration can discover supported observation sites, request selected residual vectors, or record bounded activation summaries from the next external chat request on the loaded model. The API is Midnight-specific and uses the same listener and optional bearer authentication as ordinary inference. It requires no ABSlayer package, job controller, or consumer-specific configuration.

Consumers can use the returned tensors for distillation research, evaluation, representation analysis, or model transformations. `midnight-moonshine`, ABSlayer, and independent tools consume the same contract; each owns its datasets, training or analysis objectives, artifact storage, and quality decisions. Supplying domain examples captures the model's responses to those inputs; the vectors do not identify isolated domain knowledge or establish causal importance.

Use `GET /v1/inspector/model` for discovery and `POST /v1/inspector/trace` for a dedicated capture with its own question. To observe a conversation in another client, arm `POST /v1/inspector/recordings`, then poll or cancel `GET` / `DELETE /v1/inspector/recordings/{recording_id}`. All five operations are Midnight-specific and use the listener’s authentication setting.

Observers are installed only for a dedicated trace or an explicitly armed recording. A dedicated trace owns the runner’s admission slot and uses a fresh KV cache. A recording attaches to the next admitted ordinary Chat Completions or Responses generation on that loaded runtime. Both restore the exact original normalization objects when their execution finishes. The original model operation computes the logits; recording does not substitute trained weights.

## Passive component observations

Passive recording returns optional `components` and `expertRouting` in its trace. Component entries use layer `index`, actual projection `path`, `site` (`attention_output`, dense `feed_forward_output`, or `shared_expert_output`) and residual-style summary samples. They measure actual projection outputs before residual addition, delegating to the original quantized or unquantized modules. This adds no raw sites to dedicated `/trace`.

Expert entries contain layer `index`, router `path`, `expertCount`, and samples with `tokenIndex`, `expertIDs`, and corresponding normalized `weights`. Laguna observes actual selected IDs/weights; GPT-OSS uses the pinned router's identical partition/gather/precise-softmax on its actual projection output. Bounds are 4096 experts and 64 selected experts per token. Unsupported geometry fails recording while chat continues. Routed expert output vectors and the combined GPT-OSS MoE output are not captured.

Studio lights attention and dense/shared feed-forward storage bars independently, with a fixed recording-wide RMS scale per kind. Router/norm/unmeasured capacity stays dim. Separate labelled expert ID tiles identify selected experts and show routing weights; their size does not encode storage. Unlisted experts were not selected. Neither activity nor selection proves causal importance, and individual weight elements/neurons are not measured. Old recordings explicitly retain whole-layer residual playback. Saved JSON preserves these fields for offline replay. Discovery advertises available kinds in `traceCapabilities.recordingComponents`.

## Dedicated capture workflow

1. Select a model through Midnight's normal model controls. If the server has a configured API key, supply it as a bearer token.
2. Discover its capabilities, available layers, observation sites, and runtime generation.
3. Request explicit layers and token positions with `capture: "raw"` or `"both"` to obtain individual vectors.
4. Validate and save the returned tensors with their model identity, token IDs, and positions before using them downstream.

For example, with an already loaded model and no configured server API key. If a key is configured, add `-H "Authorization: Bearer $MIDNIGHT_API_KEY"` to each curl request:

```sh
BASE="${MIDNIGHT_BASE_URL:-http://127.0.0.1:8080}"
curl --fail-with-body -sS \
  "$BASE/v1/inspector/model" > model-capabilities.json || exit 1

# Abort if the loaded model does not advertise raw activation capture.
jq -e '.traceSupported == true and (.traceCapabilities.captureModes | index("raw") != null)' \
  model-capabilities.json > /dev/null || exit 1

jq '{model: .id, runtimeGeneration: .runtimeGeneration,
     question: "Explain how a cache hit differs from a cache miss.",
     maxTokens: 0, layers: [.layers[0].index], sites: ["layer_input"],
     tokenPositions: {prefill: [-1], decode: []},
     capture: "raw", maxCaptureBytes: 1048576}' model-capabilities.json |
  curl --fail-with-body -sS \
    -H 'Content-Type: application/json' --data-binary @- \
    "$BASE/v1/inspector/trace" > activation-capture.json
```

The example captures the final rendered prompt token at the first available layer.
Its local filenames are client-owned artifacts, not files written by Midnight.
The source checkpoint revision and weight hashes should be recorded separately by
the consumer: a runtime generation guard identifies the loaded instance, not a
portable checkpoint fingerprint.

## Record an external conversation

Use this workflow to chat in OnoSendai or another client while Studio records the
measurements. Recording does not send a prompt or generate a second answer.

1. Discover the current model and check `traceSupported` and the available sites.
2. Arm one recording with explicit layers, an optional site selection, and bounds.
3. Send a normal Chat Completions or Responses request from the chat client to the
   same Midnight listener and loaded model. The next admitted text generation
   claims the recording; it is not tied to a particular client or conversation ID.
4. Poll the recording ID. Save the completed session and its model descriptor for
   later replay. Cancel with `DELETE` when collection is no longer wanted.

```json
{
  "model": "loaded-model",
  "runtimeGeneration": 4,
  "layers": [0, 4],
  "sites": ["layer_input", "after_attention"],
  "maxTokens": 32,
  "maxCaptureBytes": 4194304
}
```

The body has no `question`, `tokenPositions`, or raw `capture` option. `layers`
requires 1–128 distinct available indices. Omitted `sites` selects both advertised
sites; supplied sites must be nonempty and distinct. `maxTokens` defaults to 32
and permits 1–64. It limits the recorded decode window, not the chat response
length. `maxCaptureBytes` defaults to 4 MiB and permits up to 16 MiB. Over-budget
selections fail during arming. The dedicated trace’s 256-token prompt limit does
not apply to the external conversation.

With `model-capabilities.json` from discovery and `BASE` set as above:

```sh
jq '{model: .id, runtimeGeneration: .runtimeGeneration,
     layers: [.layers[0].index], sites: ["layer_input"],
     maxTokens: 32, maxCaptureBytes: 4194304}' model-capabilities.json |
  curl --fail-with-body -sS \
    -H 'Content-Type: application/json' --data-binary @- \
    "$BASE/v1/inspector/recordings" > recording-session.json || exit 1

RECORDING_ID=$(jq -er '.id' recording-session.json) || exit 1

# Send the chat request from your usual client, then poll this same ID.
curl --fail-with-body -sS \
  "$BASE/v1/inspector/recordings/$RECORDING_ID" > recording-session.json

# If needed, stop collection while allowing the chat request to continue.
curl --fail-with-body -sS -X DELETE \
  "$BASE/v1/inspector/recordings/$RECORDING_ID"
```

Arming returns HTTP 202 with the session. GET and DELETE return HTTP 200. Bodies
are limited to 32 KiB, and recording ID routes accept UUIDs without query
parameters.

The returned session includes UUID `id`, `status`, Unix-seconds `createdAt`, an
optional `message`, and the model descriptor in `model`. States are `armed`,
`recording`, `completed`, `cancelled`, and `failed`. A completed session contains
`trace` and, when known, `cachedPromptTokenCount`. Polling and cancellation do not
wait for the model execution lease. Completed sessions remain readable after
model unload. Unknown or evicted IDs return 404 `recording_not_found`; attempting to arm a second
active recording returns 409 `recording_busy`. Invalid selections return 400
`invalid_inspector_request`; unsupported native runtimes return 422
`unsupported_model_feature`, while non-text backends return 501. Missing or
changed loaded-model instances use the existing 503/409 lifecycle errors.

Only one recording can be armed or active per listener. The server retains at
most four recent sessions in memory, evicting the oldest when another is armed;
restarting the server clears them. Unloading or replacing the model cancels an
unclaimed armed recording. Save completed captures in the client for
long-term logging. Cancellation stops collection without aborting the external
chat, and does not delete an already completed capture. A recording failure is
reported in its session independently of the chat response.

### Recorded token coordinates and replay

Recording returns summary measurements for the final evaluated prompt token and
at most the first `maxTokens` emitted tokens. Each selected layer/site has RMS,
maximum absolute value, and 16 channel-group RMS values. The underlying measured
sites have the same residual meaning as dedicated traces. No raw vectors, logits,
full prompt, or full conversation transcript are retained by this recording mode.

A recording trace has `outputKind: "recorded"`, an empty `question`, and an `answer`
containing only the captured generated-token window. Its `tokens` array is sparse:
it contains the observed window, not every prompt or response token. `position`
and each sample’s `tokenIndex` use absolute prompt-plus-answer coordinates. The
first observed token is at `promptTokenCount - 1`; emitted token zero is at
`promptTokenCount`. `promptTokenCount` and `generatedTokenCount` describe the full
request, so their sum can exceed `tokens.count`. `unobservedDecodePositions`
reports recorded-window positions not reached before generation ended. The
`stopReason` describes the entire chat request. Token labels plus the recorded
answer have a separate 64 KiB decoded-text budget; exceeding it fails the
recording without truncating content or interrupting the conversation.

Cached prompt prefixes are not recomputed for observation. Recording accounts
for the actual uncached input and preserves absolute positions; it cannot replay
activations from cached tokens that the request never evaluated. The recording
completes when that chat request finishes, even if its captured window filled
earlier. Polling returns status, not a live activation stream.

For the captured request, speculative decoding and Laguna compiled block tails
are disabled so that each observed position corresponds to the ordinary target
forward pass. Sampling, output parsing, and chat output limits remain the
client’s settings. This capture has runtime cost; it is not a performance-neutral
profiler. Unarmed requests install no recording wrappers or summary reductions.

### Studio recording library and logs

In Studio’s layer view, `a` arms Record or stops the current recording. Chat from
OnoSendai or another client, then use Space to play/pause, Left/Right to step
tokens, and `v` to cycle component and residual views. `l` opens saved recordings; choose
one with Up/Down and Enter. Replay reads saved measurements without inference or
a loaded model. Escape cancels/returns; `q` quits.

Studio writes these client-owned files below `--state-directory`, which defaults
to `~/.midnight/studio/`:

| File | Contents |
| --- | --- |
| `recordings/<recording UUID>.json` | Completed capture and its original model topology, with format `midnight-studio-recording-v1`. |
| `recordings.jsonl` | Recording events with Unix timestamp, model, ID, status, and the relative artifact path on completion. |

New recording directories use mode `0700`; capture files and the log use `0600`.
Artifacts omit the endpoint and authentication value. They retain the recorded
token labels and bounded answer window, so they may contain text from the
conversation; this recording mode does not save the full prompt. The final prompt
token’s text is retained and may contain a fragment of it. Preserve both the
capture and topology when copying artifacts to another tool. The server’s
four-session retention limit does not remove saved Studio files.

Layer-block lengths represent stored component weights. New recordings use actual component output RMS and separate selected-expert ID tiles. Older recordings retain explicitly labelled whole-layer residual brightness. Neither display establishes causal importance.

## Discover the loaded model first

`GET /v1/inspector/model` returns model identity, runtime generation, layer indices and paths, parameter shapes, and `traceCapabilities`. Check `traceSupported` before submitting a capture. Native Laguna, Mistral3/Ministral3, and GPT-OSS runtimes currently support the two advertised observation points when their module graph matches the expected geometry:

| Site | Tensor meaning |
| --- | --- |
| `layer_input` | Decoder-layer residual input, immediately before input RMS normalization. |
| `after_attention` | Residual after attention addition, immediately before post-attention RMS normalization. |

Discovery includes each point's hidden width, supported capture modes, required explicit selections, and limits. These are residual vectors, not attention probabilities, arbitrary named intermediates, feature interpretations, or causal explanations. Architecture metadata remains available for unsupported capture runtimes; callers must not infer observation support from a model name alone.

Send the returned `id` as `model` and `runtimeGeneration` with the capture request. The managed HTTP server checks the generation while acquiring its model lease. A model replacement between discovery and capture fails instead of executing against a different runtime.

## Make an explicit selection

`POST /v1/inspector/trace` accepts camel-case fields:

```json
{
  "model": "loaded-model",
  "runtimeGeneration": 4,
  "question": "The sky is blue.",
  "maxTokens": 0,
  "layers": [0, 4],
  "sites": ["layer_input"],
  "tokenPositions": {"prefill": [-1], "decode": []},
  "capture": "raw",
  "maxCaptureBytes": 1048576
}
```

- `layers` is required: 1–128 distinct available layer indices. Full-model capture requires listing those layers explicitly.
- `sites` defaults to both advertised sites. A supplied list must be nonempty and contain no duplicates.
- `tokenPositions` is required and contains both `prefill` and `decode` arrays. At least one array must select a position. An empty phase captures no rows from that phase.
- `capture` is `summary`, `raw`, or `both`; the default is `summary`.
- `maxTokens` defaults to 32 and permits 0–64, subject to the server's output limit. Zero performs prefill only and requires an empty decode selection.
- `maxCaptureBytes` optionally lowers the capture payload ceiling; its maximum is 16 MiB. Over-budget selections fail before observers are installed. Captures are never silently reduced to fit.

The request question is limited to 16 KiB and its rendered chat prompt to 256 tokens. Prefill positions index the complete rendered prompt, including chat-template and control tokens. Negative prefill positions count from the end: `-1` selects its last token. Out-of-range positions and aliases that resolve to the same token, such as `0` and `-3` in a three-token prompt, are rejected.

Decode positions are nonnegative ordinals among emitted tokens. Decode position `0` has absolute token index `promptTokenCount`. Every emitted token, including the last token at the requested limit, is evaluated once so that its capture describes that token's input activation. This extra final forward pass belongs only to active inspection. EOS tokens are not emitted or captured as answer tokens.

The capture report returns sorted layer/site selections and normalized, sorted absolute prefill positions. If EOS arrives before a requested decode position, `unobservedDecodePositions` lists the unproduced ordinals. Missing decode rows are expected only when explained by this report. Prefill-only traces have `stopReason: "prefill_only"`.

## Read summaries or raw vectors

For dedicated `/v1/inspector/trace` captures, the `tokens` array identifies all processed tokens by absolute position and token ID. Recording sessions instead return only their observed token window, as described above. `layers` contains summaries only for selected layer/site/position combinations when summary capture was requested. RMS, maximum absolute value, and 16 contiguous hidden-channel RMS groups remain summaries rather than individual neuron interpretations.

Raw capture adds `tensors`. Each selected layer/site has one tensor:

```json
{
  "layerIndex": 4,
  "path": "model.layers.4.input_layernorm",
  "site": "layer_input",
  "encoding": "base64",
  "dtype": "float32",
  "byteOrder": "little",
  "shape": [1, 2048],
  "tokenIndices": [12],
  "tokenIDs": [1234],
  "data": "..."
}
```

Data is dense, row-major IEEE 754 float32 in little-endian byte order. Values are the selected residual activation converted to float32, without normalization or channel reduction. `shape` is `[selectedTokenCount, hiddenWidth]`; there is no batch dimension. `tokenIndices` and `tokenIDs` identify every row, in increasing absolute position order. A capture with no reached decode rows still includes its selected tensor with shape `[0, hiddenWidth]`, empty row identifiers, and an empty base64 string.

A client must validate encoding, dtype, byte order, layer/site identity, expected hidden width, row order, token IDs, and `decodedByteCount == rows × hiddenWidth × 4` before using the vectors. Reject nonfinite values and over-budget allocations. Avoid assuming that a tensor named `after_attention` is the attention-module output itself: it includes the residual addition at the advertised normalization boundary.

For Swift consumers, decode little-endian words without assuming `Data` alignment:

```swift
let values: [Float] = data.withUnsafeBytes { bytes in
    stride(from: 0, to: bytes.count, by: 4).map { offset in
        let word = bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
        return Float(bitPattern: UInt32(littleEndian: word))
    }
}
```

Validate the byte count and shape before running this conversion. The consuming application owns experiment state, transformations, artifact writing, and acceptance decisions. Midnight supplies the source vectors, token coordinates, runtime identity, and capability contract.

## Dedicated trace memory, cancellation, and performance limits

Preflight estimates summary serialization, raw float storage, base64 expansion including worst-case JSON slash escaping, tensor metadata, temporary selected-row graphs, and encoding copies. The runner includes that memory estimate in admission alongside its existing model, cache, and request bounds. `maxCaptureBytes` bounds capture payload estimates; it is not a limit on the complete HTTP response or the loaded model's memory.

Token labels, prediction labels, the question, and answer have a separate aggregate 256 KiB decoded-text cap. The vocabulary dimension is limited to 1,048,576 before copying the final probability vector to the CPU. Pathological model output or metadata fails explicitly. Observations are flushed at bounded 16-token prefill chunks; unselected rows create no recording reductions or CPU transfers, and unselected layers/sites receive no temporary wrappers.

The endpoint is a bounded JSON operation rather than an activation event stream. Disconnect cancellation follows the server's request cancellation path. Cleanup synchronizes the inspection stream, restores original modules, and releases temporary activation buffers before the admission slot becomes available to ordinary inference.

Active inspection intentionally costs time and memory. Structural isolation and exact-logit tests do not establish universal zero inactive overhead. See [the inspection performance evidence](inspection-performance.md) for the separate inactive-versus-absent acceptance gate and the hardware/source scope of each measurement.
