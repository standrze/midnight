# GPT-OSS on Midnight

GPT-OSS-20B is the speed-oriented starting point for this M5 Max / 64 GB Mac.
GPT-OSS-20B and 120B accept and produce text; neither has native image or audio
support. Midnight's separate speech backends can produce voice from their text.
See the official [20B](https://developers.openai.com/api/docs/models/gpt-oss-20b)
and [120B](https://developers.openai.com/api/docs/models/gpt-oss-120b) specifications.

## Use the existing mixed checkpoint

**MXFP4 is the preferred expert format for GPT-OSS.** OpenAI post-trained its
MoE weights with MXFP4 and used that quantization for its published evaluations.
See the [official repository](https://github.com/openai/gpt-oss#highlights).
Midnight preserves those experts in MLX's packed MXFP4 representation. The
affine Q4 dense layers and Q8 routers below are the community checkpoint's
additional local speed/memory choices, not a claim that OpenAI quantizes every
layer this way. Future format-specific tuning should target this mixed path;
replacing the expert grid with affine Q4 would need new quality evidence.

The measured checkpoint is `mlx-community/gpt-oss-20b-MXFP4-Q4`, pinned at
`f356f2747216d7e98fee755df25987459fc19089`. Its name understates its mixed policy:

| Component | Stored format |
| --- | --- |
| 72 expert projections | MXFP4, group 32 |
| Attention, embeddings, output head | Affine Q4, group 64 |
| 24 routers | Affine Q8, group 64 |

The header audit reconciled all 775 tensors: **11.178 GB of tensor payload**
(10.411 GiB). The downloaded model occupies the ignored local folder
`tmp/models/gpt-oss-20b-MXFP4-Q4-f356f274`.

Keep this combination for now. Midnight's Q4R8 policy protects GPT-OSS routers
with Q8, but converting its experts to affine Q4 has **not** demonstrated a
whole-model speed or quality advantage. Affine G64 expert metadata would add
approximately 597 MB. GPT-OSS's 2,880-wide expert inputs cannot use G128 without
changing the layout; its 4,096-wide attention output projection can.

The controlled synthetic comparisons favored affine for the query projection
and MXFP4 slightly for expert gate/up; other comparisons were inconclusive or
failed timing-stability gates. They measure random-weight primitives, not model
quality or whole-model throughput. ScaleSearch changes offline weight fitting,
not the packed inference format. Laguna's ScaleSearch/AWSS results do not establish
the best GPT-OSS quantizer.

Prefer a checkpoint already stored in MLX's packed format. The pinned upstream
GPT-OSS sanitizer expands raw official `_blocks` tensors when loading that layout;
the generic Midnight converter rejects already-quantized sources. Do not strip
quantization metadata or label an MXFP4-to-affine requantization as lossless.

## Run with explicit latency controls

From the repository root:

```bash
./run.sh --config Examples/gpt-oss-mxfp4.json

curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @Examples/gpt-oss-fast-request.json
```

The explicit [MXFP4 launch configuration](../Examples/gpt-oss-mxfp4.json)
selects the already-measured checkpoint. Its model path is relative to the
repository working directory. Reasoning effort remains a per-request choice
in the request example; the launch configuration does not override it.

`reasoning_effort` accepts `low`, `medium`, and `high`. Use `low` for latency;
more difficult work may benefit from additional reasoning. Omission preserves
the checkpoint template default (medium for this checkpoint). This setting
controls the model's reasoning instruction, not a guaranteed reasoning-token
budget or a faster matrix kernel. The output limit includes reasoning tokens;
a small cap can finish without visible text. Non-GPT-OSS model templates retain
their existing behavior.

The release launcher, resident model, measured wired-memory policy, uncompressed
KV cache, and 512-token prefill are the measured starting configuration.
GPT-OSS KV compression remains disabled because the upstream quantized-cache
attention path does not preserve its attention sinks. Laguna DFlash and fused
gate/up kernels are architecture-specific and are not GPT-OSS optimizations.

## What changed and what was measured

Midnight now forwards reasoning effort through HTTP, prompt preparation and
generation, and includes it in cache identity. Changing the effort or tool
definitions invalidates incompatible state. GPT-OSS can retain a live session
while awaiting a tool result; ordinary text turns avoid this retention because
the measured follow-up rebuilt its prompt. Conservative context checks include
the live token timeline, including reasoning absent from the public transcript.

The final short-prompt run produced a median **138.5 tokens/s** and approximately
**295 ms** to first visible text after warm-up, with four 256-token trials.
Loading briefly used about **12.13 GB** of active MLX memory. These are local
measurements of this checkpoint and workload, not a comparison against affine
Q4R8. Decode counts include reasoning; first-visible-text latency does not.

A 4,305-token prefill campaign could not establish a 512-versus-2048 speed ratio:
the 512 arms ended after 85 generated tokens while 2048 arms reached 256, and
timings also drifted. The failed comparisons are retained. Prefill defaults were
not changed; neither a faster setting nor equivalent generated quality follows
from this experiment.

See the [results and limitations](../benchmark-results/gpt-oss-20260906/README.md)
for final measurements, verification and raw reports. The runtime benchmark now
accepts `--reasoning-effort`; campaigns validate the corresponding report field.
`--hot-cache-ab` retains the old `--mistral-hot-cache-ab` alias, requires actual
reuse, and does not claim a speedup from zero cached tokens.
