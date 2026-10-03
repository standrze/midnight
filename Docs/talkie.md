> Current local inventory: only the ScaleSearch model is retained. The user
> deleted Q8, BF16/reference weights and experimental weights after verification.
> Q8 comparisons below describe historical measurements, not an installed model.

# Talkie in Midnight

Midnight runs Talkie natively in Swift/MLX. The instruction-tuned 13B checkpoint
has been tested on Metal with full BF16 weights, our own affine Q4/Q8 conversions,
and the community DWQ Q4 conversion. Our **Q8/G64** conversion is the preferred
local checkpoint because it closely preserves the BF16 model's probabilities.

Talkie was pretrained on 260 billion tokens of primarily English material from
before 1931. Its authors acknowledge modern information leakage and influence
from modern models used during instruction tuning. It is useful for exploring a
historical model's perspective, not as primary historical evidence. Read the
[project report](https://talkie-lm.com/introducing-talkie),
[official model card](https://huggingface.co/talkie-lm/talkie-1930-13b-it), and
[author implementation](https://github.com/talkie-lm/talkie).

## Run the local version

The tested build has been installed into the local `midnight` command. An older
running server keeps its existing binary until restarted. Run:

```sh
midnight \
  --model talkie-1930-13b-it-midnight-q4r8-scalesearch-attn8 \
  --context-length 2048 --max-tokens 256
```

The selected checkpoint is in `~/.midnight/models/`. The failed plain-Q4
experiment is preserved outside the available model list in
`~/.midnight/quantization-experiments/`:

| Folder | Weight bytes | Status |
| --- | ---: | --- |
| `talkie-1930-13b-it-midnight-q8` | 14,110,387,976 | Preferred for BF16 fidelity |
| `talkie-1930-13b-it-midnight-q4r8-scalesearch-attn8` | 10,533,124,415 | Compact ScaleSearch option; attention and sensitive MLPs retained at Q8 |
| `talkie-1930-13b-it-midnight-q4` (archived) | 7,470,259,558 | Failed generation; excluded from available chat models |

These were produced directly from BF16, not requantized from community weights.
Each folder contains the tokenizer, standalone chat template, generation config,
conversion recipe, and source provenance. The converter lives in the sibling
[Wick project](../../wick/Docs/talkie-quantization.md).

```sh
../wick/.build/release/wick \
  /absolute/path/talkie-1930-13b-it-hf \
  /absolute/path/new-talkie-q8 --standard-q8
```

Use `--standard-q4` for the plain Q4 control. Omitting either flag uses experimental
weight ScaleSearch. Full 13B ScaleSearch conversions and mixed-precision policies
have now been evaluated; see the [recipes and results](../benchmark-results/talkie-20260913/scalesearch/README.md).
Weight ScaleSearch is not activation-calibrated DWQ. Preserve BF16 activations
and learned gains; do not convert those to FP16.

## Source and checkpoint limits

Verified September 13, 2026:

- BF16 source: [lewtun/talkie-1930-13b-it-hf at `6311ded`](https://huggingface.co/lewtun/talkie-1930-13b-it-hf/tree/6311dedf518470856a8503f2080bb4b54fcb3323),
  a safetensors conversion of the official instruction checkpoint.
- Source weight file: 26,560,565,098 bytes; 443 BF16 tensors;
  13,280,257,721 parameters. Its full SHA256 matched upstream LFS metadata:
  `6fcd0eb3c0a9d490727f3d1f04ee4b51b7edff0906f89d78395ddb58200d2d9d`.
- Architecture: 40 layers, hidden width 5,120, 40 full attention heads of width
  128, 13,696-wide MLP, and 65,540 vocabulary entries. License: Apache-2.0.
- Checkpoint-declared context: **2,048 tokens**, including formatting, conversation,
  and generated output. Raising a command-line value does not extend this limit.

The requested 256K window is **not validated or enabled**. In matched Q8 tests,
retrieval passed at all three positions at 2K; at 4K, the beginning-position
answer failed despite full prompt processing. Generic affine Q2 KV also failed
that retrieval at 2K, while uncompressed and Q4 KV passed. The separate
[context report](talkie-context.md) contains memory calculations, exact outputs,
and the compressed-from-start diagnostic. It preserves production admission
limits and does not change the installed checkpoint's context declaration.

An experimental 40-query / eight-KV-head derivative is now supported by the
runtime. Head averaging followed by three bounded K/V recovery pilots improved
fidelity but still failed the unseen answer checks. It remains outside the
available model list and does not extend context. See the
[GQA experiment](../benchmark-results/talkie-20260913/gqa/README.md).

For exploring the model itself, try the [twelve Talkie questions](talkie-questions.md).

The HF converter reports close agreement with the original on four prompts.
That is the converter's validation, not an independently established bit-exact
PyTorch match. Native MLX fast RoPE and BF16 arithmetic can differ in rounding.
The original base model is for text completion; the chat commands here target
the instruction model. The downloader does not load GGUF or execute remote code.

## Quality evidence

The [held-out evaluation](../benchmark-results/talkie-20260913/quality-bf16-q4-q8-dwq.json)
uses 13 original synthetic prose passages and 1,932 scored next-token positions.
The 4,921-token synthetic calibration split is separate and was not used by the
standard affine conversions. This is a narrow fidelity check, not a historical
or general capability benchmark. All models receive identical token IDs.

| Checkpoint | NLL | KL(BF16 ∥ checkpoint) | Same top token as BF16 |
| --- | ---: | ---: | ---: |
| BF16 source | 3.940569 | 0 | 100% |
| Our Q8/G64 | 3.947060 | 0.002024 | 97.77% |
| Our plain Q4/G64 | 3.110095 | 0.495440 | 67.81% |
| Our mixed ScaleSearch, attention Q8 | 3.885415 | 0.028112 | 91.30% |
| Community DWQ Q4/G64 | 3.906354 | 0.091486 | 84.68% |

Plain Q4 has lower NLL on this small prose set, but also substantially changes
the source distribution and failed all four generation prompts (punctuation,
blank lines, or no visible output). Lower loss on that set alone does not establish that
it preserves instruction behaviour or avoids repetition. Q8 is selected for
fidelity; it reduces weight storage by approximately 47% versus BF16.

The compact ScaleSearch option retains BF16 embeddings, Q8 attention and output
head, and Q8 MLPs in blocks 14/37/38. The other 37 MLPs use searched Q4; about
58.6% of parameters remain Q4. It reduces weight storage by 25.4% versus our Q8
model and produced relevant non-looping prose on the four generation checks.
Factual errors remain, and the limited tests do not establish broad capability
or a speed advantage. Earlier, more aggressive ScaleSearch recipes produced
repetition or incomplete answers and remain archived.
The installed server passed seven HTTP checks with the compact artifact,
including streaming, multi-turn input, context rejection, and cancellation.

## Runtime implementation and verification

The native model preserves Talkie's inverse NeoX rotary positions, weightless
FP32-reduction RMS normalization, post-RoPE Q/K normalization, per-head query
gain, branch gains, and normalized-embedding skip. The output-head gain is
folded into the BF16 weight before quantization/multiplication, following the
author's equation. It must not be applied a second time to quantized weights.

Compatible uniform affine checkpoints can opt into concatenated Q/K/V and
gate/up projections with `MIDNIGHT_TALKIE_FUSE_PROJECTIONS=1`. The default keeps
separate projections. Heterogeneous quantization, non-affine modes, and LoRA loads retain
original projection names. `MIDNIGHT_TALKIE_FUSE_PROJECTIONS=0` explicitly selects the default for
comparison. Per-source shapes are checked before fusion so
malformed dimensions cannot be hidden by concatenation.

The standalone template renders `<|user|>PROMPT<|end|><|assistant|>` without BOS
or extra newlines. EOS IDs are 65535/65536; role markers 65537–65539 also stop
generation. Keep `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
and `generation_config.json` together.

Focused tests passed: 15 Swift Testing cases (plus parameterized variants), six
download tests, and four quantizer conversion tests. Independent tiny equations
cover FP32/BF16, packed Q4, fused/separate projections, and cached continuation.
BF16 checks account for rounding and additionally verify exact fused/separate
parity. The installed executable with our Q8 artifact passed all seven HTTP checks:
readiness, normal/streamed chat, multi-turn input, context rejection,
cancellation, and the next request. Evidence and reproducible scripts
are in [the Talkie measurement directory](../benchmark-results/talkie-20260913/).

Weight size is not total memory use. A full 2,048-token BF16 KV cache for this
MHA geometry alone is about 1.68 GB, with further workspace and activation costs.
Metal was tested on an Apple M5 Max with 64 GiB RAM. CUDA performance has not
been validated. Architecture inspection can enumerate Talkie blocks; activation
tracing is not implemented for this architecture.

## Optional community download

The existing public preset `talkie-1930-13b` refers to the community DWQ model,
not our local Q8 artifact. Its tested revision is
`1cde1b1becc99e097923ae6dfe0c51b89014fa7f`:

```sh
midnight download warshanks/talkie-1930-13b-it-mlx-4bit-DWQ \
  --revision 1cde1b1becc99e097923ae6dfe0c51b89014fa7f
```

Existing editable download catalogs are preserved; the direct repository form
works without adding a preset to `~/.midnight/config/downloads.json`.

## Projection measurements

The [paired Q8 measurements](../benchmark-results/talkie-20260913/q8-fusion/summary.json)
used four independent process pairs for each of two prompts, one warmup and two
measured generations per process, with alternating order. All generated text
was identical between separate and fused projections. Short-prompt decode was
about 35 tokens/s; the fusion ratio was 1.003 (95% paired bootstrap interval
0.988–1.022), which does not establish a speedup.

Long-prompt runs drifted sharply across both variants (roughly 34 to 12 tokens/s).
The paired fusion decode ratio was 0.866 with a wide interval of 0.711–1.054.
Background system load was observed, but its causal contribution was not
established. These runs do not justify a stable long-prompt speed claim or
enabling fusion by default. The raw trials are retained; no outliers were
discarded. Q8 is selected for storage reduction and source fidelity, independently
of this experimental projection path.

## Longer generation checks

[Four fixed prompts](../benchmark-results/talkie-20260913/generation-comparison.md)
covered railway travel, wireless education, a seaside letter, and a steam-engine
lesson. Our Q8 responses contained 278, 254, 510, and 335 tokens respectively,
each ending at a stop token within a 512-token budget. They were relevant prose
without visible sustained loops or duplicate eight-word spans. BF16 also
returned stop for all four; its steam answer ended mid-sentence, illustrating
that an EOS stop alone is not proof of complete prose. Factual accuracy was not
scored. Plain Q4 failed every prompt and was archived for diagnosis.

The final installed binary's hash matches the tested release; its packaged
Metal resources and selected model discovery were verified. Existing running
servers were not restarted by installation.

The [final default-mode fidelity check](../benchmark-results/talkie-20260913/quality-final-default-q8.json) reproduced the reported Q8 NLL, KL, and top-token agreement with projection fusion disabled.

## Experimental-weight cleanup

The user deferred further architecture work. Experimental GQA and failed Q4
weights, recovery deltas and teacher caches were deleted; historical reports
and recipes above remain, but those experimental artifact paths no longer
contain runnable weights. Working ScaleSearch and Q8 remain installed. See
[cleanup and fresh responses](../benchmark-results/talkie-20260913/cleanup-verification/README.md).
