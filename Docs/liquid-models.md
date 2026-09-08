# Liquid LFM text models

Midnight uses the native `MLXLLM` dense `lfm2` and hybrid MoE `lfm2_moe`
loaders. Model selection comes from checkpoint `config.json`, not its folder
name. Both loaders advertise Liquid's `lfm2` tool-call format.

| Checkpoint | Loader | Intended use |
| --- | --- | --- |
| LFM2.5-1.2B-Base | `lfm2` | Pretrained text continuation and fine-tuning |
| LFM2.5-1.2B-Instruct / Thinking | `lfm2` | Instruction following / reasoning |
| LFM2.5-2.6B / Base | `lfm2` | Larger dense text model |
| LFM2-8B-A1B / LFM2.5-8B-A1B | `lfm2_moe` | Sparse expert text models |
| LFM2-24B-A2B | `lfm2_moe` | 24B total-parameter expert model |

Use MLX-compatible safetensors checkpoints, including their tokenizer files and
configuration. GGUF, ONNX, and DSpark artifacts are not interchangeable with
these MLX loaders. This page covers text models, not Liquid audio, vision,
embedding, or ColBERT variants.

Place an exported checkpoint beneath `~/.midnight/models`, then run:

```sh
midnight-runner --list-models
midnight-runner --model LFM2.5-1.2B-Instruct --host 127.0.0.1 --port 8080
midnight-chat --endpoint http://127.0.0.1:8080/v1 --model LFM2.5-1.2B-Instruct
```

The folder name in these commands must match your actual checkpoint folder.
The Base checkpoint is not instruction-tuned. When no chat template is present,
the upstream text processor joins message contents as plain text; this does not
make a base model an instruction-following assistant. Prefer Instruct for chat.

Liquid's convolution layers carry recurrent state alongside attention KV state.
Midnight keeps these models outside its Mistral/GPT-OSS hot conversation cache
and does not enable Mistral/Laguna-only KV compression or DFlash for them.
Automatic tool parsing uses the upstream Liquid parser; forced tool choice is
not implemented for these models. GPT-OSS `reasoning_effort` handling is not a
Liquid Thinking control.

All expert weights must fit in memory even though only some experts run for
each token. Use checkpoint-specific context limits and measured memory budgets;
config metadata alone does not establish reliable long-context quality.

## Verification

`LiquidModelCompatibilityTests` checks the official 1.2B Base, 8B-A1B and
24B-A2B config snapshots against the pinned Swift decoders, including the MoE
attention/convolution layout, expert counts, vocabulary and nested RoPE settings.
It also guards against accidentally enabling transformer-only hot caches.

```sh
swift test --filter LiquidModelCompatibilityTests
```

These are configuration compatibility tests, not full checkpoint generation or
quality benchmarks. No model weights were downloaded for this check. End-to-end
loading, quantized weights, streaming, tool calls and multi-turn behavior still
need verification using the specific checkpoint to be deployed.

Sources checked September 7, 2026:

- [Liquid model library](https://huggingface.co/LiquidAI/models)
- [LFM2.5-1.2B-Base](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base)
- [LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B)
- [LFM2-24B-A2B](https://huggingface.co/LiquidAI/LFM2-24B-A2B)
