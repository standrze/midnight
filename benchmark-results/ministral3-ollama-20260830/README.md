# Ministral 3 14B: native MLX ScaleSearch versus Ollama Q4_K_M

Measured 2026-08-30 on a 64 GB Apple M5 Max MacBook Pro running macOS 26.6.2.

## Result

For this model, prompt, machine, and 128-token decode workload, the native MLX ScaleSearch checkpoint was **35.21% faster** than the official Ollama Q4_K_M package by pooled median decode rate:

| Runtime/package | Measured trials | Median tok/s | Mean tok/s | Range tok/s |
| --- | ---: | ---: | ---: | ---: |
| Native MLX ScaleSearch affine Q4/group-64 | 12 | 67.995 | 67.258 | 59.751–68.142 |
| Ollama Q4_K_M | 12 | 50.287 | 50.423 | 48.968–51.896 |

The median ratio is `67.995 / 50.287 = 1.3521`, or **35.21% more generated tokens per second**. Equivalently, Ollama took about 26.04% more decode time per token.

The bracketed block medians were:

| Block | Runtime | Median tok/s |
| --- | --- | ---: |
| A | Native ScaleSearch | 67.848 |
| B | Ollama Q4_K_M | 51.187 |
| C | Ollama Q4_K_M | 49.566 |
| D | Native ScaleSearch | 68.075 |

The native A-to-D change was +0.33%; Ollama B-to-C changed -3.17%. Every native measured trial was faster than every Ollama measured trial, including the single 59.751 tok/s native low outlier.

A later, slower system-state replay preserved the ratio: native ScaleSearch measured 55.566 tok/s and Ollama measured 41.044 tok/s, a **35.38% native advantage**. Absolute rates therefore moved with system state, while the relative gap remained nearly unchanged.

## What this proves

This establishes that the current native MLX package/runtime is faster than this official, typical Ollama quant on this machine and workload. It does not prove superiority over every GGUF quant, every prompt length, or every llama.cpp/Ollama release.

It also does **not** show that ScaleSearch calibration itself is the source of the speedup. The ScaleSearch and matched standard MLX checkpoints both use affine 4-bit weights with group size 64 and identical storage/kernel geometry. The earlier path-matched timing control differed by only about 0.21%, which is noise-sized. ScaleSearch's demonstrated benefit is quantization quality; the large Ollama comparison gap is a package/runtime/backend result.

An ordinary MLX Q4 diagnostic later measured 51.540 tok/s, followed by ScaleSearch at 55.566 tok/s. Because those were free-running, non-interleaved generations during visible system-state drift, that difference is not evidence of a ScaleSearch speed effect.

## Models and runtimes

- Native checkpoint: `tmp/models/Ministral-3-14B-Instruct-2512-MLX-Q4-ScaleSearch-LS2-3cea74c`
- Native source revision: `3cea74c1ebaf5ce5f5a2553de470e2ceab825142`
- Native format: affine Q4, group size 64, about 7.1 GiB on disk
- Ollama tag: `ministral-3:14b-instruct-2512-q4_K_M`
- Ollama model ID: `4760c35aeb9d`
- Ollama package: Q4_K_M, 9.1 GB download, including vision assets
- Ollama server/client: 0.33.1 / 0.31.1
- `ollama ps`: 8.8 GB resident, 100% GPU, 4096-token context
- Repository commit at measurement: `aa879f011588302a2ef16d58ca3f0f12fba63931`

The Ollama control is a practical same-model comparison, not a bit-matched quant comparison. Q4_K_M mixes K-quant tensor types, while the native checkpoint uses uniform affine Q4/group-64.

## Protocol

Prompt:

> Write a long, detailed technical tutorial about implementing a lock-free work-stealing scheduler in Swift. Continue with implementation details and code examples until the output limit; do not conclude or summarize early.

- 128 requested/generated tokens per trial
- two warmups before each measured block
- deterministic decoding: temperature 0, top-p 1, seed 0
- Ollama: `/api/chat`, non-streaming, `think: false`, `num_ctx: 4096`
- native: direct Metal runtime benchmark, excluding HTTP overhead
- Ollama decode rate: `eval_count * 1e9 / eval_duration`
- all measured generations stopped for `length` at exactly 128 tokens

The native chat template reported 570 prompt tokens; Ollama reported 593. Therefore this is user-prompt matched but not exact prompt-token-ID matched. The extra 23 Ollama prompt tokens are a limitation, although decode rate—not prompt/prefill rate—is the reported metric.

## Artifacts

- `native-a.json`: opening native block
- `ollama-primary.json`: Ollama warmups and 12 measured trials
- `native-d.json`: closing native block
- `standard-mlx-q4.json`: later ordinary MLX Q4 diagnostic
- `scalesearch-after-standard.json`: later ScaleSearch replay
- `ollama-late-replay.json`: later Ollama replay

References:

- Official Ollama model: https://ollama.com/library/ministral-3:14b-instruct-2512-q4_K_M
- Ollama timing fields: https://github.com/ollama/ollama/blob/main/docs/api/usage.mdx
- Official Mistral GGUF collection: https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-GGUF
