# Direct KV slice-update milestone — 2026-08-29/30

This campaign measures the stable-identity direct KV slice-update patch against an exact
source-equivalent baseline. Both executables were built from the same working tree and toolchain;
the baseline was produced by reverse-applying only the two direct-KV dependency patches. The
candidate then re-applied those patches. Both bundles use the same `mlx.metallib`.

## Outcome

The optimization is correctness-clean and consistently performance-positive, but the strict
12-trial ABBA publication gate was rejected because unrelated machine-state changes made the
baseline endpoints differ by more than 2% or collapse within a block.

The clearest stable high-state comparison was:

| Block | Median decode |
| --- | ---: |
| baseline-e | 166.4966 tok/s |
| candidate-e | 172.2590 tok/s |
| candidate-f | 171.9570 tok/s |

The mean candidate median is **172.1080 tok/s**, or **+3.3703%** versus the stable 166.4966
tok/s control (0.1958 ms saved per token). It is **95.85%** of the historical exact Ollama
Q4R8 ScaleSearch median of 179.5662 tok/s, leaving 7.4582 tok/s. The best stable candidate block
was 172.2590 tok/s; the highest individual trial was 174.57695 tok/s.

Every warmup and measured trial produced byte-identical text. Every report records Metal,
79 prompt tokens, 128 generated tokens, zero cached prompt tokens, and a `length` stop.

## Rejected brackets

- `baseline-a → candidate-a → candidate-b → baseline-b`: both sides were internally stable and
  the candidate won, but the two baseline medians differed by 2.55% (limit: 2%).
- `baseline-c`: severe external contention; median 98.58 tok/s and minimum below 90% of median.
- `baseline-d → candidate-c → candidate-d → baseline-e`: candidate blocks were stable, but the
  baseline endpoints represented different machine states (151.56 versus 166.50 tok/s).
- `baseline-e → candidate-e → candidate-f → baseline-f`: the closing baseline collapsed during
  the block from about 167 to 132–142 tok/s; first-four versus last-four means differed by 14.5%.

The raw reports are retained so no favorable subset is silently substituted for the rejected
publication brackets. `results.json` records the consolidated interpretation.

## Reproduction

- Model: `Laguna-XS-2.1-Abliterated-Q4R8-ScaleSearch-LS2` (fused 1,397-tensor artifact)
- Request: greedy, 128 output tokens, 2 warmups, 12 measured trials
- MLX memory limit: 24 GiB
- MLX cache limit: 256 MiB
- Wired memory: enabled; tuning horizon 513 tokens
- Swift: 6.3.3
- Xcode: 26.6 (17F113)
- `Package.resolved`: `91c1db67f3c74b19a63d0f557d8bbfbd7f3e90795f207632c86c626fea820da9`
- Baseline executable: `b7b708d9794422a932e242eaac1952014bab67db51f293e490c1d1392c235c83`
- Candidate executable: `4c7241e7e23f2c184c9d4d8ae9627a621d73c2116411339d3c43e742bfd6bbec`
- `mlx.metallib`: `903daf038bc9e65c6b77ccb3dc023df6435cf50d4d2dc78ed950a711f68be48c`
- Generated-text aggregate SHA-256: `3e30da17e2ee690531192c9b06dfc827c1fc5276334a24358cc177374ab87d57`

Model fingerprints:

- `config.json`: `e69b519e0ccb52568a736386414136e90a1b7963ffc2a6da9bdefa73b92fe7b1`
- `tokenizer.json`: `807c53a95141e77c14e45f68c51db3f84d2ea6b555a6ea832bc99c88dae6a279`
- `model.safetensors.index.json`: `10ed73f0beb7944a2b348b73d7e1faa654039601a585a8dbd431e68826485391`
