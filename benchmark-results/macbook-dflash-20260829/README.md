# M5 Max Laguna runtime benchmarks — 2026-08-29

Machine: 40-GPU-core M5 Max MacBook, 64 GiB unified memory, AC power, Low Power
Mode off. Generated-token rate excludes prompt evaluation.

## Results that support decisions

| A/B | Control median | Candidate median | Change | Output gate | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| Compiled Laguna MoE fusion, same loaded target, 5x512/mode | 139.46 tok/s | 151.17 tok/s | +8.40% | exact match | Enabled by default |
| DFlash block 3, same loaded target, 5x512/mode | 134.25 tok/s | 128.54 tok/s | -4.26% | mismatch | Disabled for Mac deployment |

The compiled-fusion trials are in
`laguna-compiled-fusion-interleaved-ab.json`. DFlash proposed 495 tokens and
accepted 263 (53.1%); its report is
`dflash-block3-interleaved-ab.json`.

Compiled fusion also raised median prompt processing from 1,006.50 to 1,032.94
tok/s (+2.63%).

## Standalone Ollama controls

`ollama/` contains an earlier five-trial NVFP4 control with a 132.96 tok/s
median. `ollama-after-fusion/` contains a later five-trial control with a
149.27 tok/s median. Both used Ollama 0.33.1, the same model name, prompt,
greedy settings, and 512 generated tokens.

The 12% swing between independent Ollama runs, plus the 116.25 tok/s standalone
Swift result in `laguna-fused-after-ollama.json`, shows that process-separated
controls are dominated by changing thermal/residency state during this long
session. They are preserved as raw observations, not used to declare a runtime
winner. The same-process alternating A/B results above are the valid evidence
for the implemented graph and DFlash decisions.

## Mistral-shaped graph probes

The checked-in Metal benchmark also tested affine-Q4 Mistral-7B dimensions
without requiring a checkpoint. Packed rows, scales, and biases were
concatenated exactly and outputs were checked before timing.

| Candidate | Relative end-to-end operation throughput | Decision on MLX 0.31.1 |
| --- | ---: | --- |
| compiled SwiGLU elementwise fragment | 0.92–0.97x | reject |
| one packed gate/up QMV versus two | 0.73–0.85x | reject |
| one packed QKV QMV versus three | 0.94–1.01x | no measured gain |

These tests prevent a Laguna-specific fusion strategy from being applied
blindly to Mistral's much larger dense matrices. Rerun them after the planned
MLX core upgrade.

See [`Docs/macbook-mlx-performance.md`](../../Docs/macbook-mlx-performance.md)
for the MLX core/graph audit and remaining Laguna/Mistral roadmap.
