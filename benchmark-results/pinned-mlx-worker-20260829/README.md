# Pinned MLX worker A/B — 2026-08-29

This directory records the macOS permanent-pthread experiment described in
[`Docs/ollama-optimization-gap.md`](../../Docs/ollama-optimization-gap.md).

The feature is intentionally opt-in:

```bash
MODEL_RUNNER_PINNED_MLX=1 bash Scripts/benchmark-runtime-model.sh \
  /path/to/Laguna-XS-2.1-Abliterated-Q4R8-ScaleSearch-LS2 \
  benchmark-results/pinned-mlx-worker-20260829/new-result.json \
  --engine metal --tokens 128 --warmups 2 --trials 12
```

`pinned-worker-128x12.json` is the primary matched block. Its median decode
rate was 166.56 tok/s, versus the prior exact-prompt native baseline of 166.09
tok/s (+0.28%). Median time to first token rose from 77.23 to 83.42 ms.

`pinned-worker-confirm-128x12.json` is an immediate confirmation block. It was
nonstationary (155.73–165.90 tok/s) and had a 158.34 tok/s median, so it is
evidence against treating the small primary-block increase as a real win.

`results.json` contains the compact comparison and decision. The raw benchmark
files retain all generated content, per-trial metrics, and warmups.
