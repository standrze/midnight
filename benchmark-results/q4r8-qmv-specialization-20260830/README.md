# Affine Q4/G64 Metal QMV specialization — 2026-08-30

Hardware: Apple M5 Max, 40-core GPU. Runtime: this repository's release
`model-runner-runtime-bench` and the local Laguna-XS-2.1 standard Q4R8 runtime
view. The generated text matched exactly across every tile and trial.

The patch adds 2-, 4-, and 8-result affine Q4/group-64 fast-QMV variants for
dense and gathered calls. It remains opt-in:

```bash
MLX_METAL_AFFINE_QMV_RESULTS_PER_SIMDGROUP=2
MLX_METAL_AFFINE_QMV_MIN_OUTPUTS=4096
```

The first variable selects 2, 4, or 8 results per SIMDgroup. The optional
minimum-output threshold leaves smaller output matrices on the stock 4-result
tile. Invalid, ineligible, non-fast, or non-divisible requests fall back to 4.

## Correctness and primitive timing

Fresh-process R=2/R=4/R=8 runs covered Laguna's dense 2048→6144,
2048→8192, 2048→16384, and 2048→100352 shapes plus gathered top-8
2048→1024 and 512→2048 expert shapes. Each specialized output was compared
with dequantize-plus-matmul/gather-matmul. All full-output hashes were identical
across the three tiles. Neither alternate tile delivered a repeatable primitive
win; R=2 was generally slower on the queued small and medium shapes, while the
LM-head result was tied.

## Full-model decode

The initial two-pass 3×256-token sweep was noisy: forced R=2 produced pass
medians of 169.16 and 158.60 tok/s, stock R=4 produced 161.75 and 161.80,
and R=8 produced 150.56 and 153.03. R=8 is a clear regression, while R=2's
large between-pass spread required a narrower follow-up.

The follow-up kept small dense and gathered calls at R=4 and used R=2 only
for outputs with at least 4096 rows. Across two counterbalanced 5×256-token
pairs, the pooled trial median was 171.41 tok/s for scoped R=2 versus
172.57 tok/s for stock R=4: a 0.68% regression. A prolonged 5×512-token
forced-all run also favored R=4 (110.89 versus 87.18 tok/s), although both
long runs showed substantial machine-state drift.

Conclusion: the specialization is implemented and numerically correct, but
R=4 remains the automatic/default tile because no alternate policy produced a
repeatable end-to-end gain. The environment controls remain available for
future MLX or hardware revisions. Since Metal did not validate a speedup, no
CUDA kernel change was promoted; CUDA requires a separately tuned `.cu`
specialization rather than a translation of this Metal tile.
