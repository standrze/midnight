# Synthetic Talkie numerical fixtures

These weights are generated data, not Talkie checkpoint excerpts. The default
shape has two layers, hidden width 16, two heads of width eight, a 32-wide MLP,
and 32 vocabulary entries. Every residual gain and embedding skip is nonzero;
the query gains differ between heads. The Q4 fixture uses hidden width 32 and
MLP width 64 so affine group size 32 divides every matrix input dimension.

`generate_reference.py` uses independent, explicit Python MLX tensor equations
based on the [official Talkie model implementation](https://github.com/talkie-lm/talkie/blob/main/src/talkie/model.py).
It imports neither Midnight nor `mlx_lm`. It computes inverse NeoX rotation
directly from sine and cosine, RMS normalization through a Float32 reduction,
dense attention with an explicit rectangular causal mask, and an independent
K/V cache. The vocabulary, weights, tokens, and gains are deterministic.

The equations retain the official placement of Q/K normalization after RoPE,
per-head query gains, residual branch gains, and the normalized embedding skip.
The output gain is folded into the head weight before multiplication, matching
the official PyTorch implementation. The merged Python MLX port instead applies
the gain to output logits, which can differ through BF16 rounding.

The small Float32 and BF16 fixtures use canonical HF keys, including bare
`lm_head` and `lm_head_gain.w_g`. Q4 stores `lm_head.weight/scales/biases` with the
gain already folded, like the public DWQ conversion. Its BF16 reference uses
MLX's packed quantized-matmul primitive inside the independent model equations.
The CPU BF16 QMM accumulator rounds differently from dense GEMM over dequantized
BF16 weights: the two paths differ by 0.109375 in this fixture, so the latter
cannot serve as a tight golden reference for the former.

A second Q4 reference promotes the same affine scales and biases to Float32,
unpacks each four-bit integer explicitly, and reconstructs `code * scale + bias`
before dense multiplication. It uses neither quantized matmul nor dequantize.
This independent Float32 calculation agrees with packed QMM within 0.0000012,
and Swift tests require the same strict 0.00002 bound as the unquantized fixture.
Together, the references cover actual BF16 checkpoint dtypes and independently
verify quantized loading and projection fusion without accumulator ambiguity.

The BF16 packed-reference check uses a per-token relative root-mean-square logit
error below two BF16 epsilons (`1/64`), finite logits, unchanged top tokens, and
exact CPU fused/unfused parity. A relative vector norm remains meaningful near
zero logits, where elementwise relative errors and ULP counts do not. The
observed worst-token relative RMS is 1.06%; the wrong-RoPE, missing-embedding-skip,
and missing-query-gain mutations produce 7.31%, 17.92%, and 2.19%, respectively.
All exceed the 1.5625% bound; their values are recorded in the manifest. The
Float32 packed reference retains the independent, tight absolute-error check.

Each expected tensor file contains the complete seven-token forward pass,
two-token prefill, three-token cached continuation, and two single-token decode
steps. The Float32 CPU reference agrees between full and cached execution within
0.000001. The manifest records checksums and fixture sensitivity to a wrong RoPE
sign, omitted query gain, and omitted embedding skip. All three mutations cause
logit errors far larger than the Float32 test tolerance.

Regenerate with Python MLX 0.31.2:

```sh
python3 Tests/Fixtures/TalkieTiny/generate_reference.py
```

The generator uses CPU execution for stable, shape-independent reference
rounding. On macOS, initialization of MLX can still require access to Metal.
Run the Swift coverage using the repository's normal test launcher and
`--filter TalkieModelTests`. Fixture paths are resolved relative to `#filePath`,
matching the other fixtures in this repository; no package resource changes are
required.
