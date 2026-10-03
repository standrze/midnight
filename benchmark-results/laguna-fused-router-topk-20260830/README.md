# Fused Laguna router top-k — 2026-08-30

## Decision

Accepted for production promotion on the isolated candidate branch. The fused
decode router is token-exact, exceeds the predeclared `+1%` gate in two
independent same-loaded campaigns, stays inside the `1%` early/late drift gate,
and is the first correct Swift arm to beat the historical exact Ollama Q4R8
ScaleSearch median.

- Confirmation medians: `183.753624693681` and `183.782052267099 tok/s`
- Same-loaded median effects: `+3.483719502080%` and `+4.177331039446%`
- Geometric/order-balanced effects: `+4.036358479396%` and
  `+4.231835277240%`
- Historical Ollama median: `179.566223067381 tok/s`
- Conservative lead over Ollama: `4.187401626300 tok/s`
  (`+2.331953947001%`)
- Exact generated output: yes

`origin/main` remained at `7238a63` while this evidence was collected. The
measured source is preserved separately before any default-on production
change.

## Default-on production confirmation

After preserving the default-off candidate and both same-loaded campaigns,
commit `8fa0f0a07b2ae0b39c21a3d20955ff97600b4ff2` enabled the compiled block tail
and fused router automatically for `LocalModelRunner` requests resolved to the
Metal engine. CPU and CUDA remain on the existing path. Scoped explicit true
and false values still override the automatic choice, so the same-loaded A/B
and rollback diagnostics remain valid.

The ordinary benchmark path used no fast-path flag:

```sh
.build/arm64-apple-macosx/release/model-runner-runtime-bench \
  /path/to/models/Laguna-XS-2.1-Abliterated-Q4R8-ScaleSearch-LS2 \
  /private/tmp/fused-router-production-default-v3.json \
  --engine metal \
  --tokens 128 \
  --warmups 2 \
  --trials 6
```

- Default-on median: `185.291435150063 tok/s`
- Default-on geometric mean: `184.905172116502 tok/s`
- Early-three versus late-three drift: `-0.147554408633%`
- Lead over Ollama: `5.725212082682 tok/s` (`+3.188356910828%`)
- Improvement over the prior `177.499062770142 tok/s` best arm:
  `+4.390092126859%`
- All warmups and trials: exact 742-byte output, 128/128 tokens, `length` stop
- Production release binary SHA-256:
  `fc9b1ad973f1fc1d27b046f3d462aae1163bf72aff697da5a1e22fd53105f385`

Two intervening diagnostic runs were excluded from performance evidence after
`mediaanalysisd` reached 131% CPU. The preserved production confirmation was
run only after that daemon fell to single-digit CPU and satisfies the drift
gate independently.

## Candidate

- Branch: `codex/fused-laguna-router-fast-path`
- Base: `9cbdd9462dda584dd8fb938af4228bddb9b24966`
- Source commit: `c6c29da497f6437fa819587eee142ea246ed28be`
- Candidate tag: `laguna-fused-router-topk-candidate-v1`
- Release binary SHA-256:
  `e6fad6d3ef1cbf6387c4ace6dd41532ad71c0bea9fed301a1fd2d544515d27a8`
- `mlx.metallib` SHA-256:
  `903daf038bc9e65c6b77ccb3dc023df6435cf50d4d2dc78ed950a711f68be48c`

The candidate keeps the existing FP32 sigmoid. One Metal threadgroup then
ranks the single 256-expert decode row with the correction bias, gathers the
eight corresponding unbiased scores, normalizes in stable slot order with the
existing `1e-20` denominator term, and casts the weights back to the projection
dtype. Prefill and every unsupported shape retain the original chain.

The benchmark owns two independent compiled tails per sparse layer. The
legacy and fused arms therefore cannot alias a first-traced TaskLocal value.
Both arms use the same compiled attention gate, compiled MoE residual, compiled
block tail, direct KV updates, model instance, prompt, and event-stream path.

## Correctness checks

- Exact router indices and weights for FP32, FP16, and BF16: passed.
- Stable all-equal and zero-valued selection-key ties: passed.
- Tiny Laguna prefill and second-token logits: bit-identical.
- Full `LagunaModelTests`: 11/11 passed.
- Dependency preparation idempotence: passed twice.
- Debug real-Q4 smoke: exact 16/16-token outputs; directional `+7.19%`.
- Release real-Q4 smoke: exact 16/16-token outputs; directional `+4.41%`.
- Both 128-token confirmation campaigns: every warmup and measured output is
  identical, 742 UTF-8 bytes, 128/128 tokens, `length` stop.
- Canonical generated-content SHA-256:
  `5b8e8efa2dd96fdc936e519d6f0922a1d4b0693dbf13167abfdea592d54d5946`

## Release benchmarks

Each campaign used:

```sh
.build/arm64-apple-macosx/release/model-runner-runtime-bench \
  /path/to/models/Laguna-XS-2.1-Abliterated-Q4R8-ScaleSearch-LS2 \
  OUTPUT.json \
  --engine metal \
  --tokens 128 \
  --warmups 2 \
  --trials 6 \
  --laguna-router-topk-ab
```

Within each run, measured order alternated legacy/fused then fused/legacy.

| Campaign | Legacy median | Fused median | Median effect | Geometric effect |
| --- | ---: | ---: | ---: | ---: |
| v1 | 177.567665307960 | 183.753624693681 | +3.483719502080% | +4.036358479396% |
| v2 | 176.412709399813 | 183.782052267099 | +4.177331039446% | +4.231835277240% |

Early-three versus late-three geometric drift stayed under the gate in both
campaigns:

| Campaign | Legacy drift | Fused drift |
| --- | ---: | ---: |
| v1 | -0.014406029975% | +0.624673883311% |
| v2 | +0.202257042723% | +0.554578039443% |

The two execution-order effects were independently positive in each campaign:
`+4.096613304819%` / `+3.976138531610%` in v1 and
`+4.575740910948%` / `+3.889060604536%` in v2.

## Preserved artifacts

- `fused-router-release-abba-v1.json` SHA-256:
  `c552387bd4f6003a8dc3776b5a05c27c8795f443b1ad73d422d5133b72f84a4c`
- `fused-router-release-abba-v2.json` SHA-256:
  `f7f4f1aec3b72cc12449c4f06af14131d8d1fed95da9978e2f3942b6f102c746`
- `fused-router-release-smoke-v1.json` SHA-256:
  `8f286429af423528886ecf9227dffdf5e57b1aadcdbeeff7a5d041ff000e2802`
- `fused-router-debug-smoke-v1.json` SHA-256:
  `8900b30b3c13c5e1e38a45c386250e93b793f1c2d3beecced7ad7119fa875a5d`
- `fused-router-production-default-v3.json` SHA-256:
  `068b5f781a38d04d427e8e6c62f2a23d1e2c81ffefd294e9a37eb5852d6b2f06`
