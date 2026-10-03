# Incremental ByteLevel streaming decoder (2026-08-30)

## Scope and controls

This milestone isolates incremental ByteLevel detokenization on top of accepted
direct-KV zero-wrapper v3. It does not include the autorelease-pool candidate or
the in-place outer token-loop decoder candidate.

- Model: `Laguna-XS-2.1-Abliterated-Q4R8-ScaleSearch-LS2`
- Engine: Metal
- Prompt: 79 uncached tokens
- Generation: greedy, 128 requested tokens
- Per block: two warmups followed by six measured target-only trials
- Memory/cache setup: 24 GiB memory limit, 256 MiB MLX cache, wired residency
- Accepted v3 control binary SHA-256:
  `59536631ba31e251bd0f71424262aa334faa48cf86fa8d9df60afdcab97ce93c`
- Final guarded candidate binary SHA-256:
  `93c85328a8d1809113df6134a13b2090bd456cdefb8a866a2a9d9b9718dbaa96`
- Metallib SHA-256:
  `903daf038bc9e65c6b77ccb3dc023df6435cf50d4d2dc78ed950a711f68be48c`
- Historical Ollama Q4R8 ScaleSearch median: `179.5662 tok/s`
- Predeclared candidate promotion gate: at least `+1%` with stable brackets

The preliminary ABBA used an earlier pre-hardening candidate build. Its token
hot path was the same, but the final source added construction-time density,
size, and malformed-configuration guards. That preliminary executable was not
retained as a durable artifact and its full SHA-256 is not present in the raw
JSON. The final reverse BAAB used the guarded candidate digest above.

## Results

| Order/block | Median decode (tok/s) |
| --- | ---: |
| Preliminary A1: v3 control | 175.308872 |
| Preliminary B1: incremental | 176.595608 |
| Preliminary B2: incremental | 176.510638 |
| Preliminary A2: v3 control | 175.865905 |
| Final B3: incremental guarded | 177.270472 |
| Final A3: v3 control | 176.226184 |
| Final A4: v3 control | 174.852671 |
| Final B4: incremental guarded | 175.503181 |

The preliminary `A -> B -> B -> A` bracket produced geometric means of
`175.587167 tok/s` for control and `176.553118 tok/s` for the candidate, an
exact `+0.550126%` effect. Control endpoint drift was `+0.317744%`; candidate
center drift was `-0.048116%`.

The final guarded `B -> A -> A -> B` bracket produced geometric means of
`175.538084 tok/s` for control and `176.384613 tok/s` for the candidate, an
exact `+0.482248%` effect. Control center drift was `-0.779404%`; candidate
endpoint drift was `-0.996946%`.

All 48 measured generations produced exactly 128 tokens with stop reason
`length`. For every raw block, this command produced the same content digest:

```sh
jq -r '.trials[].content' FILE | shasum -a 256
```

Digest:
`cde803c2ceb8ffc04563bb24b42de11575b3c56359074a5fa137f7a0fddde348`.

## Decision

Both execution orders favor incremental decoding by about one half percent.
The descriptive geometric balance across the two campaigns is `+0.516181%`,
but it is not a strict same-binary estimate because the preliminary run used
the pre-hardening build. The result is reproducible directional evidence, yet
it remains below the `+1%` promotion gate and does not close the gap to Ollama.

Preserve the implementation and measurements as a reversible candidate. Do
not promote it over accepted direct-KV v3 on this evidence alone.
