# Incremental fast path: definitive final-binary campaign (2026-08-30)

## Scope

These campaigns compare the accepted direct-KV zero-wrapper v3 binary with the
clean guarded incremental ByteLevel binary. Every forward and reverse block
uses the same two final executables and the same Metal library; there is no
in-place outer token-loop decoder, generation-loop autorelease change, QMV
experiment, or stable-string cache in the candidate.

- Model: `Laguna-XS-2.1-Abliterated-Q4R8-ScaleSearch-LS2`
- Engine: Metal
- Prompt: 79 uncached tokens
- Generation: greedy, 128 requested tokens
- Per block: two warmups followed by six measured target-only trials
- Accepted v3 control binary SHA-256:
  `59536631ba31e251bd0f71424262aa334faa48cf86fa8d9df60afdcab97ce93c`
- Incremental candidate binary SHA-256:
  `93c85328a8d1809113df6134a13b2090bd456cdefb8a866a2a9d9b9718dbaa96`
- Shared metallib SHA-256:
  `903daf038bc9e65c6b77ccb3dc023df6435cf50d4d2dc78ed950a711f68be48c`
- Historical Ollama Q4R8 ScaleSearch median: `179.5662 tok/s`
- Strict stability gate: absolute within-campaign drift must not exceed `1%`

The comparison for each bracket is the candidate block geometric mean divided
by the control block geometric mean, minus one. Drift is the later block
median divided by the earlier same-binary block median, minus one. The v2
balanced result uses the geometric mean of all four control blocks and all
four candidate blocks across the forward and reverse orders.

## Raw block medians

| Campaign and order | Block | Binary | Median decode (tok/s) |
| --- | --- | --- | ---: |
| v2 forward `A -> B -> B -> A` | A1 | accepted v3 | `172.841119828806` |
| v2 forward `A -> B -> B -> A` | B1 | incremental | `175.698834709663` |
| v2 forward `A -> B -> B -> A` | B2 | incremental | `176.538453109260` |
| v2 forward `A -> B -> B -> A` | A2 | accepted v3 | `171.108917770482` |
| v2 reverse `B -> A -> A -> B` | B3 | incremental | `176.172586197701` |
| v2 reverse `B -> A -> A -> B` | A3 | accepted v3 | `171.469488693009` |
| v2 reverse `B -> A -> A -> B` | A4 | accepted v3 | `171.785851069531` |
| v2 reverse `B -> A -> A -> B` | B4 | incremental | `175.744632102977` |
| v3 confirmation `A -> B -> B -> A` | A5 | accepted v3 | `173.569473628578` |
| v3 confirmation `A -> B -> B -> A` | B5 | incremental | `176.527249706232` |
| v3 confirmation `A -> B -> B -> A` | B6 | incremental | `176.496826573032` |
| v3 confirmation `A -> B -> B -> A` | A6 | accepted v3 | `171.813474854616` |

## v2 forward and reverse calculations

| Calculation | Control (tok/s) | Candidate (tok/s) | Effect | Control drift | Candidate drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forward ABBA | `171.972837855707` | `176.118143564890` | `+2.410442114505%` | `-1.002193262830%` | `+0.477873630172%` |
| Reverse BAAB | `171.627596987157` | `175.958479045328` | `+2.523418223058%` | `+0.184500682269%` | `-0.242917529884%` |
| Order-balanced | `171.800130699126` | `176.038293203396` | `+2.466914598390%` | — | — |

The reverse campaign passes the strict drift gate for both binaries. The
incremental candidate is also stable in the forward campaign, but the forward
control endpoint drift is `-1.002193262830%`, just outside the `1%` boundary.

## v3 confirmation calculations

| Calculation | Control (tok/s) | Candidate (tok/s) | Effect | Control drift | Candidate drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forward ABBA confirmation | `172.689242261389` | `176.512037484176` | `+2.213684635318%` | `-1.011697931239%` | `-0.017234241881%` |

The confirmation again favors the incremental binary and the candidate itself
is exceptionally stable. Its control endpoint drift is
`-1.011697931239%`, however, so this forward bracket also lands just outside
the strict stability gate.

## Correctness and evidence integrity

The three source directories are preserved byte-for-byte under:

- `forward-abba-v2/`
- `reverse-baab-v2/`
- `forward-abba-v3-confirmation/`

There are 12 JSON files and 72 measured generations. Every generation is
target-only, produces exactly 128 tokens, and stops for `length`. For every raw
file, the following command produces the same aggregate output digest:

```sh
jq -r '.trials[].content' FILE | shasum -a 256
```

Digest:
`cde803c2ceb8ffc04563bb24b42de11575b3c56359074a5fa137f7a0fddde348`.

## Decision

This is the best measured correct native candidate so far. The repeatable
direction and stable candidate blocks make the incremental ByteLevel fast path
the leading optimization branch. It is not promoted to accepted `main` yet:
the reverse bracket passes, but both forward control endpoint drifts
(`-1.002193262830%` and `-1.011697931239%`) are just outside the predeclared
strict `1%` stability gate. The candidate also remains below the historical
Ollama Q4R8 ScaleSearch median.

Preserve this state at the measured milestone tag and rerun a fully stable
forward bracket before considering promotion.
