# Native host-frontend diagnostic A/B — 2026-08-29

These reports isolate Swift-side work after the cold Metal traces showed nearly
equal GPU active time but materially lower native GPU duty.

## Common setup

- Model: `Laguna-XS-2.1-MLX-Q4R8-ScaleSearch-LS2`
- Engine: Metal
- Generation: greedy, 128 requested tokens
- Per block: two warmups plus 12 measured trials
- MLX cache cap: 256 MiB
- Wired residency: enabled in the current-tree KV bracket

## Clean results

| Report | Median decode | Interpretation |
| --- | ---: | --- |
| `current-control-128x12.json` | 163.7086 tok/s | Matched default cache-write control |
| `direct-kv-128x12.json` | 168.5972 tok/s | Generic index translation bypassed |
| `direct-kv-confirm-128x12.json` | 168.6380 tok/s | Adjacent confirmation |
| `single-pool-128x12.json` | 165.1341 tok/s | One per-token autorelease pool |
| `unpooled-128x12.json` | 166.0609 tok/s | Diagnostic only; cleanup can move past the timer |

The two direct-KV candidate medians average 168.6176 tok/s, **3.00%** above
the matched 163.7086 tok/s control. Trial-zero generated text hashes match the
control exactly. The prototype used complete slice bounds and called
`mlx_slice_update` directly instead of routing each K/V write through the
general NumPy-style Swift subscript translator.

This first prototype replaced the cache property with the returned `MLXArray`.
A production implementation should first measure the lower-risk form that
keeps wrapper identity and applies the result through `_updateInternal`, which
also matches Ollama's `Array.Set` behavior.

## Contaminated reports

The following files are retained for auditability but must not be used to
estimate code deltas:

- `direct-kv-stack-128x12.json`
- `direct-kv-stack-confirm-128x12.json`
- `direct-kv-stack-reassign-128x12.json`
- `direct-kv-identity-functional-128x12.json`
- `raw-handler-loaded-128x12.json`
- `control-handler-loaded-128x12.json`

During those runs, Codex was independently packing 30+ GiB of model files from
another repository into Git while a PDF service also saturated a core. The
affected blocks varied from about 95 to 165 tok/s and showed large TTFT swings,
so neither medians nor within-block trends are attributable to the candidate.

Diagnostic dependency edits were removed after the measurements. No
compile-flag experiment remains enabled in the checkout.
