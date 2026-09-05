# KV compression and prefill peak — 2026-09-04

The current Laguna path compresses eligible KV caches **after prompt evaluation**. This explains why compression can reduce decode cache storage without reducing a request's earlier prefill peak. Source inspection establishes that ordering; it does not identify the allocation that produced each measured peak.

## Measurements and limits

Nine matched retrieval requests completed for `none` and `affine8`, spanning approximately 4.3k, 17k and 34k prompt tokens with 512-token prefill chunks. Their maximum per-request MLX peaks were 20,881,835,488 and 20,881,842,000 bytes respectively, a 6,512-byte difference. Five generated texts match exactly; four differ. Similar peak memory does not imply compression was inactive or that generation was identical.

`Sources/ModelGenerationBenchmark/Command.swift:140` resets MLX peak before each request and line 226 records it after generation. The wired-residency wrapper can reset it again immediately before generation (`LocalModelRunner.swift:772`). These are whole-request high-water marks, without separate prefill/conversion/decode measurements. The startup `eligible=10, skipped=30` message comes from planned cache status (`LocalModelRunner.swift:449–459`), not a post-prefill measurement of realized cache bytes.

## Confirmed source path

- `Sources/ModelRunnerCore/LagunaModel.swift:1379–1386` constructs native hybrid caches. Dependency `CacheConfiguration.swift:291–316` returns `KVCacheSimple` for unbounded full attention and `RotatingKVCache` for native sliding windows; it does not construct compressed caches here.
- `LagunaModel.swift:1117–1124` selects normal preparation. Its `prepareNormally`, lines 1177–1205, runs all prefill chunks using the original cache array, then evaluates the cache. There is no compression callback between chunks.
- Dependency `Evaluate.swift:855–889` calls `model.prepare`, then evaluates any prompt remainder with `step`. In `step`, lines 905–915, the model forward precedes cache-plan application. All prompt positions and the first next-token logits therefore use the initial uncompressed cache representation. Later decode steps use the rewritten cache.
- `KVCachePlan.swift:21–32` validates initial compatibility without conversion. Lines 44–57 and 69–85 apply the strategy and stop revisiting it once application is terminal. `KVCache.swift:2507–2526` rewrites only eligible simple caches after their offset exceeds the configured threshold; rotating caches remain unchanged. `LongContextConfiguration.swift:10–15` selects the requested strategy, with its default zero threshold.
- `KVCache.swift:493–524` converts the accumulated full-precision keys/values. Their lazy quantization graphs can retain source arrays until evaluation; conversion is not guaranteed to reduce peak immediately. Wired-memory tuning has the same post-prefill application order (`WiredMemoryUtils.swift:91–115`).

These source facts make prefill a strong explanation for the similar peak. They do not prove that prefill, rather than conversion or another workspace allocation, set the measured maximum.

## Admission already budgets full precision

There is no compressed-cache discount in admission. `Sources/ModelRunnerProtocol/LongContextOptions.swift:47–49` states the conservative policy. Lines 85–86 budget FP32 K+V (`8 × kvHeads × headDim` bytes per layer-token); lines 100–104 honor native windows; lines 118–123 add prompt/output positions, 256 positions of slack, incoming prefill rows and workspace. `LocalModelRunner.swift:594–600` enforces the estimate, and lines 687–702 account for retained/live allocations. Compression does not reduce this budget. No admission fix is warranted for the suspected discount.

## Earlier compression needs an attention design

Simply moving conversion into the prefill loop is risky. `AttentionUtils.swift:72–84` routes affine caches into `quantizedScaledDotProductAttention`; `KVCache.swift:2393–2429` constructs QK scores, masks, softmax and AV explicitly. Uncompressed caches use MLXFast SDPA (`AttentionUtils.swift:85–93`). At 48 query heads, 512 query positions and 34,000 cached positions, a single full score tensor would contain 835,584,000 elements: 1.67 GB in BF16 or 3.34 GB in FP32, before other intermediates. This is a shape calculation, not a measured allocation or claim that every temporary coexists.

Next, instrument cache type/offset/realized bytes and active/peak memory after prefill chunks, before/after conversion and during decode. Synchronize only in diagnostic mode so measurement does not silently change the production execution policy. Then compare blockwise or fused compressed prefill attention, or bounded layerwise dequantization into fast SDPA. A temporary dequantization design still needs explicit lifetime and workspace limits. Keep sliding-window cache semantics and quality checks intact, and retain conservative admission until a lower bound is justified by measured behavior.

## Source identity

Repository base: `a49a5990d3f3965ad37fc202c0f46d9e323ab569`, with this worktree's changes. Dependency base: mlx-swift-lm `14414441fa44f45eee35a61e9fa0bab577cf9734`. Dependency locations above are under `.build/checkouts/mlx-swift-lm/Libraries/MLXLMCommon/`. The checkout includes local patches, notably to `Evaluate.swift` and `KVCache.swift`; upstream commit links alone would not identify the inspected content. [The provenance record](../benchmark-results/next-priorities-20260904/kv-source-review/provenance.json) contains SHA256 hashes for every cited source file and both completed measurement reports.
