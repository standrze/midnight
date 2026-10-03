# Dependency patches

Midnight applies reviewed patches to exact, pinned dependency revisions during
the build. The preparation script verifies each checkout revision before any
patch is applied and refuses an unexpected source tree.

File prefixes identify their upstream project:

| Prefix | Upstream | License |
| --- | --- | --- |
| `mlx-*.patch`, `mlx-c-*.patch`, `mlx-swift-*.patch` | `ml-explore/mlx-swift`, MLX, and MLX-C | MIT |
| `mlx-swift-lm-*.patch` | `ml-explore/mlx-swift-lm` | MIT |
| `swift-transformers-*.patch` | `huggingface/swift-transformers` | Apache-2.0 |

See [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for attribution and
license notices. Copyright and authorship lines present in upstream source are
retained inside the patch hunks.

`mlx-swift-lm-chat-cache-memory.patch` adds a read-only ChatSession cache-byte
query so admission and retention decisions do not allocate a snapshot. It
applies independently after the existing session-snapshot patch.

`mlx-swift-lm-reasoning-stream.patch` exposes semantic reasoning events through
the generation API and separates Gemma 4 thought-channel text before tool-call
parsing. Midnight controls its transport exposure with `include_reasoning`.
Gemma delimiter split/truncation tests and the real A4B HTTP matrix in
`benchmark-results/gemma4-a4b-reasoning/` cover the routing change.

`mlx-swift-lm-extra-eos-token-identity.patch` requires an additional named EOS
marker to round-trip through the tokenizer's vocabulary before its ID joins
the stop set. A missing name that resolves to the unknown-token fallback no
longer adds that fallback as an extra EOS. Explicit numeric EOS IDs, the
tokenizer's primary EOS and decoded stop strings remain unchanged. The separate
existing policy of stopping generation on unknown tokens also remains unchanged;
this resolver hardening alone makes no generation-quality or speed claim.
`Scripts/extra-eos-token-identity-patch.sh` applies the overlay atomically after
the MTP overlays, verifies the exact pinned revision and refuses conflicting or
partial source. `Tests/Shell/ExtraEOSTokenIdentityPatchTests.sh` covers replay,
idempotence and refusal cases. The four `ExtraEOSTokenIdentityTests` pass in the
stage 6 run using the real BPE fixture in
`Tests/Fixtures/UnknownFallbackTokenizer`, including intentional unknown EOS
and multi-token decoded stop strings.

`mlx-swift-lm-gemma4-dense-fusion.patch` adds an experimental compiled GELU × up
activation in Gemma 4's dense branch. Set `MIDNIGHT_GEMMA4_DENSE_FUSION=1` before
loading the model to enable it; omission keeps the existing path. A4B Mac
off/on/off measurements and matching generated outputs are recorded in
`benchmark-results/gemma4-a4b-reasoning/dense-fusion-*.json`. It remains opt-in
while broader workload and CUDA validation are outstanding.

`mlx-swift-lm-gemma4-window-mask.patch` completes the retained-cache Gemma 4
path by enforcing the sliding window for every chunk and single-token decode.
An ordinary cache's default mask omits the window for short queries, so keeping
old K/V entries requires an explicit windowed causal mask once history exceeds
the checkpoint's window. The patch changes neither full-attention layers nor
the configured attention span. The A4B long-prompt regression is recorded under
`benchmark-results/gemma4-a4b-reasoning/window-fixed/`.

`mlx-swift-lm-gemma4-window-slicing.patch` adds an opt-in decode optimization:
regular sliding-attention layers read only the trailing active window while
retaining the original cache and offsets. Full attention, prefill, and
quantized-cache paths retain their existing behavior. Enable with
`MIDNIGHT_GEMMA4_WINDOW_SLICING=1`; the benchmark's
`--gemma-window-slicing-ab` overrides it per generation for paired comparison.
The float32 CPU equivalence regression covers both settings. Real-model
performance and output checks determine whether it should become a default.

`mlx-swift-lm-gemma4-text-mtp.patch` provides Gemma text targets with the shared
embedding and opt-in hidden/KV state contracts used by the assistant drafter.
The assistant bounds its own sliding-KV view without truncating target state
needed for verification rollback. Ordinary forwards do not emit draft state.
Small CPU tests compare ordinary/speculative token sequences for both wrapper
types at block sizes 2/4. The official A4B QAT assistant weights also load.
This adapter does not by itself enable a Midnight CLI drafting option.

`mlx-sorted-gather-qmm-nax-row-bounds.patch` backports the kernel correction from
[MLX commit d73eb752ef2e6288fd95b032c0bff0a15a4a9e93](https://github.com/ml-explore/mlx/commit/d73eb752ef2e6288fd95b032c0bff0a15a4a9e93)
(Philip John Basile, co-authored by Cheng). It clamps the remaining sorted
expert-row count as an integer before narrowing to a signed 16-bit tile extent.
The companion `mlx-swift-sorted-gather-qmm-nax-row-bounds-jit.patch` applies the
same correction to both generated shader copies. Both patches are macOS-only
and apply to the existing pinned revisions; they do not change dispatch or
quantization values.

This bound counts token/expert assignments in one operation, not the whole
conversation. A 512-token Laguna prefill chunk has at most 4,096 top-8 rows and
does not reach the overflow. An unaligned 4,097-token chunk has 32,776 rows and
can reach it. Exact tile-aligned operations bypass the affected bounds branch.

Run the source-fixture checks without a build or GPU:

```sh
bash Tests/Shell/MLXSortedGatherQMMNAXPatchTests.sh
```

After dependency preparation and rebuilding the release Metal library, run the
opt-in kernel regression on macOS 26.2+ with a NAX-capable GPU (for example M5
Max), with no concurrent model or GPU workload:

```sh
MIDNIGHT_RUN_NAX_REGRESSION=1 swift test --configuration release --filter SortedGatherQMMNAXTests
```

The fixture compares sorted affine-Q4 gather results with dequantized FP32
matrix multiplication across the signed-16-bit boundary, with group sizes 32
and 64, 16 and 256 experts, aligned controls, and poisoned allocator buffers.
It is adapted from the upstream regression. An unset opt-in variable skips the
kernel test; a skipped test is not NAX validation. This is a correctness fix,
not a measured speed improvement.

`mlx-compile-cache-lifetime.patch`, `mlx-c-compile-cache-lifetime.patch`, and
`mlx-swift-compile-cache-lifetime.patch` repair cross-thread destruction in the
pinned macOS MLX 0.32 dependency stack. Its C++ compiler caches are thread-local;
the Swift function destructor previously erased only the destructor thread's
cache, leaving captured model weights in caches belonging to other live threads.
The new explicit erase-all API removes only the destroyed function's ID. The
existing per-cache erase API retains its meaning.

Each thread registers its cache once in a process-lifetime registry of weak
handles. Warm compiled calls do no registry work. Function destruction snapshots
live caches, releases the registry mutex, removes entries under each cache mutex,
and destroys removed graphs after releasing those mutexes. Swift retains its
synchronous `evalLock` barrier to prevent address-based function-ID reuse during
cleanup. This changes function teardown, not generated math or decode choices.

`Scripts/compile-cache-lifetime-patch.sh` verifies all three exact Darwin
revisions and preflights all patches before writing any checkout. Linux uses a
different pinned dependency stack and is left untouched. The binding patch keeps
both exported header copies synchronized with the MLX-C declaration. Run the
network-free patch fixture checks with:

```sh
bash Tests/Shell/MLXCompileCacheLifetimePatchTests.sh
```

The opt-in `MLXCompileCacheLifetimeTests` regression keeps two tracing threads
alive while destroying a compiled function on a third thread. Before the patch,
the process retained the entire 4 MiB capture (4,198,400 active bytes versus the
4,096-byte baseline) after destruction. After the patch, both lifetime tests
pass and an unrelated compiled function remains warm. This bounded regression
proves the cross-thread cache fix; it does not establish that all model unload
paths release every allocation.

```sh
MIDNIGHT_RUN_COMPILE_CACHE_LIFETIME=1 swift test --jobs 2 --filter MLXCompileCacheLifetimeTests
```

Run this alone; allocator measurements require an otherwise idle process.

`mlx-swift-lm-gate-up-slices.patch` works around the pinned MLX compiler's
multi-output graph lifetime problem in the shared `FusedGateUpSwitchGLU` layer.
On macOS it replaces the equal gate/up `split` with two independent views. The
native Laguna dense/shared MLP and calibration branch use the same views.
Matrix dimensions, expert sorting, activation, quantization, and compilation
remain unchanged. The original split remains in the Linux branch; dependency
preparation is a strict Linux no-op. The shared layer change also applies to
other Mac models that use `FusedGateUpSwitchGLU`.

This is a narrow workaround for [upstream MLX issue #3932](https://github.com/ml-explore/mlx/issues/3932),
not a general change to array destruction. The source investigation separately
reproduced concurrent final-sibling release and sequential descriptor-assignment
leaks. Those diagnostic fixtures remain in the campaign artifacts. No global
array destructor mutex, compiler-wide cache clear, or Metal callback change is
included in the product.

The pinned helper rejects revision or source drift and applies idempotently.
Run its network-free fixture and the numerical/lifetime suite with:

```sh
bash Tests/Shell/MLXGateUpSlicesPatchTests.sh
MIDNIGHT_RUN_GATE_UP_LIFETIME=1 swift test --jobs 2 --filter LagunaGateUpSliceTests
```

The opt-in flag enables Metal checks and allocator-sensitive repeated-lifetime
checks, which require an otherwise idle process/GPU. The suite compares eager
against eager and compiled against compiled reference graphs, including sorted
and unsorted routing, quantized and unquantized projections, and contiguous and
strided values. CPU numerical checks also run without the flag. In the real
compiled Laguna unload diagnostic, the independent views reduced the remaining
roughly 2.2 GB of active MLX memory to 6,412 bytes. That is cleanup evidence, not
a throughput claim; the campaign separately evaluates performance.
# Text-only Gemma assistant

`mlx-swift-lm-gemma4-text-assistant.patch` follows the Gemma text MTP adapter.
It adapts the upstream MIT-licensed assistant to MLXLLM's decoder and adds an
explicit shared-KV-only attention constructor mode. Ordinary decoder construction
retains its defaults. The assistant reuses text attention, MLP, norms and RoPE;
Midnight registers it only when an assistant is requested. Reference comparisons
cover float32/Q4 miniature models and optional official BF16/Q4 A4B weights.
`GemmaReasoningPatchTests.rb` verifies the complete patch stack round trip.

`mlx-swift-lm-laguna-tools.patch` layers after the reasoning-stream patch. It
preserves assistant reasoning through template rendering and cached histories,
accepts zero-argument GLM/Laguna calls, preserves string whitespace, decodes JSON
scalars, and rejects truncated or malformed argument lists. Midnight explicitly
selects Laguna's tagged tool and reasoning protocols from model metadata.

### Sparse mixed-precision module arrays (macOS)

`mlx-swift-sparse-module-array-update.patch` fixes module replacement when
selective quantization skips the first element of a transformer-block array.
Poolside's official Laguna INT4 layout keeps layer 0 unquantized and starts
quantization at layer 1; the original updater selected the update structure
from the missing first entry and crashed. The patch selects the first present
update and skips untouched nested entries. Dependency preparation applies it
on the pinned macOS checkout. `MLXSparseQuantizationTests` covers leading and
interior omissions for nested transformer blocks and direct module arrays.
This does not add compressed-tensors parsing or FP8 KV-cache support.

`mlx-affine-q4-qmv-tail.patch` and `mlx-swift-affine-q4-qmv-tail-jit.patch`
add opt-in full-tile plus tail handling for affine Q4/G64 FP16/BF16 matrix-vector
products. `MLX_METAL_AFFINE_Q4_QMV_TAIL=1` admits unaligned input widths divisible
by 64 with at least 512 columns and output widths divisible by eight. This covers
Gemma widths 640, 704, 2816 and 5376 without padding or changing checkpoint values.
The default remains the prior dispatch. `Scripts/affine-q4-qmv-tail-patch.sh`
owns the earlier specialization patches and the overlay together, preflighting
both pinned source and generated shader copies before applying either. See
`Tests/Shell/MLXAffineQ4QMVTailPatchTests.sh` for replay, bounds and scalar-oracle
checks. A passing CPU oracle alone does not establish Metal speed or correctness.

`mlx-metal-sdpa-d512-decode.patch` and
`mlx-swift-metal-sdpa-d512-generated.patch` add FP16/BF16 D512 two-pass vector
attention kernels. `MIDNIGHT_METAL_SDPA_D512=1` enables only single-query,
batch-one, unmasked decode with GQA at most eight. Prefill, array masks, FP32,
training and broader shapes retain their existing paths. The source/generated
pair is prepared by `Scripts/metal-sdpa-d512-patch.sh`. Opt-in Metal numerical
coverage lives in `MetalSDPAD512Tests`; `forceFused` prevents that test from
silently passing through the decomposed fallback. Promotion requires measured
full-model performance as well as numerical parity.

`mlx-swift-lm-gemma3-attention-layout.patch` honors explicit Gemma 3
`layer_types` and the `_sliding_window_pattern` metadata alias. Mask and cache
selection use the actual global/sliding representatives, including models with
fewer layers than the usual attention pattern. Invalid layouts fail decoding.
`Gemma3LayoutTests` covers 270M metadata, explicit layout overrides and cached
logit parity, including Q4 fixtures. This is model support, not a measured speed
claim. The complete Gemma campaign is tracked in `Docs/GemmaMetalPerformance.md`.

`mlx-swift-lm-gemma3-compiled-tail.patch` adds the default-off
`MIDNIGHT_GEMMA3_COMPILED_TAIL=1` experiment for Metal BF16 batch-one cached
Gemma 3 decode with stock affine Q4/G64 MLP projections. It compiles the
post-attention norm/residual/MLP sequence while leaving attention and KV updates
outside. Current parameters enter the graph as observed module state; child
replacement rebuilds it, training clears it, and QLoRA/custom MLP projections
fall back eagerly. `Scripts/gemma3-compiled-tail-patch.sh` follows the layout
overlay. Six opt-in `Gemma3CompiledTailTests` cover lifecycle and numerical
behavior; CPU replay and formatting checks pass, with Metal validation pending.
The quality runner's `gemma3-compiled` candidate requires positive per-layer
trace counts and a zero-trace baseline. See
`Docs/Gemma3CompiledTailExperiment.md`; this is not a measured speed claim.

`mlx-metal-sdpa-d256-mask-bounds.patch` and its generated Swift shader companion
add an opt-in NAX path for BF16 D256 masked prefill: batch one, 512 queries,
512–1536 keys and GQA up to eight. `MIDNIGHT_METAL_SDPA_D256_MASKED=1` selects
it; `MIDNIGHT_METAL_SDPA_D256_PRUNE=0` keeps the same NAX math while disabling
tile pruning for attribution. Bounds derive from the actual GPU mask. Completely
masked rows conservatively retain all real keys; the experimental path excludes
padded key slots from the denominator. The existing attention helper now owns
the D512 and D256 overlays atomically. See `MetalSDPAD256MaskTests` and
`Docs/MetalSDPAD256MaskedPrefillExperiment.md`; GPU validation is required.

`mlx-swift-lm-gemma4-bounded-window-cache.patch` introduces a chronological
`WindowedKVCache` with absolute positions and independent copy/restore. It
presents the union required for each prefill chunk. When concatenating two
nonempty inputs produces an owning allocation that fits the window, the cache
reuses that allocation through an independent array wrapper. Initial borrowed
views and cropped tails still materialize the retained tail. It retains
O(window) copies when eviction requires them; paged or ring storage remains
separate work. `MIDNIGHT_GEMMA4_BOUNDED_KV=1` enables it only
for Metal target-only generation with uncompressed KV; capacity/quantization
policies retain the old cache. The memory estimator receives the same selection
as model construction. Trimming after eviction is refused; rollback must restore
a snapshot. `Scripts/gemma4-window-cache-patch.sh` owns the nine overlapping
Gemma 4 model patches as one preflighted stack. `GemmaBoundedCacheTests` covers
absolute positions, chunk/window boundaries, branching, serialization, Q4 and
KV sharing. Stage 3's A4B marker recheck preserves exact output and saves
408.47 MB (2.63%), with TTFT 1096.62 → 1105.61 ms. Four generated tokens cannot
establish sustained decode speed. A separate 3837-position quality probe changes
117 top predictions after the window boundary. The reference NLL change has a
95% interval spanning zero, so quality improvement or equivalence is not
established. This storage path remains opt-in; see the
[campaign evidence](../benchmark-results/gemma-performance-20260929/README.md).

`mlx-swift-lm-mtp-stateless-adaptation.patch` adds the explicit
`MTPStatelessLogitProcessor` contract to the adaptive-draft experiment. Only
processors with no prompt/emitted-token state may adopt it; this does not imply
that processing a batch is equivalent to processing each row. Midnight's pad
sanitizer adopts the marker. Initialization and per-round eligibility checks
permit that marker while retaining sequential processing and the existing
readback boundary. Unknown processors, forced-prefix/repetition state and all
chains retain fixed widths. `Scripts/mtp-adaptive-drafts-patch.sh` owns all five
overlapping MTP overlays. `MTPAdaptiveIteratorTests` exercises the actual pad
processor, mixed finite/nonfinite rows, changing widths and excluded processors.
The stage 3 benchmark predates this repair and bypassed adaptive selection;
its results are not evidence for the repaired policy.

`swift-transformers-added-token-regex.patch` removes per-alternative regex
capture groups only when every added token has `lstrip=false` and
`rstrip=false` and nonempty content. In that case the existing splitter's full match is identical
to the former captured literal. Any stripping token preserves the original
capture-aware pattern for the entire table. Empty tokens also preserve captures:
`()`, unlike an empty pattern, compiles and keeps zero-width token boundaries.
Escaping, ordering, matching and
the generic splitter are unchanged. `Scripts/added-token-regex-patch.sh`
validates the pinned revision and supports repeat preparation alongside the
incremental ByteLevel decoder overlay. Its checked migration repairs the initial
stage7 variant without accepting a partially modified old fast path.
`AddedTokenRegexTests` exercises public
tokenizer segmentation and IDs; `AddedTokenRegexPatchTests.sh` checks replay,
idempotence and conflicts. A Foundation-only CPU measurement using Gemma 3's
6,415-token table reduced warm splitting of the 199-byte single-user fixture
from median 112.416 ms to 0.099 ms with identical segments. This excludes Jinja,
BPE and model execution; whole-request latency requires separate validation.
See `benchmark-results/gemma-performance-20260929/added-token-regex/README.md`.

For numerical GPU tests, use `Scripts/test-metal.sh --filter TEST_NAME` with
the desired experiment flags. This rebuilds the library and installs it next to
each test executable. Bare `swift test --skip-build` can otherwise load a stale
colocated test-bundle `mlx.metallib`, even when the serving binary has a fresh one.

## Gemma nested marker-delimited values

`mlx-swift-lm-gemma-nested-marker-values.patch` extends the pinned
`BareKeyJSONParser` only when Gemma supplies its escape marker. Complete marked
strings inside objects/arrays become escaped JSON strings; the structural scanner
treats their punctuation as opaque. Other formats retain the default parser.
Whole marked object-shaped strings stay strings and invalid bare values are not
guessed. This repairs the nested artifact-filter failure observed by Afterglow's
small model suite. `GemmaNestedToolCallingTests` covers the retained payload,
schema and no-schema parsing, punctuation/quotes/backslashes, empty strings and
malformed spans. The patch applies after Laguna/reasoning changes and is verified
against the pinned original source with an exact forward/reverse round trip.
