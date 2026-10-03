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
