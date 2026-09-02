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
