# Direction: remove Python from Midnight

The project owner set the long-term requirement on 5 September 2026: remove Python from the entire project, including training, evaluation, quantization helpers, diagnostics, and tests. Native Swift/MLX is the implementation target. This applies beyond the HTTP serving path, which already executes natively.

On 6 September 2026, training, distillation, and pruning research moved to the separate sibling Training project. The native implementation direction is retained there; Midnight owns inference and exported-model compatibility. The latent experiment remains paused, with its notes and preserved research archive managed by Training. A research result does not authorize adding Python to serving, installation, or Midnight's standard build/test workflow.

## Existing migration scope

### Completed 8 September 2026: dataset preparation

`model-runner-prepare-corpus text` and `reference` replace the Python text and
pinned quantization-corpus builders. Their replaced Python test also moved to
the sibling Backup directory after parity validation. The new CPU-only Swift
tests use frozen original outputs; the real 128-record corpus and all six
associated files reproduce their original bytes. The generated-evaluation
workflow now reads shared JSON source pins instead of importing the retired
builder. See [dataset preparation and validation](dataset-preparation.md).

### Completed 8 September 2026: external diagnostics

The entire `Diagnostics` directory moved to the sibling
`../midnight-diagnostics` folder, independent of the runner. `summarize_sse.rb` is a Ruby replacement
for the Python stream analyzer. Nine Ruby tests (26 assertions) passed, including
six exact-output comparisons with the original Python implementation and tests
of the reference launcher's argument handling and exit status. SHA-256 manifests
preserve the original scripts, Dockerfiles, requests, config, and reference patch;
the external archive includes the applicable license notices.

`transformers_reference.rb` is explicitly a Ruby **launcher** for the preserved
external Python/PyTorch/CUDA reference, not a port of that model runtime. An exact
pure-Ruby conversion is not available from the existing dependencies. No reference
model inference was run during this migration. Midnight no longer contains or
depends on these diagnostic utilities. The browser proxy remains historical in
the same external archive.

48 Python files remain: 14 in `Scripts`, 9 in `Tests/Python`, and 25 in historical
benchmark/reproduction folders. Four shell tests still invoke Python. The HTTP
integration test, benchmark/analysis tools, quantization research, and historical
benchmark archive migration remain pending. This scoped completion does not mark
the broader Python-removal goal complete.

On 8 September 2026, Studio and its worker moved to `../midnight-studio`,
and checkpoint-conversion tooling and its tests moved to
`../midnight-quantization`. These are project-boundary changes, not Python
ports; remaining interpreter-based quantization helpers now belong to that
project. See the [project layout](project-layout.md) for ownership and
verification. Runtime evaluation and corpus preparation remain in Midnight.

Midnight Studio now also owns a small native `midnight-studio-worker` for
interactive LoRA fine-tuning, adapter-based CPT, and dense shard quantization.
This worker shares Midnight's pinned model runtime. Broader distillation,
pruning, and training research remain in the sibling Training project. Studio
adds no Python training or download dependency.

The existing main checkout at `ca8d1cef8cec26c4b18c0c9f78097312249b9573` contains 52 Python files: 15 under `Scripts`, 3 under `Diagnostics`, 9 under `Tests/Python`, and 25 historical/reproduction files under `benchmark-results`. This count excludes the external latent experiment. Three shell test files additionally invoke Python directly, including inline fixture generation/inspection. Literal historical references to Python are distinct from interpreter dependencies.

| Priority | Area | Native replacement and acceptance check |
|---|---|---|
| 1 | Corpus preparation, JSONL/report validation, statistics, benchmark orchestration | Swift command-line tools with shared Foundation parsing/statistics code. Preserve split membership, hashes, metric definitions, failure behavior, and existing report compatibility. |
| 2 | HTTP integration tests and shell-test fixture helpers | Swift HTTP clients and Swift Testing fixtures. Run the same request, stopping, cancellation, and failure checks without an interpreter. Standalone reference diagnostics were moved outside Midnight; the stream analyzer is now Ruby. |
| 3 | Packing, compatibility audits, and quantization research | Extend existing Swift/MLX quantization tools. Compare packed tensors, numerical outputs, calibration scores, and model quality against preserved fixtures. |
| 4 | Exported latent-model compatibility | Training owns latent adaptation and its quality experiments. Add a native inference protocol in Midnight only after a useful exported candidate and its runtime requirements have been validated. |
| 5 | Python tests and independent reference checks | Migrate tests to Swift Testing. Preserve externally generated numerical fixtures, source/version identities, and tolerance definitions so validation remains independent of the implementation being tested. |
| 6 | Historical reproduction scripts | Archive the exact original sources outside the project with hashes and provenance, retain results as data, and supply native reproduction commands. Do not discard the audit trail while removing executable Python files. |

Python-language code-generation benchmarks need an explicit scope decision during migration: Swift can own orchestration, but executing generated Python still requires a compatible external evaluator. A fully Python-independent default should use supported non-Python evaluation paths. Preserve older benchmark results and label changed task definitions instead of implying direct equivalence.

## Completion criteria

- No Python source files or embedded Python programs remain in the project, including reproduction directories.
- Build, installation, serving, supported quantization/evaluation workflows, and the standard test suite run without a Python interpreter or Python package installation. Training workflows are owned by the separate Training project.
- Native replacements retain documented correctness, quality, performance, checkpoint, and report contracts; numeric protocol changes are versioned explicitly.
- Historical evidence remains traceable through immutable results, source hashes, and the external reference archive.

This document records direction and migration order. It does not claim that the existing Python utilities have already been replaced or require deleting working tools before their replacements are validated.
