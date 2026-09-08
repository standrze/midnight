# Native dataset preparation

The two dataset builders now share the Swift executable
`model-runner-prepare-corpus`. They use Foundation and SHA-256, with no Python,
MLX inference, model weights, or GPU work. The executable's SwiftPM resource
bundle must stay beside it when relocating an installed build.

Build once, then invoke the compiled executable for repeated runs:

```sh
swift build -c release --product model-runner-prepare-corpus
```

## Plain-text and JSON text datasets

```sh
.build/release/model-runner-prepare-corpus text \
  /absolute/input.txt /absolute/corpus.jsonl \
  --samples 64 --target-characters 2500 \
  --id-prefix heldout --category language-modeling
```

JSON inputs may contain an array of objects with string `text` fields or an
object with an `instances` array. Other extensions are plain text. The tool
chunks paragraphs, selects evenly spaced samples, and reports source/output
SHA-256 and selected character counts. It preserves the former builder's
Unicode-scalar counting, whitespace, newline normalization, and chunk-boundary
behavior. It rejects insufficient input and identical source/output paths.
As before, an existing text output is overwritten after successful validation.

## Pinned code/math reference datasets

```sh
.build/release/model-runner-prepare-corpus reference \
  --output-dir /absolute/reference-corpus \
  --cache-dir /absolute/verified-source-cache --offline
```

Omit `--offline` to allow missing sources to download from their pinned revisions.
An incorrect existing cache never silently falls back to a download. The tool
checks source hashes, transformed hashes, and the combined corpus hash before
writing output. All existing output files are checked for conflicts first;
identical files are reused without rewriting, and new files are exclusively
created. The output contains two source files, two 64-record datasets, their
combined 128-record dataset, and the original-format `provenance.json`.

The shared pins live in
`Sources/CorpusPreparationCore/Resources/pinned-sources.json`. The remaining
Python generated-answer evaluator reads this data directly; it no longer imports
the retired Python reference-corpus builder. Dataset preparation does not execute
the benchmark's Python reference answers. Python-program grading remains a
separate workflow.

## Migration and validation

The old `prepare-text-benchmark-corpus.py` command maps to the `text` subcommand;
`prepare-quantization-corpus.py` maps to `reference`. Existing options retain
their names. Historical benchmark snapshots retain their original source and
commands; use these native equivalents for current runs.

Run the native regression tests with:

```sh
swift test --filter CorpusPreparationTests
bash Tests/Shell/CorpusPreparationCLITests.sh
```

Tests use frozen original-Python outputs, including Unicode/CRLF/boundary cases,
and synthetic reference inputs. They cover all six output files, source and
output hash failures, offline behavior, pinned fetch destinations, conflict
preflight, unchanged-file reuse, invalid records, and portable SHA-256 vectors.
The fixture provenance records hashes of the retired Python sources. No Python
interpreter or network is needed for these native tests.

The original sources are preserved in the sibling
`../Backup/midnight/dataset-preparation/` directory. The independent golden
fixtures remain inside `Tests/CorpusPreparationTests/Fixtures`.

### Observed validation on 8 September 2026

- All 11 native tests passed; CLI tests also passed, including execution with
  an empty command search path to exclude interpreter/external-tool dependencies.
- The real pinned source cache reproduced all six Python output files exactly,
  including the combined hash and original `provenance.json` bytes.
- A 7,275,000-byte Unicode/CRLF text input produced identical JSONL and stdout.
- The affected generated-evaluation suite passed 14 checks; three optional
  Docker checks were skipped because no evaluation image was configured.

Six counterbalanced warm-cache pairs, with fresh output locations and process
startup included, measured these median wall times for the release executable:

| Workload | Swift | Original Python |
|---|---:|---:|
| Pinned reference corpus | 25.8 ms | 43.3 ms |
| 7.3 MB text input, selecting 64 records | 60.4 ms | 58.2 ms |

These are local CPU/file-processing measurements, not universal speed claims.
No model or GPU inference was started. Builds were completed before timing.
Download performance, cold filesystem caches, and Linux execution were not
measured. The portable SHA-256 path was exercised against known vectors on macOS.
Raw pairs and output hashes are in
[dataset-preparation-validation.json](dataset-preparation-validation.json).
