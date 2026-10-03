# Memory/context release validation — 2026-09-04

Host: Apple M5 Max, 64 GiB unified memory, macOS 26.6.2.
Model: Ministral-3-14B-Instruct-2512-MLX-Q4-Standard-3cea74c.

The real HTTP smoke harness passed all four cache modes with a 1,624-token
prompt and passed none/affine8 with an 11,344-token prompt under a configured
32,768-token ceiling. Each run checked model-discovery policy fields,
non-streaming generation, a streamed cached continuation, and JSON HTTP 400
rejection of an oversized request in both transport modes.

These are functional smoke checks, not comparative performance or quality
benchmarks. Recorded request durations include endpoint overhead and must not
be interpreted as repeatable speedups. The longer test does not establish
successful generation at the full 32,768-token ceiling. Compression remains
experimental; retrieval and broader quality evaluation remain future work.

Reproduce using `Scripts/test-long-context-http.py`; pass `--context-length
32768 --repetitions 1800 --schemes none,affine8` for the longer fixture.
