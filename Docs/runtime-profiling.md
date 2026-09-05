# Profiling the native MLX runner

Build an optimized release before profiling. Keep model downloads, conversion,
other GPU benchmarks and builds out of the measurement interval. A profiler
changes execution; collect separate unprofiled, counterbalanced comparisons
before making a speed claim.

On macOS, start `model-runner-runtime-bench` normally with enough warmups and
trials to remain running while Instruments attaches. Use its PID with:

```sh
xcrun xctrace record --template 'Time Profiler' \
  --output /absolute/output/runner.trace --time-limit 45s --attach BENCHMARK_PID
xcrun xctrace export --input /absolute/output/runner.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
  --output /absolute/output/time-profile.xml
python3 Scripts/analyze-time-profiler.py /absolute/output/time-profile.xml \
  --output /absolute/output/cpu-summary.json
```

Replace the uppercase PID and absolute paths. macOS may require access to the
profiling services. Attaching worked on the September 4 M5 Max run; direct
Instruments launch stalled inside Metal initialization and produced only startup
samples. That failed trace cannot support a decode conclusion. The attach method
also lets the benchmark finish normally and write its metrics report.

The analyzer resolves xctrace references and weights samples rather than treating
every row as an equal duration. `--start` and `--end` select seconds relative to
the recording, with an exclusive end. Select a known decode interval when a
recording includes loading or prefill. Missing call stacks are counted separately.
Leaf shares partition symbolized sampled CPU weight; inclusive shares overlap.
Neither is GPU utilization or a fraction of wall time. Profile GPU scheduling
separately with Metal System Trace if needed.

The pinned MLX backend exposes `MLX_MAX_OPS_PER_BUFFER` and
`MLX_MAX_MB_PER_BUFFER`. These affect when it submits a command buffer and are
read during initialization. Test them in fresh processes. Larger buffers can
reduce submission overhead but retain more temporary memory or delay useful
work. Use the campaign harness's environment allowlist to remove ambient values
from the stock arm and record explicit candidate values. Do not set them globally
based on one short prompt or a microbenchmark.

```json
{
  "environment_allowlist": ["MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER"],
  "models": [
    {"label": "stock", "path": "/absolute/model"},
    {"label": "larger-buffers", "path": "/absolute/model",
     "environment": {"MLX_MAX_OPS_PER_BUFFER": "200", "MLX_MAX_MB_PER_BUFFER": "256"}}
  ]
}
```

This is an excerpt to merge into a complete `benchmark-campaign.py` manifest,
not a standalone manifest. Compare short decode, longer prefill, memory, output
identity, drift and order effects. Retain defaults when confirmation fails.

The September 4 Metal System Trace attempt recorded about 3.9 GB but did not
finish finalizing after several minutes and a graceful interrupt. The recorder
was stopped; that incomplete trace is not used for GPU occupancy or bottleneck
claims. Successful CPU exports and unprofiled runtime comparisons remain the
available evidence.
