# Performance regression checks

Midnight's performance gate compares two explicit local runner builds on the same model and computer. Use it before accepting a change to generation, prompt caching, or optional runtime work. It runs only when requested; it adds no telemetry or work to normal inference.

## Run a comparison

Create a manifest for the installed baseline and the candidate. See [Scripts/Performance](../Scripts/Performance) for the command, examples, and synthetic tests. Both commands must use absolute executable paths and the harness's `{config}`, `{model}`, `{served_model}`, and `{port}` placeholders. Declare runtime resources, dependency lockfiles, and other files that identify the builds with `identity_paths`.

```sh
python3 Scripts/Performance/compare.py run \
  --manifest /absolute/path/comparison.json \
  --out /absolute/path/new-result-directory

python3 Scripts/Performance/compare.py analyze \
  /absolute/path/new-result-directory
```

The result directory must be new. The harness creates its own loopback servers and configuration files and stops its own processes. It hashes model files, executables, and declared resources before and after the run. Preserve the result directory: it contains the manifest, launch settings, raw samples, warmups, output checks, runtime snapshots, logs, and analysis.

Use identical weights, quantization, context, prefill settings, output limits, and requests. An environment or configuration change under test must be explicit in the manifest. Verify the resolved runtime settings in the result, rather than assuming the launch arguments prove they took effect.

## What is measured

The suite separates short requests, fresh long prompts, exact repeats, shared instructions with a changed question, long output, multi-turn requests, cancellation followed by another request, process/model startup, and sampled resident memory. These are different workloads; combining them into one throughput average hides important regressions.

Alternating baseline/candidate order and independent process pairs reduce ordering bias. Warmups and cache setup are preserved but excluded from measurement rows. Controlled pairs require matching inputs, visible output, reasoning/tool fields, finish information, and token counts. Expected-answer checks provide a separate bounded correctness check. Temperature zero alone is not proof of identical work.

First-text latency is measured when nonempty content arrives. The visible decode-rate estimate uses streamed content boundaries and reported token counts, so it is advisory; it does not measure individual device token times. Request completion runs through the terminal event and response close; it does not include hidden server work after that event. The cancellation check separately measures whether another request can complete. A continued Responses conversation must actually use the returned response ID.

## Read the result

The default practical regression tolerance is 3%, recorded in the manifest. A 95% paired interval must fit within that tolerance before the gate calls a measurement a non-regression. A claimed improvement needs an interval below zero. The analysis requires at least four independent process pairs; more are useful for noisy small models. A minimum sample count is not a guarantee of statistical precision.

Missing or failed samples, changed identities, output mismatches, failed quality checks, and substantial timing drift cannot produce a passing result. An inconclusive run is not evidence that a change is free. A passing small-model run does not establish performance for every model, device, or context size.

Stop builds, profilers, and owned competing model jobs before measuring. Do not stop unrelated user processes. If other activity or thermal drift interferes, retain and label that run, then arrange a separate controlled run. Traces diagnose bottlenecks; traced timings do not establish production speed.

## Additional release evidence

Use an actual Lowlight conversation alongside controlled API measurements. Changes inside the runner should benefit both Lowlight clients through the existing API. Natural conversations may diverge after generated answers are carried forward; report those as observations instead of pretending their timings compare identical inputs.

Process startup does not clear operating-system file caches. Sampled RSS excludes separate GPU allocations and may miss brief peaks. Inspect the retained runtime memory counters and, where needed, collect a separate device-memory measurement. Model switching, active voice/vision, broad model-quality testing, and optional-feature absence comparisons remain additional gates required by the [PRD](product-requirements.md); a text-only harness run does not stand in for them.

Current work and deployment decisions are recorded in the [P0 evidence report](../benchmark-results/p0-performance-20260911/README.md).
