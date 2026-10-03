# Opt-in Metal command-buffer timing

The dependency patch is integrated and disabled by default. Eight CPU Python
checks pass, and the preserved stage 3 runtime has completed a controlled A4B
off/on ABBA comparison with exact output preservation. The short screen shows
little measured timing change; it does not establish a general overhead bound
or explain earlier throughput variance.

## What changes

`Patches/mlx-metal-command-timing.patch` adds a private `command_timing.h` and
three gated call sites in MLX's `device.cpp`: queue creation, buffer commit, and
the existing `CommandEncoder::synchronize` wait. It makes no CommandEncoder
layout or public API change. The native completion/error/event handler is
byte-identical to the pre-patch source.

When disabled, each hook reads a cached opt-in gate. No records, counters, extra
completion handler, clock reads, or output are created. When enabled, storage is
allocated once, bounded by the configured sample count. The extra completion
handler reads timestamps/status and publishes scalars; it performs no allocation,
I/O, lock acquisition, event signaling, or GPU synchronization. No ordering
between Metal completion handlers is assumed. Adding an observer can affect
handler scheduling and existing wait duration, which must be measured.

Submission and callback threads write distinct scalar fields. Separate release
flags publish CPU-return and GPU-completion observations. The exit snapshot only
copies a row after both acquire loads succeed, including the case where the
callback runs before `commit()` returns. Storage is deliberately process-lived
so an asynchronous callback cannot access a destroyed collector. Initialization
allocation failure disables telemetry and preserves the normal path.

The existing wait is executed once through either the ordinary or instrumented
branch. No additional wait is introduced. The process-exit snapshot neither
waits for outstanding callbacks nor drains GPU queues; pending and overflowed
observations are reported. Output is a new mode-0600 file created with `O_EXCL`,
so it cannot replace prior evidence.

## Integration

The exact inner MLX revision is
`1f8e74e3f12f31365464a6867c6579f0e9b29d85`. The helper verifies that pin, copies the
two target files, peels/replays the exact patch, and applies a single combined
diff only after the complete preflight succeeds. Partial/drifted patches fail
without changing dependency files. Unrelated edits are preserved.

```sh
# Dependency preparation also invokes this pinned, idempotent helper.
python3 Scripts/metal-command-timing-patch.py \
  --checkout .build/checkouts/mlx-swift/Source/Cmlx/mlx
```

For the normal preparation script, source `Scripts/metal-command-timing-patch.sh`
and call `model_runner_prepare_metal_command_timing "$HOST_OS" "$ROOT_DIR"
"$MLX_CHECKOUT"`, where the third argument is the inner `Source/Cmlx/mlx`
repository. The hook is Darwin-only; runtime telemetry stays disabled by default.

## Runtime controls

| Variable | Meaning |
|---|---|
| `MIDNIGHT_METAL_COMMAND_TIMING=1` | Enable before process startup; other/unset values disable |
| `MIDNIGHT_METAL_COMMAND_TIMING_OUTPUT` | Required fresh output path; absent path disables telemetry |
| `MIDNIGHT_METAL_COMMAND_TIMING_LIMIT` | Retain at most this many command records and this many wait records; default 8192, maximum 65536, minimum 1 |
| `MIDNIGHT_METAL_COMMAND_TIMING_SKIP` | Skip this many initial command buffers before retaining a contiguous sample; default 0, maximum one billion |

Queue creation metadata retains the first 128 queue generations. Reused Metal
queue addresses are separated by creation time. Missing queue metadata or an
incomplete buffer record disables adjacent-queue gap analysis. Wait sampling has
its own bounded initial range and does not inherit the command skip count.

```sh
# Example only: use the existing controlled runtime command after rebuilding.
MIDNIGHT_METAL_COMMAND_TIMING=1 \
MIDNIGHT_METAL_COMMAND_TIMING_OUTPUT=/absolute/new/path/metal-timing.json \
  /path/to/rebuilt/model-runner-runtime-bench ...

python3 Scripts/analyze-metal-command-timing.py \
  /absolute/new/path/metal-timing.json /absolute/new/path/metal-timing-analysis.json
```

The analyzer can select `--sequence-from N --sequence-to M`. It retains the raw
report hash and separates commit-call duration, commit-to-GPU delay, GPU interval,
callback arrival delay, and the existing wait duration. Queue-local comparisons
show time between CPU commit calls, gaps/overlaps between GPU intervals, and
whether the next commit began after the prior observed GPU interval ended.
These are observations, not attribution to a specific bottleneck.

## Validation before interpretation

```sh
# Already passed: copied-tree replay, semantic preservation and synthetic analysis.
python3 -m unittest discover -s Tests/Python -p test_metal_command_timing.py -v

# CPU-only C++ publication/bounds fixture; no Metal APIs or model weights.
clang++ -std=c++20 -pthread \
  -I .build/checkouts/mlx-swift/Source/Cmlx/mlx \
  Tests/Fixtures/MetalCommandTiming/collector_fixture.cpp \
  -o /private/tmp/midnight-command-timing-collector-fixture
/private/tmp/midnight-command-timing-collector-fixture /private/tmp/new-timing-fixture.json
```

The fixture should emit submitted=8, skip=1, capacity=5, complete=3,
incomplete=2, capacity overflow=2, and one existing-wait record. Its numbers are
synthetic. For runtime validation, alternate telemetry off/on with identical
executable, weights, prompt IDs and fixed kernel flags; compare native
timing/output metrics before using instrumented observations to explain run
variance. Run no concurrent GPU work during that comparison. An overhead
measurement is specific to its workload and sampling configuration.

## Measured A4B validation

The [stage 3 A4B ABBA screen](../benchmark-results/gemma-performance-20260929/stage3-a4b-telemetry-abba/summary.json)
uses the same preserved executable and Metal library in four fresh processes,
with collection off/on/on/off. Each process has one warmup and two measured
256-token greedy trials; prefix caching and experimental kernels are disabled.
Outputs, prompt fingerprints, token counts, stop reasons and artifact identities
match. Median decode is 135.081 tok/s off and 135.169 on (+0.0653%); TTFT is
69.994 ms off and 70.079 ms on (+0.1226%). The differences are small in this
screen, without establishing a general overhead ceiling.

The enabled [first trace analysis](../benchmark-results/gemma-performance-20260929/stage3-a4b-telemetry-abba/01-B-metal-analysis.json)
and [second trace analysis](../benchmark-results/gemma-performance-20260929/stage3-a4b-telemetry-abba/02-B-metal-analysis.json)
each contain 65,536 valid completed GPU samples, zero incomplete/invalid
samples, complete queue identities and no negative clock ordering. The bounded
collector omits 3170 of 68,706 and 3067 of 68,603 submissions after capacity is
reached. Samples have no automatic phase labels and include loading, warmup and
generation work. Further workload and phase-specific measurements remain
necessary before attributing throughput differences to submission behavior.

## Clock and interpretation limits

CPU samples convert `mach_absolute_time()` with the cached timebase. Apple's
[GPU timestamp documentation](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpuendtime)
places GPU start/end values in system mach time and requires reading them after
completion. The observer follows that requirement through
[addCompletedHandler](https://developer.apple.com/documentation/metal/mtlcommandbuffer/addcompletedhandler(_:)).
The existing
[waitUntilCompleted](https://developer.apple.com/documentation/metal/mtlcommandbuffer/waituntilcompleted())
includes completion handlers; observer overhead can therefore affect it.

A GPU interval can contain event waits, dependencies, and stalls. It is not
shader busy time. Other queues/processes can execute during a queue-local gap,
so this does not measure total GPU utilization. CPU time between commits includes
encoding, runtime/model work, allocation and existing waits. Callback latency is
observer arrival, with no assumed relation to the other completion handlers.
Only this existing synchronize wait is covered; other event or condition waits
are not. Referenced bytes are MLX bookkeeping, not physical memory traffic.

The bounded sample is not randomized or automatically divided into loading,
prefill and decode phases. The exit snapshot can run before dependency
destructors; missing records are explicit. Signals/crashes that bypass normal
exit may produce no report. The Python replay tests read the exact pinned Git object, so they work before
or after the live overlay is applied. Initial staging provenance remains under
`artifacts/metal-command-timing-stage-20260929/`.

The patched `device.cpp` retains its upstream Apple copyright. The new private
collector header is Copyright © 2026 Midnight contributors. MLX is MIT-licensed;
see `THIRD_PARTY_NOTICES.md` and `Patches/README.md` for dependency attribution.
