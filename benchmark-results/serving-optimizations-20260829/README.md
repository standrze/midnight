# Serving optimization validation — 2026-08-29

These reports are the first on-hardware checks after implementing measured
wired residency, completed-message prefix snapshots, and the DFlash verifier
and passthrough scheduling fixes. They are deployment gates, not broad model
benchmarks.

| Probe | Result | Decision |
| --- | --- | --- |
| DFlash block 3, 3x128 tokens per mode, alternated | target 145.58 tok/s; DFlash 157.36 tok/s; +8.09%; 201/357 drafts accepted (56.3%) | Throughput direction is positive, but output diverged at UTF-8 byte 109; keep opt-in |
| DFlash block 2, 1x64 tokens per mode | target 150.03 tok/s; DFlash 138.83 tok/s; -7.47% | Same byte-109 divergence rules out the new multi-row argmax width as its cause |
| DFlash first-rejection diagnostic, block 3 | first normal rejection was round 1, generated index 1, top-two target margin 1.125 | Diagnostic works; this record is a draft/verifier rejection, not the later A/B divergence |
| Prefix LRU sibling branch, warmed 3x32-token trials | 111/140 prompt tokens reused; cached median TTFT 86.16 ms; cold 180.95 ms; 52.39% reduction | Snapshot restore works across branches; generated continuation was not bit-identical |

The DFlash block-3 warm-up ran at 101.29 tok/s, so three measured trials are
not enough to declare the historical outlier problem solved. The paired
reports preserve every trial, generated text, acceptance count, TTFT, and the
first output-divergence byte.

An intentionally unwarmed sibling-branch probe recorded a 677.33 ms first LRU
restore versus 198.97 ms cold. After one complete warm-up probe, all three
measured cached arms were faster than their paired cold controls. The context
was only 140 tokens and the run showed large thermal drift, so the positive
median is mechanism validation rather than a general latency forecast.
