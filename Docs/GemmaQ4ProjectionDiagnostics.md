# Real-checkpoint Q4 diagnostics

No GPU execution or build has been run by this lane. The two opt-in suites are installed in `Tests/ModelRunnerProtocolTests` for the coordinated test build. They do not modify backend arithmetic or production capture behavior.

## 270M first-divergence capture

Run Gemma270MProjectionCaptureTests twice in separate processes with MLX_METAL_AFFINE_Q4_QMV_TAIL=0 and 1. Set MIDNIGHT_CAPTURE_GEMMA270_PROJECTIONS=1, MIDNIGHT_Q4_REAL_CHECKPOINT to artifacts/gemma3-270m-it-4bit, MIDNIGHT_Q4_REAL_TOKEN_IDS to `benchmark-results/gemma-performance-20260929/quality/projection-diagnostics/gemma270-prefix-16.json`, and MIDNIGHT_Q4_REAL_CAPTURE_DIR to a new unique directory for each arm.

The first 128 CPU-tokenized IDs match the actual pilot's fnv1a64:a7b01d0b51b5c406 fingerprint. The saved first 16 IDs include its first argmax divergence. See `benchmark-results/gemma-performance-20260929/quality/projection-diagnostics/prefix-provenance.json`. Captures retain each original 640-wide quantized linear input/output without changing weights. Original graph materialization happens once per token before lossless BF16 safetensor writes. Disk writes are diagnostic overhead; no speed conclusions apply.

Run `Scripts/compare-gemma-projection-captures.py BASELINE_DIR CANDIDATE_DIR NEW_JSON` using a Python with NumPy. It reads only CPU data and reports the earliest layer/projection output divergence, whether inputs were exactly equal, and all subsequent amplification. No MLX import or GPU execution is used by the analyzer.

## Real A4B expert projection regression

Run RealGemmaExpertQ4ProjectionTests in two fresh processes. Set MIDNIGHT_RUN_REAL_GEMMA_Q4_PROJECTIONS=1, MIDNIGHT_Q4_REAL_CHECKPOINT=/Users/stephen/.midnight/models/gemma-4-26B-A4B-it-midnight, MIDNIGHT_Q4_REAL_REPORT to a new JSON per arm, and the tail flag to 0/1. It loads original layer0 gate/up/down expert weights and BF16 scale/bias tensors. No re-quantization is performed.

The exact implicit lhs-index paths are covered: gate/up x=[B,L,1,1,2816], indices=[B,L,8]; down x=[B,L,8,1,704]. Cases include B=L=1 and B=2,L=3, unscaled activations, magnitude32, cancellation pairs, large outliers, and tail-only activation. Only eight selected experts are restored for the independent FP32 reference. The reference casts stored affine metadata to FP32 before restoration. Reports include complete outputs and per-case error to compare across processes. These controlled activation cases supplement actual activation capture, not model-quality proof.

The analyzer has CPU-only fixtures in `Tests/Python/test_compare_gemma_projection_captures.py`; run with a Python environment providing NumPy. Swift formatting and parsing, source-level indexing/generated-source checks, and the analyzer fixtures pass. Projection GPU diagnostics remain pending. No tail arithmetic changes are included.
