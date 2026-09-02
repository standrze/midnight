#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
QUANTIZER="$PACKAGE_ROOT/Sources/LagunaQuantizer/main.swift"
AUDITOR="$PACKAGE_ROOT/Sources/Q4ScaleSearchAudit/main.swift"
RESCORER="$PACKAGE_ROOT/Sources/LagunaScaleSearchRescorer/LagunaScaleSearchRescorer.swift"
VERIFIER="$PACKAGE_ROOT/Sources/LagunaQ4R8Verifier/main.swift"
LAGUNA_MODEL="$PACKAGE_ROOT/Sources/ModelRunnerCore/LagunaModel.swift"
PACKER="$PACKAGE_ROOT/Scripts/pack-laguna-gate-up.py"
CUDA_AB_BENCHMARK="$PACKAGE_ROOT/Scripts/benchmark-cuda-laguna-ab.sh"
RUNTIME_BENCHMARK="$PACKAGE_ROOT/Sources/RuntimeBenchmark/main.swift"

grep -Fq 'name: "model-runner-laguna-quantize"' "$PACKAGE_ROOT/Package.swift"
grep -Fq 'name: "model-runner-scale-plan"' "$PACKAGE_ROOT/Package.swift"
grep -Fq 'name: "model-runner-q4-scale-search-audit"' "$PACKAGE_ROOT/Package.swift"
grep -Fq 'name: "model-runner-laguna-q4r8-rescore"' "$PACKAGE_ROOT/Package.swift"
grep -Fq 'name: "model-runner-laguna-q4r8-verify"' "$PACKAGE_ROOT/Package.swift"
grep -Fq 'bits: 4' "$QUANTIZER"
grep -Fq 'bits: 8' "$QUANTIZER"
grep -Fq 'var cpu = false' "$QUANTIZER"
grep -Fq 'if dryRun || cpu' "$QUANTIZER"
grep -Fq 'groupSize: 64' "$QUANTIZER"
grep -Fq 'var q4ScaleSearch = false' "$QUANTIZER"
grep -Fq 'calibration: q4Calibration' "$QUANTIZER"
grep -Fq 'ScalePlanBundle.self' "$QUANTIZER"
grep -Fq '.q4AffineScaleSearch' "$QUANTIZER"
grep -Fq 'q4AffineScaleSearchQuantized' "$AUDITOR"
grep -Fq 'passesFifteenPercentMSEGate' "$AUDITOR"
grep -Fq 'standardBytes == searchedBytes' "$AUDITOR"
grep -Fq 'q4AffineScaleSearchQuantized' "$RESCORER"
grep -Fq 'templateModel' "$RESCORER"
grep -Fq 'q8ModulesPreserved' "$RESCORER"
grep -Fq 'destination already exists' "$RESCORER"
grep -Fq 'Template identity checks passed' "$RESCORER"
grep -Fq 'Every searched Q4 module differs from the standard template' "$VERIFIER"
grep -Fq 'All preserved tensors are byte-for-byte identical' "$VERIFIER"
grep -Fq 'path.hasSuffix(".mlp.gate.proj")' "$QUANTIZER"
grep -Fq 'with: ".switch_mlp.gate_up_proj"' "$QUANTIZER"
grep -Fq 'with: ".shared_expert.gate_up_proj"' "$QUANTIZER"
grep -Fq 'path.hasSuffix(".mlp.gate_proj")' "$QUANTIZER"
grep -Fq "Poolside's BF16 checkpoint stores one tensor per expert" "$LAGUNA_MODEL"
grep -Fq 'Fusing both routed and always-active MLPs' "$LAGUNA_MODEL"
grep -Fq 'rewrite_fused_quantization_config(staging, layers)' "$PACKER"
grep -Fq 'gate/up quantization differs and cannot be fused' "$PACKER"
grep -Fq 'q4AffineScaleSearchQuantized' \
  "$PACKAGE_ROOT/Patches/mlx-swift-lm-q4-affine-scale-search.patch"
grep -Fq 'q4AffineScaleSearchFactors' \
  "$PACKAGE_ROOT/Patches/mlx-swift-lm-q4-affine-centered-scale-search.patch"
grep -Fq 'least-squares affine bias' \
  "$PACKAGE_ROOT/Patches/mlx-swift-lm-q4-affine-bias-refinement.patch"
grep -Fq 'let secondCodeValues' \
  "$PACKAGE_ROOT/Patches/mlx-swift-lm-q4-affine-joint-fit.patch"
grep -Fq 'mlx-swift-lm Q4 affine scale search' "$PACKAGE_ROOT/prepare-dependencies.sh"
grep -Fq 'mlx-swift-lm Q4 affine bias refinement' "$PACKAGE_ROOT/prepare-dependencies.sh"
grep -Fq 'mlx-swift-lm Q4 affine joint fit' "$PACKAGE_ROOT/prepare-dependencies.sh"
grep -Fq 'model-runner-q4-scale-search-audit' \
  "$PACKAGE_ROOT/Scripts/audit-laguna-q4-scale-search.sh"
grep -Fq 'model-runner-laguna-q4r8-rescore' \
  "$PACKAGE_ROOT/Scripts/rescore-laguna-q4r8.sh"
grep -Fq 'model-runner-laguna-q4r8-verify' \
  "$PACKAGE_ROOT/Scripts/verify-laguna-q4r8.sh"
grep -Fq 'one 256-token warm-up then matched greedy trials' "$CUDA_AB_BENCHMARK"
grep -Fq 'tokens_per_second=' "$CUDA_AB_BENCHMARK"
grep -Fq 'MODEL_RUNNER_MLX_MEMORY_LIMIT_GIB=20' "$CUDA_AB_BENCHMARK"
grep -Fq 'name: "model-runner-runtime-bench"' "$PACKAGE_ROOT/Package.swift"
grep -Fq 'LocalModelRunner(' "$RUNTIME_BENCHMARK"
grep -Fq 'medianDecodeTokensPerSecond' "$RUNTIME_BENCHMARK"
grep -Fq 'model-runner-runtime-bench' \
  "$PACKAGE_ROOT/Scripts/benchmark-runtime-model.sh"
grep -Fq '"$PACKAGE_ROOT/build.sh"' \
  "$PACKAGE_ROOT/Scripts/quantize-laguna-q4r8.sh"

PACKER_PYCACHE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/laguna-packer-pycache.XXXXXX")"
cleanup() {
  rm -rf "$PACKER_PYCACHE_ROOT"
}
trap cleanup EXIT
PYTHONPYCACHEPREFIX="$PACKER_PYCACHE_ROOT" python3 -m py_compile "$PACKER"

echo "Laguna Q4R8 quantizer checks passed"
