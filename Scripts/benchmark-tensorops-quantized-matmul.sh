#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURE="$ROOT/Tests/Fixtures/TensorOpsQuantizedMatmul"
BUILD="$ROOT/.build/tensorops-quantized-matmul"

if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: $0 OUTPUT_DIRECTORY [--benchmark]"
    echo "Builds a native Metal probe. The output directory must not exist."
    echo "Correctness fixtures run first; --benchmark adds paired GPU timings."
    exit 0
fi
if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && "$2" != "--benchmark" ) ]]; then
    echo "Usage: $0 OUTPUT_DIRECTORY [--benchmark]" >&2
    exit 2
fi
OUTPUT="$1"
if [[ -e "$OUTPUT" ]]; then
    echo "Refusing to overwrite existing output: $OUTPUT" >&2
    exit 2
fi
mkdir -p "$BUILD/module-cache" "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"

xcrun swiftc -O -parse-as-library -module-cache-path "$BUILD/module-cache" \
    "$FIXTURE/TensorOpsQuantizedMatmulProbe.swift" -o "$BUILD/tensorops-quantized-matmul"
xcodebuild -version > "$OUTPUT/toolchain.txt" 2>&1
xcrun --sdk macosx --show-sdk-version >> "$OUTPUT/toolchain.txt"
shasum -a 256 "$FIXTURE/affine-q4.metal" "$FIXTURE/mxfp4-direct.metal" \
    "$FIXTURE/TensorOpsQuantizedMatmulProbe.swift" "$ROOT/Package.resolved" \
    "$BUILD/tensorops-quantized-matmul" > "$OUTPUT/sha256.txt"

ARGUMENTS=("$FIXTURE" "$OUTPUT")
if [[ "${2:-}" == "--benchmark" ]]; then
    ARGUMENTS+=(--benchmark)
fi
"$BUILD/tensorops-quantized-matmul" "${ARGUMENTS[@]}" 2>&1 | tee "$OUTPUT/run.log"
