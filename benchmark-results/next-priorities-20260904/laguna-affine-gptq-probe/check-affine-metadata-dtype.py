"""Tiny CPU-only affine dtype-dispatch check; no timing or quality claim."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
output = Path(args.output)
if output.exists():
    raise FileExistsError(output)

import mlx.core as mx
mx.set_default_device(mx.cpu)
weight = (mx.arange(64 * 128).reshape(64, 128).astype(mx.float32) / 8192 - 0.5).astype(mx.bfloat16)
x = mx.ones((1, 128), dtype=mx.bfloat16)
packed, scales, biases = mx.quantize(weight, bits=4, group_size=64)
results = []
for dtype in [mx.bfloat16, mx.float16, mx.float32]:
    s, b = scales.astype(dtype), biases.astype(dtype)
    dense = mx.quantized_matmul(x, packed, s, b, transpose=True, bits=4, group_size=64)
    gathered = mx.gather_qmm(x, mx.stack([packed, packed]), mx.stack([s, s]), mx.stack([b, b]),
                            rhs_indices=mx.array([0], dtype=mx.uint32), transpose=True, bits=4, group_size=64)
    mx.eval(dense, gathered)
    results.append({"input_dtype": str(x.dtype), "metadata_dtype": str(dtype),
                    "quantized_matmul_output_dtype": str(dense.dtype),
                    "gather_qmm_output_dtype": str(gathered.dtype),
                    "finite": bool(mx.all(mx.isfinite(dense)).item() and mx.all(mx.isfinite(gathered)).item())})
report = {"mlx_version": mx.__version__, "device": str(mx.default_device()),
          "scope": "Tiny CPU dtype-dispatch mechanism, no GPU latency measurement",
          "production_embedded_core_commit": "1f8e74e3f12f31365464a6867c6579f0e9b29d85",
          "production_source": "https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/ops.cpp#L4794",
          "results": results}
with output.open("x") as destination:
    destination.write(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
