// Copyright (c) 2026 Midnight contributors.
// Experimental Laguna affine-Q4 expert GEMV; no tensor-core padded M rows.
#pragma once

#include <cutlass/array.h>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace mlx::core::cu {

// Exact-shape caller contract: K is 512 or 2048, N is a multiple of four,
// eight contiguous index pairs, and packed rows have Q4/group-64 metadata.
// Dequantization deliberately rounds multiply and add separately in T, as
// cute_vectorized_dequant does. Products/partial sums then remain FP32.
// The reduction order differs from SM80 QMM; bitwise equivalence is a test
// gate, not an assumed property of equal accumulator precision.
template <typename T, int K>
__global__ void midnight_expert_qmv_fp32_kernel(
    const T* x,
    const uint32_t* weights,
    const T* scales,
    const T* biases,
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    T* output,
    int n) {
  constexpr int values_per_lane = 8;
  constexpr int rows_per_block = 4;
  const int lane = threadIdx.x;
  const int row = blockIdx.x * rows_per_block + threadIdx.y;
  const int selection = blockIdx.y;
  const int64_t weight_row = int64_t(rhs_indices[selection]) * n + row;
  const T* input = x + int64_t(lhs_indices[selection]) * K;
  const uint32_t* packed = weights + weight_row * (K / 8);
  const T* row_scales = scales + weight_row * (K / 64);
  const T* row_biases = biases + weight_row * (K / 64);
  float partial[values_per_lane] = {};

#pragma unroll
  for (int base = lane * values_per_lane; base < K; base += 32 * values_per_lane) {
    const uint32_t word = packed[base / 8];
    const auto inputs = *reinterpret_cast<const cutlass::Array<T, 8>*>(input + base);
    cutlass::Array<T, 8> codes;
#pragma unroll
    for (int i = 0; i < values_per_lane; ++i) {
      codes[i] = T((word >> (4 * i)) & 15);
    }
    auto dequantized = codes * row_scales[base / 64];
    dequantized = dequantized + row_biases[base / 64];
#pragma unroll
    for (int i = 0; i < values_per_lane; ++i) {
      partial[i] = fmaf(float(inputs[i]), float(dequantized[i]), partial[i]);
    }
  }

  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < values_per_lane; ++i) {
    sum += partial[i];
  }
#pragma unroll
  for (int distance = 16; distance > 0; distance /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, distance);
  }
  if (lane == 0) {
    output[selection * n + row] = T(sum);
  }
}

} // namespace mlx::core::cu
