// Runtime execution probe; weights use MLX's existing affine Q4 layout.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#if USE_BFLOAT
using Value = bfloat;
#else
using Value = half;
#endif

struct Shape {
    uint rows;
    uint columns;
    uint reduction;
    uint group_size;
};

inline Value affine_value(
    const device uint* packed,
    const device Value* scales,
    const device Value* biases,
    constant Shape& shape,
    uint row,
    uint column) {
    if (row >= shape.columns || column >= shape.reduction) {
        return Value(0);
    }
    const uint word = packed[row * (shape.reduction / 8) + column / 8];
    const uint code = (word >> (4 * (column % 8))) & 15u;
    const uint group = row * (shape.reduction / shape.group_size) + column / shape.group_size;
    // Decode on the stored grid, then round once to the operand's precision.
    return Value(float(scales[group]) * float(code) + float(biases[group]));
}

template <bool stage_weights>
inline void affine_q4_matmul(
    const device Value* input,
    const device uint* packed,
    const device Value* scales,
    const device Value* biases,
    device Value* output,
    constant Shape& shape,
    threadgroup Value* staged,
    uint2 tile,
    uint lane) {
    constexpr auto descriptor = matmul2d_descriptor(
        16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroup> op;
    auto left = op.get_left_input_cooperative_tensor<Value, Value, float>();
    auto right = op.get_right_input_cooperative_tensor<Value, Value, float>();
    auto result = op.get_destination_cooperative_tensor<decltype(left), decltype(right), float>();

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < result.get_capacity(); ++i) {
        if (result.is_valid_element(i)) {
            result[i] = 0.0f;
        }
    }

    for (uint k = 0; k < shape.reduction; k += 16) {
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < left.get_capacity(); ++i) {
            if (left.is_valid_element(i)) {
                const auto coordinate = left.get_multidimensional_index(i);
                const uint row = tile.y * 16 + coordinate[1];
                const uint column = k + coordinate[0];
                left[i] = row < shape.rows && column < shape.reduction
                    ? input[row * shape.reduction + column] : Value(0);
            }
        }
        if constexpr (stage_weights) {
            // Matched control: the same decoded values, operand precision,
            // TensorOps descriptor and K order, with an explicit memory round trip.
            for (uint element = lane; element < 32 * 16; element += 32) {
                staged[element] = affine_value(
                    packed, scales, biases, shape, tile.x * 32 + element / 16, k + element % 16);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            tensor<threadgroup Value, extents<int, 16, 32>, tensor_inline> weights(
                staged, extents<int, 16, 32>());
            right.load(weights);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            // Query the cooperative layout rather than hard-coding a lane map.
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < right.get_capacity(); ++i) {
                if (right.is_valid_element(i)) {
                    const auto coordinate = right.get_multidimensional_index(i);
                    right[i] = affine_value(
                        packed, scales, biases, shape, tile.x * 32 + coordinate[1], k + coordinate[0]);
                }
            }
        }
        op.run(left, right, result);
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < result.get_capacity(); ++i) {
        if (result.is_valid_element(i)) {
            const auto coordinate = result.get_multidimensional_index(i);
            const uint row = tile.y * 16 + coordinate[1];
            const uint column = tile.x * 32 + coordinate[0];
            if (row < shape.rows && column < shape.columns) {
                output[row * shape.columns + column] = Value(result[i]);
            }
        }
    }
}

kernel void affine_q4_staged(
    const device Value* input [[buffer(0)]],
    const device uint* packed [[buffer(1)]],
    const device Value* scales [[buffer(2)]],
    const device Value* biases [[buffer(3)]],
    device Value* output [[buffer(4)]],
    constant Shape& shape [[buffer(5)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup Value staged[32 * 16];
    affine_q4_matmul<true>(input, packed, scales, biases, output, shape, staged, tile, lane);
}

kernel void affine_q4_cooperative(
    const device Value* input [[buffer(0)]],
    const device uint* packed [[buffer(1)]],
    const device Value* scales [[buffer(2)]],
    const device Value* biases [[buffer(3)]],
    device Value* output [[buffer(4)]],
    constant Shape& shape [[buffer(5)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    // The unused pointer keeps the matched template signature; no weight staging.
    affine_q4_matmul<false>(input, packed, scales, biases, output, shape, nullptr, tile, lane);
}
