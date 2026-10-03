// UNCOMPILED macOS 27 scaffold, not a validated kernel or a dispatch candidate.
// MLX weights remain [N, K], with K contiguous.
// Each E8M0 scale applies to 32 consecutive K elements, exactly as in MLX MXFP4.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

using Scales = tensor_blockwise<tensor_plane_scales, const device metal_fp8_ue8m0_format, 32, 1>;
using Weights = tensor<const device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline, Scales>;

kernel void mxfp4_direct(
    const device half* input [[buffer(0)]],
    const device uchar* packed [[buffer(1)]],
    const device uchar* scales [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint3& shape [[buffer(4)]],
    uint2 tile [[threadgroup_position_in_grid]]) {
    // shape = {M, N, K}. Strides are expressed in logical elements, not bytes.
    tensor<const device half, dextents<int, 2>, tensor_inline> x(
        input, dextents<int, 2>(shape.z, shape.x), array<int, 2>{1, int(shape.z)});
    Weights w(packed, dextents<int, 2>(shape.z, shape.y), array<int, 2>{1, int(shape.z)}, Scales(scales));
    tensor<device half, dextents<int, 2>, tensor_inline> y(
        output, dextents<int, 2>(shape.y, shape.x), array<int, 2>{1, int(shape.y)});
    constexpr auto descriptor = matmul2d_descriptor(16, 32, dynamic_length_v<int>, false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    op.run(x.slice(0, tile.y * 16), w.slice(0, tile.x * 32), y.slice(tile.x * 32, tile.y * 16));
}
