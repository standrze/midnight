"""CPU-only checks of affine Q4/G64 tail contribution ordering.

These tests compare symbolic, ordered affine contributions, not real-number
sums. Reassigning channels between lanes or merging two eight-value qdots is
therefore observable even when every channel still appears exactly once.
Source checks bind the model to the kernel geometry and unchanged arithmetic
helpers. This does not prove compiler-level or Metal/BF16 bitwise equivalence.
No compiler, model loading, or GPU execution is used.
"""

from collections import Counter
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / ".build/checkouts/mlx-swift/Source/Cmlx/mlx"
HEADER = "mlx/backend/metal/kernels/quantized.h"
CORE_REVISION = "1f8e74e3f12f31365464a6867c6579f0e9b29d85"
ELIGIBLE_WIDTHS = tuple(k for k in range(576, 8193, 64) if k % 512)


def function_source(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def compact(source):
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
    return re.sub(r"\s+", "", source)


def generic_lane_chunks(width, lane):
    """Pinned generic kernel: full tiles followed by clamped safe loads."""
    result, k = [], 0
    while k < width - 256:
        start = k + lane * 8
        result.append(tuple(range(start, start + 8)))
        k += 256
    remaining = min(8, max(0, width - k - lane * 8))
    if remaining:
        start = k + lane * 8
        result.append(tuple(range(start, start + remaining)))
    return result


def candidate_lane_chunks(width, lane, values_per_lane=8):
    """Candidate: full chunks, including the separately guarded last tile."""
    tile = 32 * values_per_lane
    result, k = [], 0
    while k < width - tile:
        start = k + lane * values_per_lane
        result.append(tuple(range(start, start + values_per_lane)))
        k += tile
    if lane * values_per_lane < width - k:
        start = k + lane * values_per_lane
        result.append(tuple(range(start, start + values_per_lane)))
    return result


def affine_contribution(channels, width, row=0, scale_offset=0):
    """An ordered qdot call, retaining its input sums and packed-word terms.

    The preserved helpers establish the operations within each four-term
    group. In particular, input addition occurs with the original T operands,
    before accumulation into float; it is not replaced by a float input sum.
    Each returned node represents one scale*qdot + input_sum*bias expression.
    """
    group = row * (width // 64) + channels[0] // 64 + scale_offset
    input_sums, packed_dot_terms = [], []
    for start in range(0, len(channels), 4):
        quad = channels[start:start + 4]
        input_sums.append(("T.add.left_associative", quad))
        packed_dot_terms.append(tuple(
            (channel, 16 ** position, row * (width // 4) + channel // 4,
             15 << (4 * position))
            for position, channel in enumerate(quad)
        ))
    return (
        "float(scale*qdot + input_sum*bias)", group,
        ("float.accumulate_in_order", tuple(packed_dot_terms)),
        ("float.accumulate_in_order", tuple(input_sums)),
    )


def ordered_simd_inputs(chunks_by_lane, width, row=0, scale_offset=0):
    # simd_sum is deliberately opaque: preserving its intrinsic and these
    # lane inputs avoids inventing a CPU reduction tree for the Metal ISA.
    return ("simd_sum", tuple(
        ("float.result_plus_equals_in_order", tuple(
            affine_contribution(channels, width, row, scale_offset)
            for channels in chunks
        ))
        for chunks in chunks_by_lane
    ))


class Q4TailLaneOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header = (CORE / HEADER).read_text()
        cls.kernel = function_source(cls.header, "METAL_FUNC void qmv_fast_impl(")
        cls.generic = function_source(cls.header, "METAL_FUNC void qmv_impl(")
        cls.original = subprocess.run(
            ["git", "-C", str(CORE), "show", f"{CORE_REVISION}:{HEADER}"],
            check=True, capture_output=True, text=True,
        ).stdout

    def test_original_arithmetic_helpers_and_generic_kernel_are_unchanged(self):
        for marker in ("inline U load_vector(", "inline U qdot(",
                       "METAL_FUNC void qmv_impl("):
            with self.subTest(function=marker):
                self.assertEqual(
                    compact(function_source(self.original, marker)),
                    compact(function_source(self.header, marker)),
                )

    def test_candidate_source_uses_generic_geometry_and_operations(self):
        source = compact(self.kernel)
        # This is the source-to-model contract. The behavioral tests below
        # compare contributions independently and exercise order mutations.
        self.assertRegex(
            source,
            r"packs_per_thread=\(?(?:bits==2\|\|has_tail|has_tail\|\|bits==2)\)?\?1:2;",
        )
        self.assertIn("results_per_simdgroup==4", source)
        for statement in (
            "static_assert(!has_tail||(bits==4&&group_size==64&&results_per_simdgroup==4));",
            "num_simdgroups=2;", "pack_factor=get_pack_factor<bits,32>();",
            "values_per_thread=pack_factor*packs_per_thread;",
            "block_size=values_per_thread*SIMD_SIZE;",
            "scale_step_per_thread=group_size/values_per_thread;",
            "simd_lid*packs_per_thread*bytes_per_pack",
            "simd_lid/scale_step_per_thread",
            "simd_lid*values_per_thread",
            "ws+=block_size*bytes_per_pack/pack_factor;",
            "scales+=block_size/group_size;", "biases+=block_size/group_size;",
            "x+=block_size;", "result[row]=simd_sum(result[row]);",
            "y[row]=static_cast<T>(result[row]);",
            "full_size=has_tail?in_vec_size-block_size:in_vec_size;",
            "intk=0;for(;k<full_size;k+=block_size)",
            "remaining=in_vec_size-k;",
            "if(simd_lid*values_per_thread<remaining)",
        ):
            self.assertIn(statement, source)
        self.assertEqual(source.count("load_vector<T,U,values_per_thread,bits>(x,x_thread)"), 2)
        self.assertEqual(source.count("result[row]+=qdot<U,values_per_thread,bits>("), 2)
        self.assertEqual(source.count("qdot<U,values_per_thread,bits>(wl,x_thread,s,b,sum)"), 2)
        self.assertEqual(source.count("Us=sl[0];Ub=bl[0];"), 2)
        self.assertEqual(source.count("simd_sum("), 1)
        for forbidden in ("load_vector_safe", "qdot_safe", "clamp(", "min("):
            self.assertNotIn(forbidden, source)

    def test_every_eligible_width_preserves_ordered_affine_contributions(self):
        for width in ELIGIBLE_WIDTHS:
            with self.subTest(width=width):
                generic = [generic_lane_chunks(width, lane) for lane in range(32)]
                candidate = [candidate_lane_chunks(width, lane) for lane in range(32)]
                self.assertEqual(
                    ordered_simd_inputs(generic, width),
                    ordered_simd_inputs(candidate, width),
                )
                channels = [channel for chunks in candidate for chunk in chunks for channel in chunk]
                self.assertEqual(Counter(channels), Counter(range(width)))
                for chunks in candidate:
                    for chunk in chunks:
                        self.assertEqual(len(chunk), 8)
                        self.assertEqual(chunk[0] // 64, chunk[-1] // 64)

    def test_final_tile_has_only_complete_eight_value_lanes(self):
        active_counts = set()
        for width in ELIGIBLE_WIDTHS:
            final_start = ((width - 1) // 256) * 256
            final = [
                chunk for lane in range(32)
                for chunk in candidate_lane_chunks(width, lane)
                if chunk[0] >= final_start
            ]
            active_counts.add(len(final))
            self.assertIn(len(final), (8, 16, 24, 32))
            self.assertEqual([channel for chunk in final for channel in chunk],
                             list(range(final_start, width)))
        self.assertEqual(active_counts, {8, 16, 24, 32})
        for width, tile_count in ((768, 3), (2816, 11)):
            self.assertIn(width, ELIGIBLE_WIDTHS)
            for lane in range(32):
                chunks = candidate_lane_chunks(width, lane)
                self.assertEqual(len(chunks), tile_count)
                self.assertEqual(chunks[-1][0], width - 256 + lane * 8)

    def test_scale_rows_and_output_tiles_remain_separate(self):
        for width in (576, 704, 768, 2816, 8128):
            generic = [generic_lane_chunks(width, lane) for lane in range(32)]
            candidate = [candidate_lane_chunks(width, lane) for lane in range(32)]
            for row in (0, 3, 4, 7, 15):
                self.assertEqual(ordered_simd_inputs(generic, width, row),
                                 ordered_simd_inputs(candidate, width, row))
                for lane_chunks in candidate:
                    for chunk in lane_chunks:
                        group = affine_contribution(chunk, width, row)[1]
                        self.assertTrue(row * (width // 64) <= group < (row + 1) * (width // 64))
        for rows in (8, 16, 24, 704, 2816):
            writes = [tile * 8 + simd * 4 + row for tile in range(rows // 8)
                      for simd in range(2) for row in range(4)]
            self.assertEqual(Counter(writes), Counter(range(rows)))

    def test_order_oracles_reject_complete_but_reassociated_computations(self):
        for width in (704, 768, 2816):
            baseline = [generic_lane_chunks(width, lane) for lane in range(32)]
            merged = [candidate_lane_chunks(width, lane, 16) for lane in range(32)]
            # The old sixteen-value geometry covers every channel too, but
            # changes qdot grouping and which partial sums enter each lane.
            self.assertEqual(Counter(c for chunks in merged for chunk in chunks for c in chunk),
                             Counter(range(width)))
            expected = ordered_simd_inputs(baseline, width)
            self.assertNotEqual(expected, ordered_simd_inputs(merged, width))
            self.assertNotEqual(expected, ordered_simd_inputs([list(reversed(c)) for c in baseline], width))
            self.assertNotEqual(expected, ordered_simd_inputs(baseline[1:] + baseline[:1], width))
            self.assertNotEqual(expected, ordered_simd_inputs(baseline, width, scale_offset=1))


if __name__ == "__main__":
    unittest.main()
