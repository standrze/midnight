"""CPU-only checks of preparation and the actual Q4 tail loop control flow.

The scalar C++ shims model Q4 arithmetic and SIMD reduction, not Metal/BF16
rounding or GPU performance. GPU numerical and timing checks remain required.
"""

from pathlib import Path
import os
import re
import shutil
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SWIFT = ROOT / ".build/checkouts/mlx-swift"
CORE = SWIFT / "Source/Cmlx/mlx"
CORE_REVISION = "1f8e74e3f12f31365464a6867c6579f0e9b29d85"
SWIFT_REVISION = "72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798"
CORE_PATHS = [
    "mlx/backend/metal/kernels/quantized.h",
    "mlx/backend/metal/kernels/quantized.metal",
    "mlx/backend/metal/quantized.cpp",
]
SWIFT_PATHS = [
    "Source/Cmlx/mlx-generated/metal/quantized.h",
    "Source/Cmlx/mlx-generated/quantized.cpp",
]
SPECS = [
    (CORE, CORE_REVISION, CORE_PATHS, "mlx-affine-q4-qmv-specialization.patch", "mlx-affine-q4-qmv-tail.patch",
     "mlx-affine-q4-qmv-tail-order.patch"),
    (SWIFT, SWIFT_REVISION, SWIFT_PATHS, "mlx-swift-affine-q4-qmv-jit.patch", "mlx-swift-affine-q4-qmv-tail-jit.patch",
     "mlx-swift-affine-q4-qmv-tail-order-jit.patch"),
]


def run(*args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, text=True, **kwargs)


def function_source(text, marker):
    start = text.index(marker)
    opening = text.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


class TailPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.originals = {}
        for checkout, revision, paths, _, _, _ in SPECS:
            for relative in paths:
                cls.originals[relative] = run(
                    "git", "-C", str(checkout), "show", f"{revision}:{relative}"
                ).stdout

    def fixture(self, directory, state="clean"):
        swift = directory / "mlx-swift"
        repositories = [swift / "Source/Cmlx/mlx", swift]
        for repository, (_, _, paths, base, overlay, order_overlay) in zip(repositories, SPECS):
            for relative in paths:
                target = repository / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(self.originals[relative])
            if state in ("baseline", "legacy", "applied"):
                run("git", "-C", str(repository), "apply", str(ROOT / "Patches" / base))
            if state in ("legacy", "applied"):
                run("git", "-C", str(repository), "apply", str(ROOT / "Patches" / overlay))
            if state == "applied":
                run("git", "-C", str(repository), "apply", str(ROOT / "Patches" / order_overlay))
        (swift / "unrelated-local-edit.txt").write_text("preserve this edit\n")
        return swift

    def prepare(self, fixture, host="Darwin", bad_revision=False, check=True):
        script = r'''
set -euo pipefail
git() {
  if [[ "$#" == 4 && "$1" == -C && "$3" == rev-parse && "$4" == HEAD ]]; then
    if [[ "${TAIL_TEST_BAD_REVISION:-0}" == 1 ]]; then
      echo 0000000000000000000000000000000000000000
      return 0
    fi
    case "$2" in
      */Source/Cmlx/mlx) echo 1f8e74e3f12f31365464a6867c6579f0e9b29d85 ;;
      */mlx-swift) echo 72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798 ;;
      *) return 1 ;;
    esac
  else
    command git "$@"
  fi
}
source "$1/Scripts/affine-q4-qmv-tail-patch.sh"
model_runner_prepare_affine_q4_qmv_tail "$2" "$1" "$3"
'''
        env = dict(os.environ, TAIL_TEST_BAD_REVISION="1" if bad_revision else "0")
        return subprocess.run(
            ["bash", "-c", script, "tail-test", str(ROOT), host, str(fixture)],
            check=check, capture_output=True, text=True, env=env,
        )

    @staticmethod
    def snapshot(fixture):
        return {str(p.relative_to(fixture)): p.read_bytes() for p in fixture.rglob("*") if p.is_file()}

    def test_prepare_clean_baseline_and_applied_twice(self):
        with tempfile.TemporaryDirectory(prefix="midnight-tail-patches-") as temporary:
            temporary = Path(temporary)
            reference = None
            for state in ("clean", "baseline", "legacy", "applied"):
                with self.subTest(state=state):
                    fixture = self.fixture(temporary / state, state)
                    self.prepare(fixture)
                    first = self.snapshot(fixture)
                    self.prepare(fixture)
                    self.assertEqual(first, self.snapshot(fixture))
                    if reference is None:
                        reference = first
                    self.assertEqual(reference, first)
                    self.assertEqual(b"preserve this edit\n", first["unrelated-local-edit.txt"])
                    for repository, (_, _, paths, base, overlay, order_overlay) in zip(
                        [fixture / "Source/Cmlx/mlx", fixture], SPECS
                    ):
                        run("git", "-C", str(repository), "apply", "--reverse", str(ROOT / "Patches" / order_overlay))
                        run("git", "-C", str(repository), "apply", "--reverse", str(ROOT / "Patches" / overlay))
                        run("git", "-C", str(repository), "apply", "--reverse", str(ROOT / "Patches" / base))
                        for relative in paths:
                            self.assertEqual(self.originals[relative], (repository / relative).read_text())

    def test_preflight_failure_changes_neither_checkout(self):
        with tempfile.TemporaryDirectory(prefix="midnight-tail-drift-") as temporary:
            fixture = self.fixture(Path(temporary), "baseline")
            before = self.snapshot(fixture)
            self.assertNotEqual(0, self.prepare(fixture, bad_revision=True, check=False).returncode)
            self.assertEqual(before, self.snapshot(fixture))
            generated = fixture / SWIFT_PATHS[-1]
            generated.write_text(generated.read_text().replace(
                "METAL_FUNC void qmv_fast_impl(", "METAL_FUNC void user_modified_qmv_fast_impl(", 1
            ))
            before = self.snapshot(fixture)
            self.assertNotEqual(0, self.prepare(fixture, check=False).returncode)
            self.assertEqual(before, self.snapshot(fixture))
            self.prepare(Path("/nonexistent"), host="Linux")
            self.assertEqual(before, self.snapshot(fixture))

    def test_readonly_clean_partial_and_applied_sources(self):
        with tempfile.TemporaryDirectory(prefix="midnight-tail-readonly-") as temporary:
            temporary = Path(temporary)
            expected = self.snapshot(self.fixture(temporary / "expected", "applied"))
            for state in ("clean", "baseline", "legacy", "applied"):
                with self.subTest(state=state):
                    fixture = self.fixture(temporary / state, state)
                    for path in fixture.rglob("*"):
                        if path.is_file():
                            path.chmod(0o444)
                    result = self.prepare(fixture, check=False)
                    self.assertEqual(0, result.returncode, result.stderr)
                    self.assertEqual(expected, self.snapshot(fixture))
                    self.assertEqual(0o444, stat.S_IMODE((fixture / "unrelated-local-edit.txt").stat().st_mode))
                    # Patch application may replace pending source files. An
                    # idempotent replay must preserve read-only bytes and modes.
                    for path in fixture.rglob("*"):
                        if path.is_file():
                            path.chmod(0o444)
                    before = {str(path.relative_to(fixture)): (
                        path.read_bytes(), stat.S_IMODE(path.stat().st_mode)
                    ) for path in fixture.rglob("*") if path.is_file()}
                    self.prepare(fixture)
                    after = {str(path.relative_to(fixture)): (
                        path.read_bytes(), stat.S_IMODE(path.stat().st_mode)
                    ) for path in fixture.rglob("*") if path.is_file()}
                    self.assertEqual(before, after)

    def test_partial_source_only_overlay_converges(self):
        with tempfile.TemporaryDirectory(prefix="midnight-tail-partial-") as temporary:
            fixture = self.fixture(Path(temporary), "baseline")
            run("git", "-C", str(fixture / "Source/Cmlx/mlx"), "apply", str(ROOT / "Patches" / SPECS[0][-2]))
            self.prepare(fixture)
            before = self.snapshot(fixture)
            self.prepare(fixture)
            self.assertEqual(before, self.snapshot(fixture))

    def test_patch_targets_generated_parity_and_registration(self):
        with tempfile.TemporaryDirectory(prefix="midnight-tail-parity-") as temporary:
            fixture = self.fixture(Path(temporary), "applied")
            headers = [fixture / "Source/Cmlx/mlx" / CORE_PATHS[0], fixture / SWIFT_PATHS[0], fixture / SWIFT_PATHS[1]]
            for marker in ("METAL_FUNC void qmv_fast_impl(", "[[kernel]] void affine_qmv_fast(", "[[kernel]] void affine_gather_qmv_fast("):
                snippets = [function_source(path.read_text(), marker) for path in headers]
                self.assertEqual(snippets[0], snippets[1])
                self.assertEqual(snippets[0], snippets[2])
            for _, _, paths, _, _, overlay in SPECS:
                patch = (ROOT / "Patches" / overlay).read_text()
                targets = re.findall(r"^diff --git a/(.+) b/(.+)$", patch, re.M)
                self.assertEqual([(path, path) for path in paths], targets)
            metal = (fixture / "Source/Cmlx/mlx" / CORE_PATHS[1]).read_text()
            self.assertIn('"_gs_64_b_4_tail_g8_batch_0"', metal)
            self.assertIn('"_gs_64_b_4_tail_g8_batch_1"', metal)
            self.assertIn('"_gs_64_b_4_tail_g8"', metal)
            self.assertIn("instantiate_affine_q4_qmv_tail(float16_t)", metal)
            self.assertIn("instantiate_affine_q4_qmv_tail(bfloat16_t)", metal)
            host = (fixture / "Source/Cmlx/mlx" / CORE_PATHS[2]).read_text()
            self.assertIn("use_affine_q4_qmv_tail(mode, x.dtype(), group_size, bits, N, K, false)",
                          function_source(host, "void qmv("))
            self.assertIn("use_affine_q4_qmv_tail(mode, x.dtype(), group_size, bits, N, K, true)",
                          function_source(host, "void gather_qmv("))

    def test_actual_control_flow_bounds_and_scalar_math(self):
        compiler = shutil.which("clang++")
        self.assertIsNotNone(compiler, "clang++ is required for the address-sanitized CPU fixture")
        with tempfile.TemporaryDirectory(prefix="midnight-tail-cpu-") as temporary:
            temporary = Path(temporary)
            fixture = self.fixture(temporary, "applied")
            header = (fixture / "Source/Cmlx/mlx" / CORE_PATHS[0]).read_text()
            marker = "METAL_FUNC void qmv_fast_impl("
            start = header.rfind("template <", 0, header.index(marker))
            kernel = header[start:header.index(marker)] + function_source(header, marker)
            kernel = re.sub(r"\[\[[^\]]+\]\]", "", kernel)
            kernel = re.sub(r"\b(METAL_FUNC|device|constant|thread)\s+", "", kernel)
            dispatch = function_source((fixture / "Source/Cmlx/mlx" / CORE_PATHS[2]).read_text(), "inline bool use_affine_q4_qmv_tail(")
            source = temporary / "tail.cpp"
            source.write_text(CPP_PREFIX + dispatch + "\n" + kernel + CPP_MAIN)
            executable = temporary / "tail"
            run(compiler, "-std=c++17", "-O1", "-fsanitize=address,undefined", "-fno-omit-frame-pointer", str(source), "-o", str(executable))
            for enabled in ("0", "1", "2", "-1"):
                for scope in (None, "all", "dense", "gather", "invalid", ""):
                    env = dict(os.environ, MLX_METAL_AFFINE_Q4_QMV_TAIL=enabled, ASAN_OPTIONS="detect_leaks=0")
                    env.pop("MLX_METAL_AFFINE_Q4_QMV_TAIL_SCOPE", None)
                    if scope is not None:
                        env["MLX_METAL_AFFINE_Q4_QMV_TAIL_SCOPE"] = scope
                    self.assertIn("Q4 tail CPU checks passed", run(str(executable), env=env).stdout)


CPP_PREFIX = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
using uint = unsigned int;
struct uint3 { uint x, y, z; };
constexpr int SIMD_SIZE = 32;
enum Dtype { float16, bfloat16, float32 };
namespace env { int get_var(const char* key, int fallback) {
  const char* value = std::getenv(key); return value ? std::atoi(value) : fallback;
}
std::string get_var(const char* key, const char* fallback) {
  const char* value = std::getenv(key); return value ? value : fallback;
} }
template<int bits, int width> constexpr int get_pack_factor() { return width / bits; }
template<int bits, int width> constexpr int get_bytes_per_pack() { return width / 8; }
template<class T, class U, int count, int bits>
U load_vector(const T* x, U* local) {
  U sum = 0; for (int i = 0; i < count; ++i) { local[i] = x[i]; sum += x[i]; } return sum;
}
template<class U, int count, int bits>
U qdot(const uint8_t* w, const U* x, U scale, U bias, U sum) {
  static_assert(bits == 4); U dot = 0;
  for (int i = 0; i < count; ++i) dot += ((w[i / 2] >> (4 * (i % 2))) & 15) * x[i];
  return scale * dot + bias * sum;
}
float partials[32][4];
int active_lane, active_row;
float simd_sum(float value) {
  int row = active_row++; partials[active_lane][row] = value;
  if (active_lane != 0) return value;
  float result = 0; for (int lane = 0; lane < 32; ++lane) result += partials[lane][row]; return result;
}
'''

CPP_MAIN = r'''
template<bool tail> void check_shape(int K, int N, int M, int expert) {
  std::vector<uint32_t> weights((expert + 1) * N * K / 8);
  std::vector<float> scales((expert + 1) * N * K / 64), biases(scales.size()), x(M * K), y(M * N);
  for (size_t i = 0; i < weights.size(); ++i) {
    for (int shift = 0; shift < 8; ++shift) weights[i] |= uint32_t((i + 3 * shift) % 16) << (4 * shift);
  }
  for (size_t i = 0; i < scales.size(); ++i) { scales[i] = float(i % 5 + 1) / 64; biases[i] = float(int(i % 7) - 3) / 128; }
  for (size_t i = 0; i < x.size(); ++i) x[i] = float(int(i % 7) - 3) / 32;
  auto* w = weights.data() + expert * N * K / 8;
  auto* s = scales.data() + expert * N * K / 64;
  auto* b = biases.data() + expert * N * K / 64;
  for (uint m = 0; m < uint(M); ++m) for (uint group = 0; group < uint(N / 8); ++group) {
    for (uint simd = 0; simd < 2; ++simd) {
      // Lane zero observes all peer lanes when the reduction is modeled.
      for (int sequence = 0; sequence < 32; ++sequence) {
        active_lane = (sequence + 1) % 32; active_row = 0;
        qmv_fast_impl<float, 64, 4, 4, tail>(w, s, b, x.data(), y.data(), K, N, {m, group, 0}, simd, active_lane);
      }
    }
  }
  for (int m = 0; m < M; ++m) for (int row = 0; row < N; ++row) {
    double reference = 0;
    for (int k = 0; k < K; ++k) {
      uint32_t q = (w[row * K / 8 + k / 8] >> (4 * (k % 8))) & 15;
      reference += x[m * K + k] * (s[row * K / 64 + k / 64] * q + b[row * K / 64 + k / 64]);
    }
    assert(std::abs(y[m * N + row] - reference) < 0.0001);
  }
}
int main() {
  bool enabled = env::get_var("MLX_METAL_AFFINE_Q4_QMV_TAIL", 0) == 1;
  std::string scope = env::get_var("MLX_METAL_AFFINE_Q4_QMV_TAIL_SCOPE", "all");
  for (int K = 512; K <= 5888; K += 64) {
    for (bool gather : {false, true}) {
      bool expected = enabled && K % 512 != 0 &&
          (scope == "all" || scope == (gather ? "gather" : "dense"));
      assert(use_affine_q4_qmv_tail("affine", bfloat16, 64, 4, 24, K, gather) == expected);
      assert(use_affine_q4_qmv_tail("affine", float16, 64, 4, 24, K, gather) == expected);
    }
    check_shape<true>(K, 24, 2, 2);
    if (K % 512 == 0) check_shape<false>(K, 24, 2, 2);
  }
  for (int K : {576, 640, 704, 768, 2112, 2816, 5376, 21504}) check_shape<true>(K, 8, 1, 0);
  for (int K : {0, 64, 128, 256, 511, 513, 703, 705})
    assert(!use_affine_q4_qmv_tail("affine", bfloat16, 64, 4, 8, K, false));
  for (int N : {0, 1, 7, 9}) assert(!use_affine_q4_qmv_tail("affine", bfloat16, 64, 4, N, 704, false));
  for (int bits : {2, 3, 5, 6, 8}) assert(!use_affine_q4_qmv_tail("affine", bfloat16, 64, bits, 8, 704, false));
  assert(!use_affine_q4_qmv_tail("affine", float32, 64, 4, 8, 704, false));
  assert(!use_affine_q4_qmv_tail("affine", bfloat16, 128, 4, 8, 704, false));
  assert(!use_affine_q4_qmv_tail("nvfp4", bfloat16, 64, 4, 8, 704, false));
  std::cout << "Q4 tail CPU checks passed\n";
}
'''


if __name__ == "__main__":
    unittest.main()
