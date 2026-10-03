// Correctness-only fixture probe for the patched MLX CUDA backend.
// No checkpoints, installed services, or throughput measurements are involved.
#include "mlx/backend/cuda/utils.h"
#include "mlx/memory.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mlx::core;

namespace {
struct Fixture {
  std::string name;
  Dtype dtype;
  int n = 512, k = 2048, experts = 256, selected = 8, m = 1, bits = 4, group = 64;
  bool eligible = false;
};

std::vector<Fixture> fixtures() {
  std::vector<Fixture> values;
  for (auto dtype : {float16, bfloat16}) {
    const std::string suffix = dtype == float16 ? "-float16" : "-bfloat16";
    values.push_back({"gate" + suffix, dtype, 512, 2048, 256, 8, 1, 4, 64, true});
    values.push_back({"gate_up" + suffix, dtype, 1024, 2048, 256, 8, 1, 4, 64, true});
    values.push_back({"down" + suffix, dtype, 2048, 512, 256, 8, 1, 4, 64, true});
  }
  values.push_back({"top7", float16, 512, 2048, 256, 7});
  values.push_back({"top9", float16, 512, 2048, 256, 9});
  values.push_back({"prefill", float16, 512, 2048, 256, 8, 2});
  values.push_back({"expert_count", float16, 512, 2048, 255});
  values.push_back({"fp32", float32});
  values.push_back({"other_projection", float16, 128});
  values.push_back({"q8", float16, 512, 2048, 256, 8, 1, 8});
  values.push_back({"group128", float16, 512, 2048, 256, 8, 1, 4, 128});
  return values;
}

uint32_t mix(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

float narrow(float value, Dtype dtype) {
  if (dtype == float16) return float(float16_t(value));
  if (dtype == bfloat16) return float(bfloat16_t(value));
  return value;
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

std::vector<float> read_float(const array& value, Stream stream) {
  auto converted = astype(value, float32, stream);
  eval(converted);
  synchronize(stream);
  std::vector<float> result(converted.size());
  check_cuda(cudaMemcpy(result.data(), gpu_ptr<void>(converted),
                        converted.nbytes(), cudaMemcpyDeviceToHost), "copy output");
  return result;
}

void run_fixture(const Fixture& f, int ordinal, const std::filesystem::path& output,
                 std::ostream& report, Stream stream) {
  std::size_t free_bytes = 0, total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "query free GPU memory");
  require(free_bytes >= (std::size_t{1} << 30), "less than 1 GiB GPU memory free; refusing fixture");
  std::cerr << "midnight_expert_fixture_begin " << f.name << std::endl;
  const int packed_per_word = 32 / f.bits;
  const std::size_t rows = std::size_t(f.experts) * f.n;
  std::vector<uint32_t> packed(rows * f.k / packed_per_word);
  std::vector<float> scales(rows * f.k / f.group), biases(scales.size());
  std::vector<float> inputs(std::size_t(f.selected) * f.m * f.k);
  const uint32_t seed = 260926U + uint32_t(ordinal) * 1009U;
  for (std::size_t i = 0; i < packed.size(); ++i) packed[i] = mix(uint32_t(i) + seed);
  const float scale_factor = 15.f / ((1U << f.bits) - 1U);
  for (std::size_t i = 0; i < scales.size(); ++i) {
    scales[i] = narrow((0.004f + float(mix(uint32_t(i) + seed + 19) & 65535U) * (0.076f / 65535.f)) * scale_factor, f.dtype);
    biases[i] = narrow(-float((1U << f.bits) - 1U) * 0.5f * scales[i], f.dtype);
  }
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    inputs[i] = narrow((float(mix(uint32_t(i) + seed + 31) & 65535U) / 32768.f - 1.f) / std::sqrt(float(f.k)), f.dtype);
  }
  const uint32_t rhs_values[] = {uint32_t(f.experts - 1), 0, 73, 73, 127, 15, 2, uint32_t(f.experts - 2), 19};
  const uint32_t lhs_values[] = {7, 2, 4, 4, 1, 0, 6, 3, 8};
  std::vector<uint32_t> lhs(f.selected), rhs(rhs_values, rhs_values + f.selected);
  for (int i = 0; i < f.selected; ++i) {
    lhs[i] = f.name.rfind("gate", 0) == 0 ? 0 : lhs_values[i] % f.selected;
  }

  // Constructor conversion rounds CPU fixture values into the declared dtype.
  array x(inputs.begin(), {f.selected, f.m, f.k}, f.dtype);
  array w(packed.begin(), {f.experts, f.n, f.k / packed_per_word}, uint32);
  array s(scales.begin(), {f.experts, f.n, f.k / f.group}, f.dtype);
  array b(biases.begin(), {f.experts, f.n, f.k / f.group}, f.dtype);
  array li(lhs.begin(), {f.selected}, uint32);
  array ri(rhs.begin(), {f.selected}, uint32);
  eval({x, w, s, b, li, ri});
  auto y = gather_qmm(x, w, s, b, li, ri, true, f.group, f.bits, "affine", false, stream);
  eval(y);
  synchronize(stream);
  require(y.shape() == Shape{f.selected, f.m, f.n}, "incorrect result shape");
  const auto values = read_float(y, stream);
  const auto binary = output / (f.name + ".f32");
  std::ofstream data(binary, std::ios::binary);
  require(bool(data), "cannot create fixture output");
  data.write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * sizeof(float)));
  require(bool(data), "cannot write fixture output");
  data.close();

  // Independent FP64 dot products of separately rounded affine reconstruction.
  // Diagnostic only: a closer reference result never waives exact stock parity.
  double max_error = 0, squared_error = 0;
  for (int selection = 0; selection < f.selected; ++selection) {
    for (int m = 0; m < f.m; ++m) {
      const auto input_base = (std::size_t(lhs[selection]) * f.m + m) * f.k;
      for (int row = 0; row < f.n; ++row) {
        const auto weight_row = std::size_t(rhs[selection]) * f.n + row;
        double reference = 0;
        for (int k = 0; k < f.k; ++k) {
          const auto word = packed[weight_row * (f.k / packed_per_word) + k / packed_per_word];
          const auto code = (word >> ((k % packed_per_word) * f.bits)) & ((1U << f.bits) - 1U);
          const auto metadata = weight_row * (f.k / f.group) + k / f.group;
          const auto product = narrow(float(code) * scales[metadata], f.dtype);
          const auto dequantized = narrow(product + biases[metadata], f.dtype);
          reference += double(inputs[input_base + k]) * double(dequantized);
        }
        const auto actual = values[(selection * f.m + m) * f.n + row];
        require(std::isfinite(actual), "nonfinite fixture output");
        const auto error = std::abs(double(actual) - reference);
        max_error = std::max(max_error, error);
        squared_error += error * error;
      }
    }
  }
  report << "{\"name\":\"" << f.name << "\",\"n\":" << f.n << ",\"k\":" << f.k
         << ",\"eligible\":" << (f.eligible ? "true" : "false")
         << ",\"output_bytes\":" << values.size() * sizeof(float)
         << ",\"max_reference_error\":" << std::setprecision(17) << max_error
         << ",\"rms_reference_error\":" << std::sqrt(squared_error / values.size()) << "}";
  std::cerr << "midnight_expert_fixture_end " << f.name << std::endl;
}

void run(const std::filesystem::path& output) {
  require(!std::filesystem::exists(output), "output directory already exists");
  require(is_available(Device::gpu), "CUDA MLX GPU is required");
  int device = 0;
  check_cuda(cudaGetDevice(&device), "query CUDA device");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device), "query CUDA properties");
  require(properties.major == 8 && properties.minor == 9, "probe requires compute capability 8.9");
  std::filesystem::create_directories(output);
  set_cache_limit(0);
  set_memory_limit(std::size_t{768} << 20);
  const auto stream = default_stream(Device::gpu);
  std::ofstream report(output / "results.json");
  require(bool(report), "cannot create results report");
  const char* mode = std::getenv("MIDNIGHT_CUDA_EXPERT_QMV");
  report << "{\"timing\":\"not_measured\",\"runtime_flag\":\"" << (mode ? mode : "0") << "\",\"cases\":[";
  const auto all = fixtures();
  for (std::size_t i = 0; i < all.size(); ++i) {
    if (i) report << ',';
    run_fixture(all[i], int(i), output, report, stream);
    clear_cache();
  }
  report << "]}\n";
  require(bool(report), "cannot finish results report");
  clear_streams();
}
} // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: cuda-expert-probe NEW_OUTPUT_DIRECTORY");
    run(argv[1]);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CUDA expert probe: " << error.what() << '\n';
    try { clear_streams(); } catch (...) {}
    return 1;
  }
}
