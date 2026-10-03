// Synthetic submission-overhead measurement, not a language-model benchmark.
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/midnight_replay.h"
#include "mlx/fast.h"
#include "mlx/memory.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mlx::core;

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
std::vector<std::uint8_t> read_completed(const array& value) {
  std::vector<std::uint8_t> bytes(value.nbytes());
  check_cuda(cudaMemcpy(bytes.data(), gpu_ptr<void>(value), bytes.size(),
                        cudaMemcpyDeviceToHost));
  return bytes;
}

void run(const std::string& workload, bool reversed) {
  require(is_available(Device::gpu), "CUDA device required");
  std::size_t free_bytes, total_bytes;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes));
  require(free_bytes >= (std::size_t{1} << 30), "At least 1 GiB free is required");
  set_memory_limit(std::size_t{256} << 20);
  set_cache_limit(std::size_t{64} << 20);
  reset_peak_memory();
  const bool small = workload == "small-fp32";
  const auto dtype = workload == "stack-fp16" ? float16 : float32;
  const int width = small ? 128 : 896;
  const int layers = small ? 4 : 24;
  auto stream = default_stream(Device::gpu);
  auto& encoder = cu::get_command_encoder(stream);
  auto original = astype(reshape(divide(arange(width, float32, stream),
                                      array(float(width)), stream), {1, width}, stream), dtype, stream);
  auto replacement = astype(sin(multiply(astype(original, float32, stream),
                                        array(7.f), stream), stream), dtype, stream);
  // MLX copy() may alias storage; allocate the mutable input independently.
  auto input = zeros({1, width}, dtype, stream);
  // All layers share one weight array: this intentionally isolates submission
  // overhead and does not represent a model's weight bandwidth or memory size.
  auto weights = astype(reshape(multiply(cos(arange(width * width, float32, stream), stream),
                                         array(.01f), stream), {width, width}, stream), dtype, stream);
  auto scale = ones({width}, dtype, stream);
  eval({original, replacement, input, weights, scale});
  synchronize(stream);
  require(gpu_ptr<void>(input) != gpu_ptr<void>(original) &&
          gpu_ptr<void>(input) != gpu_ptr<void>(replacement),
          "Mutable input must have independent storage");
  require(input.flags().row_contiguous && input.buffer_size() >= input.nbytes(),
          "Mutable input must own contiguous storage");
  check_cuda(cudaMemcpyAsync(gpu_ptr<void>(input), gpu_ptr<void>(original),
                             input.nbytes(), cudaMemcpyDeviceToDevice, encoder.stream()));
  synchronize(stream);
  auto forward = [&](const array& value) {
    array output = value;
    for (int layer = 0; layer < layers; ++layer) {
      output = add(output, matmul(fast::rms_norm(output, scale, 1e-6f, stream),
                                   weights, stream), stream);
    }
    return output;
  };
  auto first = forward(original);
  auto second = forward(replacement);
  eval({first, second});
  synchronize(stream);
  std::array<std::vector<std::uint8_t>, 2> expected{
      read_completed(first), read_completed(second)};
  require(expected[0] != expected[1], "Inputs must produce distinct outputs");
  require(all(isfinite(first, stream), stream).item<bool>() &&
          all(isfinite(second, stream), stream).item<bool>(), "Nonfinite reference");

  cu::ReplaySession::Limits limits;
  limits.retained_bytes = std::size_t{64} << 20;
  cu::ReplaySession session(encoder, limits);
  const std::vector<array> bindings{input, weights, scale};
  array captured = input;
  const auto capture_start = std::chrono::steady_clock::now();
  session.record(1, bindings, [&] { captured = forward(input); eval(captured); });
  const double capture_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - capture_start).count();
  require(read_completed(captured) == expected[0], "Capture parity failed");
  const auto stats = session.statistics();
  array ordinary = input;
  auto iteration = [&](bool replay, int which) {
    const auto& source = which ? replacement : original;
    // Include identical stable-address input-copy work in both timed arms.
    check_cuda(cudaMemcpyAsync(gpu_ptr<void>(input), gpu_ptr<void>(source),
                               input.nbytes(), cudaMemcpyDeviceToDevice, encoder.stream()));
    if (replay) {
      session.replay(1, bindings); // Includes the supported completion boundary.
    } else {
      ordinary = forward(input);
      eval(ordinary);
      synchronize(stream);
    }
  };
  auto verify = [&](bool replay, int which) {
    require(read_completed(replay ? captured : ordinary) == expected[which],
            "Exact output check failed");
  };
  // Verify every warmup launch, including input changes and ordinary/replay
  // interleaving. Timed blocks each have a separate untimed output check.
  for (int step = 0; step < 40; ++step) {
    for (bool replay : {false, true}) {
      iteration(replay, step % 2);
      verify(replay, step % 2);
    }
  }
  // Give GPU clocks time to settle before sampling either execution path.
  const auto settle_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
  int settle_step = 0;
  do {
    iteration(false, settle_step % 2);
    iteration(true, settle_step % 2);
    ++settle_step;
  } while (std::chrono::steady_clock::now() < settle_until);
  verify(false, (settle_step - 1) % 2);
  verify(true, (settle_step - 1) % 2);
  constexpr int rounds = 5;
  constexpr int samples = 100;
  std::cout << std::setprecision(12)
            << "{\"workload\":\"" << workload << "\",\"model_benchmark\":false,"
            << "\"shared_layer_weights\":true,\"width\":" << width
            << ",\"layers\":" << layers << ",\"capture_ms\":" << capture_ms
            << ",\"graph_chunks\":" << stats.graph_chunks
            << ",\"retained_bytes\":" << stats.retained_bytes
            << ",\"retained_allocations\":" << stats.retained_allocations
            << ",\"allocator_limit_bytes\":" << (std::size_t{256} << 20)
            << ",\"cache_limit_bytes\":" << (std::size_t{64} << 20)
            << ",\"order\":\"" << (reversed ? "BAAB" : "ABBA") << "\",\"blocks\":[";
  bool comma = false;
  for (int round = 0; round < rounds; ++round) {
    for (bool nominal_replay : {false, true, true, false}) {
      const bool replay = nominal_replay != reversed;
      std::vector<double> microseconds;
      microseconds.reserve(samples);
      for (int sample = 0; sample < samples; ++sample) {
        const auto start = std::chrono::steady_clock::now();
        iteration(replay, sample % 2);
        microseconds.push_back(std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count());
      }
      verify(replay, (samples - 1) % 2);
      if (comma) std::cout << ',';
      comma = true;
      std::cout << "{\"round\":" << round << ",\"mode\":\""
                << (replay ? "replay" : "ordinary") << "\",\"microseconds\":[";
      for (int sample = 0; sample < samples; ++sample) {
        if (sample) std::cout << ',';
        std::cout << microseconds[sample];
      }
      std::cout << "]}";
    }
  }
  std::cout << "],\"peak_mlx_bytes\":" << get_peak_memory()
            << ",\"exact_checks_passed\":true}\n";
  session.clear();
}
} // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 3, "Usage: benchmark small-fp32|stack-fp32|stack-fp16 ABBA|BAAB");
    const std::string workload = argv[1], order = argv[2];
    require(workload == "small-fp32" || workload == "stack-fp32" || workload == "stack-fp16",
            "Unknown workload");
    require(order == "ABBA" || order == "BAAB", "Unknown order");
    run(workload, order == "BAAB");
    clear_streams();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Replay benchmark failed: " << error.what() << '\n';
    try { clear_streams(); } catch (...) {}
    return 1;
  }
}
