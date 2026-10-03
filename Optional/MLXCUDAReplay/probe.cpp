#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/midnight_replay.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace mlx::core;

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <class Operation>
void rejects(Operation operation, const char* message) {
  bool failed = false;
  try { operation(); } catch (const std::exception&) { failed = true; }
  require(failed, message);
}
std::vector<float> read_completed(const array& value) {
  require(value.dtype() == float32, "probe read requires float32");
  std::vector<float> result(value.size());
  check_cuda(cudaMemcpy(result.data(), gpu_ptr<void>(value),
                        value.nbytes(), cudaMemcpyDeviceToHost));
  return result;
}

void run() {
  require(is_available(Device::gpu), "MLX CUDA GPU required; no CPU fallback");
  auto stream = default_stream(Device::gpu);
  auto& encoder = cu::get_command_encoder(stream);
  constexpr int width = 128;
  auto input = reshape(divide(arange(width, float32, stream), array(128.f), stream),
                       {1, width}, stream);
  auto replacement = sin(multiply(input, array(7.f), stream), stream);
  auto weights = reshape(multiply(cos(arange(width * width, float32, stream), stream),
                                  array(.01f), stream), {width, width}, stream);
  auto scale = ones({width}, float32, stream);
  eval({input, replacement, weights, scale});
  synchronize(stream);
  auto forward = [&](const array& value) {
    array output = value;
    for (int layer = 0; layer < 4; ++layer) {
      output = add(output, matmul(fast::rms_norm(output, scale, 1e-6f, stream),
                                   weights, stream), stream);
    }
    return output;
  };
  auto expected = forward(input);
  auto changed = forward(replacement);
  eval({expected, changed});
  synchronize(stream);
  const auto original_values = read_completed(expected);
  const auto changed_values = read_completed(changed);
  require(original_values != changed_values, "replacement input must affect output");

  auto bindings = [&] { return std::vector<array>{input, weights, scale}; };
  cu::ReplaySession session(encoder);
  array output = input;
  auto capture = [&] {
    session.record(1, bindings(), [&] { output = forward(input); eval(output); });
  };
  capture();
  require(read_completed(output) == original_values, "capture changed MLX arithmetic");
  const auto stats = session.statistics();
  require(stats.graph_chunks > 0 && stats.retained_allocations > 0,
          "capture did not record kernels and allocations");
  for (int step = 0; step < 20; ++step) {
    session.replay(1, bindings());
    // Intentionally no eval(output) or extra synchronization: replay is the
    // supported completion boundary for these already-evaluated MLX arrays.
    require(read_completed(output) == original_values, "replay output changed");
  }
  check_cuda(cudaMemcpyAsync(gpu_ptr<void>(input), gpu_ptr<void>(replacement),
                             input.nbytes(), cudaMemcpyDeviceToDevice,
                             encoder.stream()));
  session.replay(1, bindings());
  require(read_completed(output) == changed_values, "stable-address input update not observed");

  // An ordinary MLX graph between launches cannot overwrite retained workspace.
  auto interleaved = forward(input);
  eval(interleaved);
  session.replay(1, bindings());
  require(read_completed(output) == changed_values, "ordinary evaluation corrupted retained buffers");
  rejects([&] { session.replay(2, bindings()); }, "model mutation epoch was accepted");
  require(session.phase() == cu::ReplaySession::Phase::invalidated, "epoch mismatch did not invalidate");
  session.clear();
  capture();

  auto changed_bindings = bindings();
  changed_bindings[0] = replacement;
  rejects([&] { session.replay(1, changed_bindings); }, "changed input allocation was accepted");
  session.clear();
  rejects([&] {
    session.record(1, bindings(), [&] {
      output = forward(input);
      eval(output);
      throw std::runtime_error("simulated cancellation");
    });
  }, "capture exception was swallowed");
  require(session.statistics().retained_bytes == 0, "cancelled capture retained storage");
  session.clear();
  capture();

  rejects([&] {
    session.record(1, bindings(), [&] {
      session.record(1, bindings(), [] {});
    });
  }, "nested capture accepted");
  session.clear();

  auto other_stream = new_stream(Device::gpu);
  rejects([&] {
    session.record(1, bindings(), [&] {
      output = forward(input);
      eval(output);
      (void)cu::get_command_encoder(other_stream);
    });
  }, "another encoder was accepted during capture");
  session.clear();
  capture();

  // Destruction of an encoder invalidates borrowing sessions after MLX drains.
  auto survivor = std::make_unique<cu::ReplaySession>(encoder);
  survivor->record(1, bindings(), [&] { output = forward(input); eval(output); });
  session.clear();
  clear_streams();
  rejects([&] { survivor->replay(1, bindings()); }, "destroyed encoder remained usable");
  survivor.reset();
  std::cout << "{\"passed\":true,\"actual_mlx_kernels\":true,"
               "\"model_benchmark\":false,\"synchronous_replay\":true,"
               "\"graph_chunks\":" << stats.graph_chunks
            << ",\"retained_allocations\":" << stats.retained_allocations << "}\n";
}
} // namespace

int main() {
  try {
    run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "MLX replay probe: " << error.what() << '\n';
    // Preserve the original failure; best-effort cleanup does not turn it into
    // a successful operator or lifecycle result.
    try { clear_streams(); } catch (...) {}
    return 1;
  }
}
