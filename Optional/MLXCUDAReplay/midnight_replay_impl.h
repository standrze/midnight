// Included once at the end of MLX's device.cpp; no build manifest or encoder
// layout change is needed. See README.md for the intentionally narrow contract.
#if defined(MIDNIGHT_MLX_CUDA_REPLAY)

#include "mlx/backend/cuda/midnight_replay.h"

#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace mlx::core::cu {
namespace replay_detail {

struct EncoderLifetime {
  CommandEncoder* encoder;
  std::atomic<bool> alive{true};
};

struct Registry {
  std::mutex mutex;
  std::unordered_map<CommandEncoder*, std::weak_ptr<EncoderLifetime>> encoders;
  std::shared_ptr<ReplaySession::Impl> active;
};

Registry& registry() {
  // Encoders can be destroyed during process teardown; keep registry ordering
  // independent from thread-local encoder and CUDA static destruction ordering.
  static auto* value = new Registry;
  return *value;
}

std::shared_ptr<EncoderLifetime> lifetime(CommandEncoder& encoder) {
  auto& r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  auto& weak = r.encoders[&encoder];
  auto token = weak.lock();
  if (!token) {
    token = std::make_shared<EncoderLifetime>();
    token->encoder = &encoder;
    weak = token;
  }
  return token;
}

// One stream and one creator thread are deliberate limitations. The token lets
// a session outlive its encoder safely, but never use the destroyed stream.
struct Backend {
  std::shared_ptr<EncoderLifetime> lifetime;
  Device* device;
  cudaStream_t stream;
  CudaGraph graph;
  CudaGraphExec executable;
  cudaGraphNode_t previous = nullptr;

  explicit Backend(std::shared_ptr<EncoderLifetime> token)
      : lifetime(std::move(token)), device(&lifetime->encoder->device()),
        stream(lifetime->encoder->stream()) {}

  Backend(Backend&&) = default;
  Backend(const Backend&) = delete;

  CommandEncoder& encoder() const {
    if (!lifetime->alive.load(std::memory_order_acquire)) {
      throw std::logic_error("replay encoder has been destroyed");
    }
    return *lifetime->encoder;
  }
  void begin() {
    (void)encoder();
    device->make_current();
    graph = CudaGraph(*device);
  }
  void append(cudaGraph_t chunk) {
    cudaGraphNode_t child = nullptr;
    CHECK_CUDA_ERROR(cudaGraphAddChildGraphNode(
        &child, graph, previous ? &previous : nullptr, previous ? 1 : 0, chunk));
    previous = child;
  }
  void instantiate() {
    device->make_current();
    executable.instantiate(graph);
  }
  void launch() {
    // Flush ordinary work queued between replays; no host wait is introduced.
    encoder().commit();
    device->make_current();
    CHECK_CUDA_ERROR(cudaGraphLaunch(executable, stream));
  }
  void drain() {
    if (lifetime->alive.load(std::memory_order_acquire)) {
      device->make_current();
      CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    }
    // encoder_destroyed is called only after MLX's synchronize() has completed.
  }
  void reset() {
    device->make_current();
    executable.reset();
    graph.reset();
    previous = nullptr;
  }
};

using Core = midnight::mlx_replay::SessionCore<Backend>;

std::vector<midnight::mlx_replay::Binding> binding_descriptors(
    const std::vector<array>& arrays) {
  std::vector<midnight::mlx_replay::Binding> result;
  result.reserve(arrays.size());
  for (const auto& value : arrays) {
    if (!value.data_shared_ptr()) {
      throw std::invalid_argument("replay bindings must already be evaluated");
    }
    result.push_back({value.buffer().ptr(), gpu_ptr<void>(value),
                      value.buffer_size(), static_cast<int>(value.dtype().val()),
                      {value.shape().begin(), value.shape().end()},
                      {value.strides().begin(), value.strides().end()}});
  }
  return result;
}

std::size_t inspect_graph(cudaGraph_t graph, const ReplaySession::Limits& limits,
                          std::size_t depth, std::size_t& remaining) {
  if (depth > limits.graph_depth) {
    throw std::length_error("replay graph nesting limit exceeded");
  }
  std::size_t count = 0;
  CHECK_CUDA_ERROR(cudaGraphGetNodes(graph, nullptr, &count));
  if (count > remaining) {
    throw std::length_error("replay graph node limit exceeded");
  }
  remaining -= count;
  std::vector<cudaGraphNode_t> nodes(count);
  if (count) {
    CHECK_CUDA_ERROR(cudaGraphGetNodes(graph, nodes.data(), &count));
  }
  auto total = count;
  for (auto node : nodes) {
    cudaGraphNodeType type;
    CHECK_CUDA_ERROR(cudaGraphNodeGetType(node, &type));
    switch (type) {
      case cudaGraphNodeTypeKernel:
      case cudaGraphNodeTypeEmpty:
      case cudaGraphNodeTypeMemset:
        break;
      case cudaGraphNodeTypeGraph: {
        cudaGraph_t child = nullptr;
        CHECK_CUDA_ERROR(cudaGraphChildGraphNodeGetGraph(node, &child));
        total += inspect_graph(child, limits, depth + 1, remaining);
        break;
      }
      default:
        // Host callbacks, event handles, memory-copy host addresses, external
        // semaphores and allocations require ownership that this API lacks.
        throw std::invalid_argument("replay graph contains unsupported resource nodes");
    }
  }
  return total;
}
} // namespace replay_detail

struct ReplaySession::Impl {
  std::shared_ptr<replay_detail::EncoderLifetime> lifetime;
  std::thread::id creator = std::this_thread::get_id();
  replay_detail::Core core;
  std::atomic<const char*> rejected{nullptr};

  Impl(CommandEncoder& encoder, Limits limits)
      : lifetime(replay_detail::lifetime(encoder)),
        core(replay_detail::Backend(lifetime), limits) {}

  CommandEncoder& encoder() const {
    if (creator != std::this_thread::get_id()) {
      throw std::logic_error("replay session used from another thread");
    }
    if (!lifetime->alive.load(std::memory_order_acquire)) {
      throw std::logic_error("replay encoder has been destroyed");
    }
    return *lifetime->encoder;
  }
  bool observes(CommandEncoder& encoder) noexcept {
    if (creator != std::this_thread::get_id() || lifetime->encoder != &encoder) {
      reject("replay capture crossed a thread or CUDA encoder");
      return false;
    }
    return rejected.load(std::memory_order_acquire) == nullptr;
  }
  void reject(const char* message) noexcept {
    const char* empty = nullptr;
    rejected.compare_exchange_strong(empty, message, std::memory_order_release);
  }
};

namespace replay_detail {
std::shared_ptr<ReplaySession::Impl> active() {
  auto& r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  return r.active;
}
std::shared_ptr<ReplaySession::Impl> observing(CommandEncoder& encoder) {
  auto& r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  // Foreign-thread rejection and final capture closure share this lock. A
  // delayed hook cannot mark a session rejected after it becomes replayable.
  if (!r.active || !r.active->observes(encoder)) return {};
  return r.active;
}
void set_active(const std::shared_ptr<ReplaySession::Impl>& state) {
  auto& r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  if (r.active) {
    throw std::logic_error("another replay capture is active");
  }
  r.active = state;
}
void end_active(const std::shared_ptr<ReplaySession::Impl>& state) noexcept {
  auto& r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  if (r.active == state) {
    r.active.reset();
  }
}
const char* finish_active(const std::shared_ptr<ReplaySession::Impl>& state) {
  auto& r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  const auto* reason = state->rejected.load(std::memory_order_acquire);
  if (r.active != state) throw std::logic_error("replay capture ownership changed");
  r.active.reset();
  return reason;
}
void observe_encoder(CommandEncoder& encoder) noexcept {
  (void)observing(encoder);
}
void retain(CommandEncoder& encoder, const array& value) noexcept {
  auto state = observing(encoder);
  if (!state) return;
  try {
    auto owner = value.data_shared_ptr();
    if (!owner) {
      throw std::invalid_argument("missing MLX allocation");
    }
    state->core.retain(owner->buffer.ptr(), value.buffer_size(), owner);
  } catch (...) {
    state->reject("replay allocation retention failed or exceeded its budget");
  }
}
void capture_graph(CommandEncoder& encoder, cudaGraph_t graph) noexcept {
  auto state = observing(encoder);
  if (!state) return;
  try {
    auto remaining = state->core.limits().graph_nodes;
    const auto nodes = inspect_graph(graph, state->core.limits(), 1, remaining);
    state->core.append(graph, nodes);
  } catch (...) {
    state->reject("replay graph rejected or exceeded its budget");
  }
}
void encoder_destroyed(CommandEncoder& encoder) noexcept {
  auto& r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  if (auto found = r.encoders.find(&encoder); found != r.encoders.end()) {
    if (auto token = found->second.lock()) {
      token->alive.store(false, std::memory_order_release);
    }
    r.encoders.erase(found);
  }
  if (r.active && r.active->lifetime->encoder == &encoder) {
    r.active->reject("replay encoder destroyed during capture");
  }
}
} // namespace replay_detail

ReplaySession::ReplaySession(CommandEncoder& encoder, Limits limits)
    : impl_(std::make_shared<Impl>(encoder, limits)) {}
ReplaySession::~ReplaySession() = default;

void ReplaySession::record(std::uint64_t epoch, const std::vector<array>& bindings,
                           const std::function<void()>& evaluate) {
  auto& encoder = impl_->encoder();
  // Do this before resetting state: nested capture must not alter the outer one.
  replay_detail::set_active(impl_);
  try {
    if (!use_cuda_graphs()) {
      throw std::logic_error("replay requires MLX CUDA graphs");
    }
    // Prior ordinary work is excluded. Hooks reject until begin completes.
    impl_->rejected.store("replay is preparing capture", std::memory_order_release);
    encoder.synchronize();
    impl_->core.begin(epoch, replay_detail::binding_descriptors(bindings));
    impl_->rejected.store(nullptr, std::memory_order_release);
    for (const auto& binding : bindings) {
      replay_detail::retain(encoder, binding);
    }
    evaluate();
    impl_->encoder().synchronize();
    impl_->core.finish();
    if (const auto* reason = replay_detail::finish_active(impl_)) {
      throw std::runtime_error(reason);
    }
  } catch (...) {
    const auto original = std::current_exception();
    replay_detail::end_active(impl_);
    // Flush a partially constructed ordinary graph before releasing its owners.
    if (impl_->lifetime->alive.load(std::memory_order_acquire)) {
      try {
        encoder.synchronize();
      } catch (...) {
        impl_->core.quarantine_after_failed_drain();
        throw;
      }
    }
    impl_->core.invalidate();
    std::rethrow_exception(original);
  }
}

void ReplaySession::replay(std::uint64_t epoch, const std::vector<array>& bindings) {
  (void)impl_->encoder();
  if (replay_detail::active()) {
    throw std::logic_error("cannot replay while a capture is active");
  }
  impl_->core.replay(epoch, replay_detail::binding_descriptors(bindings));
}
void ReplaySession::synchronize() {
  (void)impl_->encoder();
  impl_->core.synchronize();
}
void ReplaySession::invalidate() {
  if (replay_detail::active() == impl_) {
    throw std::logic_error("cannot invalidate inside a capture callback");
  }
  impl_->core.invalidate();
}
void ReplaySession::clear() { impl_->core.clear(); }
ReplaySession::Statistics ReplaySession::statistics() const {
  return impl_->core.statistics();
}
ReplaySession::Phase ReplaySession::phase() const { return impl_->core.phase(); }

} // namespace mlx::core::cu
#endif
