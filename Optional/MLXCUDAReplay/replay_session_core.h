#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace midnight::mlx_replay {

// Explicit bounds apply to retained allocations, not just tensor view sizes.
struct Limits {
  std::size_t retained_bytes = std::size_t{8} << 30;
  std::size_t retained_allocations = 131072;
  std::size_t graph_nodes = 262144;
  std::size_t graph_chunks = 4096;
  std::size_t graph_depth = 16;
};

struct Binding {
  const void* allocation = nullptr;
  const void* view = nullptr;
  std::size_t bytes = 0;
  int dtype = 0;
  std::vector<int> shape;
  std::vector<std::int64_t> strides;

  bool operator==(const Binding& other) const {
    return allocation == other.allocation && view == other.view &&
        bytes == other.bytes && dtype == other.dtype && shape == other.shape &&
        strides == other.strides;
  }
};

struct Statistics {
  std::size_t retained_bytes = 0;
  std::size_t retained_allocations = 0;
  std::size_t graph_nodes = 0;
  std::size_t graph_chunks = 0;
  std::uint64_t launches = 0;
};

enum class Phase { empty, recording, ready, invalidated, quarantined };

// The CUDA adapter and the executable CPU tests use this same ownership/state
// implementation. Backend operations are begin, append, instantiate, launch,
// drain and reset. append receives an already validated backend graph handle.
template <class Backend>
class SessionCore {
 public:
  SessionCore(Backend backend, Limits limits)
      : resources_(std::make_unique<Resources>(std::move(backend))),
        limits_(limits), creator_(std::this_thread::get_id()) {
    if (!limits.retained_bytes || !limits.retained_allocations ||
        !limits.graph_nodes || !limits.graph_chunks || !limits.graph_depth) {
      throw std::invalid_argument("replay limits must be positive");
    }
  }

  SessionCore(const SessionCore&) = delete;
  SessionCore& operator=(const SessionCore&) = delete;
  SessionCore(SessionCore&&) = delete;
  SessionCore& operator=(SessionCore&&) = delete;

  ~SessionCore() noexcept {
    try {
      release_resources();
    } catch (...) {
      // A failed GPU drain gives no permission to free borrowed addresses.
      // Quarantine this one session for process lifetime on a backend failure.
      // Normal cancellation, invalidation and destruction release everything.
      (void)resources_.release();
    }
  }

  void begin(std::uint64_t epoch, std::vector<Binding> bindings) {
    require_thread();
    require_resources();
    if (phase_ == Phase::recording) {
      throw std::logic_error("nested replay capture");
    }
    if (phase_ == Phase::invalidated) {
      throw std::logic_error("invalidated replay session must be cleared");
    }
    release_resources();
    bindings_ = std::move(bindings);
    epoch_ = epoch;
    phase_ = Phase::recording;
    resources_->backend.begin();
  }

  void retain(const void* allocation, std::size_t bytes,
              std::shared_ptr<void> owner) {
    require_recording();
    if (!owner || !allocation) {
      throw std::invalid_argument("replay cannot retain missing storage");
    }
    auto found = resources_->owners.find(allocation);
    if (found != resources_->owners.end()) {
      if (found->second.bytes != bytes) {
        throw std::logic_error("replay allocation size changed during capture");
      }
      return;
    }
    if (stats_.retained_allocations == limits_.retained_allocations ||
        bytes > limits_.retained_bytes - stats_.retained_bytes) {
      throw std::length_error("replay retained allocation budget exceeded");
    }
    resources_->owners.emplace(allocation, Retained{bytes, std::move(owner)});
    stats_.retained_bytes += bytes;
    ++stats_.retained_allocations;
  }

  template <class Graph>
  void append(Graph graph, std::size_t nodes) {
    require_recording();
    if (!nodes) {
      return;
    }
    if (stats_.graph_chunks == limits_.graph_chunks ||
        nodes > limits_.graph_nodes - stats_.graph_nodes) {
      throw std::length_error("replay graph budget exceeded");
    }
    resources_->backend.append(graph);
    stats_.graph_nodes += nodes;
    ++stats_.graph_chunks;
  }

  void finish() {
    require_recording();
    resources_->backend.drain();
    if (!stats_.graph_chunks) {
      throw std::logic_error("empty replay capture");
    }
    resources_->backend.instantiate();
    phase_ = Phase::ready;
  }

  void replay(std::uint64_t epoch, const std::vector<Binding>& bindings) {
    require_thread();
    if (phase_ != Phase::ready) {
      throw std::logic_error("replay session is not ready");
    }
    if (epoch != epoch_ || bindings != bindings_) {
      invalidate();
      throw std::logic_error("replay model epoch or storage binding changed");
    }
    try {
      resources_->backend.launch();
      ++stats_.launches;
      // Raw graph launch does not update MLX's array completion events. The
      // initial supported API is synchronous so output reads/consumers are safe
      // after return without pretending eval(existing_array) tracks replay.
      resources_->backend.drain();
    } catch (...) {
      phase_ = Phase::invalidated;
      throw;
    }
  }

  void synchronize() {
    require_thread();
    require_resources();
    resources_->backend.drain();
  }

  void invalidate() {
    require_thread();
    require_resources();
    phase_ = Phase::invalidated;
    release_resources();
  }

  void clear() {
    require_thread();
    require_resources();
    if (phase_ == Phase::recording) {
      throw std::logic_error("cannot clear an active replay capture");
    }
    release_resources();
    phase_ = Phase::empty;
  }

  // An encoder drain failure may leave an unsubmitted graph referencing these
  // addresses. A subsequent raw stream drain alone cannot prove them unused.
  // Retain the entire failed session for process lifetime; never reuse it.
  void quarantine_after_failed_drain() {
    require_thread();
    (void)resources_.release();
    phase_ = Phase::quarantined;
  }

  Phase phase() const { require_thread(); return phase_; }
  Statistics statistics() const { require_thread(); return stats_; }
  const Limits& limits() const { return limits_; }

 private:
  struct Retained {
    std::size_t bytes;
    std::shared_ptr<void> owner;
  };
  struct Resources {
    explicit Resources(Backend value) : backend(std::move(value)) {}
    // Backend graph objects must be destroyed before the referenced storage.
    std::unordered_map<const void*, Retained> owners;
    Backend backend;
  };

  void require_thread() const {
    if (std::this_thread::get_id() != creator_) {
      throw std::logic_error("replay session used from another thread");
    }
  }
  void require_recording() const {
    require_thread();
    if (phase_ != Phase::recording) {
      throw std::logic_error("replay session is not recording");
    }
  }
  void require_resources() const {
    if (!resources_) {
      throw std::logic_error("replay resources quarantined after backend failure");
    }
  }
  void release_resources() {
    if (!resources_) return;
    resources_->backend.drain();
    resources_->backend.reset();
    resources_->owners.clear();
    bindings_.clear();
    stats_ = {};
  }

  std::unique_ptr<Resources> resources_;
  Limits limits_;
  std::thread::id creator_;
  Phase phase_ = Phase::empty;
  std::uint64_t epoch_ = 0;
  std::vector<Binding> bindings_;
  Statistics stats_;
};

} // namespace midnight::mlx_replay
