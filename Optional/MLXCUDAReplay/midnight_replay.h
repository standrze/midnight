#pragma once

// Installed only into an isolated MLX CUDA source tree by prepare.py. The
// CommandEncoder overlay compiles to the original path unless explicitly enabled.
#if defined(MIDNIGHT_MLX_CUDA_REPLAY)

#include "mlx/array.h"
#include "mlx/backend/cuda/cuda_utils.h"
#include "mlx/backend/cuda/replay_session_core.h"

#include <functional>
#include <memory>
#include <vector>

namespace mlx::core::cu {
class CommandEncoder;

class MLX_API ReplaySession {
 public:
  using Limits = midnight::mlx_replay::Limits;
  using Statistics = midnight::mlx_replay::Statistics;
  using Phase = midnight::mlx_replay::Phase;

  explicit ReplaySession(CommandEncoder& encoder, Limits limits = {});
  ~ReplaySession();
  ReplaySession(const ReplaySession&) = delete;
  ReplaySession& operator=(const ReplaySession&) = delete;
  ReplaySession(ReplaySession&&) = delete;
  ReplaySession& operator=(ReplaySession&&) = delete;

  // evaluate must synchronously construct AND evaluate fresh MLX outputs on the
  // supplied encoder. Bindings enumerate stable inputs, weights and KV storage;
  // the caller must increment epoch before any in-place model mutation.
  void record(std::uint64_t epoch, const std::vector<array>& bindings,
              const std::function<void()>& evaluate);
  // Synchronous: drains the replay stream before return. Raw CUDA launches do
  // not update MLX array completion events; eval(existing_output) is not a fence.
  // Read outputs or schedule cross-stream consumers only after this returns.
  void replay(std::uint64_t epoch, const std::vector<array>& bindings);
  void synchronize();
  void invalidate();
  void clear();
  Statistics statistics() const;
  Phase phase() const;

  struct Impl;

 private:
  std::shared_ptr<Impl> impl_;
};

namespace replay_detail {
// Hooks never throw through ordinary MLX evaluation: rejection is reported by
// record after all ordinary work drains. Other encoders/threads reject capture.
void observe_encoder(CommandEncoder& encoder) noexcept;
void retain(CommandEncoder& encoder, const array& value) noexcept;
void capture_graph(CommandEncoder& encoder, cudaGraph_t graph) noexcept;
void encoder_destroyed(CommandEncoder& encoder) noexcept;
} // namespace replay_detail
} // namespace mlx::core::cu

#endif
