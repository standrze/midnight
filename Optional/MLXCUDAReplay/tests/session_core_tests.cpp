#include "../replay_session_core.h"

#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

using namespace midnight::mlx_replay;

namespace {
void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template <class Operation>
void rejects(Operation operation) {
  bool rejected = false;
  try { operation(); } catch (const std::exception&) { rejected = true; }
  check(rejected, "expected rejection");
}

struct Log {
  std::vector<std::string> events;
  bool pending = false;
  bool fail_instantiate = false;
  bool fail_drain = false;
};

struct Backend {
  std::shared_ptr<Log> log;
  void begin() { log->events.push_back("begin"); }
  void append(int) { log->events.push_back("append"); }
  void instantiate() {
    log->events.push_back("instantiate");
    if (log->fail_instantiate) throw std::runtime_error("injected instantiate error");
  }
  void launch() { log->pending = true; log->events.push_back("launch"); }
  void drain() {
    log->events.push_back("drain");
    if (log->fail_drain) throw std::runtime_error("injected drain error");
    log->pending = false;
  }
  void reset() {
    check(!log->pending, "graph released before pending work drained");
    log->events.push_back("reset");
  }
};
using Core = SessionCore<Backend>;
static_assert(!std::is_copy_constructible_v<Core>);
static_assert(!std::is_move_constructible_v<Core>);

struct Buffer {
  std::shared_ptr<Log> log;
  ~Buffer() {
    check(!log->pending, "buffer released while GPU work is pending");
    check(!log->events.empty() && log->events.back() == "reset",
          "buffer released before graph reset");
    log->events.push_back("buffer_destroyed");
  }
};

Limits limits() { return {1024, 4, 16, 4, 4}; }
Binding binding(const void* pointer) {
  return {pointer, pointer, 128, 3, {1, 32}, {32, 1}};
}
void record(Core& session, const Binding& input,
            const std::shared_ptr<void>& owner, std::size_t nodes = 3) {
  session.begin(7, {input});
  session.retain(input.allocation, input.bytes, owner);
  session.append(1, nodes);
  session.finish();
}

void retention_and_deduplication() {
  auto log = std::make_shared<Log>();
  auto owner = std::make_shared<int>(1);
  std::weak_ptr<int> weak = owner;
  const auto input = binding(owner.get());
  Core session(Backend{log}, limits());
  session.begin(7, {input});
  session.retain(owner.get(), 128, owner);
  session.retain(owner.get(), 128, owner);
  check(session.statistics().retained_allocations == 1, "duplicate allocation retained twice");
  check(session.statistics().retained_bytes == 128, "alias double-counted bytes");
  session.append(1, 2);
  session.append(2, 3);
  session.finish();
  owner.reset();
  check(!weak.expired(), "capture did not retain storage");
  session.replay(7, {input});
  session.replay(7, {input});
  check(session.statistics().launches == 2, "launch count mismatch");
  session.clear();
  check(weak.expired(), "clear retained storage");
  check(session.phase() == Phase::empty, "clear did not reset phase");
  check(session.statistics().retained_bytes == 0, "clear retained byte accounting");
}

void drain_precedes_destruction() {
  auto log = std::make_shared<Log>();
  std::weak_ptr<Buffer> weak;
  {
    auto owner = std::make_shared<Buffer>();
    owner->log = log;
    weak = owner;
    const auto input = binding(owner.get());
    Core session(Backend{log}, limits());
    record(session, input, owner);
    owner.reset();
    log->fail_drain = true;
    rejects([&] { session.replay(7, {input}); });
    log->fail_drain = false;
  }
  check(weak.expired(), "destructor retained storage");
  check(log->events.back() == "buffer_destroyed", "unexpected cleanup order");
}

void invalidation_and_recapture() {
  auto log = std::make_shared<Log>();
  auto owner = std::make_shared<int>(1);
  const auto input = binding(owner.get());
  Core session(Backend{log}, limits());
  record(session, input, owner);
  session.replay(7, {input});
  rejects([&] { session.replay(8, {input}); });
  check(session.phase() == Phase::invalidated, "epoch mismatch did not invalidate");
  rejects([&] { session.begin(8, {input}); });
  session.clear();
  record(session, input, owner);
  session.replay(7, {input});
  session.invalidate();
  rejects([&] { session.replay(7, {input}); });
}

void changed_bindings_reject() {
  for (int variation = 0; variation < 7; ++variation) {
    auto log = std::make_shared<Log>();
    auto owner = std::make_shared<int>(1);
    auto other = std::make_shared<int>(2);
    const auto input = binding(owner.get());
    Core session(Backend{log}, limits());
    record(session, input, owner);
    auto changed = input;
    switch (variation) {
      case 0: changed.allocation = other.get(); break;
      case 1: changed.view = other.get(); break;
      case 2: ++changed.bytes; break;
      case 3: ++changed.dtype; break;
      case 4: changed.shape = {2, 16}; break;
      case 5: changed.strides = {64, 2}; break;
    }
    const auto bindings = variation == 6 ? std::vector<Binding>{} : std::vector{changed};
    rejects([&] { session.replay(7, bindings); });
    check(session.phase() == Phase::invalidated, "changed binding stayed replayable");
    check(session.statistics().retained_allocations == 0, "invalidated binding retained owners");
  }
}

void capture_bounds_and_failures() {
  auto log = std::make_shared<Log>();
  auto owner = std::make_shared<int>(1);
  const auto input = binding(owner.get());
  Core session(Backend{log}, limits());
  rejects([&] { session.replay(7, {input}); });
  session.begin(7, {input});
  rejects([&] { session.begin(7, {input}); });
  rejects([&] { session.clear(); });
  rejects([&] { session.finish(); });
  rejects([&] { session.retain(nullptr, 8, owner); });
  rejects([&] { session.retain(owner.get(), 8, {}); });
  session.retain(owner.get(), 128, owner);
  rejects([&] { session.retain(owner.get(), 256, owner); });
  rejects([&] { session.append(1, 17); });
  check(session.statistics().graph_chunks == 0, "rejected graph changed count");
  session.append(1, 16);
  rejects([&] { session.append(2, 1); });
  log->fail_instantiate = true;
  rejects([&] { session.finish(); });
  session.invalidate();
  check(session.statistics().retained_allocations == 0, "failed capture retained storage");
  log->fail_instantiate = false;
  session.clear();
  record(session, input, owner);
}

void byte_allocation_and_chunk_limits() {
  auto log = std::make_shared<Log>();
  std::vector<std::shared_ptr<int>> owners;
  for (int i = 0; i < 6; ++i) owners.push_back(std::make_shared<int>(i));
  Core session(Backend{log}, limits());
  session.begin(7, {});
  session.retain(owners[0].get(), 1024, owners[0]);
  rejects([&] { session.retain(owners[1].get(), 1, owners[1]); });
  session.invalidate();
  session.clear();
  session.begin(7, {});
  for (int i = 0; i < 4; ++i) session.retain(owners[i].get(), 1, owners[i]);
  rejects([&] { session.retain(owners[4].get(), 1, owners[4]); });
  for (int i = 0; i < 4; ++i) session.append(i, 1);
  rejects([&] { session.append(4, 1); });
  session.invalidate();

  auto huge_limits = limits();
  huge_limits.retained_bytes = std::numeric_limits<std::size_t>::max();
  Core huge(Backend{log}, huge_limits);
  huge.begin(7, {});
  huge.retain(owners[0].get(), huge_limits.retained_bytes, owners[0]);
  rejects([&] { huge.retain(owners[1].get(), 1, owners[1]); });
  huge.invalidate();
}

void wrong_thread_rejects_without_mutation() {
  auto log = std::make_shared<Log>();
  auto owner = std::make_shared<int>(1);
  const auto input = binding(owner.get());
  Core session(Backend{log}, limits());
  record(session, input, owner);
  const auto count = log->events.size();
  std::exception_ptr failure;
  std::thread other([&] {
    try {
      rejects([&] { session.replay(7, {input}); });
      rejects([&] { session.clear(); });
      rejects([&] { session.invalidate(); });
      rejects([&] { session.synchronize(); });
      rejects([&] { (void)session.statistics(); });
    } catch (...) { failure = std::current_exception(); }
  });
  other.join();
  if (failure) std::rethrow_exception(failure);
  check(log->events.size() == count, "wrong-thread call reached backend");
  session.replay(7, {input});
}

void failed_drain_keeps_owners_until_retry() {
  auto log = std::make_shared<Log>();
  auto owner = std::make_shared<int>(1);
  std::weak_ptr<int> weak = owner;
  const auto input = binding(owner.get());
  Core session(Backend{log}, limits());
  record(session, input, owner);
  owner.reset();
  session.replay(7, {input});
  log->fail_drain = true;
  rejects([&] { session.invalidate(); });
  check(!weak.expired(), "failed GPU drain freed storage");
  rejects([&] { session.replay(7, {input}); });
  log->fail_drain = false;
  session.clear();
  check(weak.expired(), "successful retry did not release storage");
}

void failed_encoder_drain_quarantines_through_destruction() {
  auto log = std::make_shared<Log>();
  auto owner = std::make_shared<int>(1);
  std::weak_ptr<int> weak = owner;
  const auto input = binding(owner.get());
  {
    Core session(Backend{log}, limits());
    session.begin(7, {input});
    session.retain(owner.get(), input.bytes, owner);
    session.append(1, 1);
    owner.reset();
    // Models an encoder exception with a still-unsubmitted graph. Even a later
    // successful CUDA stream drain would not make those addresses safe to free.
    session.quarantine_after_failed_drain();
    check(session.phase() == Phase::quarantined, "quarantine phase missing");
    const auto events = log->events.size();
    rejects([&] { session.clear(); });
    rejects([&] { session.begin(7, {input}); });
    rejects([&] { session.replay(7, {input}); });
    rejects([&] { session.invalidate(); });
    check(log->events.size() == events, "quarantined session touched backend");
  }
  check(!weak.expired(), "destructor freed quarantined storage");
  // This single tiny allocation is intentionally retained for process lifetime,
  // exactly as a session with an unrecoverable GPU drain failure must be.
}
} // namespace

int main() {
  try {
    retention_and_deduplication();
    drain_precedes_destruction();
    invalidation_and_recapture();
    changed_bindings_reject();
    capture_bounds_and_failures();
    byte_allocation_and_chunk_limits();
    wrong_thread_rejects_without_mutation();
    failed_drain_keeps_owners_until_retry();
    failed_encoder_drain_quarantines_through_destruction();
    std::cout << "9 replay ownership/state suites passed (CPU fake backend; no GPU claim)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
