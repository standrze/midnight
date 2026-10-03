// CPU-only collector/publication fixture; no Metal includes or GPU calls.
// Compile explicitly after review; this file is outside normal test discovery.
#include "mlx/backend/metal/command_timing.h"
#include <cassert>
#include <thread>

using namespace mlx::core::metal::command_timing;

void complete(CommandSample* sample, double start) {
  std::thread callback([=] {
    sample->row.gpu_start = start + .01;
    sample->row.gpu_end = start + .02;
    sample->row.callback_observed = start + .03;
    sample->row.status = 4;
    sample->completed.store(true, std::memory_order_release);
  });
  std::thread submission([=] {
    sample->row.commit_start = start;
    sample->row.commit_return = start + .005;
    sample->returned.store(true, std::memory_order_release);
  });
  callback.join();
  submission.join();
}

int main(int argc, char** argv) {
  assert(argc == 2);
  Collector collector(5, 1, argv[1]);
  collector.register_queue(101, 7);
  const double start = now();
  assert(!collector.reserve(101, 1, 128)); // explicit skip
  complete(collector.reserve(101, 2, 256), start + .1);
  auto* callback_only = collector.reserve(101, 3, 256);
  callback_only->row.callback_observed = start + .2;
  callback_only->completed.store(true, std::memory_order_release);
  auto* returned_only = collector.reserve(101, 3, 256);
  returned_only->row.commit_start = start + .3;
  returned_only->row.commit_return = start + .31;
  returned_only->returned.store(true, std::memory_order_release);
  complete(collector.reserve(101, 4, 512), start + .4);
  complete(collector.reserve(101, 5, 512), start + .5);
  assert(!collector.reserve(101, 9, 1024));
  assert(!collector.reserve(101, 9, 1024)); // bounded overflow
  collector.record_wait(101, start + .6, start + .7);
  collector.write_snapshot(); // must omit both partially published rows
}
