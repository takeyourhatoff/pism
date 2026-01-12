#include "gpism/sync_stats.h"

#include <atomic>
#include <cstdlib>

namespace gpism {
namespace {

std::atomic<bool> g_enabled{false};
std::atomic<int> g_env_checked{0};
std::atomic<std::size_t> g_h2d_calls{0};
std::atomic<std::size_t> g_d2h_calls{0};
std::atomic<std::size_t> g_h2d_bytes{0};
std::atomic<std::size_t> g_d2h_bytes{0};

bool env_enabled(const char* var) {
  const char* value = std::getenv(var);
  if (!value) {
    return false;
  }
  return std::atoi(value) != 0;
}

void check_env_once() {
  int expected = 0;
  if (!g_env_checked.compare_exchange_strong(expected, 1)) {
    return;
  }
  if (env_enabled("GPISM_SYNC_AUDIT")) {
    g_enabled.store(true);
  }
}

}  // namespace

void SyncStats::enable(bool on) { g_enabled.store(on); }

bool SyncStats::enabled() {
  check_env_once();
  return g_enabled.load();
}

void SyncStats::reset() {
  g_h2d_calls.store(0);
  g_d2h_calls.store(0);
  g_h2d_bytes.store(0);
  g_d2h_bytes.store(0);
}

void SyncStats::record_h2d(std::size_t bytes) {
  if (!enabled() || bytes == 0) {
    return;
  }
  g_h2d_calls.fetch_add(1);
  g_h2d_bytes.fetch_add(bytes);
}

void SyncStats::record_d2h(std::size_t bytes) {
  if (!enabled() || bytes == 0) {
    return;
  }
  g_d2h_calls.fetch_add(1);
  g_d2h_bytes.fetch_add(bytes);
}

std::size_t SyncStats::h2d_calls() { return g_h2d_calls.load(); }
std::size_t SyncStats::d2h_calls() { return g_d2h_calls.load(); }
std::size_t SyncStats::h2d_bytes() { return g_h2d_bytes.load(); }
std::size_t SyncStats::d2h_bytes() { return g_d2h_bytes.load(); }

}  // namespace gpism
