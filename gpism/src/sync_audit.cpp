#include "gpism/sync_audit.h"

#include <atomic>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gpism {
namespace {

struct ScopeState {
  bool allow_sync = true;
  std::string tag;
};

std::atomic<bool> g_enabled{false};
std::atomic<bool> g_fail_fast{false};
std::atomic<std::size_t> g_violations{0};
std::atomic<std::size_t> g_h2d_bytes{0};
std::atomic<std::size_t> g_d2h_bytes{0};

thread_local std::vector<ScopeState> g_scopes;

std::string direction_name(SyncDirection direction) {
  switch (direction) {
    case SyncDirection::HostToDevice:
      return "host->device";
    case SyncDirection::DeviceToHost:
      return "device->host";
  }
  return "unknown";
}

}  // namespace

void SyncAudit::enable(bool on) { g_enabled.store(on); }

bool SyncAudit::enabled() { return g_enabled.load(); }

void SyncAudit::set_fail_fast(bool on) { g_fail_fast.store(on); }

bool SyncAudit::fail_fast() { return g_fail_fast.load(); }

void SyncAudit::reset() {
  g_violations.store(0);
  g_h2d_bytes.store(0);
  g_d2h_bytes.store(0);
}

std::size_t SyncAudit::violations() { return g_violations.load(); }

std::size_t SyncAudit::bytes_host_to_device() { return g_h2d_bytes.load(); }

std::size_t SyncAudit::bytes_device_to_host() { return g_d2h_bytes.load(); }

void SyncAudit::record(SyncDirection direction, std::size_t bytes,
                       const char* reason) {
  if (!enabled()) {
    return;
  }

  if (direction == SyncDirection::HostToDevice) {
    g_h2d_bytes.fetch_add(bytes);
  } else if (direction == SyncDirection::DeviceToHost) {
    g_d2h_bytes.fetch_add(bytes);
  }

  if (scope_allows_sync()) {
    return;
  }

  g_violations.fetch_add(1);
  if (!fail_fast()) {
    return;
  }

  const std::string tag = current_scope_tag();
  std::string message = "sync audit violation: " + direction_name(direction);
  if (reason && reason[0] != '\0') {
    message += " (" + std::string(reason) + ")";
  }
  if (!tag.empty()) {
    message += " in scope '" + tag + "'";
  }
  throw std::runtime_error(message);
}

void SyncAudit::push_scope(bool allow_sync, const std::string& tag) {
  g_scopes.push_back({allow_sync, tag});
}

void SyncAudit::pop_scope() {
  if (!g_scopes.empty()) {
    g_scopes.pop_back();
  }
}

bool SyncAudit::scope_allows_sync() {
  if (g_scopes.empty()) {
    return true;
  }
  return g_scopes.back().allow_sync;
}

std::string SyncAudit::current_scope_tag() {
  if (g_scopes.empty()) {
    return {};
  }
  return g_scopes.back().tag;
}

ScopedSyncAudit::ScopedSyncAudit(bool allow_sync, const std::string& tag)
    : active_(SyncAudit::enabled()) {
  if (active_) {
    SyncAudit::push_scope(allow_sync, tag);
  }
}

ScopedSyncAudit::~ScopedSyncAudit() {
  if (active_) {
    SyncAudit::pop_scope();
  }
}

}  // namespace gpism
