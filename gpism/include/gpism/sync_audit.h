#pragma once

#include <cstddef>
#include <string>

namespace gpism {

enum class SyncDirection { HostToDevice, DeviceToHost };

class SyncAudit {
public:
  static void enable(bool on);
  static bool enabled();
  static void set_fail_fast(bool on);
  static bool fail_fast();
  static void reset();
  static std::size_t violations();
  static std::size_t bytes_host_to_device();
  static std::size_t bytes_device_to_host();
  static void record(SyncDirection direction, std::size_t bytes,
                     const char* reason = nullptr);

  static void push_scope(bool allow_sync, const std::string& tag);
  static void pop_scope();
  static bool scope_allows_sync();
  static std::string current_scope_tag();
};

class ScopedSyncAudit {
public:
  ScopedSyncAudit(bool allow_sync, const std::string& tag);
  ~ScopedSyncAudit();

  ScopedSyncAudit(const ScopedSyncAudit&) = delete;
  ScopedSyncAudit& operator=(const ScopedSyncAudit&) = delete;

private:
  bool active_;
};

}  // namespace gpism
