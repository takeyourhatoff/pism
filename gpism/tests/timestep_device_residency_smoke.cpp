#include "gpism/config.h"
#include "gpism/field_sync.h"
#include "gpism/sync_audit.h"

#include <iostream>
#include <stdexcept>

int main() {
  const int mx = 4;
  const int my = 4;
  const int gw = 1;
  gpism::Field2D<double> field(mx, my, gw);
  field.fill(1.0);
  gpism::sync_host_to_device(field);

  gpism::SyncAudit::enable(true);
  gpism::SyncAudit::set_fail_fast(false);
  gpism::SyncAudit::reset();

  {
    gpism::ScopedSyncAudit allowed_scope(true, "allowed");
    gpism::sync_device_to_host(field);
  }
  if (gpism::SyncAudit::violations() != 0) {
    std::cerr << "unexpected violation in allowed sync scope\n";
    return 1;
  }

  {
    gpism::ScopedSyncAudit disallow_scope(false, "disallowed");
    gpism::sync_device_to_host(field);
  }
  if (gpism::SyncAudit::violations() == 0) {
    std::cerr << "expected at least one violation in disallowed scope\n";
    return 1;
  }

  gpism::SyncAudit::set_fail_fast(true);
  bool threw = false;
  try {
    gpism::ScopedSyncAudit disallow_scope(false, "fail-fast");
    gpism::sync_device_to_host(field);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  if (!threw) {
    std::cerr << "expected fail-fast sync audit exception\n";
    return 1;
  }

  gpism::SyncAudit::enable(false);
  gpism::SyncAudit::set_fail_fast(false);
  std::cout << "timestep_device_residency_smoke passed\n";
  return 0;
}
