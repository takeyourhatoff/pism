#pragma once

#include <cstring>

#include "gpism/field2d.h"
#include "gpism/field3d.h"
#include "gpism/field_stag2d.h"
#include "gpism/sync_audit.h"
#include "gpism/sync_stats.h"

namespace gpism {

template <typename T>
void sync_host_to_device(Field2D<T>& field) {
  if (field.elements() == 0) {
    return;
  }
  field.ensure_host_staging();
  if (!field.host_staging_data()) {
    return;
  }
  const std::size_t bytes = field.elements() * sizeof(T);
  SyncAudit::record(SyncDirection::HostToDevice, bytes, "Field2D");
  SyncStats::record_h2d(bytes);
  std::memcpy(field.host_staging_data(), field.data(),
              bytes);
  field.copy_host_to_device();
}

template <typename T>
void sync_device_to_host(Field2D<T>& field) {
  if (!field.device_data()) {
    return;
  }
  if (field.elements() == 0) {
    return;
  }
  field.ensure_host_staging();
  if (!field.host_staging_data()) {
    return;
  }
  const std::size_t bytes = field.elements() * sizeof(T);
  SyncAudit::record(SyncDirection::DeviceToHost, bytes, "Field2D");
  SyncStats::record_d2h(bytes);
  field.copy_device_to_host();
  std::memcpy(field.data(), field.host_staging_data(),
              bytes);
}

template <typename T>
void sync_host_to_device(FieldStag2D<T>& field) {
  sync_host_to_device(field.component(0));
  sync_host_to_device(field.component(1));
}

template <typename T>
void sync_device_to_host(FieldStag2D<T>& field) {
  sync_device_to_host(field.component(0));
  sync_device_to_host(field.component(1));
}

template <typename T>
void sync_host_to_device(Field3D<T>& field) {
  if (field.elements() == 0) {
    return;
  }
  field.ensure_host_staging();
  if (!field.host_staging_data()) {
    return;
  }
  const std::size_t bytes = field.elements() * sizeof(T);
  SyncAudit::record(SyncDirection::HostToDevice, bytes, "Field3D");
  SyncStats::record_h2d(bytes);
  std::memcpy(field.host_staging_data(), field.data(),
              bytes);
  field.copy_host_to_device();
}

template <typename T>
void sync_device_to_host(Field3D<T>& field) {
  if (!field.device_data()) {
    return;
  }
  if (field.elements() == 0) {
    return;
  }
  field.ensure_host_staging();
  if (!field.host_staging_data()) {
    return;
  }
  const std::size_t bytes = field.elements() * sizeof(T);
  SyncAudit::record(SyncDirection::DeviceToHost, bytes, "Field3D");
  SyncStats::record_d2h(bytes);
  field.copy_device_to_host();
  std::memcpy(field.data(), field.host_staging_data(),
              bytes);
}

}  // namespace gpism
