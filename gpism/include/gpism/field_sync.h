#pragma once

#include <cstring>

#include "gpism/config.h"
#include "gpism/device_policy.h"
#include "gpism/field2d.h"
#include "gpism/field3d.h"
#include "gpism/field_stag2d.h"
#include "gpism/sync_stats.h"

namespace gpism {

template <typename T>
void sync_host_to_device(Field2D<T>& field) {
#if GPISM_HAVE_CUDA
  if (!device_enabled()) {
    return;
  }
  if (!field.host_staging_data() || field.elements() == 0) {
    return;
  }
  SyncStats::record_h2d(field.elements() * sizeof(T));
  std::memcpy(field.host_staging_data(), field.data(),
              field.elements() * sizeof(T));
  field.copy_host_to_device();
#else
  (void)field;
#endif
}

template <typename T>
void sync_device_to_host(Field2D<T>& field) {
#if GPISM_HAVE_CUDA
  if (!device_enabled()) {
    return;
  }
  if (!field.device_data()) {
    return;
  }
  if (!field.host_staging_data() || field.elements() == 0) {
    return;
  }
  SyncStats::record_d2h(field.elements() * sizeof(T));
  field.copy_device_to_host();
  std::memcpy(field.data(), field.host_staging_data(),
              field.elements() * sizeof(T));
#else
  (void)field;
#endif
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
#if GPISM_HAVE_CUDA
  if (!device_enabled()) {
    return;
  }
  if (!field.host_staging_data() || field.elements() == 0) {
    return;
  }
  SyncStats::record_h2d(field.elements() * sizeof(T));
  std::memcpy(field.host_staging_data(), field.data(),
              field.elements() * sizeof(T));
  field.copy_host_to_device();
#else
  (void)field;
#endif
}

template <typename T>
void sync_device_to_host(Field3D<T>& field) {
#if GPISM_HAVE_CUDA
  if (!device_enabled()) {
    return;
  }
  if (!field.device_data()) {
    return;
  }
  if (!field.host_staging_data() || field.elements() == 0) {
    return;
  }
  SyncStats::record_d2h(field.elements() * sizeof(T));
  field.copy_device_to_host();
  std::memcpy(field.data(), field.host_staging_data(),
              field.elements() * sizeof(T));
#else
  (void)field;
#endif
}

}  // namespace gpism
