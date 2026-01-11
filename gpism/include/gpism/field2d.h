#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "gpism/config.h"
#include "gpism/device_policy.h"

#if GPISM_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace gpism {

template <typename T>
class Field2D {
public:
  Field2D()
      : local_mx_(0),
        local_my_(0),
        ghost_width_(0),
        stride_(0),
#if GPISM_HAVE_CUDA
        host_staging_(nullptr),
        host_staging_count_(0),
#endif
        device_data_(nullptr) {}

  Field2D(int local_mx, int local_my, int ghost_width)
      : local_mx_(local_mx),
        local_my_(local_my),
        ghost_width_(ghost_width),
        stride_(local_mx + 2 * ghost_width),
#if GPISM_HAVE_CUDA
        host_staging_(nullptr),
        host_staging_count_(0),
#endif
        device_data_(nullptr) {
    resize_storage();
  }

  ~Field2D() {
    release_device();
    release_host_staging();
  }

  void resize(int local_mx, int local_my, int ghost_width) {
    local_mx_ = local_mx;
    local_my_ = local_my;
    ghost_width_ = ghost_width;
    stride_ = local_mx + 2 * ghost_width;
    resize_storage();
  }

  int local_mx() const { return local_mx_; }
  int local_my() const { return local_my_; }
  int ghost_width() const { return ghost_width_; }
  int stride() const { return stride_; }

  T* data() { return data_.data(); }
  const T* data() const { return data_.data(); }

  T* device_data() { return device_enabled() ? device_data_ : nullptr; }
  const T* device_data() const {
    return device_enabled() ? device_data_ : nullptr;
  }

  T* host_staging_data() {
#if GPISM_HAVE_CUDA
    return host_staging_;
#else
    return host_staging_.empty() ? nullptr : host_staging_.data();
#endif
  }
  const T* host_staging_data() const {
#if GPISM_HAVE_CUDA
    return host_staging_;
#else
    return host_staging_.empty() ? nullptr : host_staging_.data();
#endif
  }

  void copy_host_to_device() {
#if GPISM_HAVE_CUDA
    if (device_enabled() && device_data_ && host_staging_) {
      cudaMemcpy(device_data_, host_staging_,
                 elements() * sizeof(T), cudaMemcpyHostToDevice);
    }
#endif
  }

  void copy_device_to_host() {
#if GPISM_HAVE_CUDA
    if (device_enabled() && device_data_ && host_staging_) {
      cudaMemcpy(host_staging_, device_data_,
                 elements() * sizeof(T), cudaMemcpyDeviceToHost);
    }
#endif
  }

  std::size_t elements() const { return data_.size(); }

  T& operator()(int i, int j) {
    return data_[index(i, j)];
  }

  const T& operator()(int i, int j) const {
    return data_[index(i, j)];
  }

  bool has_device_data() const {
    return device_enabled() && device_data_ != nullptr;
  }

  void fill(const T& value) {
    std::fill(data_.begin(), data_.end(), value);
  }

private:
  int index(int i, int j) const {
    return (j + ghost_width_) * stride_ + (i + ghost_width_);
  }

  void resize_storage() {
    if (local_mx_ <= 0 || local_my_ <= 0) {
      data_.clear();
      release_host_staging();
      release_device();
      return;
    }
    const size_t total = static_cast<size_t>(stride_) *
                         static_cast<size_t>(local_my_ + 2 * ghost_width_);
    data_.assign(total, T{});
    allocate_host_staging(total);
    allocate_device(total);
  }

  void allocate_host_staging(std::size_t elements) {
#if GPISM_HAVE_CUDA
    release_host_staging();
    if (elements == 0) {
      return;
    }
    host_staging_count_ = elements;
    cudaHostAlloc(reinterpret_cast<void**>(&host_staging_),
                  elements * sizeof(T), cudaHostAllocDefault);
#else
    host_staging_.assign(elements, T{});
#endif
  }

  void release_host_staging() {
#if GPISM_HAVE_CUDA
    if (host_staging_ != nullptr) {
      cudaFreeHost(host_staging_);
      host_staging_ = nullptr;
      host_staging_count_ = 0;
    }
#else
    host_staging_.clear();
#endif
  }

  void allocate_device(std::size_t elements) {
#if GPISM_HAVE_CUDA
    release_device();
    if (elements == 0) {
      return;
    }
    cudaMalloc(reinterpret_cast<void**>(&device_data_), elements * sizeof(T));
#else
    (void)elements;
#endif
  }

  void release_device() {
#if GPISM_HAVE_CUDA
    if (device_data_ != nullptr) {
      cudaFree(device_data_);
      device_data_ = nullptr;
    }
#endif
  }

  int local_mx_;
  int local_my_;
  int ghost_width_;
  int stride_;
  std::vector<T> data_;
#if GPISM_HAVE_CUDA
  T* host_staging_;
  std::size_t host_staging_count_;
#else
  std::vector<T> host_staging_;
#endif
  T* device_data_;
};

}  // namespace gpism
