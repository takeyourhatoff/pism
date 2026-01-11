#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "gpism/config.h"

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
        device_data_(nullptr) {}

  Field2D(int local_mx, int local_my, int ghost_width)
      : local_mx_(local_mx),
        local_my_(local_my),
        ghost_width_(ghost_width),
        stride_(local_mx + 2 * ghost_width),
        device_data_(nullptr) {
    resize_storage();
  }

  ~Field2D() { release_device(); }

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

  T* device_data() { return device_data_; }
  const T* device_data() const { return device_data_; }

  T* host_staging_data() { return host_staging_.empty() ? nullptr : host_staging_.data(); }
  const T* host_staging_data() const {
    return host_staging_.empty() ? nullptr : host_staging_.data();
  }

  std::size_t elements() const { return data_.size(); }

  T& operator()(int i, int j) {
    return data_[index(i, j)];
  }

  const T& operator()(int i, int j) const {
    return data_[index(i, j)];
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
      host_staging_.clear();
      release_device();
      return;
    }
    const size_t total = static_cast<size_t>(stride_) *
                         static_cast<size_t>(local_my_ + 2 * ghost_width_);
    data_.assign(total, T{});
    host_staging_.assign(total, T{});
    allocate_device(total);
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
  std::vector<T> host_staging_;
  T* device_data_;
};

}  // namespace gpism
