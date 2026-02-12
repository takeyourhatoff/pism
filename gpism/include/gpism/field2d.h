#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

namespace gpism {

template <typename T>
class Field2D {
public:
  Field2D()
      : local_mx_(0),
        local_my_(0),
        ghost_width_(0),
        stride_(0),
        host_staging_(nullptr),
        host_staging_count_(0),
        host_staging_pinned_(false),
        device_data_(nullptr) {}

  Field2D(int local_mx, int local_my, int ghost_width)
      : local_mx_(local_mx),
        local_my_(local_my),
        ghost_width_(ghost_width),
        stride_(local_mx + 2 * ghost_width),
        host_staging_(nullptr),
        host_staging_count_(0),
        host_staging_pinned_(false),
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

  T* device_data() { return device_data_; }
  const T* device_data() const { return device_data_; }

  T* host_staging_data() {
    ensure_host_staging();
    return host_staging_;
  }
  const T* host_staging_data() const { return host_staging_; }

  void ensure_host_staging() {
    if (host_staging_ != nullptr || host_staging_count_ == 0) {
      return;
    }
    allocate_host_staging(host_staging_count_);
  }

  void copy_host_to_device() {
    ensure_host_staging();
    if (device_data_ && host_staging_) {
      cudaMemcpy(device_data_, host_staging_, elements() * sizeof(T),
                 cudaMemcpyHostToDevice);
    }
  }

  void copy_device_to_host() {
    ensure_host_staging();
    if (device_data_ && host_staging_) {
      cudaMemcpy(host_staging_, device_data_, elements() * sizeof(T),
                 cudaMemcpyDeviceToHost);
    }
  }

  std::size_t elements() const { return data_.size(); }

  T& operator()(int i, int j) { return data_[index(i, j)]; }

  const T& operator()(int i, int j) const { return data_[index(i, j)]; }

  bool has_device_data() const { return device_data_ != nullptr; }

  void fill(const T& value) { std::fill(data_.begin(), data_.end(), value); }

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
    release_host_staging();
    host_staging_count_ = total;
    allocate_device(total);
  }

  void allocate_host_staging(std::size_t elements) {
    release_host_staging();
    if (elements == 0) {
      return;
    }
    host_staging_count_ = elements;
    cudaError_t err = cudaHostAlloc(reinterpret_cast<void**>(&host_staging_),
                                    elements * sizeof(T), cudaHostAllocDefault);
    if (err == cudaSuccess) {
      host_staging_pinned_ = true;
      host_staging_fallback_.clear();
      return;
    }
    host_staging_pinned_ = false;
    host_staging_fallback_.assign(elements, T{});
    host_staging_ = host_staging_fallback_.data();
  }

  void release_host_staging() {
    if (host_staging_pinned_ && host_staging_ != nullptr) {
      cudaFreeHost(host_staging_);
    }
    host_staging_ = nullptr;
    host_staging_count_ = 0;
    host_staging_pinned_ = false;
    host_staging_fallback_.clear();
  }

  void allocate_device(std::size_t elements) {
    release_device();
    if (elements == 0) {
      return;
    }
    cudaMalloc(reinterpret_cast<void**>(&device_data_), elements * sizeof(T));
    cudaMemset(device_data_, 0, elements * sizeof(T));
  }

  void release_device() {
    if (device_data_ != nullptr) {
      cudaFree(device_data_);
      device_data_ = nullptr;
    }
  }

  int local_mx_;
  int local_my_;
  int ghost_width_;
  int stride_;
  std::vector<T> data_;
  T* host_staging_;
  std::size_t host_staging_count_;
  bool host_staging_pinned_;
  std::vector<T> host_staging_fallback_;
  T* device_data_;
};

}  // namespace gpism
