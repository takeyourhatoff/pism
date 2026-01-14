#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "gpism/config.h"
#include "gpism/device_policy.h"

#if GPISM_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace gpism {

template <typename T>
class Field3D {
public:
  Field3D()
      : local_mx_(0),
        local_my_(0),
        local_mz_(0),
        ghost_width_(0),
        stride_x_(0),
#if GPISM_HAVE_CUDA
        host_staging_(nullptr),
        host_staging_count_(0),
        host_staging_pinned_(false),
#endif
        device_data_(nullptr) {}

  Field3D(int local_mx, int local_my, int local_mz, int ghost_width)
      : local_mx_(local_mx),
        local_my_(local_my),
        local_mz_(local_mz),
        ghost_width_(ghost_width),
        stride_x_(local_mx + 2 * ghost_width),
#if GPISM_HAVE_CUDA
        host_staging_(nullptr),
        host_staging_count_(0),
        host_staging_pinned_(false),
#endif
        device_data_(nullptr) {
    resize_storage();
  }

  Field3D(const Field3D&) = delete;
  Field3D& operator=(const Field3D&) = delete;

  Field3D(Field3D&& other) noexcept
      : local_mx_(other.local_mx_),
        local_my_(other.local_my_),
        local_mz_(other.local_mz_),
        ghost_width_(other.ghost_width_),
        stride_x_(other.stride_x_),
        data_(std::move(other.data_)),
#if GPISM_HAVE_CUDA
        host_staging_(other.host_staging_),
        host_staging_count_(other.host_staging_count_),
        host_staging_pinned_(other.host_staging_pinned_),
        host_staging_fallback_(std::move(other.host_staging_fallback_)),
#else
        host_staging_(std::move(other.host_staging_)),
#endif
        device_data_(other.device_data_) {
    other.local_mx_ = 0;
    other.local_my_ = 0;
    other.local_mz_ = 0;
    other.ghost_width_ = 0;
    other.stride_x_ = 0;
#if GPISM_HAVE_CUDA
    other.host_staging_ = nullptr;
    other.host_staging_count_ = 0;
    other.host_staging_pinned_ = false;
    other.host_staging_fallback_.clear();
#endif
    other.device_data_ = nullptr;
  }

  Field3D& operator=(Field3D&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    release_device();
    release_host_staging();
    local_mx_ = other.local_mx_;
    local_my_ = other.local_my_;
    local_mz_ = other.local_mz_;
    ghost_width_ = other.ghost_width_;
    stride_x_ = other.stride_x_;
    data_ = std::move(other.data_);
#if GPISM_HAVE_CUDA
    host_staging_ = other.host_staging_;
    host_staging_count_ = other.host_staging_count_;
    host_staging_pinned_ = other.host_staging_pinned_;
    host_staging_fallback_ = std::move(other.host_staging_fallback_);
    other.host_staging_ = nullptr;
    other.host_staging_count_ = 0;
    other.host_staging_pinned_ = false;
    other.host_staging_fallback_.clear();
#else
    host_staging_ = std::move(other.host_staging_);
#endif
    device_data_ = other.device_data_;
    other.device_data_ = nullptr;
    other.local_mx_ = 0;
    other.local_my_ = 0;
    other.local_mz_ = 0;
    other.ghost_width_ = 0;
    other.stride_x_ = 0;
    return *this;
  }

  ~Field3D() {
    release_device();
    release_host_staging();
  }

  void resize(int local_mx, int local_my, int local_mz, int ghost_width) {
    local_mx_ = local_mx;
    local_my_ = local_my;
    local_mz_ = local_mz;
    ghost_width_ = ghost_width;
    stride_x_ = local_mx + 2 * ghost_width;
    resize_storage();
  }

  int local_mx() const { return local_mx_; }
  int local_my() const { return local_my_; }
  int local_mz() const { return local_mz_; }
  int ghost_width() const { return ghost_width_; }
  int stride() const { return stride_x_; }

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

  T& operator()(int i, int j, int k) { return data_[index(i, j, k)]; }
  const T& operator()(int i, int j, int k) const {
    return data_[index(i, j, k)];
  }

  bool has_device_data() const {
    return device_enabled() && device_data_ != nullptr;
  }

  void fill(const T& value) { std::fill(data_.begin(), data_.end(), value); }

private:
  int index(int i, int j, int k) const {
    return ((j + ghost_width_) * stride_x_ + (i + ghost_width_)) * local_mz_ +
           k;
  }

  void resize_storage() {
    if (local_mx_ <= 0 || local_my_ <= 0 || local_mz_ <= 0) {
      data_.clear();
      release_host_staging();
      release_device();
      return;
    }
    const size_t total = static_cast<size_t>(stride_x_) *
                         static_cast<size_t>(local_my_ + 2 * ghost_width_) *
                         static_cast<size_t>(local_mz_);
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
    cudaError_t err = cudaHostAlloc(reinterpret_cast<void**>(&host_staging_),
                                    elements * sizeof(T),
                                    cudaHostAllocDefault);
    if (err == cudaSuccess) {
      host_staging_pinned_ = true;
      host_staging_fallback_.clear();
      return;
    }
    host_staging_pinned_ = false;
    host_staging_fallback_.assign(elements, T{});
    host_staging_ = host_staging_fallback_.data();
#else
    host_staging_.assign(elements, T{});
#endif
  }

  void release_host_staging() {
#if GPISM_HAVE_CUDA
    if (host_staging_pinned_ && host_staging_ != nullptr) {
      cudaFreeHost(host_staging_);
    }
    host_staging_ = nullptr;
    host_staging_count_ = 0;
    host_staging_pinned_ = false;
    host_staging_fallback_.clear();
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
    cudaMemset(device_data_, 0, elements * sizeof(T));
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
  int local_mz_;
  int ghost_width_;
  int stride_x_;
  std::vector<T> data_;
#if GPISM_HAVE_CUDA
  T* host_staging_;
  std::size_t host_staging_count_;
  bool host_staging_pinned_;
  std::vector<T> host_staging_fallback_;
#else
  std::vector<T> host_staging_;
#endif
  T* device_data_;
};

}  // namespace gpism
