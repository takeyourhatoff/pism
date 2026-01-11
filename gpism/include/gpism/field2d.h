#pragma once

#include <algorithm>
#include <vector>

namespace gpism {

template <typename T>
class Field2D {
public:
  Field2D() : local_mx_(0), local_my_(0), ghost_width_(0), stride_(0) {}

  Field2D(int local_mx, int local_my, int ghost_width)
      : local_mx_(local_mx),
        local_my_(local_my),
        ghost_width_(ghost_width),
        stride_(local_mx + 2 * ghost_width) {
    resize_storage();
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
      return;
    }
    data_.assign(static_cast<size_t>(stride_) *
                     static_cast<size_t>(local_my_ + 2 * ghost_width_),
                 T{});
  }

  int local_mx_;
  int local_my_;
  int ghost_width_;
  int stride_;
  std::vector<T> data_;
};

}  // namespace gpism
