#pragma once

#include "gpism/field2d.h"

namespace gpism {

template <typename T>
class FieldStag2D {
public:
  FieldStag2D() = default;

  FieldStag2D(int local_mx, int local_my, int ghost_width)
      : u_(local_mx, local_my, ghost_width),
        v_(local_mx, local_my, ghost_width) {}

  void resize(int local_mx, int local_my, int ghost_width) {
    u_.resize(local_mx, local_my, ghost_width);
    v_.resize(local_mx, local_my, ghost_width);
  }

  int local_mx() const { return u_.local_mx(); }
  int local_my() const { return u_.local_my(); }
  int ghost_width() const { return u_.ghost_width(); }

  Field2D<T>& component(int comp) {
    return (comp == 0) ? u_ : v_;
  }

  const Field2D<T>& component(int comp) const {
    return (comp == 0) ? u_ : v_;
  }

  T& operator()(int i, int j, int comp) {
    return component(comp)(i, j);
  }

  const T& operator()(int i, int j, int comp) const {
    return component(comp)(i, j);
  }

  void fill(const T& value) {
    u_.fill(value);
    v_.fill(value);
  }

private:
  Field2D<T> u_;
  Field2D<T> v_;
};

}  // namespace gpism
