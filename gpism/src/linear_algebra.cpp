#include "gpism/linear_algebra.h"

#include <cmath>

#include "gpism/config.h"

#if GPISM_HAVE_CUDA
namespace gpism {
void axpy_cuda(int mx, int my, int gw, int stride_x, int stride_y,
              const double* x, double* y, double alpha);
void scal_cuda(int mx, int my, int gw, int stride, double* x, double alpha);
void copy_cuda(int mx, int my, int gw, int stride_x, int stride_y,
               const double* x, double* y);
void set_cuda(int mx, int my, int gw, int stride, double* x, double value);
double dot_cuda(int mx, int my, int gw, int stride_a, int stride_b,
                const double* a, const double* b);
double norm1_cuda(int mx, int my, int gw, int stride, const double* a);
}  // namespace gpism
#endif

namespace gpism {
namespace {

double dot_field_host(const Field2D<double>& a, const Field2D<double>& b) {
  double sum = 0.0;
  const int mx = a.local_mx();
  const int my = a.local_my();
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      sum += a(i, j) * b(i, j);
    }
  }
  return sum;
}

double norm1_field_host(const Field2D<double>& a) {
  double sum = 0.0;
  const int mx = a.local_mx();
  const int my = a.local_my();
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      sum += std::abs(a(i, j));
    }
  }
  return sum;
}

void axpy_field_host(double alpha, const Field2D<double>& x,
                     Field2D<double>& y) {
  const int mx = y.local_mx();
  const int my = y.local_my();
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      y(i, j) += alpha * x(i, j);
    }
  }
}

void scal_field_host(double alpha, Field2D<double>& x) {
  const int mx = x.local_mx();
  const int my = x.local_my();
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      x(i, j) *= alpha;
    }
  }
}

void copy_field_host(const Field2D<double>& x, Field2D<double>& y) {
  const int mx = y.local_mx();
  const int my = y.local_my();
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      y(i, j) = x(i, j);
    }
  }
}

void set_field_host(double value, Field2D<double>& x) {
  const int mx = x.local_mx();
  const int my = x.local_my();
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      x(i, j) = value;
    }
  }
}

}  // namespace

void axpy(double alpha, const FieldStag2D<double>& x, FieldStag2D<double>& y) {
#if GPISM_HAVE_CUDA
  if (x.component(0).has_device_data() && x.component(1).has_device_data() &&
      y.component(0).has_device_data() && y.component(1).has_device_data()) {
    axpy_cuda(x.local_mx(), x.local_my(), x.ghost_width(),
             x.component(0).stride(), y.component(0).stride(),
             x.component(0).device_data(), y.component(0).device_data(),
             alpha);
    axpy_cuda(x.local_mx(), x.local_my(), x.ghost_width(),
             x.component(1).stride(), y.component(1).stride(),
             x.component(1).device_data(), y.component(1).device_data(),
             alpha);
    return;
  }
#endif
  axpy_field_host(alpha, x.component(0), y.component(0));
  axpy_field_host(alpha, x.component(1), y.component(1));
}

void scal(double alpha, FieldStag2D<double>& x) {
#if GPISM_HAVE_CUDA
  if (x.component(0).has_device_data() && x.component(1).has_device_data()) {
    scal_cuda(x.local_mx(), x.local_my(), x.ghost_width(),
             x.component(0).stride(), x.component(0).device_data(), alpha);
    scal_cuda(x.local_mx(), x.local_my(), x.ghost_width(),
             x.component(1).stride(), x.component(1).device_data(), alpha);
    return;
  }
#endif
  scal_field_host(alpha, x.component(0));
  scal_field_host(alpha, x.component(1));
}

void copy(const FieldStag2D<double>& x, FieldStag2D<double>& y) {
#if GPISM_HAVE_CUDA
  if (x.component(0).has_device_data() && x.component(1).has_device_data() &&
      y.component(0).has_device_data() && y.component(1).has_device_data()) {
    copy_cuda(x.local_mx(), x.local_my(), x.ghost_width(),
             x.component(0).stride(), y.component(0).stride(),
             x.component(0).device_data(), y.component(0).device_data());
    copy_cuda(x.local_mx(), x.local_my(), x.ghost_width(),
             x.component(1).stride(), y.component(1).stride(),
             x.component(1).device_data(), y.component(1).device_data());
    return;
  }
#endif
  copy_field_host(x.component(0), y.component(0));
  copy_field_host(x.component(1), y.component(1));
}

void set(double value, FieldStag2D<double>& x) {
#if GPISM_HAVE_CUDA
  if (x.component(0).has_device_data() && x.component(1).has_device_data()) {
    set_cuda(x.local_mx(), x.local_my(), x.ghost_width(),
            x.component(0).stride(), x.component(0).device_data(), value);
    set_cuda(x.local_mx(), x.local_my(), x.ghost_width(),
            x.component(1).stride(), x.component(1).device_data(), value);
    return;
  }
#endif
  set_field_host(value, x.component(0));
  set_field_host(value, x.component(1));
}

double dot(const FieldStag2D<double>& a, const FieldStag2D<double>& b) {
#if GPISM_HAVE_CUDA
  if (a.component(0).has_device_data() && a.component(1).has_device_data() &&
      b.component(0).has_device_data() && b.component(1).has_device_data()) {
    const double sum0 = dot_cuda(a.local_mx(), a.local_my(), a.ghost_width(),
                                 a.component(0).stride(),
                                 b.component(0).stride(),
                                 a.component(0).device_data(),
                                 b.component(0).device_data());
    const double sum1 = dot_cuda(a.local_mx(), a.local_my(), a.ghost_width(),
                                 a.component(1).stride(),
                                 b.component(1).stride(),
                                 a.component(1).device_data(),
                                 b.component(1).device_data());
    return sum0 + sum1;
  }
#endif
  return dot_field_host(a.component(0), b.component(0)) +
         dot_field_host(a.component(1), b.component(1));
}

double norm2(const FieldStag2D<double>& a) {
  return std::sqrt(dot(a, a));
}

double norm1(const FieldStag2D<double>& a) {
#if GPISM_HAVE_CUDA
  if (a.component(0).has_device_data() && a.component(1).has_device_data()) {
    const double sum0 = norm1_cuda(a.local_mx(), a.local_my(), a.ghost_width(),
                                   a.component(0).stride(),
                                   a.component(0).device_data());
    const double sum1 = norm1_cuda(a.local_mx(), a.local_my(), a.ghost_width(),
                                   a.component(1).stride(),
                                   a.component(1).device_data());
    return sum0 + sum1;
  }
#endif
  return norm1_field_host(a.component(0)) + norm1_field_host(a.component(1));
}

}  // namespace gpism
