#include "gpism/linear_algebra.h"

#include <cmath>
#include <cstring>
#include <iostream>

namespace {

bool nearly_equal(double a, double b, double tol = 1e-10) {
  return std::abs(a - b) <= tol;
}

template <typename T>
void sync_host_to_device(gpism::Field2D<T>& field) {
#if GPISM_HAVE_CUDA
  if (!field.host_staging_data() || field.elements() == 0) {
    return;
  }
  std::memcpy(field.host_staging_data(), field.data(),
              field.elements() * sizeof(T));
  field.copy_host_to_device();
#else
  (void)field;
#endif
}

template <typename T>
void sync_device_to_host(gpism::Field2D<T>& field) {
#if GPISM_HAVE_CUDA
  if (!field.device_data()) {
    return;
  }
  if (!field.host_staging_data() || field.elements() == 0) {
    return;
  }
  field.copy_device_to_host();
  std::memcpy(field.data(), field.host_staging_data(),
              field.elements() * sizeof(T));
#else
  (void)field;
#endif
}

template <typename T>
void sync_host_to_device(gpism::FieldStag2D<T>& field) {
  sync_host_to_device(field.component(0));
  sync_host_to_device(field.component(1));
}

template <typename T>
void sync_device_to_host(gpism::FieldStag2D<T>& field) {
  sync_device_to_host(field.component(0));
  sync_device_to_host(field.component(1));
}

double dot_host(const gpism::FieldStag2D<double>& a,
                const gpism::FieldStag2D<double>& b) {
  double sum = 0.0;
  for (int j = 0; j < a.local_my(); ++j) {
    for (int i = 0; i < a.local_mx(); ++i) {
      sum += a(i, j, 0) * b(i, j, 0) + a(i, j, 1) * b(i, j, 1);
    }
  }
  return sum;
}

double norm1_host(const gpism::FieldStag2D<double>& a) {
  double sum = 0.0;
  for (int j = 0; j < a.local_my(); ++j) {
    for (int i = 0; i < a.local_mx(); ++i) {
      sum += std::abs(a(i, j, 0)) + std::abs(a(i, j, 1));
    }
  }
  return sum;
}

bool compare_stag(const gpism::FieldStag2D<double>& a,
                  const gpism::FieldStag2D<double>& b, const char* name,
                  double tol = 1e-10) {
  for (int j = 0; j < a.local_my(); ++j) {
    for (int i = 0; i < a.local_mx(); ++i) {
      for (int comp = 0; comp < 2; ++comp) {
        if (!nearly_equal(a(i, j, comp), b(i, j, comp), tol)) {
          std::cerr << name << " mismatch at " << i << "," << j << "," << comp
                    << " (" << a(i, j, comp) << " vs " << b(i, j, comp)
                    << ")\n";
          return false;
        }
      }
    }
  }
  return true;
}

}  // namespace

int main() {
  const int mx = 6;
  const int my = 5;
  const int gw = 1;

  gpism::FieldStag2D<double> x(mx, my, gw);
  gpism::FieldStag2D<double> y(mx, my, gw);
  gpism::FieldStag2D<double> y_expected(mx, my, gw);
  gpism::FieldStag2D<double> x_expected(mx, my, gw);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      x(i, j, 0) = static_cast<double>(i + 2 * j);
      x(i, j, 1) = static_cast<double>(3 * i - j);
      y(i, j, 0) = static_cast<double>(1 + i - j);
      y(i, j, 1) = static_cast<double>(-2 + 2 * i + j);
    }
  }

  const double expected_dot = dot_host(x, y);
  const double expected_norm1 = norm1_host(x);
  const double expected_norm2 = std::sqrt(dot_host(x, x));

  sync_host_to_device(x);
  sync_host_to_device(y);

  const double got_dot = gpism::dot(x, y);
  const double got_norm1 = gpism::norm1(x);
  const double got_norm2 = gpism::norm2(x);

  if (!nearly_equal(got_dot, expected_dot)) {
    std::cerr << "dot mismatch: " << got_dot << " vs " << expected_dot << "\n";
    return 1;
  }
  if (!nearly_equal(got_norm1, expected_norm1)) {
    std::cerr << "norm1 mismatch: " << got_norm1 << " vs " << expected_norm1
              << "\n";
    return 1;
  }
  if (!nearly_equal(got_norm2, expected_norm2)) {
    std::cerr << "norm2 mismatch: " << got_norm2 << " vs " << expected_norm2
              << "\n";
    return 1;
  }

  const double alpha = 1.75;
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const double x0 = static_cast<double>(i + 2 * j);
      const double x1 = static_cast<double>(3 * i - j);
      const double y0 = static_cast<double>(1 + i - j);
      const double y1 = static_cast<double>(-2 + 2 * i + j);
      y_expected(i, j, 0) = y0 + alpha * x0;
      y_expected(i, j, 1) = y1 + alpha * x1;
    }
  }

  gpism::axpy(alpha, x, y);
  sync_device_to_host(y);
  if (!compare_stag(y, y_expected, "axpy")) {
    return 1;
  }

  const double beta = -0.5;
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const double x0 = static_cast<double>(i + 2 * j);
      const double x1 = static_cast<double>(3 * i - j);
      x_expected(i, j, 0) = x0 * beta;
      x_expected(i, j, 1) = x1 * beta;
    }
  }

  gpism::scal(beta, x);
  sync_device_to_host(x);
  if (!compare_stag(x, x_expected, "scal")) {
    return 1;
  }

  std::cout << "linear_algebra_smoke passed\n";
  return 0;
}
