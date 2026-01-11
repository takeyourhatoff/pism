#include "gpism/gmres.h"

#include <cmath>
#include <cstring>
#include <iostream>

#include "gpism/linear_algebra.h"

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

struct IdentityOperator : public gpism::LinearOperator {
  void apply(const gpism::FieldStag2D<double>& x,
             gpism::FieldStag2D<double>& y) const override {
    gpism::copy(x, y);
  }
};

struct ScaleOperator : public gpism::LinearOperator {
  explicit ScaleOperator(double scale) : scale_(scale) {}

  void apply(const gpism::FieldStag2D<double>& x,
             gpism::FieldStag2D<double>& y) const override {
    gpism::copy(x, y);
    gpism::scal(scale_, y);
  }

  double scale_;
};

bool check_solution(const gpism::FieldStag2D<double>& x,
                    const gpism::FieldStag2D<double>& expected,
                    const char* name) {
  for (int j = 0; j < x.local_my(); ++j) {
    for (int i = 0; i < x.local_mx(); ++i) {
      for (int comp = 0; comp < 2; ++comp) {
        if (!nearly_equal(x(i, j, comp), expected(i, j, comp), 1e-8)) {
          std::cerr << name << " mismatch at " << i << "," << j << "," << comp
                    << " (" << x(i, j, comp) << " vs " << expected(i, j, comp)
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
  const int mx = 5;
  const int my = 4;
  const int gw = 1;

  gpism::FieldStag2D<double> b(mx, my, gw);
  gpism::FieldStag2D<double> x(mx, my, gw);
  gpism::FieldStag2D<double> expected(mx, my, gw);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      b(i, j, 0) = static_cast<double>(i + 2 * j + 1);
      b(i, j, 1) = static_cast<double>(-3 + 2 * i - j);
    }
  }

  sync_host_to_device(b);
  gpism::set(0.0, x);

  gpism::GMRESOptions opts;
  opts.restart = 10;
  opts.max_iter = 30;
  opts.tol = 1e-10;

  IdentityOperator I;
  gpism::GMRESResult res = gpism::gmres_solve(I, b, x, opts);
  if (!res.converged) {
    std::cerr << "GMRES identity solve did not converge\n";
    return 1;
  }

  sync_device_to_host(x);
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      expected(i, j, 0) = b(i, j, 0);
      expected(i, j, 1) = b(i, j, 1);
    }
  }
  if (!check_solution(x, expected, "identity")) {
    return 1;
  }

  gpism::set(0.0, x);
  const double scale = 2.5;
  ScaleOperator A(scale);
  res = gpism::gmres_solve(A, b, x, opts);
  if (!res.converged) {
    std::cerr << "GMRES scaled identity solve did not converge\n";
    return 1;
  }

  sync_device_to_host(x);
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      expected(i, j, 0) = b(i, j, 0) / scale;
      expected(i, j, 1) = b(i, j, 1) / scale;
    }
  }
  if (!check_solution(x, expected, "scaled")) {
    return 1;
  }

  std::cout << "gmres_smoke passed\n";
  return 0;
}
