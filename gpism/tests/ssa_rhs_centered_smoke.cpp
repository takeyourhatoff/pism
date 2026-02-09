#include "gpism/ssa_operator.h"

#include <cmath>
#include <cstring>
#include <iostream>

namespace {

bool nearly_equal(double a, double b, double tol = 1e-12) {
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

}  // namespace

int main() {
  const int mx = 5;
  const int my = 4;
  const int gw = 1;
  gpism::Grid2D grid(mx, my, 1000.0, 2000.0, gw, 0, 1);

  const double rho = 910.0;
  const double g = 9.81;
  gpism::SSAOperator op(rho, g);

  gpism::Field2D<double> thk(mx, my, gw);
  gpism::Field2D<double> dhdx(mx, my, gw);
  gpism::Field2D<double> dhdy(mx, my, gw);
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      thk(i, j) = 100.0 + 2.0 * i + 3.0 * j;
      dhdx(i, j) = 1e-3 * (1.0 + 0.25 * i - 0.1 * j);
      dhdy(i, j) = 1e-3 * (-0.5 + 0.05 * i + 0.2 * j);
    }
  }

  sync_host_to_device(thk);
  sync_host_to_device(dhdx);
  sync_host_to_device(dhdy);

  gpism::FieldStag2D<double> rhs(mx, my, gw);
  op.assemble_rhs(grid, thk, dhdx, dhdy, rhs);
  sync_device_to_host(rhs);

  const double scale = -rho * g;
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const double expected_u = scale * thk(i, j) * dhdx(i, j);
      const double expected_v = scale * thk(i, j) * dhdy(i, j);
      if (!nearly_equal(rhs(i, j, 0), expected_u)) {
        std::cerr << "rhs_u mismatch at " << i << "," << j << " ("
                  << rhs(i, j, 0) << " vs " << expected_u << ")\n";
        return 1;
      }
      if (!nearly_equal(rhs(i, j, 1), expected_v)) {
        std::cerr << "rhs_v mismatch at " << i << "," << j << " ("
                  << rhs(i, j, 1) << " vs " << expected_v << ")\n";
        return 1;
      }
    }
  }

  std::cout << "ssa_rhs_centered_smoke passed\n";
  return 0;
}

