#include "gpism/ssa_operator.h"
#include "gpism/geometry.h"

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
  const int mx = 6;
  const int my = 5;
  const int gw = 1;
  gpism::Grid2D grid(mx, my, 1000.0, 1000.0, gw, 0, 1);

  gpism::SSAOperator op(910.0, 9.81);

  gpism::Field2D<double> tauc(mx, my, gw);
  gpism::Field2D<double> u_center(mx, my, gw);
  gpism::Field2D<double> v_center(mx, my, gw);
  gpism::Field2D<double> topg(mx, my, gw);
  gpism::Field2D<double> usurf(mx, my, gw);
  gpism::Field2D<int> cell_type(mx, my, gw);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      tauc(i, j) = 100.0 + 10.0 * i + 1.0 * j;
      u_center(i, j) = 3.0;
      v_center(i, j) = -4.0;
      topg(i, j) = 0.0;
      usurf(i, j) = 0.0;
      cell_type(i, j) = gpism::GroundedIce;
    }
  }

  sync_host_to_device(tauc);
  sync_host_to_device(u_center);
  sync_host_to_device(v_center);
  sync_host_to_device(topg);
  sync_host_to_device(usurf);
  sync_host_to_device(cell_type);

  gpism::BasalResistanceParams params;
  params.law = gpism::BasalResistanceLaw::Plastic;
  params.plastic_regularization = 5.0;
  params.beta_lateral_margin = 0.0;
  params.beta_ice_free_bedrock = 123.0;  // unused in this test

  gpism::FieldStag2D<double> beta(mx, my, gw);
  op.compute_basal_drag(grid, tauc, u_center, v_center, topg, usurf, cell_type,
                        beta, params);
  sync_device_to_host(beta);

  const double denom = std::sqrt(params.plastic_regularization *
                                     params.plastic_regularization +
                                 3.0 * 3.0 + (-4.0) * (-4.0));
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const double expected = tauc(i, j) / denom;
      if (!nearly_equal(beta(i, j, 0), expected)) {
        std::cerr << "beta_u mismatch at " << i << "," << j << " ("
                  << beta(i, j, 0) << " vs " << expected << ")\n";
        return 1;
      }
      if (!nearly_equal(beta(i, j, 1), expected)) {
        std::cerr << "beta_v mismatch at " << i << "," << j << " ("
                  << beta(i, j, 1) << " vs " << expected << ")\n";
        return 1;
      }
    }
  }

  std::cout << "ssa_beta_centered_smoke passed\n";
  return 0;
}
