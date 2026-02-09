#include "gpism/geometry.h"
#include "gpism/ssa_operator.h"

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
#if !GPISM_HAVE_CUDA
  std::cout << "ssa_operator_cuda_smoke skipped (CUDA disabled)\n";
  return 0;
#else
  const int mx = 5;
  const int my = 4;
  const int gw = 1;
  gpism::Grid2D grid(mx, my, 2.5, 3.5, gw, 0, 1);

  const double rho = 910.0;
  const double g = 9.81;
  const double plastic_reg = 100.0;
  gpism::SSAOperator op(rho, g);

  gpism::Field2D<double> tauc(mx, my, gw);
  tauc.fill(100.0);
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      tauc(i, j) = 50.0 + i + 2.0 * j;
    }
  }

  gpism::Field2D<double> thk(mx, my, gw);
  gpism::Field2D<double> dhdx(mx, my, gw);
  gpism::Field2D<double> dhdy(mx, my, gw);
  gpism::Field2D<double> u_center(mx, my, gw);
  gpism::Field2D<double> v_center(mx, my, gw);
  gpism::Field2D<double> topg(mx, my, gw);
  gpism::Field2D<double> usurf(mx, my, gw);
  gpism::Field2D<int> cell_type(mx, my, gw);
  thk.fill(2.0);
  dhdx.fill(0.5);
  dhdy.fill(-0.25);
  u_center.fill(0.0);
  v_center.fill(0.0);
  topg.fill(0.0);
  usurf.fill(0.0);
  cell_type.fill(gpism::GroundedIce);
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      thk(i, j) = 2.0 + 0.1 * i - 0.05 * j;
      dhdx(i, j) = 0.5 + 0.01 * i;
      dhdy(i, j) = -0.25 + 0.02 * j;
    }
  }

  gpism::FieldStag2D<int> bc_mask(mx, my, gw);
  gpism::FieldStag2D<double> bc_values(mx, my, gw);
  bc_mask.fill(0);
  bc_values.fill(0.0);
  bc_mask(2, 1, 0) = 1;
  bc_values(2, 1, 0) = 3.25;
  bc_mask(1, 2, 1) = 1;
  bc_values(1, 2, 1) = -1.75;
  gpism::SSABoundaryCondition bc{&bc_mask, &bc_values};

  gpism::FieldStag2D<double> beta_ref(mx, my, gw);
  gpism::FieldStag2D<double> rhs_ref(mx, my, gw);
  gpism::FieldStag2D<double> nuH(mx, my, gw);
  gpism::FieldStag2D<double> vel(mx, my, gw);
  gpism::FieldStag2D<double> out_ref(mx, my, gw);

  nuH.fill(1.0);
  vel.fill(0.0);
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      nuH(i, j, 0) = 1.0 + 0.01 * i;
      nuH(i, j, 1) = 1.2 + 0.02 * j;
      vel(i, j, 0) = static_cast<double>(i - j);
      vel(i, j, 1) = static_cast<double>(2 * i + j);
    }
  }

  // Reference: host path.
  const bool prev_device = gpism::device_enabled();
  gpism::set_device_enabled(false);
  gpism::BasalResistanceParams basal_params;
  basal_params.plastic_regularization = plastic_reg;
  op.compute_basal_drag(grid, tauc, u_center, v_center, topg, usurf, cell_type,
                        beta_ref, basal_params);
  op.assemble_rhs(grid, thk, dhdx, dhdy, rhs_ref, &bc);
  op.apply(grid, nuH, beta_ref, vel, out_ref, &bc);
  gpism::set_device_enabled(prev_device);

  gpism::FieldStag2D<double> beta_gpu(mx, my, gw);
  gpism::FieldStag2D<double> rhs_gpu(mx, my, gw);
  gpism::FieldStag2D<double> out_gpu(mx, my, gw);

  gpism::set_device_enabled(true);
  sync_host_to_device(tauc);
  sync_host_to_device(thk);
  sync_host_to_device(dhdx);
  sync_host_to_device(dhdy);
  sync_host_to_device(u_center);
  sync_host_to_device(v_center);
  sync_host_to_device(topg);
  sync_host_to_device(usurf);
  sync_host_to_device(cell_type);
  sync_host_to_device(bc_mask);
  sync_host_to_device(bc_values);

  op.compute_basal_drag(grid, tauc, u_center, v_center, topg, usurf, cell_type,
                        beta_gpu, basal_params);
  sync_device_to_host(beta_gpu);
  if (!compare_stag(beta_gpu, beta_ref, "basal_drag")) {
    return 1;
  }

  op.assemble_rhs(grid, thk, dhdx, dhdy, rhs_gpu, &bc);
  sync_device_to_host(rhs_gpu);
  if (!compare_stag(rhs_gpu, rhs_ref, "rhs")) {
    return 1;
  }

  sync_host_to_device(nuH);
  sync_host_to_device(vel);
  sync_host_to_device(beta_gpu);
  op.apply(grid, nuH, beta_gpu, vel, out_gpu, &bc);
  sync_device_to_host(out_gpu);
  if (!compare_stag(out_gpu, out_ref, "apply")) {
    return 1;
  }

  std::cout << "ssa_operator_cuda_smoke passed\n";
  return 0;
#endif
}
