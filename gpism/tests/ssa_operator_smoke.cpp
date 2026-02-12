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
  if (!field.host_staging_data() || field.elements() == 0) {
    return;
  }
  std::memcpy(field.host_staging_data(), field.data(),
              field.elements() * sizeof(T));
  field.copy_host_to_device();
}

template <typename T>
void sync_device_to_host(gpism::Field2D<T>& field) {
  if (!field.device_data()) {
    return;
  }
  if (!field.host_staging_data() || field.elements() == 0) {
    return;
  }
  field.copy_device_to_host();
  std::memcpy(field.data(), field.host_staging_data(),
              field.elements() * sizeof(T));
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
  const int mx = 4;
  const int my = 3;
  const int gw = 1;
  gpism::Grid2D grid(mx, my, 2.0, 3.0, gw, 0, 1);
  gpism::SSAOperator op(910.0, 9.81);

  gpism::Field2D<double> tauc(mx, my, gw);
  gpism::Field2D<double> u_center(mx, my, gw);
  gpism::Field2D<double> v_center(mx, my, gw);
  gpism::Field2D<double> topg(mx, my, gw);
  gpism::Field2D<double> usurf(mx, my, gw);
  gpism::Field2D<int> cell_type(mx, my, gw);
  tauc.fill(50.0);
  u_center.fill(0.0);
  v_center.fill(0.0);
  topg.fill(0.0);
  usurf.fill(0.0);
  cell_type.fill(gpism::GroundedIce);
  sync_host_to_device(tauc);
  sync_host_to_device(u_center);
  sync_host_to_device(v_center);
  sync_host_to_device(topg);
  sync_host_to_device(usurf);
  sync_host_to_device(cell_type);
  gpism::FieldStag2D<double> beta(mx, my, gw);
  gpism::BasalResistanceParams basal_params;
  basal_params.plastic_regularization = 100.0;
  op.compute_basal_drag(grid, tauc, u_center, v_center, topg, usurf, cell_type,
                        beta,
                        basal_params);
  sync_device_to_host(beta);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      if (!nearly_equal(beta(i, j, 0), 0.5)) {
        std::cerr << "beta_u mismatch at " << i << "," << j << "\n";
        return 1;
      }
      if (!nearly_equal(beta(i, j, 1), 0.5)) {
        std::cerr << "beta_v mismatch at " << i << "," << j << "\n";
        return 1;
      }
    }
  }

  gpism::Field2D<double> thk(mx, my, gw);
  gpism::Field2D<double> dhdx(mx, my, gw);
  gpism::Field2D<double> dhdy(mx, my, gw);
  thk.fill(2.0);
  dhdx.fill(3.0);
  dhdy.fill(-4.0);
  sync_host_to_device(thk);
  sync_host_to_device(dhdx);
  sync_host_to_device(dhdy);
  gpism::FieldStag2D<double> rhs(mx, my, gw);
  op.assemble_rhs(grid, thk, dhdx, dhdy, rhs);
  sync_device_to_host(rhs);

  const double expected_u = -910.0 * 9.81 * 2.0 * 3.0;
  const double expected_v = -910.0 * 9.81 * 2.0 * -4.0;
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      if (!nearly_equal(rhs(i, j, 0), expected_u)) {
        std::cerr << "rhs_u mismatch at " << i << "," << j << "\n";
        return 1;
      }
      if (!nearly_equal(rhs(i, j, 1), expected_v)) {
        std::cerr << "rhs_v mismatch at " << i << "," << j << "\n";
        return 1;
      }
    }
  }

  gpism::FieldStag2D<double> nuH(mx, my, gw);
  nuH.fill(1.0);
  gpism::FieldStag2D<double> vel(mx, my, gw);
  vel.fill(0.0);
  sync_host_to_device(nuH);
  sync_host_to_device(vel);
  gpism::FieldStag2D<double> out(mx, my, gw);
  op.apply(grid, nuH, beta, vel, out);
  sync_device_to_host(out);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      if (!nearly_equal(out(i, j, 0), 0.0)) {
        std::cerr << "apply_u mismatch at " << i << "," << j << "\n";
        return 1;
      }
      if (!nearly_equal(out(i, j, 1), 0.0)) {
        std::cerr << "apply_v mismatch at " << i << "," << j << "\n";
        return 1;
      }
    }
  }

  gpism::FieldStag2D<int> bc_mask(mx, my, gw);
  gpism::FieldStag2D<double> bc_values(mx, my, gw);
  bc_mask.fill(0);
  bc_values.fill(0.0);
  bc_mask(1, 1, 0) = 1;
  bc_values(1, 1, 0) = 2.0;
  sync_host_to_device(bc_mask);
  sync_host_to_device(bc_values);
  gpism::SSABoundaryCondition bc{&bc_mask, &bc_values};
  op.assemble_rhs(grid, thk, dhdx, dhdy, rhs, &bc);
  sync_device_to_host(rhs);
  if (!nearly_equal(rhs(1, 1, 0), 2.0)) {
    std::cerr << "rhs BC override mismatch\n";
    return 1;
  }

  vel.fill(0.0);
  vel(1, 1, 0) = 7.0;
  sync_host_to_device(vel);
  op.apply(grid, nuH, beta, vel, out, &bc);
  sync_device_to_host(out);
  if (!nearly_equal(out(1, 1, 0), 7.0)) {
    std::cerr << "apply BC override mismatch\n";
    return 1;
  }

  gpism::FieldStag2D<double> vel_b(mx, my, gw);
  gpism::FieldStag2D<double> out_a(mx, my, gw);
  gpism::FieldStag2D<double> out_b(mx, my, gw);
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      vel(i, j, 0) = static_cast<double>(i + 2 * j + 1);
      vel(i, j, 1) = static_cast<double>(2 * i - j);
      vel_b(i, j, 0) = static_cast<double>(3 * i - j + 2);
      vel_b(i, j, 1) = static_cast<double>(i + 4 * j + 1);
    }
  }
  sync_host_to_device(vel);
  sync_host_to_device(vel_b);
  op.apply(grid, nuH, beta, vel, out_a);
  op.apply(grid, nuH, beta, vel_b, out_b);
  sync_device_to_host(out_a);
  sync_device_to_host(out_b);

  auto dot = [&](const gpism::FieldStag2D<double>& a,
                 const gpism::FieldStag2D<double>& b) {
    double sum = 0.0;
    const int i_start = 1;
    const int i_end = mx - 2;
    const int j_start = 1;
    const int j_end = my - 2;
    for (int j = j_start; j <= j_end; ++j) {
      for (int i = i_start; i <= i_end; ++i) {
        sum += a(i, j, 0) * b(i, j, 0) + a(i, j, 1) * b(i, j, 1);
      }
    }
    return sum;
  };

  const double lhs = dot(out_a, vel_b);
  const double rhs_sym = dot(vel, out_b);
  if (!nearly_equal(lhs, rhs_sym, 1e-8)) {
    std::cerr << "operator symmetry check failed\n";
    return 1;
  }

  std::cout << "ssa_operator_smoke passed\n";
  return 0;
}
