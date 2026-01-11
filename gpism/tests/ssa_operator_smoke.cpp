#include "gpism/ssa_operator.h"

#include <cmath>
#include <iostream>

namespace {

bool nearly_equal(double a, double b, double tol = 1e-10) {
  return std::abs(a - b) <= tol;
}

}  // namespace

int main() {
  const int mx = 4;
  const int my = 3;
  const int gw = 1;
  gpism::Grid2D grid(mx, my, 2.0, 3.0, gw, 0, 1);
  gpism::SSAOperator op(910.0, 9.81, 100.0);

  gpism::Field2D<double> tauc(mx, my, gw);
  tauc.fill(50.0);
  gpism::FieldStag2D<double> beta(mx, my, gw);
  op.compute_basal_drag(grid, tauc, beta);

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
  gpism::FieldStag2D<double> rhs(mx, my, gw);
  op.assemble_rhs(grid, thk, dhdx, dhdy, rhs);

  const double expected_u = 910.0 * 9.81 * 2.0 * 3.0;
  const double expected_v = 910.0 * 9.81 * 2.0 * -4.0;
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
  gpism::FieldStag2D<double> out(mx, my, gw);
  op.apply(grid, nuH, beta, vel, out);

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
  gpism::SSABoundaryCondition bc{&bc_mask, &bc_values};
  op.assemble_rhs(grid, thk, dhdx, dhdy, rhs, &bc);
  if (!nearly_equal(rhs(1, 1, 0), 2.0)) {
    std::cerr << "rhs BC override mismatch\n";
    return 1;
  }

  vel.fill(0.0);
  vel(1, 1, 0) = 7.0;
  op.apply(grid, nuH, beta, vel, out, &bc);
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
  op.apply(grid, nuH, beta, vel, out_a);
  op.apply(grid, nuH, beta, vel_b, out_b);

  auto dot = [&](const gpism::FieldStag2D<double>& a,
                 const gpism::FieldStag2D<double>& b) {
    double sum = 0.0;
    for (int j = 0; j < my; ++j) {
      for (int i = 0; i < mx; ++i) {
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
