#include "gpism/field_sync.h"
#include "gpism/ssa_solver.h"
#include "gpism/viscosity.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

double velocity_diff_max(const gpism::FieldStag2D<double>& a,
                         const gpism::FieldStag2D<double>& b) {
  double diff = 0.0;
  for (int j = 0; j < a.local_my(); ++j) {
    for (int i = 0; i < a.local_mx(); ++i) {
      diff = std::max(diff, std::abs(a(i, j, 0) - b(i, j, 0)));
      diff = std::max(diff, std::abs(a(i, j, 1) - b(i, j, 1)));
    }
  }
  return diff;
}

}  // namespace

int main() {
  const int mx = 8;
  const int my = 8;
  const int gw = 1;
  gpism::Grid2D grid(mx, my, 1000.0, 1000.0, gw, 0, 1);

  gpism::Field2D<double> thk(mx, my, gw);
  gpism::Field2D<double> topg(mx, my, gw);
  gpism::Field2D<double> tauc(mx, my, gw);
  gpism::FieldStag2D<double> vel_fp64(mx, my, gw);
  gpism::FieldStag2D<double> vel_fp32(mx, my, gw);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      thk(i, j) = 500.0;
      topg(i, j) = -0.25 * static_cast<double>(i);
      tauc(i, j) = 2e5;
    }
  }
  vel_fp64.fill(0.0);
  vel_fp32.fill(0.0);

  gpism::sync_host_to_device(thk);
  gpism::sync_host_to_device(topg);
  gpism::sync_host_to_device(tauc);
  gpism::sync_host_to_device(vel_fp64);
  gpism::sync_host_to_device(vel_fp32);

  gpism::ViscosityModel viscosity(1e-16, 3.0, 1.0);
  gpism::SSASolver solver(grid, 910.0, 9.81, 100.0, viscosity);

  gpism::SSASolverOptions base;
  base.max_picard = 10;
  base.tol_nuH = 1e-3;
  base.tol_vel = 1e-3;
  base.gmres_max_iter = 80;
  base.gmres_tol = 1e-8;
  base.use_mg_precond = true;
  base.use_bc = false;
  base.fail_fast = false;
  base.precond_precision = gpism::SSAPrecondPrecision::FP64;

  gpism::SSASolverResult res_fp64 =
      solver.solve(thk, topg, tauc, nullptr, nullptr, nullptr, vel_fp64, base);
  if (!res_fp64.converged) {
    std::cerr << "FP64 preconditioner solve did not converge\n";
    return 1;
  }

  gpism::SSASolverOptions mixed = base;
  mixed.precond_precision = gpism::SSAPrecondPrecision::FP32;
  gpism::SSASolverResult res_fp32 =
      solver.solve(thk, topg, tauc, nullptr, nullptr, nullptr, vel_fp32, mixed);
  if (!res_fp32.converged) {
    std::cerr << "FP32 preconditioner solve did not converge\n";
    return 1;
  }

  gpism::sync_device_to_host(vel_fp64);
  gpism::sync_device_to_host(vel_fp32);

  const double diff_max = velocity_diff_max(vel_fp64, vel_fp32);
  if (diff_max > 1e-8) {
    std::cerr << "mixed-precision preconditioner parity failed (max diff="
              << diff_max << ")\n";
    return 1;
  }

  std::cout << "mg_mixed_precision_parity_smoke passed\n";
  return 0;
}
