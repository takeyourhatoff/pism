#include "gpism/field_sync.h"
#include "gpism/ssa_solver.h"

#include <cmath>
#include <iostream>

namespace {

double vel_norm(const gpism::FieldStag2D<double>& vel) {
  double sum = 0.0;
  for (int j = 0; j < vel.local_my(); ++j) {
    for (int i = 0; i < vel.local_mx(); ++i) {
      sum += vel(i, j, 0) * vel(i, j, 0) + vel(i, j, 1) * vel(i, j, 1);
    }
  }
  return std::sqrt(sum);
}

}  // namespace

int main() {
  const int mx = 4;
  const int my = 4;
  const int gw = 1;
  gpism::Grid2D grid(mx, my, 1.0, 1.0, gw, 0, 1);

  gpism::Field2D<double> thk(mx, my, gw);
  gpism::Field2D<double> topg(mx, my, gw);
  gpism::Field2D<double> tauc(mx, my, gw);
  gpism::FieldStag2D<double> vel(mx, my, gw);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      thk(i, j) = 2.0;
      topg(i, j) = 0.1 * i - 0.05 * j;
      tauc(i, j) = 120.0;
      vel(i, j, 0) = 0.0;
      vel(i, j, 1) = 0.0;
    }
  }

  gpism::sync_host_to_device(thk);
  gpism::sync_host_to_device(topg);
  gpism::sync_host_to_device(tauc);
  gpism::sync_host_to_device(vel);

  gpism::ViscosityModel viscosity(1e-16, 3.0, 1.0);
  gpism::SSASolver solver(grid, 910.0, 9.81, 100.0, viscosity);

  gpism::SSASolverOptions options;
  options.max_picard = 25;
  options.tol_nuH = 0.7;
  options.tol_vel = 0.7;
  options.gmres_max_iter = 150;
  options.gmres_tol = 1e-7;
  options.vel_relax = 1.0;
  options.nuH_relax = 1.0;
  options.use_bc = false;

  gpism::SSASolverResult result =
      solver.solve(thk, topg, tauc, nullptr, nullptr, nullptr, vel, options);

  gpism::sync_device_to_host(vel);

  if (!result.converged) {
    std::cerr << "Picard solver did not converge for nontrivial case\n";
    std::cerr << "nuH_change=" << result.nuH_change
              << " vel_change=" << result.vel_change
              << " picard_iters=" << result.picard_iters
              << " linear_iters=" << result.linear_iters << "\n";
    return 1;
  }

  const double speed = vel_norm(vel);
  if (!std::isfinite(speed)) {
    std::cerr << "velocity norm is not finite\n";
    return 1;
  }

  if (speed <= 0.0) {
    std::cerr << "velocity norm is zero for nontrivial case\n";
    return 1;
  }

  std::cout << "ssa_picard_nontrivial_smoke passed\n";
  return 0;
}
