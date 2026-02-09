#include "gpism/device_policy.h"
#include "gpism/ssa_solver.h"
#include "gpism/time_manager.h"

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
    }
  }
  vel.fill(0.0);

  gpism::set_device_enabled(false);
  gpism::ViscosityModel viscosity(1e-16, 3.0, 1.0);
  gpism::SSASolver solver(grid, 910.0, 9.81, 100.0, viscosity);

  gpism::SSASolverOptions options;
  options.max_picard = 80;
  options.tol_nuH = 0.7;
  options.tol_vel = 0.7;
  options.gmres_max_iter = 200;
  options.gmres_tol = 1e-7;
  // Plastic basal drag is strongly nonlinear; damping improves robustness.
  options.vel_relax = 0.5;
  options.nuH_relax = 0.5;
  options.use_bc = false;

  gpism::TimeManager clock(0.0, 0.1, 1.0, 0.2);
  gpism::Field2D<double> thk_initial(mx, my, gw);
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      thk_initial(i, j) = thk(i, j);
    }
  }

  while (!clock.done()) {
    gpism::SSASolverResult result =
        solver.solve(thk, topg, tauc, nullptr, nullptr, nullptr, vel, options);
    if (!result.converged) {
      std::cerr << "SSA solver did not converge in timestep loop (picard_iters="
                << result.picard_iters << " nuH_change=" << result.nuH_change
                << " vel_change=" << result.vel_change
                << " linear_iters=" << result.linear_iters
                << " linear_residual=" << result.linear_residual << ")\n";
      return 1;
    }
    if (!std::isfinite(vel_norm(vel))) {
      std::cerr << "velocity norm invalid\n";
      return 1;
    }
    clock.advance();
  }

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      if (std::abs(thk(i, j) - thk_initial(i, j)) > 1e-12) {
        std::cerr << "thickness changed in SSA-only loop\n";
        return 1;
      }
    }
  }

  std::cout << "ssa_timestep_smoke passed\n";
  return 0;
}
