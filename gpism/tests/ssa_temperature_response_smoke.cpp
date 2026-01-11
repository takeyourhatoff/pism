#include "gpism/device_policy.h"
#include "gpism/ssa_solver.h"
#include "gpism/viscosity.h"

#include <cmath>
#include <iostream>

namespace {

double vel_norm2(const gpism::FieldStag2D<double>& vel) {
  double sum = 0.0;
  for (int j = 0; j < vel.local_my(); ++j) {
    for (int i = 0; i < vel.local_mx(); ++i) {
      sum += vel(i, j, 0) * vel(i, j, 0) + vel(i, j, 1) * vel(i, j, 1);
    }
  }
  return sum;
}

}  // namespace

int main() {
  gpism::set_device_enabled(false);

  const int mx = 6;
  const int my = 6;
  const int nz = 8;
  const int gw = 1;
  gpism::Grid2D grid(mx, my, 1000.0, 1000.0, gw, 0, 1);

  gpism::Field2D<double> thk(mx, my, gw);
  gpism::Field2D<double> topg(mx, my, gw);
  gpism::Field2D<double> tauc(mx, my, gw);
  gpism::FieldStag2D<double> vel(mx, my, gw);
  gpism::FieldStag2D<double> vel_hot(mx, my, gw);
  gpism::Field3D<double> enthalpy_cold(mx, my, nz, gw);
  gpism::Field3D<double> enthalpy_hot(mx, my, nz, gw);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      thk(i, j) = 1000.0;
      topg(i, j) = 10.0 * i - 5.0 * j;
      tauc(i, j) = 120.0;
    }
  }
  vel.fill(0.0);
  vel_hot.fill(0.0);

  enthalpy_cold.fill(0.0);
  enthalpy_hot.fill(10.0);

  gpism::ViscosityModel viscosity(1e-16, 3.0, 1.0);
  gpism::SSASolver solver(grid, 910.0, 9.81, 100.0, viscosity);

  gpism::SSASolverOptions options;
  options.max_picard = 20;
  options.tol_nuH = 0.7;
  options.tol_vel = 0.7;
  options.gmres_max_iter = 150;
  options.gmres_tol = 1e-7;
  options.use_bc = false;
  options.enthalpy_gamma = 0.1;
  options.enthalpy_ref = 0.0;

  options.enthalpy = &enthalpy_cold;
  gpism::SSASolverResult cold_result =
      solver.solve(thk, topg, tauc, nullptr, nullptr, nullptr, vel, options);
  if (!cold_result.converged) {
    std::cerr << "SSA cold solve failed\n";
    return 1;
  }

  options.enthalpy = &enthalpy_hot;
  gpism::SSASolverResult hot_result =
      solver.solve(thk, topg, tauc, nullptr, nullptr, nullptr, vel_hot, options);
  if (!hot_result.converged) {
    std::cerr << "SSA hot solve failed\n";
    return 1;
  }

  const double cold_norm = std::sqrt(vel_norm2(vel));
  const double hot_norm = std::sqrt(vel_norm2(vel_hot));
  const double rel_change = std::abs(hot_norm - cold_norm) /
                            std::max(1e-12, cold_norm);
  if (!(rel_change > 0.01)) {
    std::cerr << "Enthalpy coupling did not change SSA response\n";
    std::cerr << "cold_norm=" << cold_norm << " hot_norm=" << hot_norm << "\n";
    return 1;
  }

  std::cout << "ssa_temperature_response_smoke passed\n";
  return 0;
}
