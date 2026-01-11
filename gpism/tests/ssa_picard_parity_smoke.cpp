#include "gpism/device_policy.h"
#include "gpism/ssa_solver.h"

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

double vel_diff_norm2(const gpism::FieldStag2D<double>& a,
                      const gpism::FieldStag2D<double>& b) {
  double sum = 0.0;
  for (int j = 0; j < a.local_my(); ++j) {
    for (int i = 0; i < a.local_mx(); ++i) {
      const double du = a(i, j, 0) - b(i, j, 0);
      const double dv = a(i, j, 1) - b(i, j, 1);
      sum += du * du + dv * dv;
    }
  }
  return sum;
}

}  // namespace

int main() {
  const int mx = 6;
  const int my = 6;
  const int gw = 1;

  gpism::SSASolverOptions options;
  options.max_picard = 20;
  options.tol_nuH = 0.7;
  options.tol_vel = 0.7;
  options.gmres_max_iter = 150;
  options.gmres_tol = 1e-7;
  options.use_bc = false;

  gpism::set_device_enabled(false);
  gpism::Grid2D grid_cpu(mx, my, 1.0, 1.0, gw, 0, 1);
  gpism::Field2D<double> thk_cpu(mx, my, gw);
  gpism::Field2D<double> topg_cpu(mx, my, gw);
  gpism::Field2D<double> tauc_cpu(mx, my, gw);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      thk_cpu(i, j) = 2.0;
      topg_cpu(i, j) = 0.1 * i - 0.07 * j;
      tauc_cpu(i, j) = 120.0;
    }
  }

  gpism::ViscosityModel viscosity_cpu(1e-16, 3.0, 1.0);
  gpism::SSASolver solver_cpu(grid_cpu, 910.0, 9.81, 100.0, viscosity_cpu);
  gpism::FieldStag2D<double> vel_cpu(mx, my, gw);
  vel_cpu.fill(0.0);
  gpism::SSASolverResult cpu_result =
      solver_cpu.solve(thk_cpu, topg_cpu, tauc_cpu, nullptr, nullptr, nullptr,
                       vel_cpu, options);

  gpism::set_device_enabled(true);
  gpism::Grid2D grid_gpu(mx, my, 1.0, 1.0, gw, 0, 1);
  gpism::Field2D<double> thk_gpu(mx, my, gw);
  gpism::Field2D<double> topg_gpu(mx, my, gw);
  gpism::Field2D<double> tauc_gpu(mx, my, gw);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      thk_gpu(i, j) = 2.0;
      topg_gpu(i, j) = 0.1 * i - 0.07 * j;
      tauc_gpu(i, j) = 120.0;
    }
  }

  gpism::ViscosityModel viscosity_gpu(1e-16, 3.0, 1.0);
  gpism::SSASolver solver_gpu(grid_gpu, 910.0, 9.81, 100.0, viscosity_gpu);
  gpism::FieldStag2D<double> vel_gpu(mx, my, gw);
  vel_gpu.fill(0.0);
  gpism::SSASolverResult gpu_result =
      solver_gpu.solve(thk_gpu, topg_gpu, tauc_gpu, nullptr, nullptr, nullptr,
                       vel_gpu, options);

  if (!cpu_result.converged || !gpu_result.converged) {
    std::cerr << "Picard solver did not converge for CPU/GPU parity test\n";
    std::cerr << "cpu_iters=" << cpu_result.picard_iters
              << " cpu_res=" << cpu_result.linear_residual
              << " gpu_iters=" << gpu_result.picard_iters
              << " gpu_res=" << gpu_result.linear_residual << "\n";
    return 1;
  }

  const double denom = std::max(1e-12, vel_norm2(vel_cpu));
  const double rel_err = std::sqrt(vel_diff_norm2(vel_cpu, vel_gpu) / denom);

  if (!std::isfinite(rel_err) || rel_err > 5e-2) {
    std::cerr << "CPU/GPU parity relative error too large: " << rel_err << "\n";
    std::cerr << "cpu_norm=" << std::sqrt(vel_norm2(vel_cpu))
              << " gpu_norm=" << std::sqrt(vel_norm2(vel_gpu)) << "\n";
    std::cerr << "cpu_res=" << cpu_result.linear_residual
              << " gpu_res=" << gpu_result.linear_residual << "\n";
    return 1;
  }

  std::cout << "ssa_picard_parity_smoke passed\n";
  return 0;
}
