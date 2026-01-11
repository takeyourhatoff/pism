#include "gpism/config.h"
#include "gpism/context.h"
#include "gpism/device_policy.h"
#include "gpism/field_sync.h"
#include "gpism/ssa_solver.h"

#include <cmath>
#include <iostream>

#if GPISM_HAVE_MPI
#include <mpi.h>
#endif

#if GPISM_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace {

double local_vel_norm2(const gpism::FieldStag2D<double>& vel) {
  double sum = 0.0;
  for (int j = 0; j < vel.local_my(); ++j) {
    for (int i = 0; i < vel.local_mx(); ++i) {
      sum += vel(i, j, 0) * vel(i, j, 0) + vel(i, j, 1) * vel(i, j, 1);
    }
  }
  return sum;
}

double global_norm(const gpism::Context& context, double local_sum) {
  double global_sum = local_sum;
#if GPISM_HAVE_MPI
  if (context.mpi_enabled()) {
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
  }
#endif
  return std::sqrt(global_sum);
}

}  // namespace

int main(int argc, char** argv) {
  gpism::Context context(&argc, &argv);
  gpism::set_device_enabled(true);

#if GPISM_HAVE_CUDA
  cudaSetDevice(context.device_id());
#endif

  const int global_mx = 8;
  const int global_my = 6;
  const int gw = 1;
  gpism::Grid2D grid(global_mx, global_my, 1000.0, 1000.0, gw, context.rank(),
                     context.size());

  gpism::Field2D<double> thk(grid.local_mx(), grid.local_my(), gw);
  gpism::Field2D<double> topg(grid.local_mx(), grid.local_my(), gw);
  gpism::Field2D<double> tauc(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> vel(grid.local_mx(), grid.local_my(), gw);

  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      const int gi = grid.xs() + i;
      const int gj = grid.ys() + j;
      thk(i, j) = 1000.0;
      topg(i, j) = 10.0 * gi - 5.0 * gj;
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
  options.context = &context;

  gpism::SSASolverResult result =
      solver.solve(thk, topg, tauc, nullptr, nullptr, nullptr, vel, options);
  gpism::sync_device_to_host(vel);

  if (!result.converged) {
    std::cerr << "Picard solver did not converge under MPI\n";
    std::cerr << "nuH_change=" << result.nuH_change
              << " vel_change=" << result.vel_change
              << " picard_iters=" << result.picard_iters
              << " linear_iters=" << result.linear_iters << "\n";
    return 1;
  }

  const double speed = global_norm(context, local_vel_norm2(vel));
  if (!std::isfinite(speed) || speed <= 0.0) {
    std::cerr << "velocity norm is invalid under MPI\n";
    return 1;
  }

  std::cout << "ssa_picard_mpi_smoke passed\n";
  return 0;
}
