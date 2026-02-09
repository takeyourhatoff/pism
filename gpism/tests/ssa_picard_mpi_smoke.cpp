#include "gpism/config.h"
#include "gpism/context.h"
#include "gpism/device_policy.h"
#include "gpism/field_sync.h"
#include "gpism/sync_stats.h"
#include "gpism/ssa_solver.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

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
  options.sea_level = 0.0;
  options.rho_ice = 910.0;
  options.rho_water = 1028.0;
  options.surface_gradient_inward = false;
  options.surface_slope_uphill = true;
  options.use_cfbc = false;
  options.context = &context;

  vel.fill(0.0);
  gpism::sync_host_to_device(vel);
#if GPISM_HAVE_CUDA
  gpism::SyncStats::enable(true);
  gpism::SyncStats::reset();
#endif
  gpism::SSASolverResult result =
      solver.solve(thk, topg, tauc, nullptr, nullptr, nullptr, vel, options);
#if GPISM_HAVE_CUDA
  const std::size_t device_syncs =
      gpism::SyncStats::h2d_calls() + gpism::SyncStats::d2h_calls();
  if (context.size() > 1) {
    if (context.cuda_aware_mpi()) {
      if (device_syncs != 0) {
        std::cerr << "unexpected host staging with cuda-aware MPI (syncs="
                  << device_syncs << ")\n";
        return 1;
      }
    } else if (device_syncs == 0) {
      std::cerr << "expected host staging when cuda-aware MPI unavailable\n";
      return 1;
    }
  }
  gpism::SyncStats::enable(false);
#endif
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
    double min_val = std::numeric_limits<double>::infinity();
    double max_val = -std::numeric_limits<double>::infinity();
    for (int j = 0; j < vel.local_my(); ++j) {
      for (int i = 0; i < vel.local_mx(); ++i) {
        for (int comp = 0; comp < 2; ++comp) {
          const double value = vel(i, j, comp);
          if (!std::isfinite(value)) {
            continue;
          }
          min_val = std::min(min_val, value);
          max_val = std::max(max_val, value);
        }
      }
    }
    if (!std::isfinite(min_val)) {
      min_val = 0.0;
      max_val = 0.0;
    }
    gpism::Field2D<double> usurf(grid.local_mx(), grid.local_my(), gw);
    gpism::Field2D<double> dhdx(grid.local_mx(), grid.local_my(), gw);
    gpism::Field2D<double> dhdy(grid.local_mx(), grid.local_my(), gw);
    gpism::Field2D<int> cell_type(grid.local_mx(), grid.local_my(), gw);
    gpism::FieldStag2D<double> rhs(grid.local_mx(), grid.local_my(), gw);
    gpism::compute_cell_type(grid, thk, topg, options.sea_level,
                             options.rho_ice, options.rho_water,
                             options.ice_free_thickness_standard, cell_type);
    gpism::compute_usurf_flotation(grid, thk, topg, cell_type, options.sea_level,
                                   options.rho_ice, options.rho_water, usurf);
    gpism::compute_surface_slopes_pism(grid, usurf, cell_type, dhdx, dhdy,
                                       options.surface_gradient_inward,
                                       options.surface_slope_uphill,
                                       options.use_cfbc);
    gpism::SSAOperator op(options.rho_ice, 9.81);
    op.assemble_rhs(grid, thk, dhdx, dhdy, rhs);
    gpism::sync_device_to_host(rhs);
    double rhs_norm = 0.0;
    for (int j = 0; j < rhs.local_my(); ++j) {
      for (int i = 0; i < rhs.local_mx(); ++i) {
        rhs_norm += rhs(i, j, 0) * rhs(i, j, 0) + rhs(i, j, 1) * rhs(i, j, 1);
      }
    }
    rhs_norm = std::sqrt(rhs_norm);
    gpism::sync_device_to_host(usurf);
    gpism::sync_device_to_host(cell_type);
    gpism::sync_device_to_host(dhdx);
    gpism::sync_device_to_host(dhdy);
    double usurf_min = std::numeric_limits<double>::infinity();
    double usurf_max = -std::numeric_limits<double>::infinity();
    int cell_min = std::numeric_limits<int>::max();
    int cell_max = std::numeric_limits<int>::min();
    for (int j = 0; j < usurf.local_my(); ++j) {
      for (int i = 0; i < usurf.local_mx(); ++i) {
        usurf_min = std::min(usurf_min, usurf(i, j));
        usurf_max = std::max(usurf_max, usurf(i, j));
        cell_min = std::min(cell_min, cell_type(i, j));
        cell_max = std::max(cell_max, cell_type(i, j));
      }
    }
    const bool prev_device = gpism::device_enabled();
    gpism::set_device_enabled(false);
    gpism::Field2D<int> cell_type_cpu(grid.local_mx(), grid.local_my(), gw);
    gpism::Field2D<double> usurf_cpu(grid.local_mx(), grid.local_my(), gw);
    gpism::Field2D<double> dhdx_cpu(grid.local_mx(), grid.local_my(), gw);
    gpism::Field2D<double> dhdy_cpu(grid.local_mx(), grid.local_my(), gw);
    gpism::compute_cell_type(grid, thk, topg, options.sea_level,
                             options.rho_ice, options.rho_water,
                             options.ice_free_thickness_standard, cell_type_cpu);
    gpism::compute_usurf_flotation(grid, thk, topg, cell_type_cpu,
                                   options.sea_level, options.rho_ice,
                                   options.rho_water, usurf_cpu);
    gpism::compute_surface_slopes_pism(grid, usurf_cpu, cell_type_cpu,
                                       dhdx_cpu, dhdy_cpu,
                                       options.surface_gradient_inward,
                                       options.surface_slope_uphill,
                                       options.use_cfbc);
    gpism::FieldStag2D<double> rhs_cpu(grid.local_mx(), grid.local_my(), gw);
    op.assemble_rhs(grid, thk, dhdx_cpu, dhdy_cpu, rhs_cpu);
    double rhs_cpu_norm = 0.0;
    for (int j = 0; j < rhs_cpu.local_my(); ++j) {
      for (int i = 0; i < rhs_cpu.local_mx(); ++i) {
        rhs_cpu_norm += rhs_cpu(i, j, 0) * rhs_cpu(i, j, 0) +
                        rhs_cpu(i, j, 1) * rhs_cpu(i, j, 1);
      }
    }
    rhs_cpu_norm = std::sqrt(rhs_cpu_norm);
    gpism::set_device_enabled(prev_device);
    gpism::sync_device_to_host(thk);
    gpism::sync_device_to_host(topg);
    double thk_min = std::numeric_limits<double>::infinity();
    double thk_max = -std::numeric_limits<double>::infinity();
    double topg_min = std::numeric_limits<double>::infinity();
    double topg_max = -std::numeric_limits<double>::infinity();
    for (int j = 0; j < thk.local_my(); ++j) {
      for (int i = 0; i < thk.local_mx(); ++i) {
        thk_min = std::min(thk_min, thk(i, j));
        thk_max = std::max(thk_max, thk(i, j));
        topg_min = std::min(topg_min, topg(i, j));
        topg_max = std::max(topg_max, topg(i, j));
      }
    }
    std::cerr << "velocity norm is invalid under MPI (speed=" << speed
              << ", min=" << min_val << ", max=" << max_val
              << ", rhs_norm=" << rhs_norm
              << ", rhs_cpu_norm=" << rhs_cpu_norm
              << ", usurf[min,max]=[" << usurf_min << "," << usurf_max << "]"
              << ", cell_type[min,max]=[" << cell_min << "," << cell_max << "]"
              << ", thk[min,max]=[" << thk_min << "," << thk_max << "]"
              << ", topg[min,max]=[" << topg_min << "," << topg_max << "]"
              << ", rho_ice=" << options.rho_ice
              << ", rho_water=" << options.rho_water
              << ", sea_level=" << options.sea_level
              << ", device_enabled=" << gpism::device_enabled()
              << ")\n";
    return 1;
  }

  std::cout << "ssa_picard_mpi_smoke passed\n";
  return 0;
}
