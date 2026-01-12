#include "gpism/config.h"
#include "gpism/device_policy.h"
#include "gpism/field_sync.h"
#include "gpism/geometry.h"
#include "gpism/sync_stats.h"
#include "gpism/ssa_solver.h"
#include "gpism/thermodynamics.h"
#include "gpism/thickness.h"
#include "gpism/viscosity.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <utility>

int main() {
#if !GPISM_HAVE_CUDA
  std::cout << "timestep_gpu_smoke skipped (CUDA disabled)\n";
  return 0;
#else
  gpism::set_device_enabled(true);

  const int mx = 16;
  const int my = 16;
  const int gw = 1;
  const int nz = 16;
  const double dz = 1.0;
  const double dt = 0.1;
  const int steps = 5;

  gpism::Grid2D grid(mx, my, 1.0, 1.0, gw, 0, 1);

  gpism::Field2D<double> thk(mx, my, gw);
  gpism::Field2D<double> topg(mx, my, gw);
  gpism::Field2D<double> tauc(mx, my, gw);
  gpism::Field2D<double> smb(mx, my, gw);
  gpism::Field2D<double> uvel(mx, my, gw);
  gpism::Field2D<double> vvel(mx, my, gw);
  gpism::Field2D<double> usurf(mx, my, gw);
  gpism::Field2D<int> mask(mx, my, gw);
  gpism::FieldStag2D<double> vel(mx, my, gw);
  gpism::FieldStag2D<double> flux(mx, my, gw);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      thk(i, j) = 2.0;
      topg(i, j) = 0.0;
      tauc(i, j) = 100.0;
      smb(i, j) = 0.05;
      mask(i, j) = 1;
    }
  }
  vel.fill(0.0);

  gpism::Field3D<double> enthalpy(mx, my, nz, gw);
  gpism::Field3D<double> enthalpy_next(mx, my, nz, gw);

  gpism::VerticalDiffusionOptions thermo_opts;
  thermo_opts.kappa = 1.0;
  thermo_opts.surface_value = -5.0;
  thermo_opts.basal_value = 5.0;
  thermo_opts.dirichlet = true;

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      for (int k = 0; k < nz; ++k) {
        const double t = (nz > 1) ? static_cast<double>(k) / (nz - 1) : 0.0;
        enthalpy(i, j, k) =
            (1.0 - t) * thermo_opts.surface_value + t * thermo_opts.basal_value;
      }
    }
  }

  gpism::sync_host_to_device(thk);
  gpism::sync_host_to_device(topg);
  gpism::sync_host_to_device(tauc);
  gpism::sync_host_to_device(smb);
  gpism::sync_host_to_device(vel);
  gpism::sync_host_to_device(enthalpy);
  gpism::sync_host_to_device(enthalpy_next);

  gpism::SyncStats::reset();
  gpism::SyncStats::enable(true);

  gpism::ViscosityModel viscosity(1e-16, 3.0, 1.0);
  gpism::SSASolver solver(grid, 910.0, 9.81, 100.0, viscosity);
  gpism::SSASolverOptions ssa_options;
  ssa_options.max_picard = 3;
  ssa_options.tol_nuH = 1e3;
  ssa_options.tol_vel = 1e3;
  ssa_options.gmres_max_iter = 20;
  ssa_options.gmres_tol = 1e-10;
  ssa_options.use_bc = false;

  gpism::ThicknessUpdateOptions thickness_opts;
  gpism::GeometryDiagnostics geometry;

  for (int step = 0; step < steps; ++step) {
    gpism::SSASolverResult result =
        solver.solve(thk, topg, tauc, nullptr, nullptr, nullptr, vel,
                     ssa_options);
    if (!result.converged) {
      std::cerr << "SSA solver did not converge in GPU timestep loop\n";
      return 1;
    }

    gpism::compute_face_fluxes(grid, thk, vel, flux);
    gpism::update_thickness(grid, flux, smb, dt, thickness_opts, thk);
    gpism::update_mask(grid, thk, mask);
    geometry.compute_usurf(grid, thk, topg, usurf);
    gpism::compute_cell_center_velocity(grid, vel, uvel, vvel);

    gpism::vertical_diffusion_step(enthalpy, nz, dz, dt, thermo_opts,
                                   enthalpy_next);
    std::swap(enthalpy, enthalpy_next);
  }

  const std::size_t h2d_calls = gpism::SyncStats::h2d_calls();
  const std::size_t d2h_calls = gpism::SyncStats::d2h_calls();
  if (h2d_calls != 0 || d2h_calls != 0) {
    std::cerr << "unexpected field syncs during hot loop (h2d=" << h2d_calls
              << ", d2h=" << d2h_calls << ")\n";
    return 1;
  }
  const std::size_t h2d_misc = gpism::SyncStats::h2d_misc_calls();
  const std::size_t d2h_misc = gpism::SyncStats::d2h_misc_calls();
  if (h2d_misc != 0 || d2h_misc != 0) {
    std::cerr << "non-field device transfers during hot loop (h2d_misc="
              << h2d_misc << ", d2h_misc=" << d2h_misc
              << ", h2d_bytes=" << gpism::SyncStats::h2d_misc_bytes()
              << ", d2h_bytes=" << gpism::SyncStats::d2h_misc_bytes() << ")\n";
    if (std::getenv("GPISM_SYNC_AUDIT_STRICT")) {
      return 1;
    }
  }
  gpism::SyncStats::enable(false);

  gpism::sync_device_to_host(thk);
  gpism::sync_device_to_host(uvel);
  gpism::sync_device_to_host(vvel);
  gpism::sync_device_to_host(enthalpy);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      if (!std::isfinite(thk(i, j)) || thk(i, j) < 0.0) {
        std::cerr << "invalid thickness after GPU timestep loop\n";
        return 1;
      }
      if (!std::isfinite(uvel(i, j)) || !std::isfinite(vvel(i, j))) {
        std::cerr << "invalid velocity after GPU timestep loop\n";
        return 1;
      }
      for (int k = 0; k < nz; ++k) {
        if (!std::isfinite(enthalpy(i, j, k))) {
          std::cerr << "invalid enthalpy after GPU timestep loop\n";
          return 1;
        }
      }
    }
  }

  std::cout << "timestep_gpu_smoke passed\n";
  return 0;
#endif
}
