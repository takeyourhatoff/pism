#include "gpism/config.h"
#include "gpism/geometry.h"
#include "gpism/netcdf_io.h"
#include "gpism/ssa_solver.h"
#include "gpism/thickness.h"
#include "gpism/time_manager.h"
#include "gpism/viscosity.h"

#include <cmath>
#include <iostream>

int main() {
#if !GPISM_HAVE_NETCDF
  std::cout << "timestep_io_smoke skipped (NetCDF disabled)\n";
  return 0;
#else
  const int mx = 4;
  const int my = 4;
  const int gw = 1;
  gpism::Grid2D grid(mx, my, 1.0, 1.0, gw, 0, 1);
  gpism::IOFields2D fields;
  fields.thk.resize(mx, my, gw);
  fields.topg.resize(mx, my, gw);
  fields.tauc.resize(mx, my, gw);
  fields.uvel.resize(mx, my, gw);
  fields.vvel.resize(mx, my, gw);
  fields.usurf.resize(mx, my, gw);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      fields.thk(i, j) = 2.0;
      fields.topg(i, j) = 0.1 * i - 0.05 * j;
      fields.tauc(i, j) = 120.0;
    }
  }

  gpism::Field2D<double> smb(mx, my, gw);
  smb.fill(0.1);
  gpism::FieldStag2D<double> vel(mx, my, gw);
  gpism::FieldStag2D<double> flux(mx, my, gw);
  vel.fill(0.0);

  gpism::ViscosityModel viscosity(1e-16, 3.0, 1.0);
  gpism::SSASolver solver(grid, 910.0, 9.81, 100.0, viscosity);
  gpism::SSASolverOptions options;
  options.max_picard = 10;
  options.tol_nuH = 0.7;
  options.tol_vel = 0.7;
  options.gmres_max_iter = 100;
  options.gmres_tol = 1e-7;
  options.use_bc = false;

  gpism::NetcdfIO io;
  gpism::TimeManager clock(0.0, 0.1, 1.0, 0.5);
  gpism::ThicknessUpdateOptions thickness_opts;

  int outputs = 0;
  while (!clock.done()) {
    gpism::SSASolverResult result =
        solver.solve(fields.thk, fields.topg, fields.tauc, nullptr, nullptr, nullptr,
                     vel, options);
    if (!result.converged) {
      std::cerr << "SSA solver did not converge in IO timestep loop\n";
      return 1;
    }

    gpism::compute_face_fluxes(grid, fields.thk, vel, flux);
    gpism::update_thickness(grid, flux, smb, clock.dt(), thickness_opts,
                            fields.thk);
    gpism::GeometryDiagnostics::compute_usurf_cpu(grid, fields.thk,
                                                  fields.topg, fields.usurf);
    gpism::compute_cell_center_velocity(grid, vel, fields.uvel, fields.vvel);
    fields.has_usurf = true;
    fields.has_velocity = true;

    if (clock.should_output()) {
      const std::string path =
          "gpism_timestep_io_smoke_t" + std::to_string(outputs) + ".nc";
      if (!io.write_output(path, grid, fields, clock.time())) {
        std::cerr << "failed to write output\n";
        return 1;
      }
      clock.mark_output();
      ++outputs;
    }

    clock.advance();
  }

  if (outputs < 2) {
    std::cerr << "expected multiple outputs, got " << outputs << "\n";
    return 1;
  }

  std::cout << "timestep_io_smoke passed\n";
  return 0;
#endif
}
