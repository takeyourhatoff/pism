#include <cmath>
#include <cstdlib>
#include <iostream>

#include "gpism/context.h"
#include "gpism/netcdf_io.h"

namespace {

bool nearly_equal(double a, double b) {
  return std::fabs(a - b) < 1e-12;
}

}  // namespace

int main(int argc, char** argv) {
  gpism::Context context(&argc, &argv);
  const char* out_path = std::getenv("GPISM_IO_SMOKE_PATH");
  std::string path = out_path ? out_path : "/tmp/gpism_io_smoke.nc";

  gpism::Grid2D grid(4, 4, 1.0, 1.0, 1, context.rank(), context.size());
  gpism::IOFields2D fields;
  fields.thk.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());
  fields.topg.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());
  fields.tauc.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());
  fields.u_bc.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());
  fields.v_bc.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());
  fields.vel_bc_mask.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());
  fields.has_vel_bc = true;

  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      const int gi = grid.xs() + i;
      const int gj = grid.ys() + j;
      fields.thk(i, j) = 1.0 + gi + gj * 10.0;
      fields.topg(i, j) = -100.0 + gi;
      fields.tauc(i, j) = 42.0;
      fields.u_bc(i, j) = 5.0 + gi;
      fields.v_bc(i, j) = -3.0 + gj;
      fields.vel_bc_mask(i, j) = (gi + gj) % 2;
    }
  }

  gpism::NetcdfIO io;
  if (!io.write_output(path, context, grid, fields)) {
    std::cerr << "Failed to write NetCDF output\n";
    return 1;
  }

  gpism::Grid2D read_grid(0, 0, 1.0, 1.0, 1, 0, 1);
  gpism::IOFields2D read_fields;
  if (!io.read_restart(path, context, read_grid, read_fields)) {
    std::cerr << "Failed to read NetCDF output\n";
    return 1;
  }

  if (read_grid.global_mx() != grid.global_mx() ||
      read_grid.global_my() != grid.global_my()) {
    std::cerr << "Grid size mismatch\n";
    return 1;
  }

  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      if (!nearly_equal(fields.thk(i, j), read_fields.thk(i, j)) ||
          !nearly_equal(fields.topg(i, j), read_fields.topg(i, j)) ||
          !nearly_equal(fields.tauc(i, j), read_fields.tauc(i, j)) ||
          !nearly_equal(fields.u_bc(i, j), read_fields.u_bc(i, j)) ||
          !nearly_equal(fields.v_bc(i, j), read_fields.v_bc(i, j)) ||
          fields.vel_bc_mask(i, j) != read_fields.vel_bc_mask(i, j)) {
        std::cerr << "Round-trip mismatch at (" << i << "," << j << ")\n";
        return 1;
      }
    }
  }

  if (context.rank() == 0) {
    std::remove(path.c_str());
    std::cout << "NetCDF IO smoke test: OK\n";
  }
  return 0;
}
