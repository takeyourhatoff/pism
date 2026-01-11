#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <netcdf.h>

#include "gpism/context.h"
#include "gpism/netcdf_io.h"

namespace {

bool nearly_equal(double a, double b) {
  return std::fabs(a - b) < 1e-12;
}

bool check_nc(int status, const char* what) {
  if (status == NC_NOERR) {
    return true;
  }
  std::cerr << what << ": " << nc_strerror(status) << '\n';
  return false;
}

bool write_legacy_file(const std::string& path, int nx, int ny, double dx,
                       double dy) {
  int ncid = -1;
  if (!check_nc(nc_create(path.c_str(), NC_CLOBBER, &ncid), "nc_create")) {
    return false;
  }

  int dim_time = -1;
  int dim_x = -1;
  int dim_y = -1;
  if (!check_nc(nc_def_dim(ncid, "time", 1, &dim_time), "nc_def_dim time") ||
      !check_nc(nc_def_dim(ncid, "x1", nx, &dim_x), "nc_def_dim x1") ||
      !check_nc(nc_def_dim(ncid, "y1", ny, &dim_y), "nc_def_dim y1")) {
    nc_close(ncid);
    return false;
  }

  int var_time = -1;
  int var_x = -1;
  int var_y = -1;
  int var_thk = -1;
  int var_topg = -1;
  int var_tauc = -1;
  const int dims1[1] = {dim_time};
  const int dims_x[1] = {dim_x};
  const int dims_y[1] = {dim_y};
  const int dims3[3] = {dim_time, dim_y, dim_x};
  if (!check_nc(nc_def_var(ncid, "time", NC_DOUBLE, 1, dims1, &var_time),
                "nc_def_var time") ||
      !check_nc(nc_def_var(ncid, "x1", NC_DOUBLE, 1, dims_x, &var_x),
                "nc_def_var x1") ||
      !check_nc(nc_def_var(ncid, "y1", NC_DOUBLE, 1, dims_y, &var_y),
                "nc_def_var y1") ||
      !check_nc(nc_def_var(ncid, "thk", NC_DOUBLE, 3, dims3, &var_thk),
                "nc_def_var thk") ||
      !check_nc(nc_def_var(ncid, "topg", NC_DOUBLE, 3, dims3, &var_topg),
                "nc_def_var topg") ||
      !check_nc(nc_def_var(ncid, "tauc", NC_DOUBLE, 3, dims3, &var_tauc),
                "nc_def_var tauc")) {
    nc_close(ncid);
    return false;
  }

  if (!check_nc(nc_enddef(ncid), "nc_enddef")) {
    nc_close(ncid);
    return false;
  }

  double time = 0.0;
  std::size_t start_time[1] = {0};
  std::size_t count_time[1] = {1};
  if (!check_nc(nc_put_vara_double(ncid, var_time, start_time, count_time, &time),
                "nc_put_vara_double time")) {
    nc_close(ncid);
    return false;
  }

  std::vector<double> xvals(static_cast<std::size_t>(nx));
  std::vector<double> yvals(static_cast<std::size_t>(ny));
  for (int i = 0; i < nx; ++i) {
    xvals[static_cast<std::size_t>(i)] = i * dx;
  }
  for (int j = 0; j < ny; ++j) {
    yvals[static_cast<std::size_t>(j)] = j * dy;
  }
  std::size_t start_x[1] = {0};
  std::size_t count_x[1] = {static_cast<std::size_t>(nx)};
  std::size_t start_y[1] = {0};
  std::size_t count_y[1] = {static_cast<std::size_t>(ny)};
  if (!check_nc(nc_put_vara_double(ncid, var_x, start_x, count_x, xvals.data()),
                "nc_put_vara_double x1") ||
      !check_nc(nc_put_vara_double(ncid, var_y, start_y, count_y, yvals.data()),
                "nc_put_vara_double y1")) {
    nc_close(ncid);
    return false;
  }

  std::vector<double> thk(static_cast<std::size_t>(nx) * ny);
  std::vector<double> topg(static_cast<std::size_t>(nx) * ny);
  std::vector<double> tauc(static_cast<std::size_t>(nx) * ny);
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const std::size_t idx = static_cast<std::size_t>(j * nx + i);
      thk[idx] = 1.0 + i + j * 10.0;
      topg[idx] = -100.0 + i;
      tauc[idx] = 42.0;
    }
  }
  std::size_t start3[3] = {0, 0, 0};
  std::size_t count3[3] = {1, static_cast<std::size_t>(ny),
                           static_cast<std::size_t>(nx)};
  if (!check_nc(nc_put_vara_double(ncid, var_thk, start3, count3, thk.data()),
                "nc_put_vara_double thk") ||
      !check_nc(nc_put_vara_double(ncid, var_topg, start3, count3, topg.data()),
                "nc_put_vara_double topg") ||
      !check_nc(nc_put_vara_double(ncid, var_tauc, start3, count3, tauc.data()),
                "nc_put_vara_double tauc")) {
    nc_close(ncid);
    return false;
  }

  nc_close(ncid);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  gpism::Context context(&argc, &argv);
  if (context.size() != 1) {
    if (context.rank() == 0) {
      std::cout << "io_legacy_dims_smoke skipped (MPI size > 1)\n";
    }
    return 0;
  }

  const char* out_path = std::getenv("GPISM_IO_LEGACY_PATH");
  std::string path = out_path ? out_path : "/tmp/gpism_io_legacy_dims.nc";

  const int nx = 3;
  const int ny = 2;
  const double dx = 5000.0;
  const double dy = 5000.0;
  if (!write_legacy_file(path, nx, ny, dx, dy)) {
    std::cerr << "Failed to write legacy NetCDF file\n";
    return 1;
  }

  gpism::Grid2D read_grid(0, 0, 1.0, 1.0, 1, 0, 1);
  gpism::IOFields2D read_fields;
  gpism::NetcdfIO io;
  if (!io.read_restart(path, context, read_grid, read_fields)) {
    std::cerr << "Failed to read legacy NetCDF file\n";
    return 1;
  }

  if (read_grid.global_mx() != nx || read_grid.global_my() != ny) {
    std::cerr << "Legacy grid size mismatch\n";
    return 1;
  }
  if (!nearly_equal(read_grid.dx(), dx) || !nearly_equal(read_grid.dy(), dy)) {
    std::cerr << "Legacy grid spacing mismatch\n";
    return 1;
  }

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      if (!nearly_equal(read_fields.thk(i, j), 1.0 + i + j * 10.0) ||
          !nearly_equal(read_fields.topg(i, j), -100.0 + i) ||
          !nearly_equal(read_fields.tauc(i, j), 42.0)) {
        std::cerr << "Legacy data mismatch at (" << i << "," << j << ")\n";
        return 1;
      }
    }
  }

  if (context.rank() == 0) {
    std::remove(path.c_str());
    std::cout << "Legacy NetCDF IO smoke test: OK\n";
  }
  return 0;
}
