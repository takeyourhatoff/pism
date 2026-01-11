#include "gpism/netcdf_io.h"

#include <netcdf.h>

#include <vector>

namespace gpism {
namespace {

bool read_var_2d(int ncid, const char* name, int xs, int ys, int nx, int ny,
                 Field2D<double>& field) {
  int varid = -1;
  if (nc_inq_varid(ncid, name, &varid) != NC_NOERR) {
    return false;
  }
  std::size_t start[2] = {static_cast<std::size_t>(ys), static_cast<std::size_t>(xs)};
  std::size_t count[2] = {static_cast<std::size_t>(ny), static_cast<std::size_t>(nx)};
  std::vector<double> buffer(static_cast<std::size_t>(nx) * ny);
  if (nc_get_vara_double(ncid, varid, start, count, buffer.data()) != NC_NOERR) {
    return false;
  }
  int idx = 0;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      field(i, j) = buffer[static_cast<std::size_t>(idx++)];
    }
  }
  return true;
}

bool write_var_2d(int ncid, int varid, const Field2D<double>& field, int nx, int ny) {
  std::vector<double> buffer(static_cast<std::size_t>(nx) * ny);
  int idx = 0;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      buffer[static_cast<std::size_t>(idx++)] = field(i, j);
    }
  }
  std::size_t start[2] = {0, 0};
  std::size_t count[2] = {static_cast<std::size_t>(ny), static_cast<std::size_t>(nx)};
  return nc_put_vara_double(ncid, varid, start, count, buffer.data()) == NC_NOERR;
}

}  // namespace

bool NetcdfIO::read_restart(const std::string& path, Grid2D& grid, IOFields2D& fields) {
  int ncid = -1;
  if (nc_open(path.c_str(), NC_NOWRITE, &ncid) != NC_NOERR) {
    return false;
  }

  int dim_x = -1;
  int dim_y = -1;
  if (nc_inq_dimid(ncid, "x", &dim_x) != NC_NOERR ||
      nc_inq_dimid(ncid, "y", &dim_y) != NC_NOERR) {
    nc_close(ncid);
    return false;
  }

  std::size_t nx = 0;
  std::size_t ny = 0;
  nc_inq_dimlen(ncid, dim_x, &nx);
  nc_inq_dimlen(ncid, dim_y, &ny);

  grid = Grid2D(static_cast<int>(nx), static_cast<int>(ny), 1.0, 1.0, 1, 0, 1);
  fields.thk.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());
  fields.topg.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());
  fields.tauc.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());

  const int xs = grid.xs();
  const int ys = grid.ys();
  const int mx = grid.local_mx();
  const int my = grid.local_my();

  bool ok = true;
  ok = read_var_2d(ncid, "thk", xs, ys, mx, my, fields.thk) && ok;
  ok = read_var_2d(ncid, "topg", xs, ys, mx, my, fields.topg) && ok;
  ok = read_var_2d(ncid, "tauc", xs, ys, mx, my, fields.tauc) && ok;

  nc_close(ncid);
  return ok;
}

bool NetcdfIO::write_output(const std::string& path, const Grid2D& grid,
                             const IOFields2D& fields) {
  int ncid = -1;
  if (nc_create(path.c_str(), NC_CLOBBER, &ncid) != NC_NOERR) {
    return false;
  }

  int dim_x = -1;
  int dim_y = -1;
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  const int xs = grid.xs();
  const int ys = grid.ys();
  if (xs != 0 || ys != 0) {
    nc_close(ncid);
    return false;
  }
  if (nc_def_dim(ncid, "x", mx, &dim_x) != NC_NOERR ||
      nc_def_dim(ncid, "y", my, &dim_y) != NC_NOERR) {
    nc_close(ncid);
    return false;
  }

  int dims[2] = {dim_y, dim_x};
  int var_thk = -1;
  int var_topg = -1;
  int var_tauc = -1;
  if (nc_def_var(ncid, "thk", NC_DOUBLE, 2, dims, &var_thk) != NC_NOERR ||
      nc_def_var(ncid, "topg", NC_DOUBLE, 2, dims, &var_topg) != NC_NOERR ||
      nc_def_var(ncid, "tauc", NC_DOUBLE, 2, dims, &var_tauc) != NC_NOERR) {
    nc_close(ncid);
    return false;
  }

  if (nc_enddef(ncid) != NC_NOERR) {
    nc_close(ncid);
    return false;
  }

  bool ok = true;
  ok = write_var_2d(ncid, var_thk, fields.thk, mx, my) && ok;
  ok = write_var_2d(ncid, var_topg, fields.topg, mx, my) && ok;
  ok = write_var_2d(ncid, var_tauc, fields.tauc, mx, my) && ok;

  nc_close(ncid);
  return ok;
}

}  // namespace gpism
