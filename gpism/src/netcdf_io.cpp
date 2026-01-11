#include "gpism/netcdf_io.h"

#include <netcdf.h>

#include <vector>

#include "gpism/config.h"

#if GPISM_HAVE_MPI
#include <mpi.h>
#endif

namespace gpism {
namespace {

bool get_dim_len(int ncid, const char* name, std::size_t* len, int* dimid_out) {
  int dimid = -1;
  if (nc_inq_dimid(ncid, name, &dimid) != NC_NOERR) {
    return false;
  }
  if (nc_inq_dimlen(ncid, dimid, len) != NC_NOERR) {
    return false;
  }
  if (dimid_out) {
    *dimid_out = dimid;
  }
  return true;
}

bool read_coord_spacing(int ncid, const char* name, double* spacing) {
  int varid = -1;
  if (nc_inq_varid(ncid, name, &varid) != NC_NOERR) {
    return false;
  }
  std::size_t start[1] = {0};
  std::size_t count[1] = {2};
  double values[2] = {0.0, 0.0};
  if (nc_get_vara_double(ncid, varid, start, count, values) != NC_NOERR) {
    return false;
  }
  *spacing = values[1] - values[0];
  return true;
}

bool read_global_attr(int ncid, const char* name, double* value) {
  return nc_get_att_double(ncid, NC_GLOBAL, name, value) == NC_NOERR;
}

bool read_var_2d_slice(int ncid, const char* name, int dim_time, std::size_t t_index,
                       int xs, int ys, int nx, int ny, Field2D<double>& field,
                       bool required, double default_value) {
  int varid = -1;
  if (nc_inq_varid(ncid, name, &varid) != NC_NOERR) {
    if (required) {
      return false;
    }
    field.fill(default_value);
    return true;
  }

  int ndims = 0;
  int dimids[NC_MAX_DIMS];
  if (nc_inq_var(ncid, varid, nullptr, nullptr, &ndims, dimids, nullptr) != NC_NOERR) {
    return false;
  }

  std::vector<std::size_t> start;
  std::vector<std::size_t> count;
  if (ndims == 3 && dimids[0] == dim_time) {
    start = {t_index, static_cast<std::size_t>(ys), static_cast<std::size_t>(xs)};
    count = {1, static_cast<std::size_t>(ny), static_cast<std::size_t>(nx)};
  } else if (ndims == 2) {
    start = {static_cast<std::size_t>(ys), static_cast<std::size_t>(xs)};
    count = {static_cast<std::size_t>(ny), static_cast<std::size_t>(nx)};
  } else {
    return false;
  }

  std::vector<double> buffer(static_cast<std::size_t>(nx) * ny);
  if (nc_get_vara_double(ncid, varid, start.data(), count.data(), buffer.data()) !=
      NC_NOERR) {
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

bool write_var_2d(int ncid, int varid, const Field2D<double>& field, int nx, int ny,
                  std::size_t t_index) {
  std::vector<double> buffer(static_cast<std::size_t>(nx) * ny);
  int idx = 0;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      buffer[static_cast<std::size_t>(idx++)] = field(i, j);
    }
  }
  std::size_t start[3] = {t_index, 0, 0};
  std::size_t count[3] = {1, static_cast<std::size_t>(ny), static_cast<std::size_t>(nx)};
  return nc_put_vara_double(ncid, varid, start, count, buffer.data()) == NC_NOERR;
}

bool read_restart_impl(const std::string& path, int rank, int size, Grid2D& grid,
                       IOFields2D& fields, int time_index_request) {
  int ncid = -1;
  if (nc_open(path.c_str(), NC_NOWRITE, &ncid) != NC_NOERR) {
    return false;
  }

  std::size_t nx = 0;
  std::size_t ny = 0;
  int dim_x = -1;
  int dim_y = -1;
  if (!get_dim_len(ncid, "x", &nx, &dim_x) ||
      !get_dim_len(ncid, "y", &ny, &dim_y)) {
    nc_close(ncid);
    return false;
  }

  int dim_time = -1;
  std::size_t nt = 0;
  bool has_time = get_dim_len(ncid, "time", &nt, &dim_time);
  std::size_t time_index = (has_time && nt > 0) ? (nt - 1) : 0;
  if (time_index_request >= 0 && has_time && nt > 0) {
    time_index = static_cast<std::size_t>(time_index_request);
    if (time_index >= nt) {
      time_index = nt - 1;
    }
  }

  double dx = 1.0;
  double dy = 1.0;
  if (!read_coord_spacing(ncid, "x", &dx)) {
    read_global_attr(ncid, "dx", &dx);
  }
  if (!read_coord_spacing(ncid, "y", &dy)) {
    read_global_attr(ncid, "dy", &dy);
  }

  grid = Grid2D(static_cast<int>(nx), static_cast<int>(ny), dx, dy, 1, rank, size);
  fields.thk.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());
  fields.topg.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());
  fields.tauc.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());

  const int xs = grid.xs();
  const int ys = grid.ys();
  const int mx = grid.local_mx();
  const int my = grid.local_my();

  bool ok = true;
  ok = read_var_2d_slice(ncid, "thk", dim_time, time_index, xs, ys, mx, my,
                         fields.thk, true, 0.0) &&
       ok;
  ok = read_var_2d_slice(ncid, "topg", dim_time, time_index, xs, ys, mx, my,
                         fields.topg, true, 0.0) &&
       ok;
  ok = read_var_2d_slice(ncid, "tauc", dim_time, time_index, xs, ys, mx, my,
                         fields.tauc, false, 0.0) &&
       ok;

  nc_close(ncid);
  return ok;
}

bool write_output_impl(const std::string& path, int rank, int size, bool mpi_enabled,
                       const Grid2D& grid, const IOFields2D& fields,
                       double time_value) {
  if (mpi_enabled && size > 1) {
    if (rank == 0) {
      int ncid = -1;
      if (nc_create(path.c_str(), NC_CLOBBER, &ncid) != NC_NOERR) {
        return false;
      }

      const int mx = grid.global_mx();
      const int my = grid.global_my();
      int dim_x = -1;
      int dim_y = -1;
      int dim_time = -1;
      if (nc_def_dim(ncid, "time", NC_UNLIMITED, &dim_time) != NC_NOERR ||
          nc_def_dim(ncid, "x", mx, &dim_x) != NC_NOERR ||
          nc_def_dim(ncid, "y", my, &dim_y) != NC_NOERR) {
        nc_close(ncid);
        return false;
      }

      int dims_tyx[3] = {dim_time, dim_y, dim_x};
      int var_time = -1;
      int var_x = -1;
      int var_y = -1;
      int var_thk = -1;
      int var_topg = -1;
      int var_tauc = -1;
      if (nc_def_var(ncid, "time", NC_DOUBLE, 1, &dim_time, &var_time) != NC_NOERR ||
          nc_def_var(ncid, "x", NC_DOUBLE, 1, &dim_x, &var_x) != NC_NOERR ||
          nc_def_var(ncid, "y", NC_DOUBLE, 1, &dim_y, &var_y) != NC_NOERR ||
          nc_def_var(ncid, "thk", NC_DOUBLE, 3, dims_tyx, &var_thk) != NC_NOERR ||
          nc_def_var(ncid, "topg", NC_DOUBLE, 3, dims_tyx, &var_topg) != NC_NOERR ||
          nc_def_var(ncid, "tauc", NC_DOUBLE, 3, dims_tyx, &var_tauc) != NC_NOERR) {
        nc_close(ncid);
        return false;
      }

      const char* units_m = "m";
      const char* units_pa = "Pa";
      const char* units_years = "years";
      nc_put_att_text(ncid, var_thk, "units", 1, units_m);
      nc_put_att_text(ncid, var_topg, "units", 1, units_m);
      nc_put_att_text(ncid, var_tauc, "units", 2, units_pa);
      nc_put_att_text(ncid, var_time, "units", 5, units_years);
      nc_put_att_text(ncid, var_x, "units", 1, units_m);
      nc_put_att_text(ncid, var_y, "units", 1, units_m);
      const std::string history = "gpism write_output";
      nc_put_att_text(ncid, NC_GLOBAL, "history", history.size(), history.c_str());

      if (nc_enddef(ncid) != NC_NOERR) {
        nc_close(ncid);
        return false;
      }

      std::size_t start_time[1] = {0};
      std::size_t count_time[1] = {1};
      nc_put_vara_double(ncid, var_time, start_time, count_time, &time_value);

      std::vector<double> xvals(static_cast<std::size_t>(mx));
      std::vector<double> yvals(static_cast<std::size_t>(my));
      for (int i = 0; i < mx; ++i) {
        xvals[static_cast<std::size_t>(i)] = i * grid.dx();
      }
      for (int j = 0; j < my; ++j) {
        yvals[static_cast<std::size_t>(j)] = j * grid.dy();
      }
      std::size_t start_x[1] = {0};
      std::size_t count_x[1] = {static_cast<std::size_t>(mx)};
      std::size_t start_y[1] = {0};
      std::size_t count_y[1] = {static_cast<std::size_t>(my)};
      nc_put_vara_double(ncid, var_x, start_x, count_x, xvals.data());
      nc_put_vara_double(ncid, var_y, start_y, count_y, yvals.data());
      nc_close(ncid);
    }
#if GPISM_HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    for (int r = 0; r < size; ++r) {
      if (r == rank) {
        int ncid = -1;
        if (nc_open(path.c_str(), NC_WRITE, &ncid) != NC_NOERR) {
          return false;
        }
        int var_thk = -1;
        int var_topg = -1;
        int var_tauc = -1;
        if (nc_inq_varid(ncid, "thk", &var_thk) != NC_NOERR ||
            nc_inq_varid(ncid, "topg", &var_topg) != NC_NOERR ||
            nc_inq_varid(ncid, "tauc", &var_tauc) != NC_NOERR) {
          nc_close(ncid);
          return false;
        }

        std::size_t start[3] = {0, static_cast<std::size_t>(grid.ys()),
                                static_cast<std::size_t>(grid.xs())};
        std::size_t count[3] = {1, static_cast<std::size_t>(grid.local_my()),
                                static_cast<std::size_t>(grid.local_mx())};
        std::vector<double> buffer(static_cast<std::size_t>(grid.local_mx()) *
                                   grid.local_my());
        auto write_local = [&](int varid, const Field2D<double>& field) {
          int idx = 0;
          for (int j = 0; j < grid.local_my(); ++j) {
            for (int i = 0; i < grid.local_mx(); ++i) {
              buffer[static_cast<std::size_t>(idx++)] = field(i, j);
            }
          }
          return nc_put_vara_double(ncid, varid, start, count, buffer.data()) ==
                 NC_NOERR;
        };

        bool ok = true;
        ok = write_local(var_thk, fields.thk) && ok;
        ok = write_local(var_topg, fields.topg) && ok;
        ok = write_local(var_tauc, fields.tauc) && ok;
        nc_close(ncid);
        if (!ok) {
          return false;
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
    return true;
  }

  int ncid = -1;
  if (nc_create(path.c_str(), NC_CLOBBER, &ncid) != NC_NOERR) {
    return false;
  }

  int dim_x = -1;
  int dim_y = -1;
  int dim_time = -1;
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  const int xs = grid.xs();
  const int ys = grid.ys();
  if (xs != 0 || ys != 0) {
    nc_close(ncid);
    return false;
  }
  if (nc_def_dim(ncid, "time", NC_UNLIMITED, &dim_time) != NC_NOERR ||
      nc_def_dim(ncid, "x", mx, &dim_x) != NC_NOERR ||
      nc_def_dim(ncid, "y", my, &dim_y) != NC_NOERR) {
    nc_close(ncid);
    return false;
  }

  int dims_tyx[3] = {dim_time, dim_y, dim_x};
  int var_time = -1;
  int var_x = -1;
  int var_y = -1;
  int var_thk = -1;
  int var_topg = -1;
  int var_tauc = -1;
  if (nc_def_var(ncid, "time", NC_DOUBLE, 1, &dim_time, &var_time) != NC_NOERR ||
      nc_def_var(ncid, "x", NC_DOUBLE, 1, &dim_x, &var_x) != NC_NOERR ||
      nc_def_var(ncid, "y", NC_DOUBLE, 1, &dim_y, &var_y) != NC_NOERR ||
      nc_def_var(ncid, "thk", NC_DOUBLE, 3, dims_tyx, &var_thk) != NC_NOERR ||
      nc_def_var(ncid, "topg", NC_DOUBLE, 3, dims_tyx, &var_topg) != NC_NOERR ||
      nc_def_var(ncid, "tauc", NC_DOUBLE, 3, dims_tyx, &var_tauc) != NC_NOERR) {
    nc_close(ncid);
    return false;
  }

  const char* units_m = "m";
  const char* units_pa = "Pa";
  const char* units_years = "years";
  nc_put_att_text(ncid, var_thk, "units", 1, units_m);
  nc_put_att_text(ncid, var_topg, "units", 1, units_m);
  nc_put_att_text(ncid, var_tauc, "units", 2, units_pa);
  nc_put_att_text(ncid, var_time, "units", 5, units_years);
  nc_put_att_text(ncid, var_x, "units", 1, units_m);
  nc_put_att_text(ncid, var_y, "units", 1, units_m);
  const std::string history = "gpism write_output";
  nc_put_att_text(ncid, NC_GLOBAL, "history", history.size(), history.c_str());

  if (nc_enddef(ncid) != NC_NOERR) {
    nc_close(ncid);
    return false;
  }

  std::size_t start_time[1] = {0};
  std::size_t count_time[1] = {1};
  nc_put_vara_double(ncid, var_time, start_time, count_time, &time_value);

  std::vector<double> xvals(static_cast<std::size_t>(mx));
  std::vector<double> yvals(static_cast<std::size_t>(my));
  for (int i = 0; i < mx; ++i) {
    xvals[static_cast<std::size_t>(i)] = i * grid.dx();
  }
  for (int j = 0; j < my; ++j) {
    yvals[static_cast<std::size_t>(j)] = j * grid.dy();
  }
  std::size_t start_x[1] = {0};
  std::size_t count_x[1] = {static_cast<std::size_t>(mx)};
  std::size_t start_y[1] = {0};
  std::size_t count_y[1] = {static_cast<std::size_t>(my)};
  nc_put_vara_double(ncid, var_x, start_x, count_x, xvals.data());
  nc_put_vara_double(ncid, var_y, start_y, count_y, yvals.data());

  bool ok = true;
  ok = write_var_2d(ncid, var_thk, fields.thk, mx, my, 0) && ok;
  ok = write_var_2d(ncid, var_topg, fields.topg, mx, my, 0) && ok;
  ok = write_var_2d(ncid, var_tauc, fields.tauc, mx, my, 0) && ok;

  nc_close(ncid);
  return ok;
}

}  // namespace

bool NetcdfIO::read_restart(const std::string& path, Grid2D& grid, IOFields2D& fields,
                            int time_index) {
  return read_restart_impl(path, 0, 1, grid, fields, time_index);
}

bool NetcdfIO::read_restart(const std::string& path, const Context& context, Grid2D& grid,
                            IOFields2D& fields, int time_index) {
  return read_restart_impl(path, context.rank(), context.size(), grid, fields,
                           time_index);
}

bool NetcdfIO::write_output(const std::string& path, const Grid2D& grid,
                             const IOFields2D& fields, double time_value) {
  return write_output_impl(path, 0, 1, false, grid, fields, time_value);
}

bool NetcdfIO::write_output(const std::string& path, const Context& context,
                             const Grid2D& grid, const IOFields2D& fields,
                             double time_value) {
  return write_output_impl(path, context.rank(), context.size(), context.mpi_enabled(),
                           grid, fields, time_value);
}

}  // namespace gpism
