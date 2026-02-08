#include "gpism/netcdf_io.h"

#include <netcdf.h>
#if defined(NC_HAS_PARALLEL) && NC_HAS_PARALLEL
#include <netcdf_par.h>
#endif

#include <cstring>
#include <fstream>
#include <type_traits>
#include <vector>

#include "gpism/config.h"
#include "gpism/field_sync.h"
#include "gpism/geometry.h"
#include "gpism/version.h"

#if GPISM_HAVE_MPI
#include <mpi.h>
#endif

namespace gpism {
namespace {

bool write_output_impl(const std::string& path, int rank, int size, bool mpi_enabled,
                       const Grid2D& grid, const IOFields2D& fields,
                       double time_value);

// Keep consistent with the gpism runtime config default ("constants.seconds_per_year").
constexpr double kSecondsPerYear = 31556926.0;

// Velocity-like NetCDF variables are stored in "m/year" for compatibility with
// PISM conventions. gpism's internal SSA solver uses SI units (m/s), so we need
// to undo the scale factor when reading restart inputs.
void scale_field_2d(Field2D<double>& field, double factor) {
  const int mx = field.local_mx();
  const int my = field.local_my();
  const int gw = field.ghost_width();
  for (int j = -gw; j < my + gw; ++j) {
    for (int i = -gw; i < mx + gw; ++i) {
      field(i, j) *= factor;
    }
  }
}

std::string to_lower(std::string s) {
  for (char& c : s) {
    if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
  }
  return s;
}

bool read_units_attr(int ncid, const char* var_name, std::string* units_out) {
  units_out->clear();
  int varid = -1;
  if (nc_inq_varid(ncid, var_name, &varid) != NC_NOERR) {
    return false;
  }
  std::size_t len = 0;
  if (nc_inq_attlen(ncid, varid, "units", &len) != NC_NOERR || len == 0) {
    return false;
  }
  std::string buf(len, '\0');
  if (nc_get_att_text(ncid, varid, "units", buf.data()) != NC_NOERR) {
    return false;
  }
  *units_out = buf;
  return true;
}

bool velocity_units_are_per_year(int ncid, const char* var_name) {
  // Prefer explicit units. If missing/unknown, keep the old behavior (assume
  // "m/year") since gpism writes that convention by default.
  std::string units;
  if (!read_units_attr(ncid, var_name, &units)) {
    return true;
  }
  const std::string u = to_lower(units);
  // Common patterns: "m year^-1", "m/year", "m a-1".
  if (u.find("year") != std::string::npos || u.find("/year") != std::string::npos ||
      u.find("a-1") != std::string::npos || u.find("a^-1") != std::string::npos ||
      u.find("yr") != std::string::npos) {
    return true;
  }
  // Common PISM output: "m s^-1".
  if (u.find("s-1") != std::string::npos || u.find("s^-1") != std::string::npos ||
      u.find("second") != std::string::npos || u.find("/s") != std::string::npos) {
    return false;
  }
  return true;
}

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

bool read_coord_origin(int ncid, const char* name, double* origin) {
  int varid = -1;
  if (nc_inq_varid(ncid, name, &varid) != NC_NOERR) {
    return false;
  }
  std::size_t start[1] = {0};
  std::size_t count[1] = {1};
  double value = 0.0;
  if (nc_get_vara_double(ncid, varid, start, count, &value) != NC_NOERR) {
    return false;
  }
  *origin = value;
  return true;
}

bool read_global_attr(int ncid, const char* name, double* value) {
  return nc_get_att_double(ncid, NC_GLOBAL, name, value) == NC_NOERR;
}

bool read_var_2d_slice(int ncid, const char* name, int dim_time, std::size_t t_index,
                       int xs, int ys, int nx, int ny, Field2D<double>& field,
                       bool required, double default_value, bool* found = nullptr) {
  int varid = -1;
  if (nc_inq_varid(ncid, name, &varid) != NC_NOERR) {
    if (found) {
      *found = false;
    }
    if (required) {
      return false;
    }
    field.fill(default_value);
    return true;
  }
  if (found) {
    *found = true;
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

bool read_var_2d_slice(int ncid, const char* name, int dim_time, std::size_t t_index,
                       int xs, int ys, int nx, int ny, Field2D<int>& field,
                       bool required, int default_value, bool* found = nullptr) {
  int varid = -1;
  if (nc_inq_varid(ncid, name, &varid) != NC_NOERR) {
    if (found) {
      *found = false;
    }
    if (required) {
      return false;
    }
    field.fill(default_value);
    return true;
  }
  if (found) {
    *found = true;
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

  std::vector<int> buffer(static_cast<std::size_t>(nx) * ny);
  if (nc_get_vara_int(ncid, varid, start.data(), count.data(), buffer.data()) !=
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

bool put_global_attr_text(int ncid, const char* name, const std::string& value) {
  if (value.empty()) {
    return true;
  }
  return nc_put_att_text(ncid, NC_GLOBAL, name, value.size(), value.c_str()) ==
         NC_NOERR;
}

void put_vel_bc_mask_attrs(int ncid, int varid) {
  if (varid < 0) {
    return;
  }
  const char* long_name = "SSA velocity Dirichlet mask";
  nc_put_att_text(ncid, varid, "long_name", std::strlen(long_name), long_name);
  const int flag_values[2] = {0, 1};
  nc_put_att_int(ncid, varid, "flag_values", NC_INT, 2, flag_values);
  const char* flag_meanings = "free dirichlet";
  nc_put_att_text(ncid, varid, "flag_meanings", std::strlen(flag_meanings),
                  flag_meanings);
  const char* comment =
      "1 enforces prescribed velocity (u_bc/v_bc) at faces; 0 leaves unconstrained";
  nc_put_att_text(ncid, varid, "comment", std::strlen(comment), comment);
}

bool file_exists(const std::string& path) {
  std::ifstream input(path);
  return input.good();
}

bool get_time_len(int ncid, std::size_t* len, int* dimid_out) {
  return get_dim_len(ncid, "time", len, dimid_out);
}

bool write_var_2d_scaled(int ncid, int varid, const Field2D<double>& field, int nx,
                         int ny, std::size_t t_index, double scale) {
  std::vector<double> buffer(static_cast<std::size_t>(nx) * ny);
  int idx = 0;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      buffer[static_cast<std::size_t>(idx++)] = scale * field(i, j);
    }
  }
  std::size_t start[3] = {t_index, 0, 0};
  std::size_t count[3] = {1, static_cast<std::size_t>(ny), static_cast<std::size_t>(nx)};
  return nc_put_vara_double(ncid, varid, start, count, buffer.data()) == NC_NOERR;
}

bool write_var_2d(int ncid, int varid, const Field2D<double>& field, int nx, int ny,
                  std::size_t t_index) {
  return write_var_2d_scaled(ncid, varid, field, nx, ny, t_index, 1.0);
}

bool write_var_2d(int ncid, int varid, const Field2D<int>& field, int nx, int ny,
                  std::size_t t_index) {
  std::vector<int> buffer(static_cast<std::size_t>(nx) * ny);
  int idx = 0;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      buffer[static_cast<std::size_t>(idx++)] = field(i, j);
    }
  }
  std::size_t start[3] = {t_index, 0, 0};
  std::size_t count[3] = {1, static_cast<std::size_t>(ny), static_cast<std::size_t>(nx)};
  return nc_put_vara_int(ncid, varid, start, count, buffer.data()) == NC_NOERR;
}

#if GPISM_HAVE_MPI && defined(NC_HAS_PARALLEL) && NC_HAS_PARALLEL
bool write_output_parallel(const std::string& path, MPI_Comm comm, int rank,
                           const Grid2D& grid, const IOFields2D& fields,
                           double time_value) {
  int ncid = -1;
  if (nc_create_par(path.c_str(), NC_NETCDF4 | NC_MPIIO, comm, MPI_INFO_NULL,
                    &ncid) != NC_NOERR) {
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
  int var_uvel = -1;
  int var_vvel = -1;
  int var_u_ssa = -1;
  int var_v_ssa = -1;
  int var_usurf = -1;
  int var_u_bc = -1;
  int var_v_bc = -1;
  int var_vel_bc_mask = -1;
  if (nc_def_var(ncid, "time", NC_DOUBLE, 1, &dim_time, &var_time) != NC_NOERR ||
      nc_def_var(ncid, "x", NC_DOUBLE, 1, &dim_x, &var_x) != NC_NOERR ||
      nc_def_var(ncid, "y", NC_DOUBLE, 1, &dim_y, &var_y) != NC_NOERR ||
      nc_def_var(ncid, "thk", NC_DOUBLE, 3, dims_tyx, &var_thk) != NC_NOERR ||
      nc_def_var(ncid, "topg", NC_DOUBLE, 3, dims_tyx, &var_topg) != NC_NOERR ||
      nc_def_var(ncid, "tauc", NC_DOUBLE, 3, dims_tyx, &var_tauc) != NC_NOERR) {
    nc_close(ncid);
    return false;
  }
  if (fields.has_velocity) {
    if (nc_def_var(ncid, "uvel", NC_DOUBLE, 3, dims_tyx, &var_uvel) != NC_NOERR ||
        nc_def_var(ncid, "vvel", NC_DOUBLE, 3, dims_tyx, &var_vvel) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }
  if (fields.has_ssa_velocity) {
    if (nc_def_var(ncid, "u_ssa", NC_DOUBLE, 3, dims_tyx, &var_u_ssa) != NC_NOERR ||
        nc_def_var(ncid, "v_ssa", NC_DOUBLE, 3, dims_tyx, &var_v_ssa) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }
  if (fields.has_usurf) {
    if (nc_def_var(ncid, "usurf", NC_DOUBLE, 3, dims_tyx, &var_usurf) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }
  if (fields.has_vel_bc) {
    if (nc_def_var(ncid, "u_bc", NC_DOUBLE, 3, dims_tyx, &var_u_bc) != NC_NOERR ||
        nc_def_var(ncid, "v_bc", NC_DOUBLE, 3, dims_tyx, &var_v_bc) != NC_NOERR ||
        nc_def_var(ncid, "vel_bc_mask", NC_INT, 3, dims_tyx,
                   &var_vel_bc_mask) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }

  const char* units_m = "m";
  const char* units_pa = "Pa";
  const char* units_years = "years";
  const char* units_velocity = "m year^-1";
  nc_put_att_text(ncid, var_thk, "units", 1, units_m);
  nc_put_att_text(ncid, var_topg, "units", 1, units_m);
  nc_put_att_text(ncid, var_tauc, "units", 2, units_pa);
  nc_put_att_text(ncid, var_time, "units", 5, units_years);
  nc_put_att_text(ncid, var_x, "units", 1, units_m);
  nc_put_att_text(ncid, var_y, "units", 1, units_m);
  if (fields.has_velocity) {
    nc_put_att_text(ncid, var_uvel, "units", 9, units_velocity);
    nc_put_att_text(ncid, var_vvel, "units", 9, units_velocity);
  }
  if (fields.has_ssa_velocity) {
    nc_put_att_text(ncid, var_u_ssa, "units", 9, units_velocity);
    nc_put_att_text(ncid, var_v_ssa, "units", 9, units_velocity);
  }
  if (fields.has_usurf) {
    nc_put_att_text(ncid, var_usurf, "units", 1, units_m);
  }
  if (fields.has_vel_bc) {
    nc_put_att_text(ncid, var_u_bc, "units", 9, units_velocity);
    nc_put_att_text(ncid, var_v_bc, "units", 9, units_velocity);
    put_vel_bc_mask_attrs(ncid, var_vel_bc_mask);
  }
  const std::string history = "gpism write_output";
  nc_put_att_text(ncid, NC_GLOBAL, "history", history.size(), history.c_str());
  put_global_attr_text(ncid, "gpism_version", GPISM_VERSION);
  put_global_attr_text(ncid, "gpism_build_type", GPISM_BUILD_TYPE);
  if (std::strcmp(GPISM_GIT_SHA, "unknown") != 0) {
    put_global_attr_text(ncid, "gpism_git_sha", GPISM_GIT_SHA);
  }

  if (nc_enddef(ncid) != NC_NOERR) {
    nc_close(ncid);
    return false;
  }

  if (nc_var_par_access(ncid, var_time, NC_INDEPENDENT) != NC_NOERR ||
      nc_var_par_access(ncid, var_x, NC_INDEPENDENT) != NC_NOERR ||
      nc_var_par_access(ncid, var_y, NC_INDEPENDENT) != NC_NOERR ||
      nc_var_par_access(ncid, var_thk, NC_INDEPENDENT) != NC_NOERR ||
      nc_var_par_access(ncid, var_topg, NC_INDEPENDENT) != NC_NOERR ||
      nc_var_par_access(ncid, var_tauc, NC_INDEPENDENT) != NC_NOERR) {
    nc_close(ncid);
    return false;
  }
  if (fields.has_velocity) {
    if (nc_var_par_access(ncid, var_uvel, NC_INDEPENDENT) != NC_NOERR ||
        nc_var_par_access(ncid, var_vvel, NC_INDEPENDENT) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }
  if (fields.has_ssa_velocity) {
    if (nc_var_par_access(ncid, var_u_ssa, NC_INDEPENDENT) != NC_NOERR ||
        nc_var_par_access(ncid, var_v_ssa, NC_INDEPENDENT) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }
  if (fields.has_usurf) {
    if (nc_var_par_access(ncid, var_usurf, NC_INDEPENDENT) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }
  if (fields.has_vel_bc) {
    if (nc_var_par_access(ncid, var_u_bc, NC_INDEPENDENT) != NC_NOERR ||
        nc_var_par_access(ncid, var_v_bc, NC_INDEPENDENT) != NC_NOERR ||
        nc_var_par_access(ncid, var_vel_bc_mask, NC_INDEPENDENT) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }

  if (rank == 0) {
    std::size_t start_time[1] = {0};
    std::size_t count_time[1] = {1};
    nc_put_vara_double(ncid, var_time, start_time, count_time, &time_value);

    std::vector<double> xvals(static_cast<std::size_t>(mx));
    std::vector<double> yvals(static_cast<std::size_t>(my));
    for (int i = 0; i < mx; ++i) {
      xvals[static_cast<std::size_t>(i)] = grid.x0() + i * grid.dx();
    }
    for (int j = 0; j < my; ++j) {
      yvals[static_cast<std::size_t>(j)] = grid.y0() + j * grid.dy();
    }
    std::size_t start_x[1] = {0};
    std::size_t count_x[1] = {static_cast<std::size_t>(mx)};
    std::size_t start_y[1] = {0};
    std::size_t count_y[1] = {static_cast<std::size_t>(my)};
    nc_put_vara_double(ncid, var_x, start_x, count_x, xvals.data());
    nc_put_vara_double(ncid, var_y, start_y, count_y, yvals.data());
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
  auto write_local_scaled = [&](int varid, const Field2D<double>& field,
                                double scale) {
    int idx = 0;
    for (int j = 0; j < grid.local_my(); ++j) {
      for (int i = 0; i < grid.local_mx(); ++i) {
        buffer[static_cast<std::size_t>(idx++)] = scale * field(i, j);
      }
    }
    return nc_put_vara_double(ncid, varid, start, count, buffer.data()) ==
           NC_NOERR;
  };
  std::vector<int> mask_buffer(static_cast<std::size_t>(grid.local_mx()) *
                               grid.local_my());
  auto write_local_mask = [&](int varid, const Field2D<int>& field) {
    int idx = 0;
    for (int j = 0; j < grid.local_my(); ++j) {
      for (int i = 0; i < grid.local_mx(); ++i) {
        mask_buffer[static_cast<std::size_t>(idx++)] = field(i, j);
      }
    }
    return nc_put_vara_int(ncid, varid, start, count, mask_buffer.data()) ==
           NC_NOERR;
  };

  bool ok = true;
  ok = write_local(var_thk, fields.thk) && ok;
  ok = write_local(var_topg, fields.topg) && ok;
  ok = write_local(var_tauc, fields.tauc) && ok;
  if (fields.has_velocity) {
    ok = write_local_scaled(var_uvel, fields.uvel, kSecondsPerYear) && ok;
    ok = write_local_scaled(var_vvel, fields.vvel, kSecondsPerYear) && ok;
  }
  if (fields.has_ssa_velocity) {
    ok = write_local_scaled(var_u_ssa, fields.u_ssa, kSecondsPerYear) && ok;
    ok = write_local_scaled(var_v_ssa, fields.v_ssa, kSecondsPerYear) && ok;
  }
  if (fields.has_usurf) {
    ok = write_local(var_usurf, fields.usurf) && ok;
  }
  if (fields.has_vel_bc) {
    ok = write_local_scaled(var_u_bc, fields.u_bc, kSecondsPerYear) && ok;
    ok = write_local_scaled(var_v_bc, fields.v_bc, kSecondsPerYear) && ok;
    ok = write_local_mask(var_vel_bc_mask, fields.vel_bc_mask) && ok;
  }
  nc_close(ncid);
  return ok;
}
#endif

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
  const char* dim_x_name = nullptr;
  const char* dim_y_name = nullptr;
  if (get_dim_len(ncid, "x", &nx, &dim_x)) {
    dim_x_name = "x";
  } else if (get_dim_len(ncid, "x1", &nx, &dim_x)) {
    dim_x_name = "x1";
  }
  if (get_dim_len(ncid, "y", &ny, &dim_y)) {
    dim_y_name = "y";
  } else if (get_dim_len(ncid, "y1", &ny, &dim_y)) {
    dim_y_name = "y1";
  }
  if (!dim_x_name || !dim_y_name) {
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
  if (!read_coord_spacing(ncid, dim_x_name, &dx)) {
    read_global_attr(ncid, "dx", &dx);
  }
  if (!read_coord_spacing(ncid, dim_y_name, &dy)) {
    read_global_attr(ncid, "dy", &dy);
  }

  double x0 = 0.0;
  double y0 = 0.0;
  (void)read_coord_origin(ncid, dim_x_name, &x0);
  (void)read_coord_origin(ncid, dim_y_name, &y0);
  grid = Grid2D(static_cast<int>(nx), static_cast<int>(ny), dx, dy, x0, y0, 1,
                rank, size);
  fields.thk.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());
  fields.topg.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());
  fields.tauc.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());
  fields.u_bc.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());
  fields.v_bc.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());
  fields.vel_bc_mask.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());
  fields.uvel.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());
  fields.vvel.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());
  fields.u_ssa.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());
  fields.v_ssa.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());
  fields.usurf.resize(grid.local_mx(), grid.local_my(), grid.ghost_width());

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
  bool has_tauc = false;
  ok = read_var_2d_slice(ncid, "tauc", dim_time, time_index, xs, ys, mx, my,
                         fields.tauc, false, 0.0, &has_tauc) &&
       ok;
  bool has_u_bc = false;
  bool has_v_bc = false;
  bool has_vel_bc_mask = false;
  ok = read_var_2d_slice(ncid, "u_bc", dim_time, time_index, xs, ys, mx, my,
                         fields.u_bc, false, 0.0, &has_u_bc) &&
       ok;
  ok = read_var_2d_slice(ncid, "v_bc", dim_time, time_index, xs, ys, mx, my,
                         fields.v_bc, false, 0.0, &has_v_bc) &&
       ok;
  ok = read_var_2d_slice(ncid, "vel_bc_mask", dim_time, time_index, xs, ys, mx, my,
                         fields.vel_bc_mask, false, 0, &has_vel_bc_mask) &&
       ok;
  fields.has_vel_bc = has_u_bc || has_v_bc || has_vel_bc_mask;
  fields.has_tauc = has_tauc;

  bool has_uvel = false;
  bool has_vvel = false;
  bool has_u_ssa = false;
  bool has_v_ssa = false;
  bool has_usurf = false;
  ok = read_var_2d_slice(ncid, "uvel", dim_time, time_index, xs, ys, mx, my,
                         fields.uvel, false, 0.0, &has_uvel) &&
       ok;
  ok = read_var_2d_slice(ncid, "vvel", dim_time, time_index, xs, ys, mx, my,
                         fields.vvel, false, 0.0, &has_vvel) &&
       ok;
  ok = read_var_2d_slice(ncid, "u_ssa", dim_time, time_index, xs, ys, mx, my,
                         fields.u_ssa, false, 0.0, &has_u_ssa) &&
       ok;
  ok = read_var_2d_slice(ncid, "v_ssa", dim_time, time_index, xs, ys, mx, my,
                         fields.v_ssa, false, 0.0, &has_v_ssa) &&
       ok;
  ok = read_var_2d_slice(ncid, "usurf", dim_time, time_index, xs, ys, mx, my,
                         fields.usurf, false, 0.0, &has_usurf) &&
       ok;
  fields.has_velocity = has_uvel || has_vvel;
  fields.has_ssa_velocity = has_u_ssa || has_v_ssa;
  fields.has_usurf = has_usurf;

  const double inv_seconds_per_year =
      (kSecondsPerYear > 0.0) ? (1.0 / kSecondsPerYear) : 1.0;
  if (has_u_bc) {
    if (velocity_units_are_per_year(ncid, "u_bc")) {
      scale_field_2d(fields.u_bc, inv_seconds_per_year);
    }
  }
  if (has_v_bc) {
    if (velocity_units_are_per_year(ncid, "v_bc")) {
      scale_field_2d(fields.v_bc, inv_seconds_per_year);
    }
  }
  if (has_uvel) {
    if (velocity_units_are_per_year(ncid, "uvel")) {
      scale_field_2d(fields.uvel, inv_seconds_per_year);
    }
  }
  if (has_vvel) {
    if (velocity_units_are_per_year(ncid, "vvel")) {
      scale_field_2d(fields.vvel, inv_seconds_per_year);
    }
  }
  if (has_u_ssa) {
    if (velocity_units_are_per_year(ncid, "u_ssa")) {
      scale_field_2d(fields.u_ssa, inv_seconds_per_year);
    }
  }
  if (has_v_ssa) {
    if (velocity_units_are_per_year(ncid, "v_ssa")) {
      scale_field_2d(fields.v_ssa, inv_seconds_per_year);
    }
  }

  nc_close(ncid);
  return ok;
}

bool write_output_append_serial(const std::string& path, const Grid2D& grid,
                                const IOFields2D& fields, double time_value) {
  int ncid = -1;
  if (nc_open(path.c_str(), NC_WRITE, &ncid) != NC_NOERR) {
    return false;
  }

  std::size_t nt = 0;
  int dim_time = -1;
  if (!get_time_len(ncid, &nt, &dim_time)) {
    nc_close(ncid);
    return false;
  }
  const std::size_t t_index = nt;

  int var_time = -1;
  int var_thk = -1;
  int var_topg = -1;
  int var_tauc = -1;
  int var_uvel = -1;
  int var_vvel = -1;
  int var_u_ssa = -1;
  int var_v_ssa = -1;
  int var_usurf = -1;
  int var_u_bc = -1;
  int var_v_bc = -1;
  int var_vel_bc_mask = -1;
  if (nc_inq_varid(ncid, "time", &var_time) != NC_NOERR ||
      nc_inq_varid(ncid, "thk", &var_thk) != NC_NOERR ||
      nc_inq_varid(ncid, "topg", &var_topg) != NC_NOERR ||
      nc_inq_varid(ncid, "tauc", &var_tauc) != NC_NOERR) {
    nc_close(ncid);
    return false;
  }
  if (fields.has_velocity) {
    if (nc_inq_varid(ncid, "uvel", &var_uvel) != NC_NOERR ||
        nc_inq_varid(ncid, "vvel", &var_vvel) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }
  if (fields.has_ssa_velocity) {
    if (nc_inq_varid(ncid, "u_ssa", &var_u_ssa) != NC_NOERR ||
        nc_inq_varid(ncid, "v_ssa", &var_v_ssa) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }
  if (fields.has_usurf) {
    if (nc_inq_varid(ncid, "usurf", &var_usurf) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }
  if (fields.has_vel_bc) {
    if (nc_inq_varid(ncid, "u_bc", &var_u_bc) != NC_NOERR ||
        nc_inq_varid(ncid, "v_bc", &var_v_bc) != NC_NOERR ||
        nc_inq_varid(ncid, "vel_bc_mask", &var_vel_bc_mask) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }

  std::size_t start_time[1] = {t_index};
  std::size_t count_time[1] = {1};
  nc_put_vara_double(ncid, var_time, start_time, count_time, &time_value);

  const int mx = grid.local_mx();
  const int my = grid.local_my();
  bool ok = true;
  ok = write_var_2d(ncid, var_thk, fields.thk, mx, my, t_index) && ok;
  ok = write_var_2d(ncid, var_topg, fields.topg, mx, my, t_index) && ok;
  ok = write_var_2d(ncid, var_tauc, fields.tauc, mx, my, t_index) && ok;
  if (fields.has_velocity) {
    ok = write_var_2d_scaled(ncid, var_uvel, fields.uvel, mx, my, t_index,
                             kSecondsPerYear) &&
         ok;
    ok = write_var_2d_scaled(ncid, var_vvel, fields.vvel, mx, my, t_index,
                             kSecondsPerYear) &&
         ok;
  }
  if (fields.has_ssa_velocity) {
    ok = write_var_2d_scaled(ncid, var_u_ssa, fields.u_ssa, mx, my, t_index,
                             kSecondsPerYear) &&
         ok;
    ok = write_var_2d_scaled(ncid, var_v_ssa, fields.v_ssa, mx, my, t_index,
                             kSecondsPerYear) &&
         ok;
  }
  if (fields.has_usurf) {
    ok = write_var_2d(ncid, var_usurf, fields.usurf, mx, my, t_index) && ok;
  }
  if (fields.has_vel_bc) {
    ok = write_var_2d_scaled(ncid, var_u_bc, fields.u_bc, mx, my, t_index,
                             kSecondsPerYear) &&
         ok;
    ok = write_var_2d_scaled(ncid, var_v_bc, fields.v_bc, mx, my, t_index,
                             kSecondsPerYear) &&
         ok;
    ok = write_var_2d(ncid, var_vel_bc_mask, fields.vel_bc_mask, mx, my, t_index) &&
         ok;
  }

  nc_close(ncid);
  return ok;
}

#if GPISM_HAVE_MPI && defined(NC_HAS_PARALLEL) && NC_HAS_PARALLEL
bool write_output_append_parallel(const std::string& path, MPI_Comm comm,
                                  int rank, const Grid2D& grid,
                                  const IOFields2D& fields,
                                  double time_value) {
  int ncid = -1;
  if (nc_open_par(path.c_str(), NC_WRITE | NC_MPIIO, comm, MPI_INFO_NULL, &ncid) !=
      NC_NOERR) {
    return false;
  }

  std::size_t nt = 0;
  int dim_time = -1;
  if (!get_time_len(ncid, &nt, &dim_time)) {
    nc_close(ncid);
    return false;
  }
  const std::size_t t_index = nt;

  int var_time = -1;
  int var_thk = -1;
  int var_topg = -1;
  int var_tauc = -1;
  int var_uvel = -1;
  int var_vvel = -1;
  int var_u_ssa = -1;
  int var_v_ssa = -1;
  int var_usurf = -1;
  int var_u_bc = -1;
  int var_v_bc = -1;
  int var_vel_bc_mask = -1;
  if (nc_inq_varid(ncid, "time", &var_time) != NC_NOERR ||
      nc_inq_varid(ncid, "thk", &var_thk) != NC_NOERR ||
      nc_inq_varid(ncid, "topg", &var_topg) != NC_NOERR ||
      nc_inq_varid(ncid, "tauc", &var_tauc) != NC_NOERR) {
    nc_close(ncid);
    return false;
  }
  if (fields.has_velocity) {
    if (nc_inq_varid(ncid, "uvel", &var_uvel) != NC_NOERR ||
        nc_inq_varid(ncid, "vvel", &var_vvel) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }
  if (fields.has_ssa_velocity) {
    if (nc_inq_varid(ncid, "u_ssa", &var_u_ssa) != NC_NOERR ||
        nc_inq_varid(ncid, "v_ssa", &var_v_ssa) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }
  if (fields.has_usurf) {
    if (nc_inq_varid(ncid, "usurf", &var_usurf) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }
  if (fields.has_vel_bc) {
    if (nc_inq_varid(ncid, "u_bc", &var_u_bc) != NC_NOERR ||
        nc_inq_varid(ncid, "v_bc", &var_v_bc) != NC_NOERR ||
        nc_inq_varid(ncid, "vel_bc_mask", &var_vel_bc_mask) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }

  if (nc_var_par_access(ncid, var_time, NC_INDEPENDENT) != NC_NOERR ||
      nc_var_par_access(ncid, var_thk, NC_INDEPENDENT) != NC_NOERR ||
      nc_var_par_access(ncid, var_topg, NC_INDEPENDENT) != NC_NOERR ||
      nc_var_par_access(ncid, var_tauc, NC_INDEPENDENT) != NC_NOERR) {
    nc_close(ncid);
    return false;
  }
  if (fields.has_velocity) {
    if (nc_var_par_access(ncid, var_uvel, NC_INDEPENDENT) != NC_NOERR ||
        nc_var_par_access(ncid, var_vvel, NC_INDEPENDENT) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }
  if (fields.has_ssa_velocity) {
    if (nc_var_par_access(ncid, var_u_ssa, NC_INDEPENDENT) != NC_NOERR ||
        nc_var_par_access(ncid, var_v_ssa, NC_INDEPENDENT) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }
  if (fields.has_usurf) {
    if (nc_var_par_access(ncid, var_usurf, NC_INDEPENDENT) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }
  if (fields.has_vel_bc) {
    if (nc_var_par_access(ncid, var_u_bc, NC_INDEPENDENT) != NC_NOERR ||
        nc_var_par_access(ncid, var_v_bc, NC_INDEPENDENT) != NC_NOERR ||
        nc_var_par_access(ncid, var_vel_bc_mask, NC_INDEPENDENT) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }

  if (rank == 0) {
    std::size_t start_time[1] = {t_index};
    std::size_t count_time[1] = {1};
    nc_put_vara_double(ncid, var_time, start_time, count_time, &time_value);
  }

  std::size_t start[3] = {t_index, static_cast<std::size_t>(grid.ys()),
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
  auto write_local_scaled = [&](int varid, const Field2D<double>& field,
                                double scale) {
    int idx = 0;
    for (int j = 0; j < grid.local_my(); ++j) {
      for (int i = 0; i < grid.local_mx(); ++i) {
        buffer[static_cast<std::size_t>(idx++)] = scale * field(i, j);
      }
    }
    return nc_put_vara_double(ncid, varid, start, count, buffer.data()) ==
           NC_NOERR;
  };
  std::vector<int> mask_buffer(static_cast<std::size_t>(grid.local_mx()) *
                               grid.local_my());
  auto write_local_mask = [&](int varid, const Field2D<int>& field) {
    int idx = 0;
    for (int j = 0; j < grid.local_my(); ++j) {
      for (int i = 0; i < grid.local_mx(); ++i) {
        mask_buffer[static_cast<std::size_t>(idx++)] = field(i, j);
      }
    }
    return nc_put_vara_int(ncid, varid, start, count, mask_buffer.data()) ==
           NC_NOERR;
  };

  bool ok = true;
  ok = write_local(var_thk, fields.thk) && ok;
  ok = write_local(var_topg, fields.topg) && ok;
  ok = write_local(var_tauc, fields.tauc) && ok;
  if (fields.has_velocity) {
    ok = write_local_scaled(var_uvel, fields.uvel, kSecondsPerYear) && ok;
    ok = write_local_scaled(var_vvel, fields.vvel, kSecondsPerYear) && ok;
  }
  if (fields.has_usurf) {
    ok = write_local(var_usurf, fields.usurf) && ok;
  }
  if (fields.has_vel_bc) {
    ok = write_local_scaled(var_u_bc, fields.u_bc, kSecondsPerYear) && ok;
    ok = write_local_scaled(var_v_bc, fields.v_bc, kSecondsPerYear) && ok;
    ok = write_local_mask(var_vel_bc_mask, fields.vel_bc_mask) && ok;
  }

  nc_close(ncid);
  return ok;
}
#endif

bool write_output_append_serial_mpi(const std::string& path, int rank, int size,
                                    const Grid2D& grid,
                                    const IOFields2D& fields,
                                    double time_value) {
#if GPISM_HAVE_MPI
  if (rank == 0) {
    int ncid = -1;
    if (nc_open(path.c_str(), NC_WRITE, &ncid) != NC_NOERR) {
      return false;
    }
    std::size_t nt = 0;
    int dim_time = -1;
    if (!get_time_len(ncid, &nt, &dim_time)) {
      nc_close(ncid);
      return false;
    }
    int var_time = -1;
    if (nc_inq_varid(ncid, "time", &var_time) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
    std::size_t start_time[1] = {nt};
    std::size_t count_time[1] = {1};
    nc_put_vara_double(ncid, var_time, start_time, count_time, &time_value);
    nc_close(ncid);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  std::size_t nt = 0;
  if (rank == 0) {
    int ncid = -1;
    if (nc_open(path.c_str(), NC_NOWRITE, &ncid) == NC_NOERR) {
      int dim_time = -1;
      get_time_len(ncid, &nt, &dim_time);
      nc_close(ncid);
    }
  }
  MPI_Bcast(&nt, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  if (nt == 0) {
    return false;
  }
  const std::size_t t_index = nt - 1;

  for (int r = 0; r < size; ++r) {
    if (r == rank) {
      int ncid = -1;
      if (nc_open(path.c_str(), NC_WRITE, &ncid) != NC_NOERR) {
        return false;
      }
      int var_thk = -1;
      int var_topg = -1;
      int var_tauc = -1;
      int var_uvel = -1;
      int var_vvel = -1;
      int var_u_ssa = -1;
      int var_v_ssa = -1;
      int var_usurf = -1;
      int var_u_bc = -1;
      int var_v_bc = -1;
      int var_vel_bc_mask = -1;
      if (nc_inq_varid(ncid, "thk", &var_thk) != NC_NOERR ||
          nc_inq_varid(ncid, "topg", &var_topg) != NC_NOERR ||
          nc_inq_varid(ncid, "tauc", &var_tauc) != NC_NOERR) {
        nc_close(ncid);
        return false;
      }
      if (fields.has_velocity) {
        if (nc_inq_varid(ncid, "uvel", &var_uvel) != NC_NOERR ||
            nc_inq_varid(ncid, "vvel", &var_vvel) != NC_NOERR) {
          nc_close(ncid);
          return false;
        }
      }
      if (fields.has_ssa_velocity) {
        if (nc_inq_varid(ncid, "u_ssa", &var_u_ssa) != NC_NOERR ||
            nc_inq_varid(ncid, "v_ssa", &var_v_ssa) != NC_NOERR) {
          nc_close(ncid);
          return false;
        }
      }
      if (fields.has_usurf) {
        if (nc_inq_varid(ncid, "usurf", &var_usurf) != NC_NOERR) {
          nc_close(ncid);
          return false;
        }
      }
      if (fields.has_vel_bc) {
        if (nc_inq_varid(ncid, "u_bc", &var_u_bc) != NC_NOERR ||
            nc_inq_varid(ncid, "v_bc", &var_v_bc) != NC_NOERR ||
            nc_inq_varid(ncid, "vel_bc_mask", &var_vel_bc_mask) != NC_NOERR) {
          nc_close(ncid);
          return false;
        }
      }

      std::size_t start[3] = {t_index, static_cast<std::size_t>(grid.ys()),
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
      auto write_local_scaled = [&](int varid, const Field2D<double>& field,
                                    double scale) {
        int idx = 0;
        for (int j = 0; j < grid.local_my(); ++j) {
          for (int i = 0; i < grid.local_mx(); ++i) {
            buffer[static_cast<std::size_t>(idx++)] = scale * field(i, j);
          }
        }
        return nc_put_vara_double(ncid, varid, start, count, buffer.data()) ==
               NC_NOERR;
      };
      std::vector<int> mask_buffer(static_cast<std::size_t>(grid.local_mx()) *
                                   grid.local_my());
      auto write_local_mask = [&](int varid, const Field2D<int>& field) {
        int idx = 0;
        for (int j = 0; j < grid.local_my(); ++j) {
          for (int i = 0; i < grid.local_mx(); ++i) {
            mask_buffer[static_cast<std::size_t>(idx++)] = field(i, j);
          }
        }
        return nc_put_vara_int(ncid, varid, start, count, mask_buffer.data()) ==
               NC_NOERR;
      };

      bool ok = true;
      ok = write_local(var_thk, fields.thk) && ok;
      ok = write_local(var_topg, fields.topg) && ok;
      ok = write_local(var_tauc, fields.tauc) && ok;
      if (fields.has_velocity) {
        ok = write_local_scaled(var_uvel, fields.uvel, kSecondsPerYear) && ok;
        ok = write_local_scaled(var_vvel, fields.vvel, kSecondsPerYear) && ok;
      }
      if (fields.has_ssa_velocity) {
        ok = write_local_scaled(var_u_ssa, fields.u_ssa, kSecondsPerYear) && ok;
        ok = write_local_scaled(var_v_ssa, fields.v_ssa, kSecondsPerYear) && ok;
      }
      if (fields.has_usurf) {
        ok = write_local(var_usurf, fields.usurf) && ok;
      }
      if (fields.has_vel_bc) {
        ok = write_local_scaled(var_u_bc, fields.u_bc, kSecondsPerYear) && ok;
        ok = write_local_scaled(var_v_bc, fields.v_bc, kSecondsPerYear) && ok;
        ok = write_local_mask(var_vel_bc_mask, fields.vel_bc_mask) && ok;
      }
      nc_close(ncid);
      if (!ok) {
        return false;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  return true;
#else
  (void)path;
  (void)rank;
  (void)size;
  (void)grid;
  (void)fields;
  (void)time_value;
  return false;
#endif
}

bool write_output_append_impl(const std::string& path, int rank, int size,
                              bool mpi_enabled, const Grid2D& grid,
                              const IOFields2D& fields, double time_value) {
  if (!file_exists(path)) {
    return write_output_impl(path, rank, size, mpi_enabled, grid, fields,
                             time_value);
  }
#if GPISM_HAVE_MPI && defined(NC_HAS_PARALLEL) && NC_HAS_PARALLEL
  if (mpi_enabled && size > 1) {
    return write_output_append_parallel(path, MPI_COMM_WORLD, rank, grid, fields,
                                        time_value);
  }
#endif
  if (mpi_enabled && size > 1) {
    return write_output_append_serial_mpi(path, rank, size, grid, fields,
                                          time_value);
  }
  return write_output_append_serial(path, grid, fields, time_value);
}

bool write_ssa_debug_bundle_impl(const std::string& path, int rank, int size,
                                 bool mpi_enabled, const Grid2D& grid,
                                 const SSADebugBundle2D& bundle,
                                 double time_value) {
  if (!bundle.thk || !bundle.topg || !bundle.usurf || !bundle.dhdx ||
      !bundle.dhdy || !bundle.cell_type || !bundle.beta || !bundle.rhs ||
      !bundle.nuH || !bundle.vel_prev || !bundle.vel) {
    return false;
  }

  // NetCDF writers use host buffers; sync device-produced fields when present.
#if GPISM_HAVE_CUDA
  auto sync2d = [](const auto* field_ptr) {
    auto& field = *const_cast<std::remove_const_t<decltype(*field_ptr)>*>(field_ptr);
    if (field.has_device_data()) {
      sync_device_to_host(field);
    }
  };
  auto sync_stag = [](const auto* field_ptr) {
    auto& field = *const_cast<std::remove_const_t<decltype(*field_ptr)>*>(field_ptr);
    if (field.component(0).has_device_data() || field.component(1).has_device_data()) {
      sync_device_to_host(field);
    }
  };
  sync2d(bundle.thk);
  sync2d(bundle.topg);
  sync2d(bundle.usurf);
  sync2d(bundle.dhdx);
  sync2d(bundle.dhdy);
  sync2d(bundle.cell_type);
  sync_stag(bundle.beta);
  sync_stag(bundle.rhs);
  sync_stag(bundle.nuH);
  sync_stag(bundle.vel_prev);
  sync_stag(bundle.vel);
#endif

#if GPISM_HAVE_MPI && defined(NC_HAS_PARALLEL) && NC_HAS_PARALLEL
  if (mpi_enabled && size > 1) {
    int ncid = -1;
    if (nc_create_par(path.c_str(), NC_NETCDF4 | NC_MPIIO, MPI_COMM_WORLD,
                      MPI_INFO_NULL, &ncid) != NC_NOERR) {
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
    int var_usurf = -1;
    int var_dhdx = -1;
    int var_dhdy = -1;
    int var_cell_type = -1;
    int var_beta_u = -1;
    int var_beta_v = -1;
    int var_rhs_u = -1;
    int var_rhs_v = -1;
    int var_nuH_u = -1;
    int var_nuH_v = -1;
    int var_vel_u = -1;
    int var_vel_v = -1;
    int var_vel_prev_u = -1;
    int var_vel_prev_v = -1;

    if (nc_def_var(ncid, "time", NC_DOUBLE, 1, &dim_time, &var_time) != NC_NOERR ||
        nc_def_var(ncid, "x", NC_DOUBLE, 1, &dim_x, &var_x) != NC_NOERR ||
        nc_def_var(ncid, "y", NC_DOUBLE, 1, &dim_y, &var_y) != NC_NOERR ||
        nc_def_var(ncid, "thk", NC_DOUBLE, 3, dims_tyx, &var_thk) != NC_NOERR ||
        nc_def_var(ncid, "topg", NC_DOUBLE, 3, dims_tyx, &var_topg) != NC_NOERR ||
        nc_def_var(ncid, "usurf", NC_DOUBLE, 3, dims_tyx, &var_usurf) != NC_NOERR ||
        nc_def_var(ncid, "dhdx", NC_DOUBLE, 3, dims_tyx, &var_dhdx) != NC_NOERR ||
        nc_def_var(ncid, "dhdy", NC_DOUBLE, 3, dims_tyx, &var_dhdy) != NC_NOERR ||
        nc_def_var(ncid, "cell_type", NC_INT, 3, dims_tyx, &var_cell_type) != NC_NOERR ||
        nc_def_var(ncid, "beta_u", NC_DOUBLE, 3, dims_tyx, &var_beta_u) != NC_NOERR ||
        nc_def_var(ncid, "beta_v", NC_DOUBLE, 3, dims_tyx, &var_beta_v) != NC_NOERR ||
        nc_def_var(ncid, "rhs_u", NC_DOUBLE, 3, dims_tyx, &var_rhs_u) != NC_NOERR ||
        nc_def_var(ncid, "rhs_v", NC_DOUBLE, 3, dims_tyx, &var_rhs_v) != NC_NOERR ||
        nc_def_var(ncid, "nuH_u", NC_DOUBLE, 3, dims_tyx, &var_nuH_u) != NC_NOERR ||
        nc_def_var(ncid, "nuH_v", NC_DOUBLE, 3, dims_tyx, &var_nuH_v) != NC_NOERR ||
        nc_def_var(ncid, "vel_u", NC_DOUBLE, 3, dims_tyx, &var_vel_u) != NC_NOERR ||
        nc_def_var(ncid, "vel_v", NC_DOUBLE, 3, dims_tyx, &var_vel_v) != NC_NOERR ||
        nc_def_var(ncid, "vel_prev_u", NC_DOUBLE, 3, dims_tyx, &var_vel_prev_u) != NC_NOERR ||
        nc_def_var(ncid, "vel_prev_v", NC_DOUBLE, 3, dims_tyx, &var_vel_prev_v) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }

    const char* units_m = "m";
    const char* units_1 = "1";
    const char* units_years = "years";
    const char* units_vel = "m s^-1";
    const char* units_beta = "Pa s m^-1";
    const char* units_rhs = "Pa";
    const char* units_nuH = "Pa s m";
    nc_put_att_text(ncid, var_time, "units", 5, units_years);
    nc_put_att_text(ncid, var_x, "units", 1, units_m);
    nc_put_att_text(ncid, var_y, "units", 1, units_m);
    nc_put_att_text(ncid, var_thk, "units", 1, units_m);
    nc_put_att_text(ncid, var_topg, "units", 1, units_m);
    nc_put_att_text(ncid, var_usurf, "units", 1, units_m);
    nc_put_att_text(ncid, var_dhdx, "units", 1, units_1);
    nc_put_att_text(ncid, var_dhdy, "units", 1, units_1);
    nc_put_att_text(ncid, var_beta_u, "units", 9, units_beta);
    nc_put_att_text(ncid, var_beta_v, "units", 9, units_beta);
    nc_put_att_text(ncid, var_rhs_u, "units", 2, units_rhs);
    nc_put_att_text(ncid, var_rhs_v, "units", 2, units_rhs);
    nc_put_att_text(ncid, var_nuH_u, "units", 6, units_nuH);
    nc_put_att_text(ncid, var_nuH_v, "units", 6, units_nuH);
    nc_put_att_text(ncid, var_vel_u, "units", 6, units_vel);
    nc_put_att_text(ncid, var_vel_v, "units", 6, units_vel);
    nc_put_att_text(ncid, var_vel_prev_u, "units", 6, units_vel);
    nc_put_att_text(ncid, var_vel_prev_v, "units", 6, units_vel);

    if (nc_enddef(ncid) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }

    auto par_ind = [&](int varid) {
      return nc_var_par_access(ncid, varid, NC_INDEPENDENT) == NC_NOERR;
    };
    if (!par_ind(var_time) || !par_ind(var_x) || !par_ind(var_y) ||
        !par_ind(var_thk) || !par_ind(var_topg) || !par_ind(var_usurf) ||
        !par_ind(var_dhdx) || !par_ind(var_dhdy) || !par_ind(var_cell_type) ||
        !par_ind(var_beta_u) || !par_ind(var_beta_v) || !par_ind(var_rhs_u) ||
        !par_ind(var_rhs_v) || !par_ind(var_nuH_u) || !par_ind(var_nuH_v) ||
        !par_ind(var_vel_u) || !par_ind(var_vel_v) || !par_ind(var_vel_prev_u) ||
        !par_ind(var_vel_prev_v)) {
      nc_close(ncid);
      return false;
    }

    if (rank == 0) {
      std::size_t start_time[1] = {0};
      std::size_t count_time[1] = {1};
      nc_put_vara_double(ncid, var_time, start_time, count_time, &time_value);

      std::vector<double> xvals(static_cast<std::size_t>(mx));
      std::vector<double> yvals(static_cast<std::size_t>(my));
      for (int i = 0; i < mx; ++i) {
        xvals[static_cast<std::size_t>(i)] = grid.x0() + i * grid.dx();
      }
      for (int j = 0; j < my; ++j) {
        yvals[static_cast<std::size_t>(j)] = grid.y0() + j * grid.dy();
      }
      std::size_t start_x[1] = {0};
      std::size_t count_x[1] = {static_cast<std::size_t>(mx)};
      std::size_t start_y[1] = {0};
      std::size_t count_y[1] = {static_cast<std::size_t>(my)};
      nc_put_vara_double(ncid, var_x, start_x, count_x, xvals.data());
      nc_put_vara_double(ncid, var_y, start_y, count_y, yvals.data());
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
    std::vector<int> mask_buffer(static_cast<std::size_t>(grid.local_mx()) *
                                 grid.local_my());
    auto write_local_mask = [&](int varid, const Field2D<int>& field) {
      int idx = 0;
      for (int j = 0; j < grid.local_my(); ++j) {
        for (int i = 0; i < grid.local_mx(); ++i) {
          mask_buffer[static_cast<std::size_t>(idx++)] = field(i, j);
        }
      }
      return nc_put_vara_int(ncid, varid, start, count, mask_buffer.data()) ==
             NC_NOERR;
    };

    bool ok = true;
    ok = write_local(var_thk, *bundle.thk) && ok;
    ok = write_local(var_topg, *bundle.topg) && ok;
    ok = write_local(var_usurf, *bundle.usurf) && ok;
    ok = write_local(var_dhdx, *bundle.dhdx) && ok;
    ok = write_local(var_dhdy, *bundle.dhdy) && ok;
    ok = write_local_mask(var_cell_type, *bundle.cell_type) && ok;
    ok = write_local(var_beta_u, bundle.beta->component(0)) && ok;
    ok = write_local(var_beta_v, bundle.beta->component(1)) && ok;
    ok = write_local(var_rhs_u, bundle.rhs->component(0)) && ok;
    ok = write_local(var_rhs_v, bundle.rhs->component(1)) && ok;
    ok = write_local(var_nuH_u, bundle.nuH->component(0)) && ok;
    ok = write_local(var_nuH_v, bundle.nuH->component(1)) && ok;
    ok = write_local(var_vel_prev_u, bundle.vel_prev->component(0)) && ok;
    ok = write_local(var_vel_prev_v, bundle.vel_prev->component(1)) && ok;
    ok = write_local(var_vel_u, bundle.vel->component(0)) && ok;
    ok = write_local(var_vel_v, bundle.vel->component(1)) && ok;

    nc_close(ncid);
    return ok;
  }
#endif

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
      int var_usurf = -1;
      int var_dhdx = -1;
      int var_dhdy = -1;
      int var_cell_type = -1;
      int var_beta_u = -1;
      int var_beta_v = -1;
      int var_rhs_u = -1;
      int var_rhs_v = -1;
      int var_nuH_u = -1;
      int var_nuH_v = -1;
      int var_vel_u = -1;
      int var_vel_v = -1;
      int var_vel_prev_u = -1;
      int var_vel_prev_v = -1;
      if (nc_def_var(ncid, "time", NC_DOUBLE, 1, &dim_time, &var_time) != NC_NOERR ||
          nc_def_var(ncid, "x", NC_DOUBLE, 1, &dim_x, &var_x) != NC_NOERR ||
          nc_def_var(ncid, "y", NC_DOUBLE, 1, &dim_y, &var_y) != NC_NOERR ||
          nc_def_var(ncid, "thk", NC_DOUBLE, 3, dims_tyx, &var_thk) != NC_NOERR ||
          nc_def_var(ncid, "topg", NC_DOUBLE, 3, dims_tyx, &var_topg) != NC_NOERR ||
          nc_def_var(ncid, "usurf", NC_DOUBLE, 3, dims_tyx, &var_usurf) != NC_NOERR ||
          nc_def_var(ncid, "dhdx", NC_DOUBLE, 3, dims_tyx, &var_dhdx) != NC_NOERR ||
          nc_def_var(ncid, "dhdy", NC_DOUBLE, 3, dims_tyx, &var_dhdy) != NC_NOERR ||
          nc_def_var(ncid, "cell_type", NC_INT, 3, dims_tyx, &var_cell_type) != NC_NOERR ||
          nc_def_var(ncid, "beta_u", NC_DOUBLE, 3, dims_tyx, &var_beta_u) != NC_NOERR ||
          nc_def_var(ncid, "beta_v", NC_DOUBLE, 3, dims_tyx, &var_beta_v) != NC_NOERR ||
          nc_def_var(ncid, "rhs_u", NC_DOUBLE, 3, dims_tyx, &var_rhs_u) != NC_NOERR ||
          nc_def_var(ncid, "rhs_v", NC_DOUBLE, 3, dims_tyx, &var_rhs_v) != NC_NOERR ||
          nc_def_var(ncid, "nuH_u", NC_DOUBLE, 3, dims_tyx, &var_nuH_u) != NC_NOERR ||
          nc_def_var(ncid, "nuH_v", NC_DOUBLE, 3, dims_tyx, &var_nuH_v) != NC_NOERR ||
          nc_def_var(ncid, "vel_u", NC_DOUBLE, 3, dims_tyx, &var_vel_u) != NC_NOERR ||
          nc_def_var(ncid, "vel_v", NC_DOUBLE, 3, dims_tyx, &var_vel_v) != NC_NOERR ||
          nc_def_var(ncid, "vel_prev_u", NC_DOUBLE, 3, dims_tyx, &var_vel_prev_u) != NC_NOERR ||
          nc_def_var(ncid, "vel_prev_v", NC_DOUBLE, 3, dims_tyx, &var_vel_prev_v) != NC_NOERR) {
        nc_close(ncid);
        return false;
      }
      const char* units_m = "m";
      const char* units_1 = "1";
      const char* units_years = "years";
      const char* units_vel = "m s^-1";
      const char* units_beta = "Pa s m^-1";
      const char* units_rhs = "Pa";
      const char* units_nuH = "Pa s m";
      nc_put_att_text(ncid, var_time, "units", 5, units_years);
      nc_put_att_text(ncid, var_x, "units", 1, units_m);
      nc_put_att_text(ncid, var_y, "units", 1, units_m);
      nc_put_att_text(ncid, var_thk, "units", 1, units_m);
      nc_put_att_text(ncid, var_topg, "units", 1, units_m);
      nc_put_att_text(ncid, var_usurf, "units", 1, units_m);
      nc_put_att_text(ncid, var_dhdx, "units", 1, units_1);
      nc_put_att_text(ncid, var_dhdy, "units", 1, units_1);
      nc_put_att_text(ncid, var_beta_u, "units", 9, units_beta);
      nc_put_att_text(ncid, var_beta_v, "units", 9, units_beta);
      nc_put_att_text(ncid, var_rhs_u, "units", 2, units_rhs);
      nc_put_att_text(ncid, var_rhs_v, "units", 2, units_rhs);
      nc_put_att_text(ncid, var_nuH_u, "units", 6, units_nuH);
      nc_put_att_text(ncid, var_nuH_v, "units", 6, units_nuH);
      nc_put_att_text(ncid, var_vel_u, "units", 6, units_vel);
      nc_put_att_text(ncid, var_vel_v, "units", 6, units_vel);
      nc_put_att_text(ncid, var_vel_prev_u, "units", 6, units_vel);
      nc_put_att_text(ncid, var_vel_prev_v, "units", 6, units_vel);
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
        xvals[static_cast<std::size_t>(i)] = grid.x0() + i * grid.dx();
      }
      for (int j = 0; j < my; ++j) {
        yvals[static_cast<std::size_t>(j)] = grid.y0() + j * grid.dy();
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
    {
      int ncid = -1;
      if (nc_open(path.c_str(), NC_WRITE, &ncid) != NC_NOERR) {
        return false;
      }
      int var_thk = -1, var_topg = -1, var_usurf = -1, var_dhdx = -1, var_dhdy = -1;
      int var_cell_type = -1, var_beta_u = -1, var_beta_v = -1;
      int var_rhs_u = -1, var_rhs_v = -1, var_nuH_u = -1, var_nuH_v = -1;
      int var_vel_u = -1, var_vel_v = -1, var_vel_prev_u = -1, var_vel_prev_v = -1;
      if (nc_inq_varid(ncid, "thk", &var_thk) != NC_NOERR ||
          nc_inq_varid(ncid, "topg", &var_topg) != NC_NOERR ||
          nc_inq_varid(ncid, "usurf", &var_usurf) != NC_NOERR ||
          nc_inq_varid(ncid, "dhdx", &var_dhdx) != NC_NOERR ||
          nc_inq_varid(ncid, "dhdy", &var_dhdy) != NC_NOERR ||
          nc_inq_varid(ncid, "cell_type", &var_cell_type) != NC_NOERR ||
          nc_inq_varid(ncid, "beta_u", &var_beta_u) != NC_NOERR ||
          nc_inq_varid(ncid, "beta_v", &var_beta_v) != NC_NOERR ||
          nc_inq_varid(ncid, "rhs_u", &var_rhs_u) != NC_NOERR ||
          nc_inq_varid(ncid, "rhs_v", &var_rhs_v) != NC_NOERR ||
          nc_inq_varid(ncid, "nuH_u", &var_nuH_u) != NC_NOERR ||
          nc_inq_varid(ncid, "nuH_v", &var_nuH_v) != NC_NOERR ||
          nc_inq_varid(ncid, "vel_u", &var_vel_u) != NC_NOERR ||
          nc_inq_varid(ncid, "vel_v", &var_vel_v) != NC_NOERR ||
          nc_inq_varid(ncid, "vel_prev_u", &var_vel_prev_u) != NC_NOERR ||
          nc_inq_varid(ncid, "vel_prev_v", &var_vel_prev_v) != NC_NOERR) {
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
      std::vector<int> mask_buffer(static_cast<std::size_t>(grid.local_mx()) *
                                   grid.local_my());
      auto write_local_mask = [&](int varid, const Field2D<int>& field) {
        int idx = 0;
        for (int j = 0; j < grid.local_my(); ++j) {
          for (int i = 0; i < grid.local_mx(); ++i) {
            mask_buffer[static_cast<std::size_t>(idx++)] = field(i, j);
          }
        }
        return nc_put_vara_int(ncid, varid, start, count, mask_buffer.data()) ==
               NC_NOERR;
      };

      bool ok = true;
      ok = write_local(var_thk, *bundle.thk) && ok;
      ok = write_local(var_topg, *bundle.topg) && ok;
      ok = write_local(var_usurf, *bundle.usurf) && ok;
      ok = write_local(var_dhdx, *bundle.dhdx) && ok;
      ok = write_local(var_dhdy, *bundle.dhdy) && ok;
      ok = write_local_mask(var_cell_type, *bundle.cell_type) && ok;
      ok = write_local(var_beta_u, bundle.beta->component(0)) && ok;
      ok = write_local(var_beta_v, bundle.beta->component(1)) && ok;
      ok = write_local(var_rhs_u, bundle.rhs->component(0)) && ok;
      ok = write_local(var_rhs_v, bundle.rhs->component(1)) && ok;
      ok = write_local(var_nuH_u, bundle.nuH->component(0)) && ok;
      ok = write_local(var_nuH_v, bundle.nuH->component(1)) && ok;
      ok = write_local(var_vel_prev_u, bundle.vel_prev->component(0)) && ok;
      ok = write_local(var_vel_prev_v, bundle.vel_prev->component(1)) && ok;
      ok = write_local(var_vel_u, bundle.vel->component(0)) && ok;
      ok = write_local(var_vel_v, bundle.vel->component(1)) && ok;
      nc_close(ncid);
      if (!ok) {
        return false;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
#else
    (void)rank;
    return false;
#endif
  }

  // Single-rank serial write.
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
  int var_usurf = -1;
  int var_dhdx = -1;
  int var_dhdy = -1;
  int var_cell_type = -1;
  int var_beta_u = -1;
  int var_beta_v = -1;
  int var_rhs_u = -1;
  int var_rhs_v = -1;
  int var_nuH_u = -1;
  int var_nuH_v = -1;
  int var_vel_u = -1;
  int var_vel_v = -1;
  int var_vel_prev_u = -1;
  int var_vel_prev_v = -1;
  if (nc_def_var(ncid, "time", NC_DOUBLE, 1, &dim_time, &var_time) != NC_NOERR ||
      nc_def_var(ncid, "x", NC_DOUBLE, 1, &dim_x, &var_x) != NC_NOERR ||
      nc_def_var(ncid, "y", NC_DOUBLE, 1, &dim_y, &var_y) != NC_NOERR ||
      nc_def_var(ncid, "thk", NC_DOUBLE, 3, dims_tyx, &var_thk) != NC_NOERR ||
      nc_def_var(ncid, "topg", NC_DOUBLE, 3, dims_tyx, &var_topg) != NC_NOERR ||
      nc_def_var(ncid, "usurf", NC_DOUBLE, 3, dims_tyx, &var_usurf) != NC_NOERR ||
      nc_def_var(ncid, "dhdx", NC_DOUBLE, 3, dims_tyx, &var_dhdx) != NC_NOERR ||
      nc_def_var(ncid, "dhdy", NC_DOUBLE, 3, dims_tyx, &var_dhdy) != NC_NOERR ||
      nc_def_var(ncid, "cell_type", NC_INT, 3, dims_tyx, &var_cell_type) != NC_NOERR ||
      nc_def_var(ncid, "beta_u", NC_DOUBLE, 3, dims_tyx, &var_beta_u) != NC_NOERR ||
      nc_def_var(ncid, "beta_v", NC_DOUBLE, 3, dims_tyx, &var_beta_v) != NC_NOERR ||
      nc_def_var(ncid, "rhs_u", NC_DOUBLE, 3, dims_tyx, &var_rhs_u) != NC_NOERR ||
      nc_def_var(ncid, "rhs_v", NC_DOUBLE, 3, dims_tyx, &var_rhs_v) != NC_NOERR ||
      nc_def_var(ncid, "nuH_u", NC_DOUBLE, 3, dims_tyx, &var_nuH_u) != NC_NOERR ||
      nc_def_var(ncid, "nuH_v", NC_DOUBLE, 3, dims_tyx, &var_nuH_v) != NC_NOERR ||
      nc_def_var(ncid, "vel_u", NC_DOUBLE, 3, dims_tyx, &var_vel_u) != NC_NOERR ||
      nc_def_var(ncid, "vel_v", NC_DOUBLE, 3, dims_tyx, &var_vel_v) != NC_NOERR ||
      nc_def_var(ncid, "vel_prev_u", NC_DOUBLE, 3, dims_tyx, &var_vel_prev_u) != NC_NOERR ||
      nc_def_var(ncid, "vel_prev_v", NC_DOUBLE, 3, dims_tyx, &var_vel_prev_v) != NC_NOERR) {
    nc_close(ncid);
    return false;
  }

  const char* units_m = "m";
  const char* units_1 = "1";
  const char* units_years = "years";
  const char* units_vel = "m s^-1";
  const char* units_beta = "Pa s m^-1";
  const char* units_rhs = "Pa";
  const char* units_nuH = "Pa s m";
  nc_put_att_text(ncid, var_time, "units", 5, units_years);
  nc_put_att_text(ncid, var_x, "units", 1, units_m);
  nc_put_att_text(ncid, var_y, "units", 1, units_m);
  nc_put_att_text(ncid, var_thk, "units", 1, units_m);
  nc_put_att_text(ncid, var_topg, "units", 1, units_m);
  nc_put_att_text(ncid, var_usurf, "units", 1, units_m);
  nc_put_att_text(ncid, var_dhdx, "units", 1, units_1);
  nc_put_att_text(ncid, var_dhdy, "units", 1, units_1);
  nc_put_att_text(ncid, var_beta_u, "units", 9, units_beta);
  nc_put_att_text(ncid, var_beta_v, "units", 9, units_beta);
  nc_put_att_text(ncid, var_rhs_u, "units", 2, units_rhs);
  nc_put_att_text(ncid, var_rhs_v, "units", 2, units_rhs);
  nc_put_att_text(ncid, var_nuH_u, "units", 6, units_nuH);
  nc_put_att_text(ncid, var_nuH_v, "units", 6, units_nuH);
  nc_put_att_text(ncid, var_vel_u, "units", 6, units_vel);
  nc_put_att_text(ncid, var_vel_v, "units", 6, units_vel);
  nc_put_att_text(ncid, var_vel_prev_u, "units", 6, units_vel);
  nc_put_att_text(ncid, var_vel_prev_v, "units", 6, units_vel);

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
    xvals[static_cast<std::size_t>(i)] = grid.x0() + i * grid.dx();
  }
  for (int j = 0; j < my; ++j) {
    yvals[static_cast<std::size_t>(j)] = grid.y0() + j * grid.dy();
  }
  std::size_t start_x[1] = {0};
  std::size_t count_x[1] = {static_cast<std::size_t>(mx)};
  std::size_t start_y[1] = {0};
  std::size_t count_y[1] = {static_cast<std::size_t>(my)};
  nc_put_vara_double(ncid, var_x, start_x, count_x, xvals.data());
  nc_put_vara_double(ncid, var_y, start_y, count_y, yvals.data());

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
  std::vector<int> mask_buffer(static_cast<std::size_t>(grid.local_mx()) *
                               grid.local_my());
  auto write_local_mask = [&](int varid, const Field2D<int>& field) {
    int idx = 0;
    for (int j = 0; j < grid.local_my(); ++j) {
      for (int i = 0; i < grid.local_mx(); ++i) {
        mask_buffer[static_cast<std::size_t>(idx++)] = field(i, j);
      }
    }
    return nc_put_vara_int(ncid, varid, start, count, mask_buffer.data()) ==
           NC_NOERR;
  };

  bool ok = true;
  ok = write_local(var_thk, *bundle.thk) && ok;
  ok = write_local(var_topg, *bundle.topg) && ok;
  ok = write_local(var_usurf, *bundle.usurf) && ok;
  ok = write_local(var_dhdx, *bundle.dhdx) && ok;
  ok = write_local(var_dhdy, *bundle.dhdy) && ok;
  ok = write_local_mask(var_cell_type, *bundle.cell_type) && ok;
  ok = write_local(var_beta_u, bundle.beta->component(0)) && ok;
  ok = write_local(var_beta_v, bundle.beta->component(1)) && ok;
  ok = write_local(var_rhs_u, bundle.rhs->component(0)) && ok;
  ok = write_local(var_rhs_v, bundle.rhs->component(1)) && ok;
  ok = write_local(var_nuH_u, bundle.nuH->component(0)) && ok;
  ok = write_local(var_nuH_v, bundle.nuH->component(1)) && ok;
  ok = write_local(var_vel_prev_u, bundle.vel_prev->component(0)) && ok;
  ok = write_local(var_vel_prev_v, bundle.vel_prev->component(1)) && ok;
  ok = write_local(var_vel_u, bundle.vel->component(0)) && ok;
  ok = write_local(var_vel_v, bundle.vel->component(1)) && ok;

  nc_close(ncid);
  return ok;
}

bool write_output_impl(const std::string& path, int rank, int size, bool mpi_enabled,
                       const Grid2D& grid, const IOFields2D& fields,
                       double time_value) {
#if GPISM_HAVE_MPI && defined(NC_HAS_PARALLEL) && NC_HAS_PARALLEL
  if (mpi_enabled && size > 1) {
    return write_output_parallel(path, MPI_COMM_WORLD, rank, grid, fields,
                                 time_value);
  }
#endif
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
      int var_uvel = -1;
      int var_vvel = -1;
      int var_u_ssa = -1;
      int var_v_ssa = -1;
      int var_usurf = -1;
      int var_u_bc = -1;
      int var_v_bc = -1;
      int var_vel_bc_mask = -1;
      if (nc_def_var(ncid, "time", NC_DOUBLE, 1, &dim_time, &var_time) != NC_NOERR ||
          nc_def_var(ncid, "x", NC_DOUBLE, 1, &dim_x, &var_x) != NC_NOERR ||
          nc_def_var(ncid, "y", NC_DOUBLE, 1, &dim_y, &var_y) != NC_NOERR ||
          nc_def_var(ncid, "thk", NC_DOUBLE, 3, dims_tyx, &var_thk) != NC_NOERR ||
          nc_def_var(ncid, "topg", NC_DOUBLE, 3, dims_tyx, &var_topg) != NC_NOERR ||
          nc_def_var(ncid, "tauc", NC_DOUBLE, 3, dims_tyx, &var_tauc) != NC_NOERR) {
        nc_close(ncid);
        return false;
      }
      if (fields.has_velocity) {
        if (nc_def_var(ncid, "uvel", NC_DOUBLE, 3, dims_tyx, &var_uvel) != NC_NOERR ||
            nc_def_var(ncid, "vvel", NC_DOUBLE, 3, dims_tyx, &var_vvel) != NC_NOERR) {
          nc_close(ncid);
          return false;
        }
      }
      if (fields.has_ssa_velocity) {
        if (nc_def_var(ncid, "u_ssa", NC_DOUBLE, 3, dims_tyx, &var_u_ssa) != NC_NOERR ||
            nc_def_var(ncid, "v_ssa", NC_DOUBLE, 3, dims_tyx, &var_v_ssa) != NC_NOERR) {
          nc_close(ncid);
          return false;
        }
      }
      if (fields.has_usurf) {
        if (nc_def_var(ncid, "usurf", NC_DOUBLE, 3, dims_tyx, &var_usurf) != NC_NOERR) {
          nc_close(ncid);
          return false;
        }
      }
      if (fields.has_vel_bc) {
        if (nc_def_var(ncid, "u_bc", NC_DOUBLE, 3, dims_tyx, &var_u_bc) != NC_NOERR ||
            nc_def_var(ncid, "v_bc", NC_DOUBLE, 3, dims_tyx, &var_v_bc) != NC_NOERR ||
            nc_def_var(ncid, "vel_bc_mask", NC_INT, 3, dims_tyx,
                       &var_vel_bc_mask) != NC_NOERR) {
          nc_close(ncid);
          return false;
        }
      }

      const char* units_m = "m";
      const char* units_pa = "Pa";
      const char* units_years = "years";
      const char* units_velocity = "m year^-1";
      nc_put_att_text(ncid, var_thk, "units", 1, units_m);
      nc_put_att_text(ncid, var_topg, "units", 1, units_m);
      nc_put_att_text(ncid, var_tauc, "units", 2, units_pa);
      nc_put_att_text(ncid, var_time, "units", 5, units_years);
      nc_put_att_text(ncid, var_x, "units", 1, units_m);
      nc_put_att_text(ncid, var_y, "units", 1, units_m);
      if (fields.has_velocity) {
        nc_put_att_text(ncid, var_uvel, "units", 9, units_velocity);
        nc_put_att_text(ncid, var_vvel, "units", 9, units_velocity);
      }
      if (fields.has_ssa_velocity) {
        nc_put_att_text(ncid, var_u_ssa, "units", 9, units_velocity);
        nc_put_att_text(ncid, var_v_ssa, "units", 9, units_velocity);
      }
      if (fields.has_usurf) {
        nc_put_att_text(ncid, var_usurf, "units", 1, units_m);
      }
      if (fields.has_vel_bc) {
        nc_put_att_text(ncid, var_u_bc, "units", 9, units_velocity);
        nc_put_att_text(ncid, var_v_bc, "units", 9, units_velocity);
        put_vel_bc_mask_attrs(ncid, var_vel_bc_mask);
      }
      const std::string history = "gpism write_output";
      nc_put_att_text(ncid, NC_GLOBAL, "history", history.size(), history.c_str());
      put_global_attr_text(ncid, "gpism_version", GPISM_VERSION);
      put_global_attr_text(ncid, "gpism_build_type", GPISM_BUILD_TYPE);
      if (std::strcmp(GPISM_GIT_SHA, "unknown") != 0) {
        put_global_attr_text(ncid, "gpism_git_sha", GPISM_GIT_SHA);
      }

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
        xvals[static_cast<std::size_t>(i)] = grid.x0() + i * grid.dx();
      }
      for (int j = 0; j < my; ++j) {
        yvals[static_cast<std::size_t>(j)] = grid.y0() + j * grid.dy();
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
        int var_uvel = -1;
        int var_vvel = -1;
        int var_u_ssa = -1;
        int var_v_ssa = -1;
        int var_usurf = -1;
        int var_u_bc = -1;
        int var_v_bc = -1;
        int var_vel_bc_mask = -1;
        if (nc_inq_varid(ncid, "thk", &var_thk) != NC_NOERR ||
            nc_inq_varid(ncid, "topg", &var_topg) != NC_NOERR ||
            nc_inq_varid(ncid, "tauc", &var_tauc) != NC_NOERR) {
          nc_close(ncid);
          return false;
        }
        if (fields.has_velocity) {
          if (nc_inq_varid(ncid, "uvel", &var_uvel) != NC_NOERR ||
              nc_inq_varid(ncid, "vvel", &var_vvel) != NC_NOERR) {
            nc_close(ncid);
            return false;
          }
        }
        if (fields.has_ssa_velocity) {
          if (nc_inq_varid(ncid, "u_ssa", &var_u_ssa) != NC_NOERR ||
              nc_inq_varid(ncid, "v_ssa", &var_v_ssa) != NC_NOERR) {
            nc_close(ncid);
            return false;
          }
        }
        if (fields.has_usurf) {
          if (nc_inq_varid(ncid, "usurf", &var_usurf) != NC_NOERR) {
            nc_close(ncid);
            return false;
          }
        }
        if (fields.has_vel_bc) {
          if (nc_inq_varid(ncid, "u_bc", &var_u_bc) != NC_NOERR ||
              nc_inq_varid(ncid, "v_bc", &var_v_bc) != NC_NOERR ||
              nc_inq_varid(ncid, "vel_bc_mask", &var_vel_bc_mask) != NC_NOERR) {
            nc_close(ncid);
            return false;
          }
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
        auto write_local_scaled = [&](int varid, const Field2D<double>& field,
                                      double scale) {
          int idx = 0;
          for (int j = 0; j < grid.local_my(); ++j) {
            for (int i = 0; i < grid.local_mx(); ++i) {
              buffer[static_cast<std::size_t>(idx++)] = scale * field(i, j);
            }
          }
          return nc_put_vara_double(ncid, varid, start, count, buffer.data()) ==
                 NC_NOERR;
        };
        std::vector<int> mask_buffer(static_cast<std::size_t>(grid.local_mx()) *
                                     grid.local_my());
        auto write_local_mask = [&](int varid, const Field2D<int>& field) {
          int idx = 0;
          for (int j = 0; j < grid.local_my(); ++j) {
            for (int i = 0; i < grid.local_mx(); ++i) {
              mask_buffer[static_cast<std::size_t>(idx++)] = field(i, j);
            }
          }
          return nc_put_vara_int(ncid, varid, start, count, mask_buffer.data()) ==
                 NC_NOERR;
        };

        bool ok = true;
        ok = write_local(var_thk, fields.thk) && ok;
        ok = write_local(var_topg, fields.topg) && ok;
        ok = write_local(var_tauc, fields.tauc) && ok;
        if (fields.has_velocity) {
          ok = write_local_scaled(var_uvel, fields.uvel, kSecondsPerYear) && ok;
          ok = write_local_scaled(var_vvel, fields.vvel, kSecondsPerYear) && ok;
        }
        if (fields.has_ssa_velocity) {
          ok = write_local_scaled(var_u_ssa, fields.u_ssa, kSecondsPerYear) && ok;
          ok = write_local_scaled(var_v_ssa, fields.v_ssa, kSecondsPerYear) && ok;
        }
        if (fields.has_usurf) {
          ok = write_local(var_usurf, fields.usurf) && ok;
        }
        if (fields.has_vel_bc) {
          ok = write_local_scaled(var_u_bc, fields.u_bc, kSecondsPerYear) && ok;
          ok = write_local_scaled(var_v_bc, fields.v_bc, kSecondsPerYear) && ok;
          ok = write_local_mask(var_vel_bc_mask, fields.vel_bc_mask) && ok;
        }
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
  int var_uvel = -1;
  int var_vvel = -1;
  int var_u_ssa = -1;
  int var_v_ssa = -1;
  int var_usurf = -1;
  int var_u_bc = -1;
  int var_v_bc = -1;
  int var_vel_bc_mask = -1;
  if (nc_def_var(ncid, "time", NC_DOUBLE, 1, &dim_time, &var_time) != NC_NOERR ||
      nc_def_var(ncid, "x", NC_DOUBLE, 1, &dim_x, &var_x) != NC_NOERR ||
      nc_def_var(ncid, "y", NC_DOUBLE, 1, &dim_y, &var_y) != NC_NOERR ||
      nc_def_var(ncid, "thk", NC_DOUBLE, 3, dims_tyx, &var_thk) != NC_NOERR ||
      nc_def_var(ncid, "topg", NC_DOUBLE, 3, dims_tyx, &var_topg) != NC_NOERR ||
      nc_def_var(ncid, "tauc", NC_DOUBLE, 3, dims_tyx, &var_tauc) != NC_NOERR) {
    nc_close(ncid);
    return false;
  }
  if (fields.has_velocity) {
    if (nc_def_var(ncid, "uvel", NC_DOUBLE, 3, dims_tyx, &var_uvel) != NC_NOERR ||
        nc_def_var(ncid, "vvel", NC_DOUBLE, 3, dims_tyx, &var_vvel) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }
  if (fields.has_ssa_velocity) {
    if (nc_def_var(ncid, "u_ssa", NC_DOUBLE, 3, dims_tyx, &var_u_ssa) != NC_NOERR ||
        nc_def_var(ncid, "v_ssa", NC_DOUBLE, 3, dims_tyx, &var_v_ssa) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }
  if (fields.has_usurf) {
    if (nc_def_var(ncid, "usurf", NC_DOUBLE, 3, dims_tyx, &var_usurf) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }
  if (fields.has_vel_bc) {
    if (nc_def_var(ncid, "u_bc", NC_DOUBLE, 3, dims_tyx, &var_u_bc) != NC_NOERR ||
        nc_def_var(ncid, "v_bc", NC_DOUBLE, 3, dims_tyx, &var_v_bc) != NC_NOERR ||
        nc_def_var(ncid, "vel_bc_mask", NC_INT, 3, dims_tyx,
                   &var_vel_bc_mask) != NC_NOERR) {
      nc_close(ncid);
      return false;
    }
  }

  const char* units_m = "m";
  const char* units_pa = "Pa";
  const char* units_years = "years";
  const char* units_velocity = "m year^-1";
  nc_put_att_text(ncid, var_thk, "units", 1, units_m);
  nc_put_att_text(ncid, var_topg, "units", 1, units_m);
  nc_put_att_text(ncid, var_tauc, "units", 2, units_pa);
  nc_put_att_text(ncid, var_time, "units", 5, units_years);
  nc_put_att_text(ncid, var_x, "units", 1, units_m);
  nc_put_att_text(ncid, var_y, "units", 1, units_m);
  if (fields.has_velocity) {
    nc_put_att_text(ncid, var_uvel, "units", 9, units_velocity);
    nc_put_att_text(ncid, var_vvel, "units", 9, units_velocity);
  }
  if (fields.has_ssa_velocity) {
    nc_put_att_text(ncid, var_u_ssa, "units", 9, units_velocity);
    nc_put_att_text(ncid, var_v_ssa, "units", 9, units_velocity);
  }
  if (fields.has_usurf) {
    nc_put_att_text(ncid, var_usurf, "units", 1, units_m);
  }
  if (fields.has_vel_bc) {
    nc_put_att_text(ncid, var_u_bc, "units", 9, units_velocity);
    nc_put_att_text(ncid, var_v_bc, "units", 9, units_velocity);
    put_vel_bc_mask_attrs(ncid, var_vel_bc_mask);
  }
  const std::string history = "gpism write_output";
  nc_put_att_text(ncid, NC_GLOBAL, "history", history.size(), history.c_str());
  put_global_attr_text(ncid, "gpism_version", GPISM_VERSION);
  put_global_attr_text(ncid, "gpism_build_type", GPISM_BUILD_TYPE);
  if (std::strcmp(GPISM_GIT_SHA, "unknown") != 0) {
    put_global_attr_text(ncid, "gpism_git_sha", GPISM_GIT_SHA);
  }

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
    xvals[static_cast<std::size_t>(i)] = grid.x0() + i * grid.dx();
  }
  for (int j = 0; j < my; ++j) {
    yvals[static_cast<std::size_t>(j)] = grid.y0() + j * grid.dy();
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
  if (fields.has_velocity) {
    ok = write_var_2d_scaled(ncid, var_uvel, fields.uvel, mx, my, 0,
                             kSecondsPerYear) &&
         ok;
    ok = write_var_2d_scaled(ncid, var_vvel, fields.vvel, mx, my, 0,
                             kSecondsPerYear) &&
         ok;
  }
  if (fields.has_ssa_velocity) {
    ok = write_var_2d_scaled(ncid, var_u_ssa, fields.u_ssa, mx, my, 0,
                             kSecondsPerYear) &&
         ok;
    ok = write_var_2d_scaled(ncid, var_v_ssa, fields.v_ssa, mx, my, 0,
                             kSecondsPerYear) &&
         ok;
  }
  if (fields.has_usurf) {
    ok = write_var_2d(ncid, var_usurf, fields.usurf, mx, my, 0) && ok;
  }
  if (fields.has_vel_bc) {
    ok = write_var_2d_scaled(ncid, var_u_bc, fields.u_bc, mx, my, 0,
                             kSecondsPerYear) &&
         ok;
    ok = write_var_2d_scaled(ncid, var_v_bc, fields.v_bc, mx, my, 0,
                             kSecondsPerYear) &&
         ok;
    ok = write_var_2d(ncid, var_vel_bc_mask, fields.vel_bc_mask, mx, my, 0) &&
         ok;
  }

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

bool NetcdfIO::write_output_append(const std::string& path, const Grid2D& grid,
                                   const IOFields2D& fields,
                                   double time_value) {
  return write_output_append_impl(path, 0, 1, false, grid, fields, time_value);
}

bool NetcdfIO::write_output_append(const std::string& path,
                                   const Context& context, const Grid2D& grid,
                                   const IOFields2D& fields,
                                   double time_value) {
  return write_output_append_impl(path, context.rank(), context.size(),
                                  context.mpi_enabled(), grid, fields,
                                  time_value);
}

bool NetcdfIO::write_ssa_debug_bundle(const std::string& path, const Grid2D& grid,
                                      const SSADebugBundle2D& bundle,
                                      double time_value) {
  return write_ssa_debug_bundle_impl(path, 0, 1, false, grid, bundle, time_value);
}

bool NetcdfIO::write_ssa_debug_bundle(const std::string& path, const Context& context,
                                      const Grid2D& grid,
                                      const SSADebugBundle2D& bundle,
                                      double time_value) {
  return write_ssa_debug_bundle_impl(path, context.rank(), context.size(),
                                     context.mpi_enabled(), grid, bundle,
                                     time_value);
}

}  // namespace gpism
