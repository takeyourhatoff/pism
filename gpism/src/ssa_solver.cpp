#include "gpism/ssa_solver.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gpism/context.h"
#include "gpism/device_policy.h"
#include "gpism/field_sync.h"
#include "gpism/gmres.h"
#include "gpism/halo_exchange.h"
#include "gpism/linear_algebra.h"
#include "gpism/mg_preconditioner.h"
#include "gpism/thickness.h"

#if GPISM_HAVE_NETCDF
#include "gpism/netcdf_io.h"
#endif

#if GPISM_HAVE_MPI
#include <mpi.h>
#endif

#if GPISM_HAVE_CUDA
namespace gpism {
void ssa_relax_nuH_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                        const double* nuH_prev_u, const double* nuH_prev_v,
                        double* nuH_u, double* nuH_v, double nuH_min,
                        double nuH_max, double nuH_relax);
void ssa_relax_vel_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                        const double* vel_prev_u, const double* vel_prev_v,
                        double* vel_u, double* vel_v, double vel_relax);
}  // namespace gpism
#endif

namespace gpism {
namespace {

template <typename T>
bool can_use_device_exchange(const FieldStag2D<T>& field,
                             const Context& context) {
#if GPISM_HAVE_CUDA
  return context.cuda_aware_mpi() &&
         field.component(0).device_data() != nullptr &&
         field.component(1).device_data() != nullptr;
#else
  (void)field;
  (void)context;
  return false;
#endif
}

template <typename T>
void exchange_for_device(FieldStag2D<T>& field, const Grid2D& grid,
                         const Context& context) {
  HaloExchange2D exchange;
  if (can_use_device_exchange(field, context)) {
    exchange.exchange(field, grid, context, HaloExchange2D::Mode::Device);
    return;
  }
  sync_device_to_host(field);
  exchange.exchange(field, grid, context, HaloExchange2D::Mode::Host);
  sync_host_to_device(field);
}

template <typename T>
bool can_use_device_exchange(const Field2D<T>& field, const Context& context) {
#if GPISM_HAVE_CUDA
  return context.cuda_aware_mpi() && field.device_data() != nullptr;
#else
  (void)field;
  (void)context;
  return false;
#endif
}

template <typename T>
void exchange_for_device(Field2D<T>& field, const Grid2D& grid,
                         const Context& context) {
  HaloExchange2D exchange;
  if (can_use_device_exchange(field, context)) {
    exchange.exchange(field, grid, context, HaloExchange2D::Mode::Device);
    return;
  }
  sync_device_to_host(field);
  exchange.exchange(field, grid, context, HaloExchange2D::Mode::Host);
  sync_host_to_device(field);
}

double global_sum(const Context* context, double local_value) {
#if GPISM_HAVE_MPI
  if (context && context->mpi_enabled()) {
    double global_value = 0.0;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    return global_value;
  }
#endif
  return local_value;
}

double global_min(const Context* context, double local_value) {
#if GPISM_HAVE_MPI
  if (context && context->mpi_enabled()) {
    double global_value = 0.0;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    return global_value;
  }
#endif
  return local_value;
}

double global_max(const Context* context, double local_value) {
#if GPISM_HAVE_MPI
  if (context && context->mpi_enabled()) {
    double global_value = 0.0;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    return global_value;
  }
#endif
  return local_value;
}

bool is_rank0(const Context* context) {
  if (!context || !context->mpi_enabled()) {
    return true;
  }
  return context->rank() == 0;
}

std::vector<std::pair<int, int>> parse_diag_points() {
  std::vector<std::pair<int, int>> points;
  const char* env = std::getenv("SSA_DIAG_POINTS");
  if (!env || !*env) {
    return points;
  }
  std::string s(env);
  std::size_t pos = 0;
  while (pos < s.size()) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == ';')) {
      ++pos;
    }
    if (pos >= s.size()) {
      break;
    }
    char* end = nullptr;
    long i = std::strtol(s.c_str() + pos, &end, 10);
    if (end == s.c_str() + pos) {
      break;
    }
    pos = static_cast<std::size_t>(end - s.c_str());
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == ',')) {
      ++pos;
    }
    long j = std::strtol(s.c_str() + pos, &end, 10);
    if (end == s.c_str() + pos) {
      break;
    }
    pos = static_cast<std::size_t>(end - s.c_str());
    points.emplace_back(static_cast<int>(i), static_cast<int>(j));
    while (pos < s.size() && s[pos] != ';') {
      ++pos;
    }
  }
  return points;
}

const std::vector<std::pair<int, int>>& diag_points() {
  static const std::vector<std::pair<int, int>> points = parse_diag_points();
  return points;
}

bool file_exists(const std::string& path) {
  std::ifstream input(path);
  return input.good();
}

bool copy_file_best_effort(const std::string& src, const std::string& dst) {
  if (src.empty() || dst.empty() || !file_exists(src)) {
    return false;
  }
  std::ifstream in(src, std::ios::binary);
  if (!in) {
    return false;
  }
  std::ofstream out(dst, std::ios::binary);
  if (!out) {
    return false;
  }
  out << in.rdbuf();
  return out.good();
}

void log_diag_points(const Grid2D& grid, int iter, const Field2D<int>& cell_type,
                     const Field2D<double>& thk, const Field2D<double>& topg,
                     const Field2D<double>& usurf, const Field2D<double>& dhdx,
                     const Field2D<double>& dhdy,
                     const Field2D<double>& u_center,
                     const Field2D<double>& v_center,
                     const FieldStag2D<double>& nuH,
                     const FieldStag2D<double>& beta,
                     const FieldStag2D<double>& rhs,
                     const Context* context) {
  const auto& points = diag_points();
  if (points.empty() || !is_rank0(context)) {
    return;
  }
  sync_device_to_host(const_cast<Field2D<double>&>(thk));
  sync_device_to_host(const_cast<Field2D<double>&>(topg));
  sync_device_to_host(const_cast<Field2D<double>&>(usurf));
  sync_device_to_host(const_cast<Field2D<double>&>(dhdx));
  sync_device_to_host(const_cast<Field2D<double>&>(dhdy));
  sync_device_to_host(const_cast<Field2D<double>&>(u_center));
  sync_device_to_host(const_cast<Field2D<double>&>(v_center));
  sync_device_to_host(const_cast<Field2D<int>&>(cell_type));
  sync_device_to_host(const_cast<FieldStag2D<double>&>(nuH));
  sync_device_to_host(const_cast<FieldStag2D<double>&>(beta));
  sync_device_to_host(const_cast<FieldStag2D<double>&>(rhs));

  const int xs = grid.xs();
  const int ys = grid.ys();
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  const double inv_dx2 = 1.0 / (grid.dx() * grid.dx());
  const double inv_dy2 = 1.0 / (grid.dy() * grid.dy());

  std::cout << "SSA diag points (iter " << iter << ", xs=" << xs
            << " ys=" << ys << ")\n";
  for (const auto& point : points) {
    const int gi = point.first;
    const int gj = point.second;
    const int i = gi - xs;
    const int j = gj - ys;
    if (i < 0 || j < 0 || i >= mx || j >= my) {
      std::cout << "  point (" << gi << "," << gj << ") not local\n";
      continue;
    }
    const double nu_u = nuH(i, j, 0);
    const double nu_v = nuH(i, j, 1);
    const double beta_u = beta(i, j, 0);
    const double beta_v = beta(i, j, 1);
    const double c_n = nuH(i, j, 1);
    const double c_s = nuH(i, j - 1, 1);
    const double c_e = nuH(i, j, 0);
    const double c_w = nuH(i - 1, j, 0);
    const double diag_u =
        beta_u + (c_n + c_s) * inv_dy2 + 4.0 * (c_e + c_w) * inv_dx2;
    const double diag_v =
        beta_v + 4.0 * (c_n + c_s) * inv_dy2 + (c_e + c_w) * inv_dx2;
    std::cout << "  (" << gi << "," << gj << ") type=" << cell_type(i, j)
              << " thk=" << thk(i, j) << " topg=" << topg(i, j)
              << " usurf=" << usurf(i, j)
              << " dhdx=" << dhdx(i, j) << " dhdy=" << dhdy(i, j)
              << " u_center=" << u_center(i, j)
              << " v_center=" << v_center(i, j)
              << " nuH_u=" << nu_u << " nuH_v=" << nu_v
              << " beta_u=" << beta_u << " beta_v=" << beta_v
              << " rhs_u=" << rhs(i, j, 0) << " rhs_v=" << rhs(i, j, 1)
              << " diag_u=" << diag_u << " diag_v=" << diag_v << '\n';
  }
}

bool mg_device_ready(const MultigridHierarchy& mg,
                     const SSABoundaryCondition* bc) {
#if GPISM_HAVE_CUDA
  if (!device_enabled()) {
    return false;
  }
  const bool has_bc = bc && bc->mask;
  const bool has_values = has_bc && bc->values;
  if (has_bc) {
    if (!bc->mask->component(0).has_device_data() ||
        !bc->mask->component(1).has_device_data()) {
      return false;
    }
  }
  if (has_values) {
    if (!bc->values->component(0).has_device_data() ||
        !bc->values->component(1).has_device_data()) {
      return false;
    }
  }
  auto has_device = [](const FieldStag2D<double>& field) {
    return field.component(0).has_device_data() &&
           field.component(1).has_device_data();
  };
  for (int level = 0; level < mg.num_levels(); ++level) {
    const MGLevel& lvl = mg.level(level);
    if (!has_device(lvl.u) || !has_device(lvl.rhs) || !has_device(lvl.r) ||
        !has_device(lvl.z) ||
        !has_device(lvl.nuH) || !has_device(lvl.beta) ||
        !has_device(lvl.diag) || !has_device(lvl.Ax) ||
        !has_device(lvl.corr)) {
      return false;
    }
  }
  return true;
#else
  (void)mg;
  (void)bc;
  return false;
#endif
}

struct FieldStats {
  double min = 0.0;
  double max = 0.0;
  bool all_finite = true;
};

FieldStats field_stats(FieldStag2D<double>& field) {
  if (field.component(0).has_device_data() ||
      field.component(1).has_device_data()) {
    sync_device_to_host(field);
  }
  FieldStats stats;
  stats.min = std::numeric_limits<double>::infinity();
  stats.max = -std::numeric_limits<double>::infinity();
  const int mx = field.local_mx();
  const int my = field.local_my();
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      for (int comp = 0; comp < 2; ++comp) {
        const double value = field(i, j, comp);
        if (!std::isfinite(value)) {
          stats.all_finite = false;
          continue;
        }
        stats.min = std::min(stats.min, value);
        stats.max = std::max(stats.max, value);
      }
    }
  }
  if (stats.min == std::numeric_limits<double>::infinity()) {
    stats.min = 0.0;
    stats.max = 0.0;
  }
  return stats;
}

FieldStats field_stats(Field2D<double>& field) {
  if (field.has_device_data()) {
    sync_device_to_host(field);
  }
  FieldStats stats;
  stats.min = std::numeric_limits<double>::infinity();
  stats.max = -std::numeric_limits<double>::infinity();
  const int mx = field.local_mx();
  const int my = field.local_my();
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const double value = field(i, j);
      if (!std::isfinite(value)) {
        stats.all_finite = false;
        continue;
      }
      stats.min = std::min(stats.min, value);
      stats.max = std::max(stats.max, value);
    }
  }
  if (stats.min == std::numeric_limits<double>::infinity()) {
    stats.min = 0.0;
    stats.max = 0.0;
  }
  return stats;
}

FieldStats diag_stats(const Grid2D& grid, FieldStag2D<double>& nuH,
                      FieldStag2D<double>& beta) {
  if (nuH.component(0).has_device_data() || nuH.component(1).has_device_data()) {
    sync_device_to_host(nuH);
  }
  if (beta.component(0).has_device_data() ||
      beta.component(1).has_device_data()) {
    sync_device_to_host(beta);
  }
  FieldStats stats;
  stats.min = std::numeric_limits<double>::infinity();
  stats.max = -std::numeric_limits<double>::infinity();
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  const double inv_dx2 = 1.0 / (grid.dx() * grid.dx());
  const double inv_dy2 = 1.0 / (grid.dy() * grid.dy());
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const double dxx_u =
          (nuH(i + 1, j, 0) + nuH(i - 1, j, 0)) * inv_dx2;
      const double dyy_u =
          (nuH(i, j + 1, 0) + nuH(i, j - 1, 0)) * inv_dy2;
      const double diag_u = beta(i, j, 0) + dxx_u + dyy_u;
      const double dxx_v =
          (nuH(i + 1, j, 1) + nuH(i - 1, j, 1)) * inv_dx2;
      const double dyy_v =
          (nuH(i, j + 1, 1) + nuH(i, j - 1, 1)) * inv_dy2;
      const double diag_v = beta(i, j, 1) + dxx_v + dyy_v;
      const double values[2] = {diag_u, diag_v};
      for (double value : values) {
        if (!std::isfinite(value)) {
          stats.all_finite = false;
          continue;
        }
        stats.min = std::min(stats.min, value);
        stats.max = std::max(stats.max, value);
      }
    }
  }
  if (stats.min == std::numeric_limits<double>::infinity()) {
    stats.min = 0.0;
    stats.max = 0.0;
  }
  return stats;
}

double clamp_value(double value, double min_value, double max_value) {
  if (max_value > 0.0 && value > max_value) {
    value = max_value;
  }
  if (min_value > 0.0 && value < min_value) {
    value = min_value;
  }
  return value;
}

void apply_nuH_constraints(FieldStag2D<double>& nuH,
                           const FieldStag2D<double>& nuH_prev,
                           const SSASolverOptions& options) {
#if GPISM_HAVE_CUDA
  if (nuH.component(0).has_device_data() && nuH.component(1).has_device_data() &&
      nuH_prev.component(0).has_device_data() &&
      nuH_prev.component(1).has_device_data()) {
    ssa_relax_nuH_cuda(nuH.local_mx(), nuH.local_my(), nuH.ghost_width(),
                       nuH.component(0).stride(), nuH.component(1).stride(),
                       nuH_prev.component(0).device_data(),
                       nuH_prev.component(1).device_data(),
                       nuH.component(0).device_data(),
                       nuH.component(1).device_data(), options.nuH_min,
                       options.nuH_max, options.nuH_relax);
    return;
  }
#endif
  for (int j = 0; j < nuH.local_my(); ++j) {
    for (int i = 0; i < nuH.local_mx(); ++i) {
      for (int comp = 0; comp < 2; ++comp) {
        double value = nuH(i, j, comp);
        value = clamp_value(value, options.nuH_min, options.nuH_max);
        if (options.nuH_relax < 1.0) {
          value = options.nuH_relax * value +
                  (1.0 - options.nuH_relax) * nuH_prev(i, j, comp);
        }
        nuH(i, j, comp) = value;
      }
    }
  }
}

void apply_vel_relax(FieldStag2D<double>& vel,
                     const FieldStag2D<double>& vel_prev,
                     double vel_relax) {
#if GPISM_HAVE_CUDA
  if (vel.component(0).has_device_data() && vel.component(1).has_device_data() &&
      vel_prev.component(0).has_device_data() &&
      vel_prev.component(1).has_device_data()) {
    ssa_relax_vel_cuda(vel.local_mx(), vel.local_my(), vel.ghost_width(),
                       vel.component(0).stride(), vel.component(1).stride(),
                       vel_prev.component(0).device_data(),
                       vel_prev.component(1).device_data(),
                       vel.component(0).device_data(),
                       vel.component(1).device_data(), vel_relax);
    return;
  }
#endif
  for (int j = 0; j < vel.local_my(); ++j) {
    for (int i = 0; i < vel.local_mx(); ++i) {
      vel(i, j, 0) = vel_relax * vel(i, j, 0) +
                     (1.0 - vel_relax) * vel_prev(i, j, 0);
      vel(i, j, 1) = vel_relax * vel(i, j, 1) +
                     (1.0 - vel_relax) * vel_prev(i, j, 1);
    }
  }
}

class SSAApplyOperator : public LinearOperator {
public:
  SSAApplyOperator(const SSAOperator& op, const Grid2D& grid,
                   const FieldStag2D<double>& nuH,
                   const FieldStag2D<double>& beta,
                   const SSABoundaryCondition* bc, const Context* context)
      : op_(op),
        grid_(grid),
        nuH_(nuH),
        beta_(beta),
        bc_(bc),
        context_(context) {}

  void apply(const FieldStag2D<double>& x,
             FieldStag2D<double>& y) const override {
    if (context_ && context_->mpi_enabled()) {
      const int gw = x.ghost_width();
      if (gw > 0) {
        auto& mutable_x = const_cast<FieldStag2D<double>&>(x);
        HaloExchange2D exchange;
        auto handle =
            exchange.start_exchange(mutable_x, grid_, *context_,
                                    HaloExchange2D::Mode::Auto);

        const int mx = grid_.local_mx();
        const int my = grid_.local_my();
        const int i0 = gw;
        const int i1 = mx - gw;
        const int j0 = gw;
        const int j1 = my - gw;
        if (i0 < i1 && j0 < j1) {
          op_.apply_region(grid_, nuH_, beta_, x, y, i0, i1, j0, j1, bc_);
          exchange.finish_exchange(handle);

          auto apply_band = [&](int is, int ie, int js, int je) {
            if (is < ie && js < je) {
              op_.apply_region(grid_, nuH_, beta_, x, y, is, ie, js, je, bc_);
            }
          };
          apply_band(0, gw, 0, my);
          apply_band(mx - gw, mx, 0, my);
          apply_band(gw, mx - gw, 0, gw);
          apply_band(gw, mx - gw, my - gw, my);
          return;
        }

        exchange.finish_exchange(handle);
      }
    }
    op_.apply(grid_, nuH_, beta_, x, y, bc_);
  }

private:
  const SSAOperator& op_;
  const Grid2D& grid_;
  const FieldStag2D<double>& nuH_;
  const FieldStag2D<double>& beta_;
  const SSABoundaryCondition* bc_;
  const Context* context_;
};

void build_bc_stag(const Grid2D& grid, const Field2D<double>* u_bc,
                   const Field2D<double>* v_bc,
                   const Field2D<int>* vel_bc_mask,
                   const Field2D<int>* cell_type,
                   FieldStag2D<int>& mask_stag,
                   FieldStag2D<double>& values_stag) {
  mask_stag.fill(0);
  values_stag.fill(0.0);
  if (!vel_bc_mask || !u_bc || !v_bc) {
    // continue: still may enforce ice-free mask below
  } else {
    for (int j = 0; j < grid.local_my(); ++j) {
      for (int i = 0; i < grid.local_mx(); ++i) {
        const int mask = (*vel_bc_mask)(i, j);
        mask_stag(i, j, 0) = mask;
        mask_stag(i, j, 1) = mask;
        values_stag(i, j, 0) = (*u_bc)(i, j);
        values_stag(i, j, 1) = (*v_bc)(i, j);
      }
    }
  }

  if (!cell_type) {
    return;
  }
  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      const int type = (*cell_type)(i, j);
      if (type == IceFreeOcean || type == IceFreeBedrock) {
        if (mask_stag(i, j, 0) == 0) {
          values_stag(i, j, 0) = 0.0;
        }
        if (mask_stag(i, j, 1) == 0) {
          values_stag(i, j, 1) = 0.0;
        }
        mask_stag(i, j, 0) = 1;
        mask_stag(i, j, 1) = 1;
      }
    }
  }
}

void compute_speed_scale(const Grid2D& grid, const Field2D<double>& u_center,
                         const Field2D<double>& v_center, double max_speed,
                         Field2D<double>& speed_scale) {
  if (max_speed <= 0.0) {
    speed_scale.fill(1.0);
    return;
  }
#if GPISM_HAVE_CUDA
  // On the CUDA path, u_center/v_center are often computed on the device.
  // Make sure we use the up-to-date values when computing the scale factor.
  if (u_center.has_device_data()) {
    sync_device_to_host(const_cast<Field2D<double>&>(u_center));
  }
  if (v_center.has_device_data()) {
    sync_device_to_host(const_cast<Field2D<double>&>(v_center));
  }
#endif
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const double u = u_center(i, j);
      const double v = v_center(i, j);
      const double s = std::sqrt(u * u + v * v);
      speed_scale(i, j) = (s > max_speed && s > 0.0) ? (max_speed / s) : 1.0;
    }
  }
}

double max_center_speed(const Grid2D& grid, const Field2D<double>& u_center,
                        const Field2D<double>& v_center,
                        const Context* context) {
#if GPISM_HAVE_CUDA
  // u_center/v_center are often device-produced; ensure host view is current
  // before scanning.
  if (u_center.has_device_data()) {
    sync_device_to_host(const_cast<Field2D<double>&>(u_center));
  }
  if (v_center.has_device_data()) {
    sync_device_to_host(const_cast<Field2D<double>&>(v_center));
  }
#endif
  double local_max = 0.0;
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const double u = u_center(i, j);
      const double v = v_center(i, j);
      const double s = std::sqrt(u * u + v * v);
      if (s > local_max) {
        local_max = s;
      }
    }
  }
  return global_max(context, local_max);
}

void apply_speed_scale(const Grid2D& grid, const Field2D<double>& speed_scale,
                       FieldStag2D<double>& vel) {
#if GPISM_HAVE_CUDA
  const bool have_device = vel.component(0).has_device_data() &&
                           vel.component(1).has_device_data();
  if (have_device) {
    sync_device_to_host(vel);
  }
#endif
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const int ie = (i == mx - 1) ? i : i + 1;
      const int jn = (j == my - 1) ? j : j + 1;
      const double su = std::min(speed_scale(i, j), speed_scale(ie, j));
      const double sv = std::min(speed_scale(i, j), speed_scale(i, jn));
      vel(i, j, 0) *= su;
      vel(i, j, 1) *= sv;
    }
  }
#if GPISM_HAVE_CUDA
  if (have_device) {
    sync_host_to_device(vel);
  }
#endif
}

}  // namespace

SSASolver::SSASolver(const Grid2D& grid, double rho, double g, double u_threshold,
                     const ViscosityModel& viscosity_model)
    : grid_(grid), ssa_(rho, g), viscosity_(viscosity_model) {
  (void)u_threshold;
}

SSASolverResult SSASolver::solve(const Field2D<double>& thk,
                                 const Field2D<double>& topg,
                                 const Field2D<double>& tauc,
                                 const Field2D<double>* u_bc,
                                 const Field2D<double>* v_bc,
                                 const Field2D<int>* vel_bc_mask,
                                 FieldStag2D<double>& vel,
                                 const SSASolverOptions& options) {
  SSASolverResult result{};
  const Context* context = options.context;

  workspace_.ensure(grid_);
  auto& usurf = workspace_.usurf;
  auto& dhdx = workspace_.dhdx;
  auto& dhdy = workspace_.dhdy;
  auto& cell_type = workspace_.cell_type;
  auto& u_center = workspace_.u_center;
  auto& v_center = workspace_.v_center;
  auto& beta = workspace_.beta;
  auto& rhs = workspace_.rhs;
  auto& nuH = workspace_.nuH;
  auto& nuH_prev = workspace_.nuH_prev;
  auto& vel_prev = workspace_.vel_prev;
  auto& speed_scale = workspace_.speed_scale;
  auto& bc_mask = workspace_.bc_mask;
  auto& bc_values = workspace_.bc_values;
  SSABoundaryCondition bc;

  GeometryDiagnostics geometry;
  compute_cell_type(grid_, thk, topg, options.sea_level, options.rho_ice,
                    options.rho_water, cell_type);
  if (context && context->mpi_enabled() && context->size() > 1) {
    exchange_for_device(cell_type, grid_, *context);
  }

  const bool use_bc = options.use_bc || options.enforce_ice_free_bc;
  if (use_bc) {
    if (cell_type.has_device_data()) {
      sync_device_to_host(cell_type);
    }
    build_bc_stag(grid_, u_bc, v_bc, vel_bc_mask,
                  options.enforce_ice_free_bc ? &cell_type : nullptr,
                  bc_mask, bc_values);
    bc.mask = &bc_mask;
    bc.values = &bc_values;
    sync_host_to_device(bc_mask);
    sync_host_to_device(bc_values);
  }

  compute_usurf_flotation(grid_, thk, topg, cell_type, options.sea_level,
                          options.rho_ice, options.rho_water, usurf);
  if (context && context->mpi_enabled() && context->size() > 1) {
    exchange_for_device(usurf, grid_, *context);
  }

  compute_surface_slopes_pism(grid_, usurf, cell_type, dhdx, dhdy,
                              options.surface_gradient_inward,
                              options.surface_slope_uphill,
                              options.use_cfbc);

  if (options.use_mg_precond) {
    const int min_size = std::max(2, options.mg_min_size);
    const bool needs_rebuild =
        !workspace_.mg || workspace_.mg_min_size != min_size ||
        workspace_.mg->level(0).grid.local_mx() != grid_.local_mx() ||
        workspace_.mg->level(0).grid.local_my() != grid_.local_my() ||
        workspace_.mg->level(0).grid.ghost_width() != grid_.ghost_width();
    if (needs_rebuild) {
      workspace_.mg = std::make_unique<MultigridHierarchy>(grid_, min_size);
      workspace_.mg_min_size = min_size;
    }
    (void)0;
  }

  ssa_.assemble_rhs(grid_, thk, dhdx, dhdy, rhs,
                    use_bc ? &bc : nullptr);
  if (context && context->mpi_enabled() && context->size() > 1) {
    exchange_for_device(rhs, grid_, *context);
  }

  for (int iter = 0; iter < options.max_picard; ++iter) {
    copy(nuH, nuH_prev);
    copy(vel, vel_prev);

    compute_cell_center_velocity(grid_, vel, u_center, v_center);
    if (context && context->mpi_enabled() && context->size() > 1) {
      exchange_for_device(u_center, grid_, *context);
      exchange_for_device(v_center, grid_, *context);
    }

    ssa_.compute_basal_drag(grid_, tauc, u_center, v_center, topg, usurf,
                            cell_type, beta, options.basal_params);
    if (context && context->mpi_enabled() && context->size() > 1) {
      exchange_for_device(beta, grid_, *context);
    }

    if (context && context->mpi_enabled() && context->size() > 1) {
      exchange_for_device(vel, grid_, *context);
    }
    viscosity_.compute_nuH(grid_, thk, vel, nuH, options.nuH_regularization,
                           options.strength_extension_nu,
                           options.strength_extension_min_thickness,
                           options.enthalpy, options.enthalpy_gamma,
                           options.enthalpy_ref);
    if (options.nuH_min > 0.0 || options.nuH_max > 0.0 ||
        options.nuH_relax < 1.0) {
      apply_nuH_constraints(nuH, nuH_prev, options);
    }
    if (context && context->mpi_enabled() && context->size() > 1) {
      exchange_for_device(nuH, grid_, *context);
    }

    if (options.diagnostic && iter == 0) {
      log_diag_points(grid_, iter, cell_type, thk, topg, usurf, dhdx, dhdy,
                      u_center, v_center, nuH, beta, rhs, context);
    }

    if (options.diagnostic && is_rank0(context)) {
      const FieldStats nuH_stats = field_stats(nuH);
      const FieldStats beta_stats = field_stats(beta);
      const FieldStats rhs_stats = field_stats(rhs);
      const FieldStats diag = diag_stats(grid_, nuH, beta);
      const FieldStats u_stats = field_stats(u_center);
      const FieldStats v_stats = field_stats(v_center);
      const double nuH_min = global_min(context, nuH_stats.min);
      const double nuH_max = global_max(context, nuH_stats.max);
      const double beta_min = global_min(context, beta_stats.min);
      const double beta_max = global_max(context, beta_stats.max);
      const double rhs_min = global_min(context, rhs_stats.min);
      const double rhs_max = global_max(context, rhs_stats.max);
      const double diag_min = global_min(context, diag.min);
      const double diag_max = global_max(context, diag.max);
      const double u_min = global_min(context, u_stats.min);
      const double u_max = global_max(context, u_stats.max);
      const double v_min = global_min(context, v_stats.min);
      const double v_max = global_max(context, v_stats.max);
      std::cout << "SSA diag iter " << iter
                << ": nuH[min,max]=[" << nuH_min << ", " << nuH_max << "] "
                << "beta[min,max]=[" << beta_min << ", " << beta_max << "] "
                << "diag[min,max]=[" << diag_min << ", " << diag_max << "] "
                << "rhs[min,max]=[" << rhs_min << ", " << rhs_max << "] "
                << "u_center[min,max]=[" << u_min << ", " << u_max << "] "
                << "v_center[min,max]=[" << v_min << ", " << v_max << "]";
      if (!(nuH_stats.all_finite && beta_stats.all_finite && rhs_stats.all_finite &&
            diag.all_finite && u_stats.all_finite && v_stats.all_finite)) {
        std::cout << " (non-finite values detected)";
      }
      std::cout << '\n';
    }

    if (options.replace_zero_diagonal_entries) {
      ssa_.replace_zero_diagonal_entries(
          grid_, nuH, beta, options.use_bc ? &bc : nullptr,
          options.basal_params.beta_ice_free_bedrock);
      if (context && context->mpi_enabled() && context->size() > 1) {
        exchange_for_device(beta, grid_, *context);
      }
    }

    if (options.use_mg_precond) {
      MultigridHierarchy& mg = *workspace_.mg;
      copy(beta, mg.level(0).beta);
      for (int level = 1; level < mg.num_levels(); ++level) {
        restrict_stag(mg.level(level - 1).beta, mg.level(level).beta);
      }
    }

    const MultigridPreconditioner* precond_ptr = nullptr;
    std::optional<MultigridPreconditioner> precond;
    if (options.use_mg_precond) {
      MultigridHierarchy& mg = *workspace_.mg;
      copy(nuH, mg.level(0).nuH);
      for (int level = 1; level < mg.num_levels(); ++level) {
        restrict_stag(mg.level(level - 1).nuH, mg.level(level).nuH);
      }
      precond.emplace(mg, options.mg_pre_iters, options.mg_post_iters,
                      options.mg_coarse_iters, options.mg_omega,
                      options.mg_smoother, options.mg_cheby_lambda_min,
                      options.mg_cheby_lambda_max,
                      options.mg_cheby_estimate,
                      options.mg_cheby_estimate_iters,
                      options.mg_cheby_estimate_min_factor,
                      options.mg_cheby_estimate_max_factor,
                      options.use_bc ? &bc : nullptr, context,
                      options.mg_diagnostic && iter == 0,
                      options.basal_params.beta_ice_free_bedrock);
      precond_ptr = &(*precond);
      if (options.mg_diagnostic && iter == 0) {
        auto& fine = mg.level(0);
        sync_device_to_host(fine.nuH);
        sync_device_to_host(fine.beta);
        sync_device_to_host(rhs);
        const FieldStats nuH_stats = field_stats(fine.nuH);
        const FieldStats beta_stats = field_stats(fine.beta);
        const FieldStats diag = diag_stats(fine.grid, fine.nuH, fine.beta);
        const FieldStats rhs_stats = field_stats(rhs);
        const double nuH_min = global_min(context, nuH_stats.min);
        const double nuH_max = global_max(context, nuH_stats.max);
        const double beta_min = global_min(context, beta_stats.min);
        const double beta_max = global_max(context, beta_stats.max);
        const double diag_min = global_min(context, diag.min);
        const double diag_max = global_max(context, diag.max);
        const double rhs_min = global_min(context, rhs_stats.min);
        const double rhs_max = global_max(context, rhs_stats.max);
        if (is_rank0(context)) {
          const bool mg_cuda =
              mg_device_ready(mg, options.use_bc ? &bc : nullptr);
          std::cout << "SSA GMRES preconditioner: multigrid\n";
          std::cout << "MG params: pre=" << options.mg_pre_iters
                    << " post=" << options.mg_post_iters
                    << " coarse=" << options.mg_coarse_iters
                    << " omega=" << options.mg_omega
                    << " min_size=" << options.mg_min_size
                    << " levels=" << mg.num_levels()
                    << " device_path=" << (mg_cuda ? "cuda" : "host")
                    << " bc=" << (use_bc ? "values" : "none")
                    << '\n';
          if (options.mg_smoother == MGSmoother::Chebyshev) {
            std::cout << "MG smoother: chebyshev"
                      << " lambda_min=" << options.mg_cheby_lambda_min
                      << " lambda_max=" << options.mg_cheby_lambda_max
                      << " estimate=" << (options.mg_cheby_estimate ? "on" : "off")
                      << " estimate_iters=" << options.mg_cheby_estimate_iters
                      << " estimate_min_factor="
                      << options.mg_cheby_estimate_min_factor
                      << " estimate_max_factor="
                      << options.mg_cheby_estimate_max_factor
                      << '\n';
          } else {
            std::cout << "MG smoother: jacobi\n";
          }
          std::cout << "MG fine-level stats: nuH[min,max]=[" << nuH_min << ", "
                    << nuH_max << "] beta[min,max]=[" << beta_min << ", "
                    << beta_max << "] diag[min,max]=[" << diag_min << ", "
                    << diag_max << "] rhs[min,max]=[" << rhs_min << ", "
                    << rhs_max << "]";
          if (!(nuH_stats.all_finite && beta_stats.all_finite &&
                diag.all_finite && rhs_stats.all_finite)) {
            std::cout << " (non-finite values detected)";
          }
          std::cout << '\n';
        }
      }
    }

    GMRESOptions gmres_opts;
    gmres_opts.restart = options.gmres_restart;
    gmres_opts.max_iter = options.gmres_max_iter;
    gmres_opts.tol = options.gmres_tol;
    gmres_opts.tol_relative = options.gmres_tol_relative;
    gmres_opts.verbose = options.gmres_verbose;
    gmres_opts.precond_diagnostic =
        options.gmres_precond_diagnostic && (iter == 0);
    gmres_opts.context = context;

    SSAApplyOperator op(ssa_, grid_, nuH, beta, use_bc ? &bc : nullptr,
                        context);
    GMRESResult gmres_result = gmres_solve(op, rhs, vel, gmres_opts, precond_ptr);
    result.linear_iters = gmres_result.iterations;
    result.linear_residual = gmres_result.residual;

    if (options.fail_fast) {
      bool ok = true;
      std::string reason;
      if (!std::isfinite(gmres_result.residual)) {
        ok = false;
        reason = "GMRES residual is non-finite";
      } else if (options.fail_fast_residual_max > 0.0 &&
                 gmres_result.residual > options.fail_fast_residual_max) {
        ok = false;
        reason = "GMRES residual exceeds ssa.fail_fast_residual_max";
      } else if (options.fail_fast_require_converged && !gmres_result.converged) {
        ok = false;
        reason = "GMRES did not converge";
      } else if (!field_stats(vel).all_finite) {
        ok = false;
        reason = "SSA velocity contains non-finite values";
      }

      if (!ok) {
        if (is_rank0(context)) {
          std::cerr << "SSA fail-fast at Picard iter " << iter << ": " << reason
                    << " (iters=" << gmres_result.iterations
                    << " residual=" << gmres_result.residual << ")\n";
        }

#if GPISM_HAVE_NETCDF
        if (!options.fail_fast_dump_prefix.empty()) {
          try {
            std::filesystem::path base(options.fail_fast_dump_prefix);
            std::filesystem::path dir =
                base / ("ssa_fail_iter" + std::to_string(iter));
            std::filesystem::create_directories(dir);

            // Copy config override file for exact reproduction.
            if (!options.config_override_path.empty()) {
              (void)copy_file_best_effort(
                  options.config_override_path,
                  (dir / "config_override.cfg").string());
            }

            SSADebugBundle2D bundle;
            bundle.thk = &thk;
            bundle.topg = &topg;
            bundle.usurf = &usurf;
            bundle.dhdx = &dhdx;
            bundle.dhdy = &dhdy;
            bundle.cell_type = &cell_type;
            bundle.beta = &beta;
            bundle.rhs = &rhs;
            bundle.nuH = &nuH;
            bundle.vel_prev = &vel_prev;
            bundle.vel = &vel;

            NetcdfIO io;
            const std::string dump_path = (dir / "ssa_bundle.nc").string();
            if (context) {
              (void)io.write_ssa_debug_bundle(dump_path, *context, grid_, bundle, 0.0);
            } else {
              (void)io.write_ssa_debug_bundle(dump_path, grid_, bundle, 0.0);
            }
          } catch (const std::exception& exc) {
            if (is_rank0(context)) {
              std::cerr << "SSA fail-fast: debug dump failed: " << exc.what() << "\n";
            }
          }
        }
#endif
        throw std::runtime_error("SSA fail-fast: " + reason);
      }
    }

    if (options.vel_relax < 1.0) {
      apply_vel_relax(vel, vel_prev, options.vel_relax);
    }

    if (options.max_speed > 0.0) {
      compute_cell_center_velocity(grid_, vel, u_center, v_center);
      compute_speed_scale(grid_, u_center, v_center, options.max_speed,
                          speed_scale);
      apply_speed_scale(grid_, speed_scale, vel);

      // The staggered representation + boundary clamping can allow small
      // violations if scaling is imperfect. Enforce a strict cap by checking
      // the resulting cell-center speed and applying a global rescale if needed.
      compute_cell_center_velocity(grid_, vel, u_center, v_center);
      const double smax = max_center_speed(grid_, u_center, v_center, context);
      if (smax > options.max_speed * (1.0 + 1e-12) && smax > 0.0) {
        const double factor = options.max_speed / smax;
        scal(factor, vel);
        if (options.diagnostic && is_rank0(context)) {
          std::cout << "SSA: max_speed fallback rescale applied (smax=" << smax
                    << " max_speed=" << options.max_speed
                    << " factor=" << factor << ")\n";
        }
      }
    }

    double nuH_diff_local = 0.0;
    double nuH_norm_local = 0.0;
    double vel_diff_local = 0.0;
    double vel_norm_local = 0.0;
    if (options.force_host_convergence) {
      const bool was_device = device_enabled();
      if (was_device) {
        sync_device_to_host(nuH);
        sync_device_to_host(nuH_prev);
        sync_device_to_host(vel);
        sync_device_to_host(vel_prev);
        set_device_enabled(false);
      }
      nuH_diff_local = diff_norm1(nuH, nuH_prev);
      nuH_norm_local = norm1(nuH_prev);
      vel_diff_local = diff_norm1(vel, vel_prev);
      vel_norm_local = norm1(vel_prev);
      if (was_device) {
        set_device_enabled(true);
      }
    } else {
      nuH_diff_local = diff_norm1(nuH, nuH_prev);
      nuH_norm_local = norm1(nuH_prev);
      vel_diff_local = diff_norm1(vel, vel_prev);
      vel_norm_local = norm1(vel_prev);
    }

    const double nuH_diff = global_sum(context, nuH_diff_local);
    const double nuH_norm = std::max(global_sum(context, nuH_norm_local), 1e-12);
    result.nuH_change = nuH_diff / nuH_norm;

    const double vel_diff = global_sum(context, vel_diff_local);
    const double vel_norm = std::max(global_sum(context, vel_norm_local), 1e-12);
    result.vel_change = vel_diff / vel_norm;

    const bool converged = result.nuH_change <= options.tol_nuH &&
                           result.vel_change <= options.tol_vel;
    if (options.diagnostic && is_rank0(context)) {
      std::cout << "SSA Picard iter " << iter
                << ": GMRES iters=" << gmres_result.iterations
                << " residual=" << gmres_result.residual
                << " nuH_change=" << result.nuH_change
                << " vel_change=" << result.vel_change
                << " converged=" << (converged ? "yes" : "no") << '\n';
    }

    result.picard_iters = iter + 1;
    if (converged) {
      result.converged = true;
      break;
    }
  }

  return result;
}

}  // namespace gpism
