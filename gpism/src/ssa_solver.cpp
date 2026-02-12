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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpism/context.h"
#include "gpism/field_sync.h"
#include "gpism/gmres.h"
#include "gpism/halo_exchange.h"
#include "gpism/linear_algebra.h"
#include "gpism/mg_preconditioner.h"
#include "gpism/thickness.h"

#include <cuda_runtime.h>
#include "gpism/netcdf_io.h"

#if GPISM_HAVE_MPI
#include <mpi.h>
#endif

namespace gpism {
void ssa_relax_nuH_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                        const double* nuH_prev_u, const double* nuH_prev_v,
                        double* nuH_u, double* nuH_v, double nuH_min,
                        double nuH_max, double nuH_relax);
void ssa_relax_vel_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                        const double* vel_prev_u, const double* vel_prev_v,
                        double* vel_u, double* vel_v, double vel_relax);
void ssa_extrapolate_velocity_cuda(int mx, int my, int gw, int stride_cell_type,
                                   int stride_u, int stride_v,
                                   const int* cell_type,
                                   double* u, double* v);
}  // namespace gpism

namespace gpism {
namespace {

template <typename T>
bool can_use_device_exchange(const FieldStag2D<T>& field,
                             const Context& context) {
  return context.cuda_aware_mpi() &&
         field.component(0).device_data() != nullptr &&
         field.component(1).device_data() != nullptr;
}

template <typename T>
void exchange_for_device(FieldStag2D<T>& field, const Grid2D& grid,
                         const Context& context, SSAHaloMode mode,
                         bool require_cuda_aware_mpi) {
  (void)mode;
  if (!context.mpi_enabled() || context.size() <= 1) {
    return;
  }
  const bool can_device = can_use_device_exchange(field, context);
  if (!can_device) {
    if (require_cuda_aware_mpi) {
      throw std::runtime_error(
          "device.require_cuda_aware_mpi=1 but CUDA-aware MPI exchange is "
          "unavailable for FieldStag2D");
    }
    throw std::runtime_error(
        "Multi-rank run requires CUDA-aware MPI device-buffer exchange "
        "for FieldStag2D");
  }
  HaloExchange2D exchange;
  exchange.exchange(field, grid, context, HaloExchange2D::Mode::Device);
}

template <typename T>
bool can_use_device_exchange(const Field2D<T>& field, const Context& context) {
  return context.cuda_aware_mpi() && field.device_data() != nullptr;
}

template <typename T>
void exchange_for_device(Field2D<T>& field, const Grid2D& grid,
                         const Context& context, SSAHaloMode mode,
                         bool require_cuda_aware_mpi) {
  (void)mode;
  if (!context.mpi_enabled() || context.size() <= 1) {
    return;
  }
  const bool can_device = can_use_device_exchange(field, context);
  if (!can_device) {
    if (require_cuda_aware_mpi) {
      throw std::runtime_error(
          "device.require_cuda_aware_mpi=1 but CUDA-aware MPI exchange is "
          "unavailable for Field2D");
    }
    throw std::runtime_error(
        "Multi-rank run requires CUDA-aware MPI device-buffer exchange "
        "for Field2D");
  }
  HaloExchange2D exchange;
  exchange.exchange(field, grid, context, HaloExchange2D::Mode::Device);
}

double global_sum(const Context* context, double local_value) {
#if GPISM_HAVE_MPI
  if (context && context->mpi_enabled() && context->size() > 1) {
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
  if (context && context->mpi_enabled() && context->size() > 1) {
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
  if (context && context->mpi_enabled() && context->size() > 1) {
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

void apply_nuH_constraints(FieldStag2D<double>& nuH,
                           const FieldStag2D<double>& nuH_prev, double nuH_min,
                           double nuH_max, double nuH_relax) {
  if (!(nuH.component(0).has_device_data() &&
        nuH.component(1).has_device_data() &&
        nuH_prev.component(0).has_device_data() &&
        nuH_prev.component(1).has_device_data())) {
    throw std::runtime_error(
        "apply_nuH_constraints requires device-resident fields");
  }

  ssa_relax_nuH_cuda(nuH.local_mx(), nuH.local_my(), nuH.ghost_width(),
                     nuH.component(0).stride(), nuH.component(1).stride(),
                     nuH_prev.component(0).device_data(),
                     nuH_prev.component(1).device_data(),
                     nuH.component(0).device_data(),
                     nuH.component(1).device_data(), nuH_min, nuH_max,
                     nuH_relax);
}

void apply_vel_relax(FieldStag2D<double>& vel,
                     const FieldStag2D<double>& vel_prev,
                     double vel_relax) {
  if (!(vel.component(0).has_device_data() &&
        vel.component(1).has_device_data() &&
        vel_prev.component(0).has_device_data() &&
        vel_prev.component(1).has_device_data())) {
    throw std::runtime_error(
        "apply_vel_relax requires device-resident fields");
  }

  ssa_relax_vel_cuda(vel.local_mx(), vel.local_my(), vel.ghost_width(),
                     vel.component(0).stride(), vel.component(1).stride(),
                     vel_prev.component(0).device_data(),
                     vel_prev.component(1).device_data(),
                     vel.component(0).device_data(),
                     vel.component(1).device_data(), vel_relax);
}

class SSAApplyOperator : public LinearOperator {
public:
  SSAApplyOperator(const SSAOperator& op, const Grid2D& grid,
                   const FieldStag2D<double>& nuH,
                   const FieldStag2D<double>& beta,
                   const SSABoundaryCondition* bc, const Context* context,
                   SSAHaloMode halo_mode, bool require_cuda_aware_mpi)
      : op_(op),
        grid_(grid),
        nuH_(nuH),
        beta_(beta),
        bc_(bc),
        context_(context),
        halo_mode_(halo_mode),
        require_cuda_aware_mpi_(require_cuda_aware_mpi) {}

  void apply(const FieldStag2D<double>& x,
             FieldStag2D<double>& y) const override {
    if (context_ && context_->mpi_enabled() && context_->size() > 1) {
      const int gw = x.ghost_width();
      if (gw > 0) {
        auto& mutable_x = const_cast<FieldStag2D<double>&>(x);
        HaloExchange2D exchange;
        const bool can_device = can_use_device_exchange(mutable_x, *context_);
        if (!can_device && require_cuda_aware_mpi_) {
          throw std::runtime_error(
              "device.require_cuda_aware_mpi=1 but SSA operator halo "
              "exchange cannot use device buffers");
        }
        if (!can_device) {
          throw std::runtime_error(
              "Multi-rank run requires CUDA-aware MPI device-buffer exchange "
              "for SSA operator halo exchange");
        }
        auto handle = exchange.start_exchange(mutable_x, grid_, *context_,
                                              HaloExchange2D::Mode::Device);

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
  SSAHaloMode halo_mode_;
  bool require_cuda_aware_mpi_;
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
  const SSAHaloMode halo_mode = options.halo_mode;
  const bool require_cuda_aware_mpi = options.require_cuda_aware_mpi;

  if (require_cuda_aware_mpi && context && context->mpi_enabled() &&
      context->size() > 1 && !context->cuda_aware_mpi()) {
    throw std::runtime_error(
        "device.require_cuda_aware_mpi=1 but CUDA-aware MPI is disabled");
  }

  if (options.precond_precision == SSAPrecondPrecision::FP32 &&
      options.use_mg_precond && is_rank0(context) && options.diagnostic) {
    std::cout
        << "SSA: mixed preconditioner precision requested (fp32 workspace).\n";
  }

  workspace_.ensure(grid_);
  auto& usurf = workspace_.usurf;
  auto& dhdx = workspace_.dhdx;
  auto& dhdy = workspace_.dhdy;
  auto& cell_type = workspace_.cell_type;
  auto& beta = workspace_.beta;
  auto& rhs = workspace_.rhs;
  auto& nuH = workspace_.nuH;
  auto& nuH_prev = workspace_.nuH_prev;
  auto& vel_prev = workspace_.vel_prev;
  auto& bc_mask = workspace_.bc_mask;
  auto& bc_values = workspace_.bc_values;
  SSABoundaryCondition bc;

  Field2D<double>& u_center = vel.component(0);
  Field2D<double>& v_center = vel.component(1);

  GeometryDiagnostics geometry;
  compute_cell_type(grid_, thk, topg, options.sea_level, options.rho_ice,
                    options.rho_water, options.ice_free_thickness_standard,
                    cell_type);
  if (context && context->mpi_enabled() && context->size() > 1) {
    exchange_for_device(cell_type, grid_, *context, halo_mode,
                        require_cuda_aware_mpi);
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
    exchange_for_device(usurf, grid_, *context, halo_mode,
                        require_cuda_aware_mpi);
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
    exchange_for_device(rhs, grid_, *context, halo_mode,
                        require_cuda_aware_mpi);
  }

  // Initialize nuH from the initial velocity guess so that nuH relaxation (if
  // enabled) starts from a physically meaningful field. Starting from all-zeros
  // can slow Picard convergence dramatically and lead to large divergences
  // from PISM on real-world problems.
  if (context && context->mpi_enabled() && context->size() > 1) {
    exchange_for_device(vel, grid_, *context, halo_mode,
                        require_cuda_aware_mpi);
  }
  viscosity_.compute_nuH(grid_, thk, vel, nuH, options.nuH_regularization,
                         options.strength_extension_nu,
                         options.strength_extension_min_thickness,
                         options.enthalpy, options.enthalpy_gamma,
                         options.enthalpy_ref);
  copy(nuH, nuH_prev);
  if (options.nuH_min > 0.0 || options.nuH_max > 0.0 ||
      options.nuH_relax < 1.0) {
    apply_nuH_constraints(nuH, nuH_prev, options.nuH_min, options.nuH_max,
                          options.nuH_relax);
  }
  if (context && context->mpi_enabled() && context->size() > 1) {
    exchange_for_device(nuH, grid_, *context, halo_mode,
                        require_cuda_aware_mpi);
  }

  const int picard_convergence_check_interval =
      std::max(1, options.picard_convergence_check_interval);
  const bool batch_device_metrics = options.device_metrics_batch;
  DeviceScalarBuffer picard_metrics;
  if (batch_device_metrics) {
    picard_metrics.ensure(7);
  }

  for (int iter = 0; iter < options.max_picard; ++iter) {
    copy(nuH, nuH_prev);
    copy(vel, vel_prev);

    ssa_.compute_basal_drag(grid_, tauc, u_center, v_center, topg, usurf,
                            cell_type, beta, options.basal_params);
    if (context && context->mpi_enabled() && context->size() > 1) {
      exchange_for_device(beta, grid_, *context, halo_mode,
                          require_cuda_aware_mpi);
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
        exchange_for_device(beta, grid_, *context, halo_mode,
                            require_cuda_aware_mpi);
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
                      options.mg_jacobi_sweeps_per_launch,
                      options.use_bc ? &bc : nullptr, context,
                      options.mg_diagnostic && iter == 0,
                      options.basal_params.beta_ice_free_bedrock,
                      options.precond_precision == SSAPrecondPrecision::FP32
                          ? MGPrecondPrecision::FP32
                          : MGPrecondPrecision::FP64);
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
    gmres_opts.tol_relative_to_rhs = options.gmres_tol_relative_to_rhs;
    gmres_opts.residual_check_interval =
        std::max(1, options.gmres_residual_check_interval);
    gmres_opts.verbose = options.gmres_verbose;
    gmres_opts.precond_diagnostic =
        options.gmres_precond_diagnostic && (iter == 0);
    gmres_opts.context = context;

    SSAApplyOperator op(ssa_, grid_, nuH, beta, use_bc ? &bc : nullptr,
                        context, halo_mode, require_cuda_aware_mpi);
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
      }

      if (!ok) {
        if (is_rank0(context)) {
          std::cerr << "SSA fail-fast at Picard iter " << iter << ": " << reason
                    << " (iters=" << gmres_result.iterations
                    << " residual=" << gmres_result.residual << ")\n";
        }

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
        throw std::runtime_error("SSA fail-fast: " + reason);
      }
    }

    if (options.vel_relax < 1.0) {
      apply_vel_relax(vel, vel_prev, options.vel_relax);
    }

    // Update viscosity using the new velocity iterate and check nonlinear
    // convergence based on nuH changes (PISM SSAFD semantics).
    if (context && context->mpi_enabled() && context->size() > 1) {
      exchange_for_device(vel, grid_, *context, halo_mode,
                          require_cuda_aware_mpi);
    }
    viscosity_.compute_nuH(grid_, thk, vel, nuH, options.nuH_regularization,
                           options.strength_extension_nu,
                           options.strength_extension_min_thickness,
                           options.enthalpy, options.enthalpy_gamma,
                           options.enthalpy_ref);
    if (options.nuH_min > 0.0 || options.nuH_max > 0.0 ||
        options.nuH_relax < 1.0) {
      // Apply nuH damping only after the first Picard iteration. Workspace
      // storage starts at zero, so damping on iter=0 would artificially halve
      // nuH (including the epsilon regularization) and can destabilize the
      // solve.
      const double nuH_relax = (iter == 0) ? 1.0 : options.nuH_relax;
      apply_nuH_constraints(nuH, nuH_prev, options.nuH_min, options.nuH_max,
                            nuH_relax);
    }
    if (context && context->mpi_enabled() && context->size() > 1) {
      exchange_for_device(nuH, grid_, *context, halo_mode,
                          require_cuda_aware_mpi);
    }

    const bool do_convergence_check =
        (((iter + 1) % picard_convergence_check_interval) == 0) ||
        (iter + 1 == options.max_picard);
    if (do_convergence_check) {
      // PISM SSAFD convergence uses an L1-based norm for nuH.
      double nuH_norm1_u_local = 0.0;
      double nuH_norm1_v_local = 0.0;
      double nuH_diff_norm1_u_local = 0.0;
      double nuH_diff_norm1_v_local = 0.0;
      double vel_diff_norm2_sq_local = 0.0;
      double vel_norm2_sq_local = 0.0;

      if (batch_device_metrics) {
        norm1_device_async(nuH.component(0), picard_metrics, 0);
        norm1_device_async(nuH.component(1), picard_metrics, 1);
        diff_norm1_device_async(nuH.component(0), nuH_prev.component(0),
                                picard_metrics, 2);
        diff_norm1_device_async(nuH.component(1), nuH_prev.component(1),
                                picard_metrics, 3);
        dot_device_async(vel_prev, vel_prev, picard_metrics, 4);
        dot_device_async(vel, vel, picard_metrics, 5);
        dot_device_async(vel, vel_prev, picard_metrics, 6);
        copy_device_scalars_to_host(picard_metrics, 7);
        const double* metrics = picard_metrics.host_data();
        nuH_norm1_u_local = metrics[0];
        nuH_norm1_v_local = metrics[1];
        nuH_diff_norm1_u_local = metrics[2];
        nuH_diff_norm1_v_local = metrics[3];
        const double vel_prev_norm2_sq = metrics[4];
        const double vel_norm2_sq = metrics[5];
        const double vel_cross = metrics[6];
        vel_diff_norm2_sq_local =
            vel_norm2_sq + vel_prev_norm2_sq - 2.0 * vel_cross;
        vel_norm2_sq_local = vel_norm2_sq;
      } else {
        nuH_norm1_u_local = norm1(nuH.component(0));
        nuH_norm1_v_local = norm1(nuH.component(1));
        nuH_diff_norm1_u_local =
            diff_norm1(nuH.component(0), nuH_prev.component(0));
        nuH_diff_norm1_v_local =
            diff_norm1(nuH.component(1), nuH_prev.component(1));

        const double vel_prev_norm2_sq = dot(vel_prev, vel_prev);
        const double vel_norm2_sq = dot(vel, vel);
        const double vel_cross = dot(vel, vel_prev);
        vel_diff_norm2_sq_local =
            vel_norm2_sq + vel_prev_norm2_sq - 2.0 * vel_cross;
        vel_norm2_sq_local = vel_norm2_sq;
      }

      const double nuH_norm1_u =
          std::max(0.0, global_sum(context, nuH_norm1_u_local));
      const double nuH_norm1_v =
          std::max(0.0, global_sum(context, nuH_norm1_v_local));
      const double nuH_diff_norm1_u =
          std::max(0.0, global_sum(context, nuH_diff_norm1_u_local));
      const double nuH_diff_norm1_v =
          std::max(0.0, global_sum(context, nuH_diff_norm1_v_local));

      const double nuH_norm =
          std::sqrt(nuH_norm1_u * nuH_norm1_u + nuH_norm1_v * nuH_norm1_v);
      const double nuH_norm_change =
          std::sqrt(nuH_diff_norm1_u * nuH_diff_norm1_u +
                    nuH_diff_norm1_v * nuH_diff_norm1_v);
      result.nuH_change = nuH_norm_change / std::max(nuH_norm, 1e-24);

      const double vel_diff_norm2_sq =
          std::max(0.0, global_sum(context, vel_diff_norm2_sq_local));
      const double vel_norm2_sq_global =
          std::max(global_sum(context, vel_norm2_sq_local), 1e-24);
      result.vel_change = std::sqrt(vel_diff_norm2_sq / vel_norm2_sq_global);
    }

    const bool vel_convergence_enabled = options.tol_vel > 0.0;
    const bool converged = do_convergence_check &&
                           result.nuH_change <= options.tol_nuH &&
                           (!vel_convergence_enabled ||
                            result.vel_change <= options.tol_vel);
    if (options.fail_fast && do_convergence_check &&
        (!std::isfinite(result.nuH_change) ||
         !std::isfinite(result.vel_change))) {
      throw std::runtime_error(
          "SSA fail-fast: Picard convergence metrics are non-finite");
    }
    if (options.diagnostic && is_rank0(context)) {
      std::cout << "SSA Picard iter " << iter
                << ": GMRES iters=" << gmres_result.iterations
                << " residual=" << gmres_result.residual
                << " nuH_change=" << result.nuH_change
                << " vel_change=" << result.vel_change
                << " vel_check=" << (vel_convergence_enabled ? "on" : "off")
                << " check="
                << (do_convergence_check ? "on" : "skip")
                << " converged=" << (converged ? "yes" : "no") << '\n';
    }

    result.picard_iters = iter + 1;
    if (converged) {
      result.converged = true;
      break;
    }
  }

  // Log final, converged field values at diagnostic points. At this point `nuH`
  // corresponds to the latest velocity iterate, but `beta` was computed using
  // the velocity at the start of the last Picard iteration. Recompute `beta`
  // using the final velocity so diagnostics are comparable to PISM.
  if (options.diagnostic && result.converged) {
    ssa_.compute_basal_drag(grid_, tauc, u_center, v_center, topg, usurf,
                            cell_type, beta, options.basal_params);
    if (context && context->mpi_enabled() && context->size() > 1) {
      exchange_for_device(beta, grid_, *context, halo_mode,
                          require_cuda_aware_mpi);
    }
    log_diag_points(grid_, result.picard_iters, cell_type, thk, topg, usurf,
                    dhdx, dhdy, u_center, v_center, nuH, beta, rhs, context);
  }

  if (options.extrapolate_at_margins && result.converged) {
    if (!(cell_type.has_device_data() && u_center.has_device_data() &&
          v_center.has_device_data())) {
      throw std::runtime_error(
          "ssa.extrapolate_at_margins requires device-resident fields");
    }

    ssa_extrapolate_velocity_cuda(grid_.local_mx(), grid_.local_my(),
                                  grid_.ghost_width(), cell_type.stride(),
                                  u_center.stride(), v_center.stride(),
                                  cell_type.device_data(), u_center.device_data(),
                                  v_center.device_data());
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("ssa_extrapolate_velocity_cuda launch "
                                           "failed: ") +
                               cudaGetErrorString(err));
    }

    if (context && context->mpi_enabled() && context->size() > 1) {
      exchange_for_device(vel, grid_, *context, halo_mode,
                          require_cuda_aware_mpi);
    }
  }

  return result;
}

}  // namespace gpism
