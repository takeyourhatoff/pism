#include "gpism/ssa_solver.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>

#include "gpism/context.h"
#include "gpism/device_policy.h"
#include "gpism/field_sync.h"
#include "gpism/gmres.h"
#include "gpism/halo_exchange.h"
#include "gpism/linear_algebra.h"
#include "gpism/mg_preconditioner.h"

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

FieldStats field_stats(const FieldStag2D<double>& field) {
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

FieldStats diag_stats(const Grid2D& grid, const FieldStag2D<double>& nuH,
                      const FieldStag2D<double>& beta) {
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
                   FieldStag2D<int>& mask_stag,
                   FieldStag2D<double>& values_stag) {
  mask_stag.fill(0);
  values_stag.fill(0.0);
  if (!vel_bc_mask || !u_bc || !v_bc) {
    return;
  }
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

}  // namespace

SSASolver::SSASolver(const Grid2D& grid, double rho, double g, double u_threshold,
                     const ViscosityModel& viscosity_model)
    : grid_(grid), ssa_(rho, g, u_threshold), viscosity_(viscosity_model) {}

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
  auto& beta = workspace_.beta;
  auto& rhs = workspace_.rhs;
  auto& nuH = workspace_.nuH;
  auto& nuH_prev = workspace_.nuH_prev;
  auto& vel_prev = workspace_.vel_prev;
  auto& bc_mask = workspace_.bc_mask;
  auto& bc_values = workspace_.bc_values;
  SSABoundaryCondition bc;
  if (options.use_bc) {
    build_bc_stag(grid_, u_bc, v_bc, vel_bc_mask, bc_mask, bc_values);
    bc.mask = &bc_mask;
    bc.values = &bc_values;
    sync_host_to_device(bc_mask);
    sync_host_to_device(bc_values);
  }

  GeometryDiagnostics geometry;
  geometry.compute_usurf(grid_, thk, topg, usurf);
  geometry.compute_surface_slopes(grid_, usurf, dhdx, dhdy);

  ssa_.compute_basal_drag(grid_, tauc, beta);
  if (context && context->mpi_enabled()) {
    exchange_for_device(beta, grid_, *context);
  }

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
    MultigridHierarchy& mg = *workspace_.mg;
    copy(beta, mg.level(0).beta);
    for (int level = 1; level < mg.num_levels(); ++level) {
      restrict_stag(mg.level(level - 1).beta, mg.level(level).beta);
    }
  }

  ssa_.assemble_rhs(grid_, thk, dhdx, dhdy, rhs,
                    options.use_bc ? &bc : nullptr);
  if (context && context->mpi_enabled()) {
    exchange_for_device(rhs, grid_, *context);
  }

  for (int iter = 0; iter < options.max_picard; ++iter) {
    copy(nuH, nuH_prev);
    copy(vel, vel_prev);

    if (context && context->mpi_enabled()) {
      exchange_for_device(vel, grid_, *context);
    }
    viscosity_.compute_nuH(grid_, thk, vel, nuH, options.enthalpy,
                           options.enthalpy_gamma, options.enthalpy_ref);
    if (options.nuH_min > 0.0 || options.nuH_max > 0.0 ||
        options.nuH_relax < 1.0) {
      apply_nuH_constraints(nuH, nuH_prev, options);
    }
    if (context && context->mpi_enabled()) {
      exchange_for_device(nuH, grid_, *context);
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
                      options.use_bc ? &bc : nullptr, context);
      precond_ptr = &(*precond);
      if (options.mg_diagnostic && iter == 0) {
        auto& fine = mg.level(0);
        sync_device_to_host(fine.nuH);
        sync_device_to_host(fine.beta);
        const FieldStats nuH_stats = field_stats(fine.nuH);
        const FieldStats beta_stats = field_stats(fine.beta);
        const FieldStats diag = diag_stats(fine.grid, fine.nuH, fine.beta);
        const double nuH_min = global_min(context, nuH_stats.min);
        const double nuH_max = global_max(context, nuH_stats.max);
        const double beta_min = global_min(context, beta_stats.min);
        const double beta_max = global_max(context, beta_stats.max);
        const double diag_min = global_min(context, diag.min);
        const double diag_max = global_max(context, diag.max);
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
                    << '\n';
          std::cout << "MG fine-level stats: nuH[min,max]=[" << nuH_min << ", "
                    << nuH_max << "] beta[min,max]=[" << beta_min << ", "
                    << beta_max << "] diag[min,max]=[" << diag_min << ", "
                    << diag_max << "]";
          if (!(nuH_stats.all_finite && beta_stats.all_finite && diag.all_finite)) {
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
    gmres_opts.precond_diagnostic =
        options.gmres_precond_diagnostic && (iter == 0);
    gmres_opts.context = context;

    SSAApplyOperator op(ssa_, grid_, nuH, beta, options.use_bc ? &bc : nullptr,
                        context);
    GMRESResult gmres_result = gmres_solve(op, rhs, vel, gmres_opts, precond_ptr);
    result.linear_iters = gmres_result.iterations;
    result.linear_residual = gmres_result.residual;

    if (options.vel_relax < 1.0) {
      apply_vel_relax(vel, vel_prev, options.vel_relax);
    }

    const double nuH_diff_local = diff_norm1(nuH, nuH_prev);
    const double nuH_norm_local = norm1(nuH_prev);
    const double nuH_diff = global_sum(context, nuH_diff_local);
    const double nuH_norm = std::max(global_sum(context, nuH_norm_local), 1e-12);
    result.nuH_change = nuH_diff / nuH_norm;

    const double vel_diff_local = diff_norm1(vel, vel_prev);
    const double vel_norm_local = norm1(vel_prev);
    const double vel_diff = global_sum(context, vel_diff_local);
    const double vel_norm = std::max(global_sum(context, vel_norm_local), 1e-12);
    result.vel_change = vel_diff / vel_norm;

    result.picard_iters = iter + 1;
    if (result.nuH_change <= options.tol_nuH &&
        result.vel_change <= options.tol_vel) {
      result.converged = true;
      break;
    }
  }

  return result;
}

}  // namespace gpism
