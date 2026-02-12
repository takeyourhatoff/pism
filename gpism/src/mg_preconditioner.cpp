#include "gpism/mg_preconditioner.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "gpism/context.h"
#include "gpism/field_sync.h"
#include "gpism/linear_algebra.h"

namespace gpism {
void ssa_replace_zero_diagonal_entries_cuda(
    int mx, int my, int gw, int stride_nu_u, int stride_nu_v,
    int stride_beta_u, int stride_beta_v, double* beta_u, double* beta_v,
    const double* nu_u, const double* nu_v, double inv_dx2, double inv_dy2,
    int stride_mask_u, int stride_mask_v, const int* mask_u,
    const int* mask_v, int has_bc, double beta_ice_free_bedrock, int periodic);
}  // namespace gpism

#if GPISM_HAVE_MPI
#include <mpi.h>
#endif

namespace {

double global_sum(const gpism::Context* context, double local_value) {
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

bool is_rank0(const gpism::Context* context) {
  if (!context || !context->mpi_enabled()) {
    return true;
  }
  return context->rank() == 0;
}

void copy_stag_mask(const gpism::FieldStag2D<int>& src,
                    gpism::FieldStag2D<int>& dst) {
  for (int comp = 0; comp < 2; ++comp) {
    const auto& s = src.component(comp);
    auto& d = dst.component(comp);
    const int mx = d.local_mx();
    const int my = d.local_my();
    for (int j = 0; j < my; ++j) {
      for (int i = 0; i < mx; ++i) {
        d(i, j) = s(i, j);
      }
    }
  }
}

void copy_stag_values(const gpism::FieldStag2D<double>& src,
                      gpism::FieldStag2D<double>& dst) {
  for (int comp = 0; comp < 2; ++comp) {
    const auto& s = src.component(comp);
    auto& d = dst.component(comp);
    const int mx = d.local_mx();
    const int my = d.local_my();
    for (int j = 0; j < my; ++j) {
      for (int i = 0; i < mx; ++i) {
        d(i, j) = s(i, j);
      }
    }
  }
}

void restrict_stag_mask(const gpism::FieldStag2D<int>& fine,
                        gpism::FieldStag2D<int>& coarse) {
  const int fine_mx = fine.local_mx();
  const int fine_my = fine.local_my();
  for (int comp = 0; comp < 2; ++comp) {
    const auto& f = fine.component(comp);
    auto& c = coarse.component(comp);
    const int coarse_mx = c.local_mx();
    const int coarse_my = c.local_my();
    for (int j = 0; j < coarse_my; ++j) {
      const int fj0 = std::min(2 * j, fine_my - 1);
      const int fj1 = std::min(fj0 + 1, fine_my - 1);
      for (int i = 0; i < coarse_mx; ++i) {
        const int fi0 = std::min(2 * i, fine_mx - 1);
        const int fi1 = std::min(fi0 + 1, fine_mx - 1);
        const int v00 = f(fi0, fj0);
        const int v10 = f(fi1, fj0);
        const int v01 = f(fi0, fj1);
        const int v11 = f(fi1, fj1);
        c(i, j) = std::max(std::max(v00, v10), std::max(v01, v11));
      }
    }
  }
}

void restrict_stag_values_host(const gpism::FieldStag2D<double>& fine,
                               gpism::FieldStag2D<double>& coarse) {
  const int fine_mx = fine.local_mx();
  const int fine_my = fine.local_my();
  for (int comp = 0; comp < 2; ++comp) {
    const auto& f = fine.component(comp);
    auto& c = coarse.component(comp);
    const int coarse_mx = c.local_mx();
    const int coarse_my = c.local_my();
    for (int j = 0; j < coarse_my; ++j) {
      const int fj0 = std::min(2 * j, fine_my - 1);
      const int fj1 = std::min(fj0 + 1, fine_my - 1);
      for (int i = 0; i < coarse_mx; ++i) {
        const int fi0 = std::min(2 * i, fine_mx - 1);
        const int fi1 = std::min(fi0 + 1, fine_mx - 1);
        const double v00 = f(fi0, fj0);
        const double v10 = f(fi1, fj0);
        const double v01 = f(fi0, fj1);
        const double v11 = f(fi1, fj1);
        c(i, j) = 0.25 * (v00 + v10 + v01 + v11);
      }
    }
  }
}

void guard_zero_diag(gpism::MGLevel& level,
                     const gpism::SSABoundaryCondition* bc,
                     double beta_ice_free_bedrock) {
  if (beta_ice_free_bedrock <= 0.0) {
    return;
  }

  const bool has_bc = bc && bc->mask;
  if (!(level.nuH.component(0).has_device_data() &&
        level.nuH.component(1).has_device_data() &&
        level.beta.component(0).has_device_data() &&
        level.beta.component(1).has_device_data() &&
        (!has_bc ||
         (bc->mask->component(0).has_device_data() &&
          bc->mask->component(1).has_device_data())))) {
    throw std::runtime_error(
        "MultigridPreconditioner requires device-resident nuH/beta/bc_mask");
  }
  const int periodic =
      (level.grid.dims_x() == 1 && level.grid.dims_y() == 1) ? 1 : 0;
  gpism::ssa_replace_zero_diagonal_entries_cuda(
      level.grid.local_mx(), level.grid.local_my(), level.grid.ghost_width(),
      level.nuH.component(0).stride(), level.nuH.component(1).stride(),
      level.beta.component(0).stride(), level.beta.component(1).stride(),
      level.beta.component(0).device_data(), level.beta.component(1).device_data(),
      level.nuH.component(0).device_data(), level.nuH.component(1).device_data(),
      1.0 / (level.grid.dx() * level.grid.dx()),
      1.0 / (level.grid.dy() * level.grid.dy()),
      has_bc ? bc->mask->component(0).stride() : 0,
      has_bc ? bc->mask->component(1).stride() : 0,
      has_bc ? bc->mask->component(0).device_data() : nullptr,
      has_bc ? bc->mask->component(1).device_data() : nullptr, has_bc ? 1 : 0,
      beta_ice_free_bedrock, periodic);
}

}  // namespace

namespace gpism {

void MultigridPreconditioner::apply(const FieldStag2D<double>& x,
                                    FieldStag2D<double>& y) const {
  if (mg_.num_levels() == 0) {
    return;
  }

  MGLevel& fine = mg_.level(0);
  copy(x, fine.rhs);
  set(0.0, fine.u);

  double r_norm = 0.0;
  const bool do_diag = diagnostic_ && !diagnostic_printed_;
  if (do_diag) {
    r_norm = std::sqrt(global_sum(context_, dot(x, x)));
    if (precision_ == MGPrecondPrecision::FP32 && is_rank0(context_)) {
      std::cout << "MG preconditioner precision: fp32 workspace\n";
    }
  }

  if (bc_ && bc_->mask && !bc_levels_cached_) {
    const int levels = mg_.num_levels();
    bc_levels_.assign(levels, {});
    MGLevel& fine = mg_.level(0);
    if (bc_->mask->component(0).has_device_data()) {
      sync_device_to_host(*const_cast<FieldStag2D<int>*>(bc_->mask));
    }
    copy_stag_mask(*bc_->mask, fine.bc_mask);
    if (bc_->values) {
      if (bc_->values->component(0).has_device_data()) {
        sync_device_to_host(*const_cast<FieldStag2D<double>*>(bc_->values));
      }
      copy_stag_values(*bc_->values, fine.bc_values);
    }
    bc_levels_[0].mask = &fine.bc_mask;
    bc_levels_[0].values = bc_->values ? &fine.bc_values : nullptr;
    for (int level = 1; level < levels; ++level) {
      MGLevel& coarse = mg_.level(level);
      MGLevel& prev = mg_.level(level - 1);
      restrict_stag_mask(prev.bc_mask, coarse.bc_mask);
      if (bc_->values) {
        restrict_stag_values_host(prev.bc_values, coarse.bc_values);
      }
      bc_levels_[level].mask = &coarse.bc_mask;
      bc_levels_[level].values = bc_->values ? &coarse.bc_values : nullptr;
    }
    for (int level = 0; level < levels; ++level) {
      sync_host_to_device(mg_.level(level).bc_mask);
      if (bc_->values) {
        sync_host_to_device(mg_.level(level).bc_values);
      }
    }
    bc_levels_cached_ = true;
  }

  if (beta_ice_free_bedrock_ > 0.0 && !beta_guarded_) {
    const int levels = mg_.num_levels();
    for (int level = 0; level < levels; ++level) {
      const SSABoundaryCondition* level_bc = bc_;
      if (!bc_levels_.empty() &&
          level < static_cast<int>(bc_levels_.size())) {
        level_bc = &bc_levels_[level];
      }
      guard_zero_diag(mg_.level(level), level_bc, beta_ice_free_bedrock_);
    }
    beta_guarded_ = true;
  }

  const std::vector<ChebyBounds>* bounds_ptr = nullptr;
  if (smoother_ == MGSmoother::Chebyshev) {
    if (!cheby_bounds_cached_ ||
        static_cast<int>(cheby_bounds_cache_.size()) != mg_.num_levels()) {
      cheby_bounds_cache_ = estimate_cheby_bounds(
          mg_, cheby_lambda_min_, cheby_lambda_max_, cheby_estimate_,
          cheby_estimate_iters_, cheby_estimate_min_factor_,
          cheby_estimate_max_factor_, bc_, context_,
          bc_levels_.empty() ? nullptr : &bc_levels_);
      cheby_bounds_cached_ = true;
    }
    bounds_ptr = &cheby_bounds_cache_;
  }

  v_cycle(mg_, pre_iters_, post_iters_, coarse_iters_, omega_, smoother_,
          cheby_lambda_min_, cheby_lambda_max_, cheby_estimate_,
          cheby_estimate_iters_, cheby_estimate_min_factor_,
          cheby_estimate_max_factor_, jacobi_sweeps_per_launch_, bc_, context_,
          bounds_ptr,
          bc_levels_.empty() ? nullptr : &bc_levels_,
          diagnostic_ && !diagnostic_printed_);

  copy(fine.u, y);

  if (do_diag) {
    compute_residual(fine.grid, fine.nuH, fine.beta, x, y, fine.r, fine.Ax,
                     bc_, context_);
    const double r_az_norm =
        std::sqrt(global_sum(context_, dot(fine.r, fine.r)));
    if (is_rank0(context_)) {
      std::cout << "MG V-cycle diagnostic: ||r||=" << r_norm
                << " ||r - A M^{-1} r||=" << r_az_norm;
      if (r_norm > 0.0) {
        std::cout << " ratio=" << (r_az_norm / r_norm);
      }
      std::cout << '\n';
    }
    diagnostic_printed_ = true;
  }
}

}  // namespace gpism
