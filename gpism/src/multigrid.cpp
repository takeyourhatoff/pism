#include "gpism/multigrid.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "gpism/field_sync.h"
#include "gpism/halo_exchange.h"
#include "gpism/linear_algebra.h"
#include "gpism/ssa_operator.h"

#if GPISM_HAVE_MPI
#include <mpi.h>
#endif

namespace gpism {

namespace {

double global_sum(const Context* context, double local_value);

bool is_rank0(const Context* context) {
  if (!context || !context->mpi_enabled()) {
    return true;
  }
  return context->rank() == 0;
}

void log_mg_residual(int level, const char* phase,
                     const FieldStag2D<double>& r, double base,
                     const Context* context) {
  const double norm = std::sqrt(global_sum(context, dot(r, r)));
  if (!is_rank0(context)) {
    return;
  }
  std::cout << "MG level " << level << " " << phase << " ||r||=" << norm;
  if (base > 0.0) {
    std::cout << " ratio=" << (norm / base);
  }
  std::cout << '\n';
}

}  // namespace

#if GPISM_HAVE_CUDA
void mg_restrict_stag_cuda(int fine_mx, int fine_my, int fine_gw,
                           int fine_stride_u, int fine_stride_v, int coarse_mx,
                           int coarse_my, int coarse_gw, int coarse_stride_u,
                           int coarse_stride_v, const double* fine_u,
                           const double* fine_v, double* coarse_u,
                           double* coarse_v);
void mg_prolong_stag_cuda(int coarse_mx, int coarse_my, int coarse_gw,
                          int coarse_stride_u, int coarse_stride_v, int fine_mx,
                          int fine_my, int fine_gw, int fine_stride_u,
                          int fine_stride_v, const double* coarse_u,
                          const double* coarse_v, double* fine_u,
                          double* fine_v);
void mg_compute_diag_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                          const double* nu_u, const double* nu_v,
                          const double* beta_u, const double* beta_v,
                          double* diag_u, double* diag_v, double inv_dx2,
                          double inv_dy2, int stride_mask_u, int stride_mask_v,
                          const int* mask_u, const int* mask_v, int has_bc);
void mg_residual_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                      const double* b_u, const double* b_v, const double* Ax_u,
                      const double* Ax_v, double* r_u, double* r_v,
                      int stride_mask_u, int stride_mask_v, const int* mask_u,
                      const int* mask_v, int has_bc);
void mg_jacobi_update_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                           const double* b_u, const double* b_v,
                           const double* Ax_u, const double* Ax_v,
                           const double* diag_u, const double* diag_v,
                           double* x_u, double* x_v, double omega,
                           int stride_mask_u, int stride_mask_v,
                           const int* mask_u, const int* mask_v,
                           int stride_bc_u, int stride_bc_v, const double* bc_u,
                           const double* bc_v, int has_bc, int has_values);
void mg_jacobi_fused_cuda(
    int mx, int my, int gw, int stride_u, int stride_v, int stride_nu_u,
    int stride_nu_v, int stride_beta_u, int stride_beta_v, int stride_b_u,
    int stride_b_v, const double* x_old_u, const double* x_old_v, double* x_u,
    double* x_v, const double* nu_u, const double* nu_v, const double* beta_u,
    const double* beta_v, const double* b_u, const double* b_v, double omega,
    double inv_dx2, double inv_dy2, double inv_2dx, double inv_2dy,
    int stride_mask_u, int stride_mask_v, const int* mask_u,
    const int* mask_v, int stride_bc_u, int stride_bc_v, const double* bc_u,
    const double* bc_v, int has_bc, int has_values, int periodic);
void mg_cheby_compute_z_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                             const double* r_u, const double* r_v,
                             const double* diag_u, const double* diag_v,
                             double* z_u, double* z_v, int stride_mask_u,
                             int stride_mask_v, const int* mask_u,
                             const int* mask_v, int has_bc);
void mg_cheby_update_p_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                            const double* z_u, const double* z_v,
                            double* p_u, double* p_v, double beta_coeff,
                            int stride_mask_u, int stride_mask_v,
                            const int* mask_u, const int* mask_v, int has_bc,
                            int first_iter);
void mg_cheby_update_x_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                            const double* p_u, const double* p_v, double* x_u,
                            double* x_v, double alpha, int stride_mask_u,
                            int stride_mask_v, const int* mask_u,
                            const int* mask_v, int stride_bc_u,
                            int stride_bc_v, const double* bc_u,
                            const double* bc_v, int has_bc, int has_values);
void mg_cheby_update_r_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                            double* r_u, double* r_v, const double* Ap_u,
                            const double* Ap_v, double alpha,
                            int stride_mask_u, int stride_mask_v,
                            const int* mask_u, const int* mask_v, int has_bc);

void ssa_apply_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                    int stride_nu_u, int stride_nu_v, int stride_beta_u,
                    int stride_beta_v, int stride_out_u, int stride_out_v,
                    const double* u, const double* v, const double* nu_u,
                    const double* nu_v, const double* beta_u,
                    const double* beta_v, double* out_u, double* out_v,
                    double inv_dx2, double inv_dy2, double inv_2dx,
                    double inv_2dy, int stride_mask_u, int stride_mask_v,
                    const int* mask_u, const int* mask_v, int has_bc,
                    int periodic);
void ssa_apply_region_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                           int stride_nu_u, int stride_nu_v, int stride_beta_u,
                           int stride_beta_v, int stride_out_u, int stride_out_v,
                           const double* u, const double* v, const double* nu_u,
                           const double* nu_v, const double* beta_u,
                           const double* beta_v, double* out_u, double* out_v,
                           double inv_dx2, double inv_dy2, double inv_2dx,
                           double inv_2dy, int stride_mask_u, int stride_mask_v,
                           const int* mask_u, const int* mask_v, int has_bc,
                           int i_start, int i_end, int j_start, int j_end,
                           int periodic);
#endif

void apply_operator(const Grid2D& grid, const FieldStag2D<double>& nuH,
                    const FieldStag2D<double>& beta,
                    const FieldStag2D<double>& x, FieldStag2D<double>& Ax,
                    const SSABoundaryCondition* bc,
                    const Context* context);

namespace {

int coarsen_dim(int n) { return (n + 1) / 2; }

void compute_jacobi_diag(const Grid2D& grid, const FieldStag2D<double>& nuH,
                         const FieldStag2D<double>& beta,
                         FieldStag2D<double>& diag,
                         const SSABoundaryCondition* bc);

bool is_dirichlet(const SSABoundaryCondition* bc, int i, int j, int comp) {
  if (!bc || !bc->mask) {
    return false;
  }
  return (*bc->mask)(i, j, comp) != 0;
}

template <typename T>
void exchange_for_device(FieldStag2D<T>& field, const Grid2D& grid,
                         const Context* context) {
  if (!context || !context->mpi_enabled() || context->size() <= 1) {
    return;
  }
  HaloExchange2D exchange;
#if GPISM_HAVE_CUDA
  const bool can_device = context->cuda_aware_mpi() &&
                          field.component(0).device_data() != nullptr &&
                          field.component(1).device_data() != nullptr;
  if (can_device) {
    exchange.exchange(field, grid, *context, HaloExchange2D::Mode::Device);
    return;
  }
#endif
  sync_device_to_host(field);
  exchange.exchange(field, grid, *context, HaloExchange2D::Mode::Host);
  sync_host_to_device(field);
}

void ssa_apply_host(const Grid2D& grid, const FieldStag2D<double>& nuH,
                    const FieldStag2D<double>& beta,
                    const FieldStag2D<double>& vel, FieldStag2D<double>& out,
                    const SSABoundaryCondition* bc) {
  const double dx = grid.dx();
  const double dy = grid.dy();
  const double inv_dx2 = 1.0 / (dx * dx);
  const double inv_dy2 = 1.0 / (dy * dy);
  const double inv_d4 = 1.0 / (4.0 * dx * dy);
  const double inv_d2 = 2.0 * inv_d4;

  const Field2D<double>& u = vel.component(0);
  const Field2D<double>& v = vel.component(1);
  const Field2D<double>& nu_u = nuH.component(0);
  const Field2D<double>& nu_v = nuH.component(1);
  const Field2D<double>& beta_u = beta.component(0);
  const Field2D<double>& beta_v = beta.component(1);

  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      const int im1 = (i == 0) ? i : i - 1;
      const int ip1 = (i == grid.local_mx() - 1) ? i : i + 1;
      const int jm1 = (j == 0) ? j : j - 1;
      const int jp1 = (j == grid.local_my() - 1) ? j : j + 1;
      if (is_dirichlet(bc, i, j, 0)) {
        out(i, j, 0) = u(i, j);
      } else {
        const double u_c = u(i, j);
        const double c_n = nu_v(i, j);
        const double c_s = nu_v(i, jm1);
        const double c_e = nu_u(i, j);
        const double c_w = nu_u(im1, j);
        double sum =
            (-c_n * u(i, jp1) - c_s * u(i, jm1) +
             (c_n + c_s) * u_c) *
                inv_dy2 +
            (-4.0 * c_e * u(ip1, j) - 4.0 * c_w * u(im1, j) +
             4.0 * (c_e + c_w) * u_c) *
                inv_dx2;
        sum += (c_w * inv_d2 + c_n * inv_d4) * v(im1, jp1);
        sum += (c_w - c_e) * inv_d2 * v(i, jp1);
        sum += (-c_e * inv_d2 - c_n * inv_d4) * v(ip1, jp1);
        sum += (c_n - c_s) * inv_d4 * v(im1, j);
        sum += (c_s - c_n) * inv_d4 * v(ip1, j);
        sum += (-c_w * inv_d2 - c_s * inv_d4) * v(im1, jm1);
        sum += (c_e - c_w) * inv_d2 * v(i, jm1);
        sum += (c_e * inv_d2 + c_s * inv_d4) * v(ip1, jm1);
        out(i, j, 0) = sum + beta_u(i, j) * u_c;
      }

      if (is_dirichlet(bc, i, j, 1)) {
        out(i, j, 1) = v(i, j);
      } else {
        const double v_c = v(i, j);
        const double c_n = nu_v(i, j);
        const double c_s = nu_v(i, jm1);
        const double c_e = nu_u(i, j);
        const double c_w = nu_u(im1, j);
        double sum =
            (-4.0 * c_n * v(i, jp1) - 4.0 * c_s * v(i, jm1) +
             4.0 * (c_n + c_s) * v_c) *
                inv_dy2 +
            (-c_e * v(ip1, j) - c_w * v(im1, j) +
             (c_e + c_w) * v_c) *
                inv_dx2;
        sum += (c_w * inv_d4 + c_n * inv_d2) * u(im1, jp1);
        sum += (c_w - c_e) * inv_d4 * u(i, jp1);
        sum += (-c_e * inv_d4 - c_n * inv_d2) * u(ip1, jp1);
        sum += (c_n - c_s) * inv_d2 * u(im1, j);
        sum += (c_s - c_n) * inv_d2 * u(ip1, j);
        sum += (-c_w * inv_d4 - c_s * inv_d2) * u(im1, jm1);
        sum += (c_e - c_w) * inv_d4 * u(i, jm1);
        sum += (c_e * inv_d4 + c_s * inv_d2) * u(ip1, jm1);
        out(i, j, 1) = sum + beta_v(i, j) * v_c;
      }
    }
  }
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

void compute_diag(const Grid2D& grid, const FieldStag2D<double>& nuH,
                  const FieldStag2D<double>& beta, FieldStag2D<double>& diag,
                  const SSABoundaryCondition* bc, const Context* context) {
#if GPISM_HAVE_CUDA
  const bool has_bc = bc && bc->mask;
  const bool device_ready = nuH.component(0).has_device_data() &&
                            nuH.component(1).has_device_data() &&
                            beta.component(0).has_device_data() &&
                            beta.component(1).has_device_data() &&
                            diag.component(0).has_device_data() &&
                            diag.component(1).has_device_data() &&
                            (!has_bc ||
                             (bc->mask->component(0).has_device_data() &&
                              bc->mask->component(1).has_device_data()));
  if (device_ready) {
    const double dx = grid.dx();
    const double dy = grid.dy();
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const int mask_stride_u =
        has_bc ? bc->mask->component(0).stride() : 0;
    const int mask_stride_v =
        has_bc ? bc->mask->component(1).stride() : 0;
    mg_compute_diag_cuda(
        grid.local_mx(), grid.local_my(), diag.ghost_width(),
        diag.component(0).stride(), diag.component(1).stride(),
        nuH.component(0).device_data(), nuH.component(1).device_data(),
        beta.component(0).device_data(), beta.component(1).device_data(),
        diag.component(0).device_data(), diag.component(1).device_data(),
        inv_dx2, inv_dy2, mask_stride_u, mask_stride_v,
        has_bc ? bc->mask->component(0).device_data() : nullptr,
        has_bc ? bc->mask->component(1).device_data() : nullptr,
        has_bc ? 1 : 0);
    return;
  }
#endif
  compute_jacobi_diag(grid, nuH, beta, diag, bc);
}

double estimate_lambda_max(const Grid2D& grid, const FieldStag2D<double>& nuH,
                           const FieldStag2D<double>& beta,
                           FieldStag2D<double>& x, FieldStag2D<double>& y,
                           FieldStag2D<double>& Ax, FieldStag2D<double>& diag,
                           const SSABoundaryCondition* bc,
                           const Context* context, int iterations) {
  const int iters = std::max(1, iterations);
  compute_diag(grid, nuH, beta, diag, bc, context);
  set(1.0, x);

  double lambda = 0.0;
  for (int iter = 0; iter < iters; ++iter) {
    apply_operator(grid, nuH, beta, x, Ax, bc, context);

#if GPISM_HAVE_CUDA
    const bool has_bc = bc && bc->mask;
    const bool device_ready = Ax.component(0).has_device_data() &&
                              Ax.component(1).has_device_data() &&
                              diag.component(0).has_device_data() &&
                              diag.component(1).has_device_data() &&
                              y.component(0).has_device_data() &&
                              y.component(1).has_device_data() &&
                              (!has_bc ||
                               (bc->mask->component(0).has_device_data() &&
                                bc->mask->component(1).has_device_data()));
    if (device_ready) {
      const int mask_stride_u =
          has_bc ? bc->mask->component(0).stride() : 0;
      const int mask_stride_v =
          has_bc ? bc->mask->component(1).stride() : 0;
      mg_cheby_compute_z_cuda(
          grid.local_mx(), grid.local_my(), y.ghost_width(),
          y.component(0).stride(), y.component(1).stride(),
          Ax.component(0).device_data(), Ax.component(1).device_data(),
          diag.component(0).device_data(), diag.component(1).device_data(),
          y.component(0).device_data(), y.component(1).device_data(),
          mask_stride_u, mask_stride_v,
          has_bc ? bc->mask->component(0).device_data() : nullptr,
          has_bc ? bc->mask->component(1).device_data() : nullptr,
          has_bc ? 1 : 0);
    } else
#endif
    {
      for (int j = 0; j < grid.local_my(); ++j) {
        for (int i = 0; i < grid.local_mx(); ++i) {
          for (int comp = 0; comp < 2; ++comp) {
            if (is_dirichlet(bc, i, j, comp)) {
              y(i, j, comp) = 0.0;
            } else {
              const double dloc = diag(i, j, comp);
              y(i, j, comp) = (dloc != 0.0) ? (Ax(i, j, comp) / dloc) : 0.0;
            }
          }
        }
      }
    }

    const double num = std::sqrt(global_sum(context, dot(y, y)));
    const double den = std::sqrt(global_sum(context, dot(x, x)));
    lambda = (den > 0.0) ? (num / den) : 0.0;
    if (num > 0.0) {
      copy(y, x);
      scal(1.0 / num, x);
    }
  }

  return lambda;
}

ChebyBounds cheby_bounds_for_level(MGLevel& level, double lambda_min,
                                   double lambda_max, bool estimate,
                                   int estimate_iters, double min_factor,
                                   double max_factor,
                                   const SSABoundaryCondition* bc,
                                   const Context* context) {
  if (!estimate) {
    return {lambda_min, lambda_max};
  }
  const double lambda_est =
      estimate_lambda_max(level.grid, level.nuH, level.beta, level.r, level.z,
                          level.Ax, level.diag, bc, context, estimate_iters);
  if (!std::isfinite(lambda_est) || lambda_est <= 0.0) {
    return {lambda_min, lambda_max};
  }
  const double min_val = lambda_est * min_factor;
  const double max_val = lambda_est * max_factor;
  if (!std::isfinite(min_val) || !std::isfinite(max_val) || min_val <= 0.0 ||
      max_val <= min_val) {
    return {lambda_min, lambda_max};
  }
  return {min_val, max_val};
}

std::vector<ChebyBounds> estimate_cheby_bounds_impl(
    MultigridHierarchy& mg, double lambda_min, double lambda_max,
    bool estimate, int estimate_iters, double min_factor, double max_factor,
    const SSABoundaryCondition* bc, const Context* context,
    const std::vector<SSABoundaryCondition>* bc_levels) {
  const int levels = mg.num_levels();
  std::vector<ChebyBounds> bounds;
  bounds.reserve(levels);
  for (int level = 0; level < levels; ++level) {
    const SSABoundaryCondition* level_bc = bc;
    if (bc_levels && level < static_cast<int>(bc_levels->size())) {
      level_bc = &(*bc_levels)[level];
    }
    bounds.push_back(
        cheby_bounds_for_level(mg.level(level), lambda_min, lambda_max,
                               estimate, estimate_iters, min_factor, max_factor,
                               level_bc, context));
  }
  return bounds;
}
void compute_jacobi_diag(const Grid2D& grid, const FieldStag2D<double>& nuH,
                         const FieldStag2D<double>& beta,
                         FieldStag2D<double>& diag,
                         const SSABoundaryCondition* bc) {
  const double dx = grid.dx();
  const double dy = grid.dy();
  const double inv_dx2 = 1.0 / (dx * dx);
  const double inv_dy2 = 1.0 / (dy * dy);

  const Field2D<double>& nu_u = nuH.component(0);
  const Field2D<double>& nu_v = nuH.component(1);
  const Field2D<double>& beta_u = beta.component(0);
  const Field2D<double>& beta_v = beta.component(1);

  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      const int jm1 = (j == 0) ? j : j - 1;
      const int im1 = (i == 0) ? i : i - 1;
      if (is_dirichlet(bc, i, j, 0)) {
        diag(i, j, 0) = 1.0;
      } else {
        const double c_n = nu_v(i, j);
        const double c_s = nu_v(i, jm1);
        const double c_e = nu_u(i, j);
        const double c_w = nu_u(im1, j);
        diag(i, j, 0) =
            beta_u(i, j) + (c_n + c_s) * inv_dy2 +
            4.0 * (c_e + c_w) * inv_dx2;
      }

      if (is_dirichlet(bc, i, j, 1)) {
        diag(i, j, 1) = 1.0;
      } else {
        const double c_n = nu_v(i, j);
        const double c_s = nu_v(i, jm1);
        const double c_e = nu_u(i, j);
        const double c_w = nu_u(im1, j);
        diag(i, j, 1) =
            beta_v(i, j) + 4.0 * (c_n + c_s) * inv_dy2 +
            (c_e + c_w) * inv_dx2;
      }
    }
  }
}

}  // namespace

std::vector<ChebyBounds> estimate_cheby_bounds(
    MultigridHierarchy& mg, double lambda_min, double lambda_max,
    bool estimate, int estimate_iters, double min_factor, double max_factor,
    const SSABoundaryCondition* bc, const Context* context,
    const std::vector<SSABoundaryCondition>* bc_levels) {
  return estimate_cheby_bounds_impl(mg, lambda_min, lambda_max, estimate,
                                    estimate_iters, min_factor, max_factor, bc,
                                    context, bc_levels);
}

MultigridHierarchy::MultigridHierarchy(const Grid2D& fine_grid, int min_size) {
  const int gw = fine_grid.ghost_width();
  const int dims_x = fine_grid.dims_x();
  const int dims_y = fine_grid.dims_y();
  const int size = dims_x * dims_y;
  const int rank = fine_grid.coord_y() * dims_x + fine_grid.coord_x();

  const int min_dim = std::max(2, min_size);

  std::vector<std::pair<int, int>> dims;
  dims.emplace_back(fine_grid.global_mx(), fine_grid.global_my());
  while (dims.back().first > min_dim && dims.back().second > min_dim) {
    const int next_mx = coarsen_dim(dims.back().first);
    const int next_my = coarsen_dim(dims.back().second);
    if (next_mx == dims.back().first && next_my == dims.back().second) {
      break;
    }
    dims.emplace_back(next_mx, next_my);
  }

  levels_.reserve(dims.size());
  double dx = fine_grid.dx();
  double dy = fine_grid.dy();
  for (std::size_t i = 0; i < dims.size(); ++i) {
    Grid2D grid(dims[i].first, dims[i].second, dx, dy, gw, rank, size);
    levels_.emplace_back(grid);
    dx *= 2.0;
    dy *= 2.0;
  }
}

void restrict_stag(const FieldStag2D<double>& fine, FieldStag2D<double>& coarse) {
#if GPISM_HAVE_CUDA
  if (fine.component(0).has_device_data() &&
      fine.component(1).has_device_data() &&
      coarse.component(0).has_device_data() &&
      coarse.component(1).has_device_data()) {
    mg_restrict_stag_cuda(
        fine.local_mx(), fine.local_my(), fine.ghost_width(),
        fine.component(0).stride(), fine.component(1).stride(),
        coarse.local_mx(), coarse.local_my(), coarse.ghost_width(),
        coarse.component(0).stride(), coarse.component(1).stride(),
        fine.component(0).device_data(), fine.component(1).device_data(),
        coarse.component(0).device_data(), coarse.component(1).device_data());
    return;
  }
#endif
  const int fine_mx = fine.local_mx();
  const int fine_my = fine.local_my();
  const int coarse_mx = coarse.local_mx();
  const int coarse_my = coarse.local_my();
  for (int j = 0; j < coarse_my; ++j) {
    const int fj0 = std::min(2 * j, fine_my - 1);
    const int fj1 = std::min(fj0 + 1, fine_my - 1);
    for (int i = 0; i < coarse_mx; ++i) {
      const int fi0 = std::min(2 * i, fine_mx - 1);
      const int fi1 = std::min(fi0 + 1, fine_mx - 1);
      for (int comp = 0; comp < 2; ++comp) {
        // Clamp fine indices so odd-sized fine grids restrict correctly without
        // pulling in (potentially stale) ghost values.
        const double v00 = fine(fi0, fj0, comp);
        const double v10 = fine(fi1, fj0, comp);
        const double v01 = fine(fi0, fj1, comp);
        const double v11 = fine(fi1, fj1, comp);
        coarse(i, j, comp) = 0.25 * (v00 + v10 + v01 + v11);
      }
    }
  }
}

void prolong_stag(const FieldStag2D<double>& coarse, FieldStag2D<double>& fine) {
#if GPISM_HAVE_CUDA
  if (coarse.component(0).has_device_data() &&
      coarse.component(1).has_device_data() &&
      fine.component(0).has_device_data() &&
      fine.component(1).has_device_data()) {
    mg_prolong_stag_cuda(
        coarse.local_mx(), coarse.local_my(), coarse.ghost_width(),
        coarse.component(0).stride(), coarse.component(1).stride(),
        fine.local_mx(), fine.local_my(), fine.ghost_width(),
        fine.component(0).stride(), fine.component(1).stride(),
        coarse.component(0).device_data(), coarse.component(1).device_data(),
        fine.component(0).device_data(), fine.component(1).device_data());
    return;
  }
#endif
  const int coarse_mx = coarse.local_mx();
  const int coarse_my = coarse.local_my();
  const int fine_mx = fine.local_mx();
  const int fine_my = fine.local_my();

  auto sample = [&](int i, int j, int comp) {
    const int ii = std::min(std::max(i, 0), coarse_mx - 1);
    const int jj = std::min(std::max(j, 0), coarse_my - 1);
    return coarse(ii, jj, comp);
  };

  for (int j = 0; j < fine_my; ++j) {
    for (int i = 0; i < fine_mx; ++i) {
      const int ic = i / 2;
      const int jc = j / 2;
      const int di = i % 2;
      const int dj = j % 2;
      for (int comp = 0; comp < 2; ++comp) {
        if (di == 0 && dj == 0) {
          fine(i, j, comp) = sample(ic, jc, comp);
        } else if (di == 1 && dj == 0) {
          fine(i, j, comp) =
              0.5 * (sample(ic, jc, comp) + sample(ic + 1, jc, comp));
        } else if (di == 0 && dj == 1) {
          fine(i, j, comp) =
              0.5 * (sample(ic, jc, comp) + sample(ic, jc + 1, comp));
        } else {
          fine(i, j, comp) = 0.25 * (sample(ic, jc, comp) +
                                     sample(ic + 1, jc, comp) +
                                     sample(ic, jc + 1, comp) +
                                     sample(ic + 1, jc + 1, comp));
        }
      }
    }
  }
}

void compute_residual(const Grid2D& grid, const FieldStag2D<double>& nuH,
                      const FieldStag2D<double>& beta,
                      const FieldStag2D<double>& b,
                      const FieldStag2D<double>& x, FieldStag2D<double>& r,
                      FieldStag2D<double>& Ax,
                      const SSABoundaryCondition* bc,
                      const Context* context) {
  HaloExchange2D exchange;
  HaloExchange2D::StagExchangeHandle<double> handle{};
  bool exchange_active = false;
#if GPISM_HAVE_MPI
  if (context && context->mpi_enabled() && context->size() > 1) {
    exchange_for_device(const_cast<FieldStag2D<double>&>(nuH), grid, context);
    exchange_for_device(const_cast<FieldStag2D<double>&>(beta), grid, context);
    if (x.ghost_width() > 0) {
      auto& mutable_x = const_cast<FieldStag2D<double>&>(x);
      handle =
          exchange.start_exchange(mutable_x, grid, *context,
                                  HaloExchange2D::Mode::Auto);
      exchange_active = handle.u.active || handle.v.active;
    }
  }
#else
  (void)context;
#endif
#if GPISM_HAVE_CUDA
  const bool has_bc = bc && bc->mask;
  const bool device_ready =
      nuH.component(0).has_device_data() &&
      nuH.component(1).has_device_data() &&
      beta.component(0).has_device_data() &&
      beta.component(1).has_device_data() &&
      b.component(0).has_device_data() &&
      b.component(1).has_device_data() &&
      x.component(0).has_device_data() &&
      x.component(1).has_device_data() &&
      r.component(0).has_device_data() &&
      r.component(1).has_device_data() &&
      Ax.component(0).has_device_data() &&
      Ax.component(1).has_device_data() &&
      (!has_bc ||
       (bc->mask->component(0).has_device_data() &&
        bc->mask->component(1).has_device_data()));
  if (device_ready) {
    const int mx = grid.local_mx();
    const int my = grid.local_my();
    const int gw = x.ghost_width();
    const double dx = grid.dx();
    const double dy = grid.dy();
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);
    const int periodic = (grid.dims_x() == 1 && grid.dims_y() == 1) ? 1 : 0;
    const int mask_stride_u =
        has_bc ? bc->mask->component(0).stride() : 0;
    const int mask_stride_v =
        has_bc ? bc->mask->component(1).stride() : 0;
    if (exchange_active) {
      const int i0 = gw;
      const int i1 = mx - gw;
      const int j0 = gw;
      const int j1 = my - gw;
      if (i0 < i1 && j0 < j1) {
        ssa_apply_region_cuda(
            mx, my, gw, x.component(0).stride(), x.component(1).stride(),
            nuH.component(0).stride(), nuH.component(1).stride(),
            beta.component(0).stride(), beta.component(1).stride(),
            Ax.component(0).stride(), Ax.component(1).stride(),
            x.component(0).device_data(), x.component(1).device_data(),
            nuH.component(0).device_data(), nuH.component(1).device_data(),
            beta.component(0).device_data(), beta.component(1).device_data(),
            Ax.component(0).device_data(), Ax.component(1).device_data(),
            inv_dx2, inv_dy2, inv_2dx, inv_2dy, mask_stride_u, mask_stride_v,
            has_bc ? bc->mask->component(0).device_data() : nullptr,
            has_bc ? bc->mask->component(1).device_data() : nullptr,
            has_bc ? 1 : 0, i0, i1, j0, j1, periodic);
      }
#if GPISM_HAVE_MPI
      exchange.finish_exchange(handle);
#endif
      auto apply_band = [&](int is, int ie, int js, int je) {
        if (is < ie && js < je) {
          ssa_apply_region_cuda(
              mx, my, gw, x.component(0).stride(), x.component(1).stride(),
              nuH.component(0).stride(), nuH.component(1).stride(),
              beta.component(0).stride(), beta.component(1).stride(),
              Ax.component(0).stride(), Ax.component(1).stride(),
              x.component(0).device_data(), x.component(1).device_data(),
              nuH.component(0).device_data(), nuH.component(1).device_data(),
              beta.component(0).device_data(), beta.component(1).device_data(),
              Ax.component(0).device_data(), Ax.component(1).device_data(),
              inv_dx2, inv_dy2, inv_2dx, inv_2dy, mask_stride_u, mask_stride_v,
              has_bc ? bc->mask->component(0).device_data() : nullptr,
              has_bc ? bc->mask->component(1).device_data() : nullptr,
              has_bc ? 1 : 0, is, ie, js, je, periodic);
        }
      };
      apply_band(0, gw, 0, my);
      apply_band(mx - gw, mx, 0, my);
      apply_band(gw, mx - gw, 0, gw);
      apply_band(gw, mx - gw, my - gw, my);
    } else {
      ssa_apply_cuda(
          mx, my, gw, x.component(0).stride(), x.component(1).stride(),
          nuH.component(0).stride(), nuH.component(1).stride(),
          beta.component(0).stride(), beta.component(1).stride(),
          Ax.component(0).stride(), Ax.component(1).stride(),
          x.component(0).device_data(), x.component(1).device_data(),
          nuH.component(0).device_data(), nuH.component(1).device_data(),
          beta.component(0).device_data(), beta.component(1).device_data(),
          Ax.component(0).device_data(), Ax.component(1).device_data(),
          inv_dx2, inv_dy2, inv_2dx, inv_2dy, mask_stride_u, mask_stride_v,
          has_bc ? bc->mask->component(0).device_data() : nullptr,
          has_bc ? bc->mask->component(1).device_data() : nullptr,
          has_bc ? 1 : 0, periodic);
    }
    mg_residual_cuda(
        mx, my, r.ghost_width(),
        r.component(0).stride(), r.component(1).stride(),
        b.component(0).device_data(), b.component(1).device_data(),
        Ax.component(0).device_data(), Ax.component(1).device_data(),
        r.component(0).device_data(), r.component(1).device_data(),
        mask_stride_u, mask_stride_v,
        has_bc ? bc->mask->component(0).device_data() : nullptr,
        has_bc ? bc->mask->component(1).device_data() : nullptr,
        has_bc ? 1 : 0);
    return;
  }
#endif
#if GPISM_HAVE_MPI
  if (exchange_active) {
    exchange.finish_exchange(handle);
  }
#endif
  ssa_apply_host(grid, nuH, beta, x, Ax, bc);
  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      for (int comp = 0; comp < 2; ++comp) {
        if (is_dirichlet(bc, i, j, comp)) {
          r(i, j, comp) = 0.0;
        } else {
          r(i, j, comp) = b(i, j, comp) - Ax(i, j, comp);
        }
      }
    }
  }
}

void apply_operator(const Grid2D& grid, const FieldStag2D<double>& nuH,
                    const FieldStag2D<double>& beta,
                    const FieldStag2D<double>& x, FieldStag2D<double>& Ax,
                    const SSABoundaryCondition* bc,
                    const Context* context) {
  HaloExchange2D exchange;
  HaloExchange2D::StagExchangeHandle<double> handle{};
  bool exchange_active = false;
#if GPISM_HAVE_MPI
  if (context && context->mpi_enabled() && context->size() > 1) {
    exchange_for_device(const_cast<FieldStag2D<double>&>(nuH), grid, context);
    exchange_for_device(const_cast<FieldStag2D<double>&>(beta), grid, context);
    if (x.ghost_width() > 0) {
      auto& mutable_x = const_cast<FieldStag2D<double>&>(x);
      handle =
          exchange.start_exchange(mutable_x, grid, *context,
                                  HaloExchange2D::Mode::Auto);
      exchange_active = handle.u.active || handle.v.active;
    }
  }
#else
  (void)context;
#endif
#if GPISM_HAVE_CUDA
  const bool has_bc = bc && bc->mask;
  const bool device_ready =
      nuH.component(0).has_device_data() &&
      nuH.component(1).has_device_data() &&
      beta.component(0).has_device_data() &&
      beta.component(1).has_device_data() &&
      x.component(0).has_device_data() &&
      x.component(1).has_device_data() &&
      Ax.component(0).has_device_data() &&
      Ax.component(1).has_device_data() &&
      (!has_bc ||
       (bc->mask->component(0).has_device_data() &&
        bc->mask->component(1).has_device_data()));
  if (device_ready) {
    const int mx = grid.local_mx();
    const int my = grid.local_my();
    const int gw = x.ghost_width();
    const double dx = grid.dx();
    const double dy = grid.dy();
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);
    const int periodic = (grid.dims_x() == 1 && grid.dims_y() == 1) ? 1 : 0;
    const int mask_stride_u =
        has_bc ? bc->mask->component(0).stride() : 0;
    const int mask_stride_v =
        has_bc ? bc->mask->component(1).stride() : 0;
    if (exchange_active) {
      const int i0 = gw;
      const int i1 = mx - gw;
      const int j0 = gw;
      const int j1 = my - gw;
      if (i0 < i1 && j0 < j1) {
        ssa_apply_region_cuda(
            mx, my, gw, x.component(0).stride(), x.component(1).stride(),
            nuH.component(0).stride(), nuH.component(1).stride(),
            beta.component(0).stride(), beta.component(1).stride(),
            Ax.component(0).stride(), Ax.component(1).stride(),
            x.component(0).device_data(), x.component(1).device_data(),
            nuH.component(0).device_data(), nuH.component(1).device_data(),
            beta.component(0).device_data(), beta.component(1).device_data(),
            Ax.component(0).device_data(), Ax.component(1).device_data(),
            inv_dx2, inv_dy2, inv_2dx, inv_2dy, mask_stride_u, mask_stride_v,
            has_bc ? bc->mask->component(0).device_data() : nullptr,
            has_bc ? bc->mask->component(1).device_data() : nullptr,
            has_bc ? 1 : 0, i0, i1, j0, j1, periodic);
      }
#if GPISM_HAVE_MPI
      exchange.finish_exchange(handle);
#endif
      auto apply_band = [&](int is, int ie, int js, int je) {
        if (is < ie && js < je) {
          ssa_apply_region_cuda(
              mx, my, gw, x.component(0).stride(), x.component(1).stride(),
              nuH.component(0).stride(), nuH.component(1).stride(),
              beta.component(0).stride(), beta.component(1).stride(),
              Ax.component(0).stride(), Ax.component(1).stride(),
              x.component(0).device_data(), x.component(1).device_data(),
              nuH.component(0).device_data(), nuH.component(1).device_data(),
              beta.component(0).device_data(), beta.component(1).device_data(),
              Ax.component(0).device_data(), Ax.component(1).device_data(),
              inv_dx2, inv_dy2, inv_2dx, inv_2dy, mask_stride_u, mask_stride_v,
              has_bc ? bc->mask->component(0).device_data() : nullptr,
              has_bc ? bc->mask->component(1).device_data() : nullptr,
              has_bc ? 1 : 0, is, ie, js, je, periodic);
        }
      };
      apply_band(0, gw, 0, my);
      apply_band(mx - gw, mx, 0, my);
      apply_band(gw, mx - gw, 0, gw);
      apply_band(gw, mx - gw, my - gw, my);
    } else {
      ssa_apply_cuda(
          mx, my, gw, x.component(0).stride(), x.component(1).stride(),
          nuH.component(0).stride(), nuH.component(1).stride(),
          beta.component(0).stride(), beta.component(1).stride(),
          Ax.component(0).stride(), Ax.component(1).stride(),
          x.component(0).device_data(), x.component(1).device_data(),
          nuH.component(0).device_data(), nuH.component(1).device_data(),
          beta.component(0).device_data(), beta.component(1).device_data(),
          Ax.component(0).device_data(), Ax.component(1).device_data(),
          inv_dx2, inv_dy2, inv_2dx, inv_2dy, mask_stride_u, mask_stride_v,
          has_bc ? bc->mask->component(0).device_data() : nullptr,
          has_bc ? bc->mask->component(1).device_data() : nullptr,
          has_bc ? 1 : 0, periodic);
    }
    return;
  }
#endif
#if GPISM_HAVE_MPI
  if (exchange_active) {
    exchange.finish_exchange(handle);
  }
#endif
  ssa_apply_host(grid, nuH, beta, x, Ax, bc);
}

void jacobi_smooth(const Grid2D& grid, const FieldStag2D<double>& nuH,
                   const FieldStag2D<double>& beta, const FieldStag2D<double>& b,
                   FieldStag2D<double>& x, int iterations, double omega,
                   FieldStag2D<double>& diag, FieldStag2D<double>& Ax,
                   const SSABoundaryCondition* bc,
                   const Context* context) {
  if (iterations <= 0) {
    return;
  }

#if GPISM_HAVE_MPI
  if (context && context->mpi_enabled() && context->size() > 1) {
    exchange_for_device(const_cast<FieldStag2D<double>&>(nuH), grid, context);
    exchange_for_device(const_cast<FieldStag2D<double>&>(beta), grid, context);
  }
#else
  (void)context;
#endif

#if GPISM_HAVE_CUDA
  const bool has_bc = bc && bc->mask;
  const bool has_values =
      has_bc && bc->values &&
      bc->values->component(0).has_device_data() &&
      bc->values->component(1).has_device_data();
  const bool device_ready =
      nuH.component(0).has_device_data() &&
      nuH.component(1).has_device_data() &&
      beta.component(0).has_device_data() &&
      beta.component(1).has_device_data() &&
      b.component(0).has_device_data() &&
      b.component(1).has_device_data() &&
      x.component(0).has_device_data() &&
      x.component(1).has_device_data() &&
      diag.component(0).has_device_data() &&
      diag.component(1).has_device_data() &&
      Ax.component(0).has_device_data() &&
      Ax.component(1).has_device_data() &&
      (!has_bc ||
       (bc->mask->component(0).has_device_data() &&
        bc->mask->component(1).has_device_data()));
  if (device_ready) {
    const double dx = grid.dx();
    const double dy = grid.dy();
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);
    const int periodic = (grid.dims_x() == 1 && grid.dims_y() == 1) ? 1 : 0;
    const int mask_stride_u =
        has_bc ? bc->mask->component(0).stride() : 0;
    const int mask_stride_v =
        has_bc ? bc->mask->component(1).stride() : 0;
    const int bc_stride_u =
        has_values ? bc->values->component(0).stride() : 0;
    const int bc_stride_v =
        has_values ? bc->values->component(1).stride() : 0;
    const bool single_rank =
        !(context && context->mpi_enabled() && context->size() > 1);
    if (single_rank) {
      // Ping-pong between x and Ax to avoid a full-field copy every Jacobi
      // iteration. This relies on mg_jacobi_fused_cuda writing x_new, not doing
      // an in-place += update.
      const bool odd_iters = (iterations & 1) != 0;
      if (odd_iters) {
        // Seed Ax with the initial x so we can start from Ax and end in x.
        copy(x, Ax);
      }
      bool input_is_x = !odd_iters;
      const double* x0_u = x.component(0).device_data();
      const double* x0_v = x.component(1).device_data();
      const double* x1_u = Ax.component(0).device_data();
      const double* x1_v = Ax.component(1).device_data();
      for (int iter = 0; iter < iterations; ++iter) {
        const double* in_u = input_is_x ? x0_u : x1_u;
        const double* in_v = input_is_x ? x0_v : x1_v;
        double* out_u = input_is_x ? Ax.component(0).device_data()
                                   : x.component(0).device_data();
        double* out_v = input_is_x ? Ax.component(1).device_data()
                                   : x.component(1).device_data();
        mg_jacobi_fused_cuda(
            grid.local_mx(), grid.local_my(), x.ghost_width(),
            x.component(0).stride(), x.component(1).stride(),
            nuH.component(0).stride(), nuH.component(1).stride(),
            beta.component(0).stride(), beta.component(1).stride(),
            b.component(0).stride(), b.component(1).stride(),
            in_u, in_v, out_u, out_v,
            nuH.component(0).device_data(), nuH.component(1).device_data(),
            beta.component(0).device_data(), beta.component(1).device_data(),
            b.component(0).device_data(), b.component(1).device_data(), omega,
            inv_dx2, inv_dy2, inv_2dx, inv_2dy, mask_stride_u, mask_stride_v,
            has_bc ? bc->mask->component(0).device_data() : nullptr,
            has_bc ? bc->mask->component(1).device_data() : nullptr,
            bc_stride_u, bc_stride_v,
            has_values ? bc->values->component(0).device_data() : nullptr,
            has_values ? bc->values->component(1).device_data() : nullptr,
            has_bc ? 1 : 0, has_values ? 1 : 0, periodic);
        input_is_x = !input_is_x;
      }
      return;
    }
    mg_compute_diag_cuda(
        grid.local_mx(), grid.local_my(), diag.ghost_width(),
        diag.component(0).stride(), diag.component(1).stride(),
        nuH.component(0).device_data(), nuH.component(1).device_data(),
        beta.component(0).device_data(), beta.component(1).device_data(),
        diag.component(0).device_data(), diag.component(1).device_data(),
        inv_dx2, inv_dy2, mask_stride_u, mask_stride_v,
        has_bc ? bc->mask->component(0).device_data() : nullptr,
        has_bc ? bc->mask->component(1).device_data() : nullptr,
        has_bc ? 1 : 0);
    for (int iter = 0; iter < iterations; ++iter) {
      HaloExchange2D exchange;
      HaloExchange2D::StagExchangeHandle<double> handle{};
      bool exchange_active = false;
#if GPISM_HAVE_MPI
      if (context && context->mpi_enabled() && context->size() > 1) {
        if (x.ghost_width() > 0) {
          auto& mutable_x = const_cast<FieldStag2D<double>&>(x);
          handle =
              exchange.start_exchange(mutable_x, grid, *context,
                                      HaloExchange2D::Mode::Auto);
          exchange_active = handle.u.active || handle.v.active;
        }
      }
#endif
      const int mx = grid.local_mx();
      const int my = grid.local_my();
      const int gw = x.ghost_width();
      if (exchange_active) {
        const int i0 = gw;
        const int i1 = mx - gw;
        const int j0 = gw;
        const int j1 = my - gw;
        if (i0 < i1 && j0 < j1) {
          ssa_apply_region_cuda(
              mx, my, gw, x.component(0).stride(), x.component(1).stride(),
              nuH.component(0).stride(), nuH.component(1).stride(),
              beta.component(0).stride(), beta.component(1).stride(),
              Ax.component(0).stride(), Ax.component(1).stride(),
              x.component(0).device_data(), x.component(1).device_data(),
              nuH.component(0).device_data(), nuH.component(1).device_data(),
              beta.component(0).device_data(), beta.component(1).device_data(),
              Ax.component(0).device_data(), Ax.component(1).device_data(),
              inv_dx2, inv_dy2, inv_2dx, inv_2dy, mask_stride_u, mask_stride_v,
              has_bc ? bc->mask->component(0).device_data() : nullptr,
              has_bc ? bc->mask->component(1).device_data() : nullptr,
              has_bc ? 1 : 0, i0, i1, j0, j1, periodic);
        }
#if GPISM_HAVE_MPI
        exchange.finish_exchange(handle);
#endif
        auto apply_band = [&](int is, int ie, int js, int je) {
          if (is < ie && js < je) {
            ssa_apply_region_cuda(
                mx, my, gw, x.component(0).stride(), x.component(1).stride(),
                nuH.component(0).stride(), nuH.component(1).stride(),
                beta.component(0).stride(), beta.component(1).stride(),
                Ax.component(0).stride(), Ax.component(1).stride(),
                x.component(0).device_data(), x.component(1).device_data(),
                nuH.component(0).device_data(), nuH.component(1).device_data(),
                beta.component(0).device_data(),
                beta.component(1).device_data(),
                Ax.component(0).device_data(), Ax.component(1).device_data(),
                inv_dx2, inv_dy2, inv_2dx, inv_2dy, mask_stride_u, mask_stride_v,
                has_bc ? bc->mask->component(0).device_data() : nullptr,
                has_bc ? bc->mask->component(1).device_data() : nullptr,
                has_bc ? 1 : 0, is, ie, js, je, periodic);
          }
        };
        apply_band(0, gw, 0, my);
        apply_band(mx - gw, mx, 0, my);
        apply_band(gw, mx - gw, 0, gw);
        apply_band(gw, mx - gw, my - gw, my);
      } else {
        ssa_apply_cuda(
            mx, my, gw, x.component(0).stride(), x.component(1).stride(),
            nuH.component(0).stride(), nuH.component(1).stride(),
            beta.component(0).stride(), beta.component(1).stride(),
            Ax.component(0).stride(), Ax.component(1).stride(),
            x.component(0).device_data(), x.component(1).device_data(),
            nuH.component(0).device_data(), nuH.component(1).device_data(),
            beta.component(0).device_data(), beta.component(1).device_data(),
            Ax.component(0).device_data(), Ax.component(1).device_data(),
            inv_dx2, inv_dy2, inv_2dx, inv_2dy, mask_stride_u, mask_stride_v,
            has_bc ? bc->mask->component(0).device_data() : nullptr,
            has_bc ? bc->mask->component(1).device_data() : nullptr,
            has_bc ? 1 : 0, periodic);
      }
      mg_jacobi_update_cuda(
          grid.local_mx(), grid.local_my(), x.ghost_width(),
          x.component(0).stride(), x.component(1).stride(),
          b.component(0).device_data(), b.component(1).device_data(),
          Ax.component(0).device_data(), Ax.component(1).device_data(),
          diag.component(0).device_data(), diag.component(1).device_data(),
          x.component(0).device_data(), x.component(1).device_data(), omega,
          mask_stride_u, mask_stride_v,
          has_bc ? bc->mask->component(0).device_data() : nullptr,
          has_bc ? bc->mask->component(1).device_data() : nullptr,
          bc_stride_u, bc_stride_v,
          has_values ? bc->values->component(0).device_data() : nullptr,
          has_values ? bc->values->component(1).device_data() : nullptr,
          has_bc ? 1 : 0, has_values ? 1 : 0);
    }
    return;
  }
#endif
  compute_jacobi_diag(grid, nuH, beta, diag, bc);

  for (int iter = 0; iter < iterations; ++iter) {
#if GPISM_HAVE_MPI
    if (context && context->mpi_enabled() && context->size() > 1) {
      exchange_for_device(x, grid, context);
    }
#endif
    ssa_apply_host(grid, nuH, beta, x, Ax, bc);
    for (int j = 0; j < grid.local_my(); ++j) {
      for (int i = 0; i < grid.local_mx(); ++i) {
        for (int comp = 0; comp < 2; ++comp) {
          if (is_dirichlet(bc, i, j, comp)) {
            if (bc && bc->values) {
              x(i, j, comp) = (*bc->values)(i, j, comp);
            }
            continue;
          }
          const double r = b(i, j, comp) - Ax(i, j, comp);
          const double d = diag(i, j, comp);
          if (d != 0.0) {
            x(i, j, comp) += omega * r / d;
          }
        }
      }
    }
  }
}

void chebyshev_smooth(const Grid2D& grid, const FieldStag2D<double>& nuH,
                      const FieldStag2D<double>& beta,
                      const FieldStag2D<double>& b, FieldStag2D<double>& x,
                      int iterations, double lambda_min, double lambda_max,
                      FieldStag2D<double>& diag, FieldStag2D<double>& Ax,
                      FieldStag2D<double>& r, FieldStag2D<double>& z,
                      FieldStag2D<double>& p,
                      const SSABoundaryCondition* bc,
                      const Context* context) {
  if (iterations <= 0) {
    return;
  }

#if GPISM_HAVE_CUDA
  const bool has_bc = bc && bc->mask;
  const bool has_values =
      has_bc && bc->values &&
      bc->values->component(0).has_device_data() &&
      bc->values->component(1).has_device_data();
  const bool device_ready =
      nuH.component(0).has_device_data() &&
      nuH.component(1).has_device_data() &&
      beta.component(0).has_device_data() &&
      beta.component(1).has_device_data() &&
      b.component(0).has_device_data() &&
      b.component(1).has_device_data() &&
      x.component(0).has_device_data() &&
      x.component(1).has_device_data() &&
      diag.component(0).has_device_data() &&
      diag.component(1).has_device_data() &&
      Ax.component(0).has_device_data() &&
      Ax.component(1).has_device_data() &&
      r.component(0).has_device_data() &&
      r.component(1).has_device_data() &&
      z.component(0).has_device_data() &&
      z.component(1).has_device_data() &&
      p.component(0).has_device_data() &&
      p.component(1).has_device_data() &&
      (!has_bc ||
       (bc->mask->component(0).has_device_data() &&
        bc->mask->component(1).has_device_data()));
  if (device_ready) {
    const double dx = grid.dx();
    const double dy = grid.dy();
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const int mask_stride_u =
        has_bc ? bc->mask->component(0).stride() : 0;
    const int mask_stride_v =
        has_bc ? bc->mask->component(1).stride() : 0;
    const int bc_stride_u =
        has_values ? bc->values->component(0).stride() : 0;
    const int bc_stride_v =
        has_values ? bc->values->component(1).stride() : 0;
    mg_compute_diag_cuda(
        grid.local_mx(), grid.local_my(), diag.ghost_width(),
        diag.component(0).stride(), diag.component(1).stride(),
        nuH.component(0).device_data(), nuH.component(1).device_data(),
        beta.component(0).device_data(), beta.component(1).device_data(),
        diag.component(0).device_data(), diag.component(1).device_data(),
        inv_dx2, inv_dy2, mask_stride_u, mask_stride_v,
        has_bc ? bc->mask->component(0).device_data() : nullptr,
        has_bc ? bc->mask->component(1).device_data() : nullptr,
        has_bc ? 1 : 0);

    compute_residual(grid, nuH, beta, b, x, r, Ax, bc, context);
    mg_cheby_compute_z_cuda(
        grid.local_mx(), grid.local_my(), r.ghost_width(),
        r.component(0).stride(), r.component(1).stride(),
        r.component(0).device_data(), r.component(1).device_data(),
        diag.component(0).device_data(), diag.component(1).device_data(),
        z.component(0).device_data(), z.component(1).device_data(),
        mask_stride_u, mask_stride_v,
        has_bc ? bc->mask->component(0).device_data() : nullptr,
        has_bc ? bc->mask->component(1).device_data() : nullptr,
        has_bc ? 1 : 0);

    const double d = 0.5 * (lambda_max + lambda_min);
    const double c = 0.5 * (lambda_max - lambda_min);
    double alpha = (d != 0.0) ? (1.0 / d) : 0.0;
    double beta_coeff = 0.0;

    for (int iter = 0; iter < iterations; ++iter) {
      mg_cheby_update_p_cuda(
          grid.local_mx(), grid.local_my(), p.ghost_width(),
          p.component(0).stride(), p.component(1).stride(),
          z.component(0).device_data(), z.component(1).device_data(),
          p.component(0).device_data(), p.component(1).device_data(),
          beta_coeff, mask_stride_u, mask_stride_v,
          has_bc ? bc->mask->component(0).device_data() : nullptr,
          has_bc ? bc->mask->component(1).device_data() : nullptr,
          has_bc ? 1 : 0, iter == 0 ? 1 : 0);

      mg_cheby_update_x_cuda(
          grid.local_mx(), grid.local_my(), x.ghost_width(),
          x.component(0).stride(), x.component(1).stride(),
          p.component(0).device_data(), p.component(1).device_data(),
          x.component(0).device_data(), x.component(1).device_data(), alpha,
          mask_stride_u, mask_stride_v,
          has_bc ? bc->mask->component(0).device_data() : nullptr,
          has_bc ? bc->mask->component(1).device_data() : nullptr,
          bc_stride_u, bc_stride_v,
          has_values ? bc->values->component(0).device_data() : nullptr,
          has_values ? bc->values->component(1).device_data() : nullptr,
          has_bc ? 1 : 0, has_values ? 1 : 0);

      apply_operator(grid, nuH, beta, p, Ax, bc, context);
      mg_cheby_update_r_cuda(
          grid.local_mx(), grid.local_my(), r.ghost_width(),
          r.component(0).stride(), r.component(1).stride(),
          r.component(0).device_data(), r.component(1).device_data(),
          Ax.component(0).device_data(), Ax.component(1).device_data(), alpha,
          mask_stride_u, mask_stride_v,
          has_bc ? bc->mask->component(0).device_data() : nullptr,
          has_bc ? bc->mask->component(1).device_data() : nullptr,
          has_bc ? 1 : 0);
      mg_cheby_compute_z_cuda(
          grid.local_mx(), grid.local_my(), r.ghost_width(),
          r.component(0).stride(), r.component(1).stride(),
          r.component(0).device_data(), r.component(1).device_data(),
          diag.component(0).device_data(), diag.component(1).device_data(),
          z.component(0).device_data(), z.component(1).device_data(),
          mask_stride_u, mask_stride_v,
          has_bc ? bc->mask->component(0).device_data() : nullptr,
          has_bc ? bc->mask->component(1).device_data() : nullptr,
          has_bc ? 1 : 0);

      const double coeff = (c * alpha * 0.5);
      beta_coeff = coeff * coeff;
      const double denom = d - (c * c * 0.25) * alpha;
      if (denom != 0.0) {
        alpha = 1.0 / denom;
      }
    }
    return;
  }
#endif

  compute_jacobi_diag(grid, nuH, beta, diag, bc);
  compute_residual(grid, nuH, beta, b, x, r, Ax, bc, context);

  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      for (int comp = 0; comp < 2; ++comp) {
        if (is_dirichlet(bc, i, j, comp)) {
          z(i, j, comp) = 0.0;
          r(i, j, comp) = 0.0;
        } else {
          const double dloc = diag(i, j, comp);
          z(i, j, comp) = (dloc != 0.0) ? (r(i, j, comp) / dloc) : 0.0;
        }
      }
    }
  }

  const double d = 0.5 * (lambda_max + lambda_min);
  const double c = 0.5 * (lambda_max - lambda_min);
  double alpha = (d != 0.0) ? (1.0 / d) : 0.0;
  double beta_coeff = 0.0;

  for (int iter = 0; iter < iterations; ++iter) {
    for (int j = 0; j < grid.local_my(); ++j) {
      for (int i = 0; i < grid.local_mx(); ++i) {
        for (int comp = 0; comp < 2; ++comp) {
          if (is_dirichlet(bc, i, j, comp)) {
            p(i, j, comp) = 0.0;
          } else if (iter == 0) {
            p(i, j, comp) = z(i, j, comp);
          } else {
            p(i, j, comp) = z(i, j, comp) + beta_coeff * p(i, j, comp);
          }
        }
      }
    }

    for (int j = 0; j < grid.local_my(); ++j) {
      for (int i = 0; i < grid.local_mx(); ++i) {
        for (int comp = 0; comp < 2; ++comp) {
          if (is_dirichlet(bc, i, j, comp)) {
            if (bc && bc->values) {
              x(i, j, comp) = (*bc->values)(i, j, comp);
            }
          } else {
            x(i, j, comp) += alpha * p(i, j, comp);
          }
        }
      }
    }

    ssa_apply_host(grid, nuH, beta, p, Ax, bc);
    for (int j = 0; j < grid.local_my(); ++j) {
      for (int i = 0; i < grid.local_mx(); ++i) {
        for (int comp = 0; comp < 2; ++comp) {
          if (is_dirichlet(bc, i, j, comp)) {
            r(i, j, comp) = 0.0;
            z(i, j, comp) = 0.0;
            continue;
          }
          r(i, j, comp) -= alpha * Ax(i, j, comp);
          const double dloc = diag(i, j, comp);
          z(i, j, comp) = (dloc != 0.0) ? (r(i, j, comp) / dloc) : 0.0;
        }
      }
    }

    const double coeff = (c * alpha * 0.5);
    beta_coeff = coeff * coeff;
    const double denom = d - (c * c * 0.25) * alpha;
    if (denom != 0.0) {
      alpha = 1.0 / denom;
    }
  }
}

void chebyshev_jacobi_smooth(const Grid2D& grid, const FieldStag2D<double>& nuH,
                             const FieldStag2D<double>& beta,
                             const FieldStag2D<double>& b,
                             FieldStag2D<double>& x, int iterations,
                             double lambda_min, double lambda_max,
                             const SSABoundaryCondition* bc) {
  FieldStag2D<double> diag(grid.local_mx(), grid.local_my(), grid.ghost_width());
  FieldStag2D<double> Ax(grid.local_mx(), grid.local_my(), grid.ghost_width());
  FieldStag2D<double> r(grid.local_mx(), grid.local_my(), grid.ghost_width());
  FieldStag2D<double> z(grid.local_mx(), grid.local_my(), grid.ghost_width());
  FieldStag2D<double> p(grid.local_mx(), grid.local_my(), grid.ghost_width());

  const bool use_device = device_enabled();
  if (use_device) {
    sync_host_to_device(const_cast<FieldStag2D<double>&>(nuH));
    sync_host_to_device(const_cast<FieldStag2D<double>&>(beta));
    sync_host_to_device(const_cast<FieldStag2D<double>&>(b));
    sync_host_to_device(x);
  }

  chebyshev_smooth(grid, nuH, beta, b, x, iterations, lambda_min, lambda_max,
                   diag, Ax, r, z, p, bc, nullptr);

  if (use_device) {
    sync_device_to_host(x);
  }
}

void v_cycle(MultigridHierarchy& mg, int pre_iters, int post_iters,
             int coarse_iters, double omega, MGSmoother smoother,
             double cheby_lambda_min, double cheby_lambda_max,
             bool cheby_estimate, int cheby_estimate_iters,
             double cheby_estimate_min_factor,
             double cheby_estimate_max_factor,
             const SSABoundaryCondition* bc, const Context* context,
             const std::vector<ChebyBounds>* cheby_bounds_in,
             const std::vector<SSABoundaryCondition>* bc_levels,
             bool diagnostic) {
  const int levels = mg.num_levels();
  if (levels == 0) {
    return;
  }

  double base_norm = 0.0;
  if (diagnostic) {
    base_norm =
        std::sqrt(global_sum(context, dot(mg.level(0).rhs, mg.level(0).rhs)));
    if (is_rank0(context)) {
      std::cout << "MG level 0 rhs ||r||=" << base_norm << '\n';
    }
  }

  std::vector<ChebyBounds> cheby_bounds;
  const std::vector<ChebyBounds>* bounds_ptr = cheby_bounds_in;
  if (smoother == MGSmoother::Chebyshev) {
    if (bounds_ptr && static_cast<int>(bounds_ptr->size()) == levels) {
      // Use cached bounds.
    } else {
      cheby_bounds =
          estimate_cheby_bounds(mg, cheby_lambda_min, cheby_lambda_max,
                                cheby_estimate, cheby_estimate_iters,
                                cheby_estimate_min_factor,
                                cheby_estimate_max_factor, bc, context,
                                bc_levels);
      bounds_ptr = &cheby_bounds;
    }
  }

  for (int level = 0; level < levels - 1; ++level) {
    MGLevel& fine = mg.level(level);
    MGLevel& coarse = mg.level(level + 1);
    const SSABoundaryCondition* level_bc = bc;
    if (bc_levels && level < static_cast<int>(bc_levels->size())) {
      level_bc = &(*bc_levels)[level];
    }

    if (smoother == MGSmoother::Chebyshev) {
      const ChebyBounds bounds = bounds_ptr->at(level);
      chebyshev_smooth(fine.grid, fine.nuH, fine.beta, fine.rhs, fine.u,
                       pre_iters, bounds.min, bounds.max,
                       fine.diag, fine.Ax, fine.r, fine.z, fine.corr, level_bc,
                       context);
    } else {
      jacobi_smooth(fine.grid, fine.nuH, fine.beta, fine.rhs, fine.u, pre_iters,
                    omega, fine.diag, fine.Ax, level_bc, context);
    }
    compute_residual(fine.grid, fine.nuH, fine.beta, fine.rhs, fine.u, fine.r,
                     fine.Ax, level_bc, context);
    if (diagnostic) {
      log_mg_residual(level, "down", fine.r, base_norm, context);
    }
    restrict_stag(fine.r, coarse.rhs);
    set(0.0, coarse.u);
  }

  MGLevel& coarsest = mg.level(levels - 1);
  const SSABoundaryCondition* coarsest_bc = bc;
  if (bc_levels && (levels - 1) < static_cast<int>(bc_levels->size())) {
    coarsest_bc = &(*bc_levels)[levels - 1];
  }
  if (smoother == MGSmoother::Chebyshev) {
    const ChebyBounds bounds = bounds_ptr->at(levels - 1);
    chebyshev_smooth(coarsest.grid, coarsest.nuH, coarsest.beta, coarsest.rhs,
                     coarsest.u, coarse_iters, bounds.min, bounds.max,
                     coarsest.diag, coarsest.Ax, coarsest.r,
                     coarsest.z, coarsest.corr, coarsest_bc, context);
  } else {
    jacobi_smooth(coarsest.grid, coarsest.nuH, coarsest.beta, coarsest.rhs,
                  coarsest.u, coarse_iters, omega, coarsest.diag, coarsest.Ax,
                  coarsest_bc, context);
  }
  if (diagnostic) {
    compute_residual(coarsest.grid, coarsest.nuH, coarsest.beta, coarsest.rhs,
                     coarsest.u, coarsest.r, coarsest.Ax, coarsest_bc, context);
    log_mg_residual(levels - 1, "coarse", coarsest.r, base_norm, context);
  }

  for (int level = levels - 2; level >= 0; --level) {
    MGLevel& fine = mg.level(level);
    MGLevel& coarse = mg.level(level + 1);
    const SSABoundaryCondition* level_bc = bc;
    if (bc_levels && level < static_cast<int>(bc_levels->size())) {
      level_bc = &(*bc_levels)[level];
    }
    prolong_stag(coarse.u, fine.corr);
    axpy(1.0, fine.corr, fine.u);
    if (smoother == MGSmoother::Chebyshev) {
      const ChebyBounds bounds = bounds_ptr->at(level);
      chebyshev_smooth(fine.grid, fine.nuH, fine.beta, fine.rhs, fine.u,
                       post_iters, bounds.min, bounds.max,
                       fine.diag, fine.Ax, fine.r, fine.z, fine.corr, level_bc,
                       context);
    } else {
      jacobi_smooth(fine.grid, fine.nuH, fine.beta, fine.rhs, fine.u, post_iters,
                    omega, fine.diag, fine.Ax, level_bc, context);
    }
    if (diagnostic) {
      compute_residual(fine.grid, fine.nuH, fine.beta, fine.rhs, fine.u, fine.r,
                       fine.Ax, level_bc, context);
      log_mg_residual(level, "up", fine.r, base_norm, context);
    }
  }
}

}  // namespace gpism
