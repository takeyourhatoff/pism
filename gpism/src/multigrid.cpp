#include "gpism/multigrid.h"

#include <algorithm>
#include <vector>

#include "gpism/field_sync.h"
#include "gpism/halo_exchange.h"
#include "gpism/linear_algebra.h"
#include "gpism/ssa_operator.h"

namespace gpism {

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

void ssa_apply_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                    int stride_nu_u, int stride_nu_v, int stride_beta_u,
                    int stride_beta_v, int stride_out_u, int stride_out_v,
                    const double* u, const double* v, const double* nu_u,
                    const double* nu_v, const double* beta_u,
                    const double* beta_v, double* out_u, double* out_v,
                    double inv_dx2, double inv_dy2, double inv_2dx,
                    double inv_2dy, int stride_mask_u, int stride_mask_v,
                    const int* mask_u, const int* mask_v, int has_bc);
#endif

namespace {

int coarsen_dim(int n) { return (n + 1) / 2; }

bool is_dirichlet(const SSABoundaryCondition* bc, int i, int j, int comp) {
  if (!bc || !bc->mask) {
    return false;
  }
  return (*bc->mask)(i, j, comp) != 0;
}

template <typename T>
void exchange_for_device(FieldStag2D<T>& field, const Grid2D& grid,
                         const Context* context) {
  if (!context || !context->mpi_enabled()) {
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
  const double inv_2dx = 1.0 / (2.0 * dx);
  const double inv_2dy = 1.0 / (2.0 * dy);

  const Field2D<double>& u = vel.component(0);
  const Field2D<double>& v = vel.component(1);
  const Field2D<double>& nu_u = nuH.component(0);
  const Field2D<double>& nu_v = nuH.component(1);
  const Field2D<double>& beta_u = beta.component(0);
  const Field2D<double>& beta_v = beta.component(1);

  auto shear = [&](int i, int j) {
    const double du_dy = (u(i, j + 1) - u(i, j - 1)) * inv_2dy;
    const double dv_dx = (v(i + 1, j) - v(i - 1, j)) * inv_2dx;
    return du_dy + dv_dx;
  };

  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      if (is_dirichlet(bc, i, j, 0)) {
        out(i, j, 0) = u(i, j);
      } else {
        const double u_c = u(i, j);
        const double flux_x =
            nu_u(i + 1, j) * (u(i + 1, j) - u_c) -
            nu_u(i - 1, j) * (u_c - u(i - 1, j));
        const double flux_y =
            nu_u(i, j + 1) * (u(i, j + 1) - u_c) -
            nu_u(i, j - 1) * (u_c - u(i, j - 1));
        double coupling = 0.0;
        if (j >= 1 && j <= grid.local_my() - 2) {
          const double shear_p = shear(i, j + 1);
          const double shear_m = shear(i, j - 1);
          coupling = nu_u(i, j) * (shear_p - shear_m) * inv_2dy;
        }
        out(i, j, 0) = flux_x * inv_dx2 + flux_y * inv_dy2 + coupling +
                       beta_u(i, j) * u_c;
      }

      if (is_dirichlet(bc, i, j, 1)) {
        out(i, j, 1) = v(i, j);
      } else {
        const double v_c = v(i, j);
        const double flux_x =
            nu_v(i + 1, j) * (v(i + 1, j) - v_c) -
            nu_v(i - 1, j) * (v_c - v(i - 1, j));
        const double flux_y =
            nu_v(i, j + 1) * (v(i, j + 1) - v_c) -
            nu_v(i, j - 1) * (v_c - v(i, j - 1));
        double coupling = 0.0;
        if (i >= 1 && i <= grid.local_mx() - 2) {
          const double shear_p = shear(i + 1, j);
          const double shear_m = shear(i - 1, j);
          coupling = nu_v(i, j) * (shear_p - shear_m) * inv_2dx;
        }
        out(i, j, 1) = flux_x * inv_dx2 + flux_y * inv_dy2 + coupling +
                       beta_v(i, j) * v_c;
      }
    }
  }
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
      if (is_dirichlet(bc, i, j, 0)) {
        diag(i, j, 0) = 1.0;
      } else {
        const double dxx = (nu_u(i + 1, j) + nu_u(i - 1, j)) * inv_dx2;
        const double dyy = (nu_u(i, j + 1) + nu_u(i, j - 1)) * inv_dy2;
        diag(i, j, 0) = beta_u(i, j) + dxx + dyy;
      }

      if (is_dirichlet(bc, i, j, 1)) {
        diag(i, j, 1) = 1.0;
      } else {
        const double dxx = (nu_v(i + 1, j) + nu_v(i - 1, j)) * inv_dx2;
        const double dyy = (nu_v(i, j + 1) + nu_v(i, j - 1)) * inv_dy2;
        diag(i, j, 1) = beta_v(i, j) + dxx + dyy;
      }
    }
  }
}

}  // namespace

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
  const int coarse_mx = coarse.local_mx();
  const int coarse_my = coarse.local_my();
  for (int j = 0; j < coarse_my; ++j) {
    for (int i = 0; i < coarse_mx; ++i) {
      const int fi = 2 * i;
      const int fj = 2 * j;
      for (int comp = 0; comp < 2; ++comp) {
        const double v00 = fine(fi, fj, comp);
        const double v10 = fine(fi + 1, fj, comp);
        const double v01 = fine(fi, fj + 1, comp);
        const double v11 = fine(fi + 1, fj + 1, comp);
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
#if GPISM_HAVE_MPI
  if (context && context->mpi_enabled()) {
    exchange_for_device(const_cast<FieldStag2D<double>&>(nuH), grid, context);
    exchange_for_device(const_cast<FieldStag2D<double>&>(beta), grid, context);
    exchange_for_device(const_cast<FieldStag2D<double>&>(x), grid, context);
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
    const double dx = grid.dx();
    const double dy = grid.dy();
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);
    const int mask_stride_u =
        has_bc ? bc->mask->component(0).stride() : 0;
    const int mask_stride_v =
        has_bc ? bc->mask->component(1).stride() : 0;
    ssa_apply_cuda(
        grid.local_mx(), grid.local_my(), x.ghost_width(),
        x.component(0).stride(), x.component(1).stride(),
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
        has_bc ? 1 : 0);
    mg_residual_cuda(
        grid.local_mx(), grid.local_my(), r.ghost_width(),
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
  if (context && context->mpi_enabled()) {
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
    for (int iter = 0; iter < iterations; ++iter) {
#if GPISM_HAVE_MPI
      if (context && context->mpi_enabled()) {
        exchange_for_device(x, grid, context);
      }
#endif
      ssa_apply_cuda(
          grid.local_mx(), grid.local_my(), x.ghost_width(),
          x.component(0).stride(), x.component(1).stride(),
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
          has_bc ? 1 : 0);
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
    if (context && context->mpi_enabled()) {
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

void chebyshev_jacobi_smooth(const Grid2D& grid, const FieldStag2D<double>& nuH,
                             const FieldStag2D<double>& beta,
                             const FieldStag2D<double>& b,
                             FieldStag2D<double>& x, int iterations,
                             double lambda_min, double lambda_max,
                             const SSABoundaryCondition* bc) {
  if (iterations <= 0) {
    return;
  }

  FieldStag2D<double> diag(grid.local_mx(), grid.local_my(), grid.ghost_width());
  FieldStag2D<double> Ax(grid.local_mx(), grid.local_my(), grid.ghost_width());
  FieldStag2D<double> r(grid.local_mx(), grid.local_my(), grid.ghost_width());
  FieldStag2D<double> z(grid.local_mx(), grid.local_my(), grid.ghost_width());
  FieldStag2D<double> p(grid.local_mx(), grid.local_my(), grid.ghost_width());
  FieldStag2D<double> Ap(grid.local_mx(), grid.local_my(), grid.ghost_width());

  compute_jacobi_diag(grid, nuH, beta, diag, bc);
  ssa_apply_host(grid, nuH, beta, x, Ax, bc);

  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      for (int comp = 0; comp < 2; ++comp) {
        if (is_dirichlet(bc, i, j, comp)) {
          r(i, j, comp) = 0.0;
          z(i, j, comp) = 0.0;
          continue;
        }
        r(i, j, comp) = b(i, j, comp) - Ax(i, j, comp);
        const double d = diag(i, j, comp);
        z(i, j, comp) = (d != 0.0) ? (r(i, j, comp) / d) : 0.0;
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

    ssa_apply_host(grid, nuH, beta, p, Ap, bc);
    for (int j = 0; j < grid.local_my(); ++j) {
      for (int i = 0; i < grid.local_mx(); ++i) {
        for (int comp = 0; comp < 2; ++comp) {
          if (is_dirichlet(bc, i, j, comp)) {
            r(i, j, comp) = 0.0;
            z(i, j, comp) = 0.0;
            continue;
          }
          r(i, j, comp) -= alpha * Ap(i, j, comp);
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

void v_cycle(MultigridHierarchy& mg, int pre_iters, int post_iters,
             int coarse_iters, double omega,
             const SSABoundaryCondition* bc, const Context* context) {
  const int levels = mg.num_levels();
  if (levels == 0) {
    return;
  }

  for (int level = 0; level < levels - 1; ++level) {
    MGLevel& fine = mg.level(level);
    MGLevel& coarse = mg.level(level + 1);

    jacobi_smooth(fine.grid, fine.nuH, fine.beta, fine.rhs, fine.u, pre_iters,
                  omega, fine.diag, fine.Ax, bc, context);
    compute_residual(fine.grid, fine.nuH, fine.beta, fine.rhs, fine.u, fine.r,
                     fine.Ax, bc, context);
    restrict_stag(fine.r, coarse.rhs);
    set(0.0, coarse.u);
  }

  MGLevel& coarsest = mg.level(levels - 1);
  jacobi_smooth(coarsest.grid, coarsest.nuH, coarsest.beta, coarsest.rhs,
                coarsest.u, coarse_iters, omega, coarsest.diag, coarsest.Ax,
                bc, context);

  for (int level = levels - 2; level >= 0; --level) {
    MGLevel& fine = mg.level(level);
    MGLevel& coarse = mg.level(level + 1);
    prolong_stag(coarse.u, fine.corr);
    axpy(1.0, fine.corr, fine.u);
    jacobi_smooth(fine.grid, fine.nuH, fine.beta, fine.rhs, fine.u, post_iters,
                  omega, fine.diag, fine.Ax, bc, context);
  }
}

}  // namespace gpism
