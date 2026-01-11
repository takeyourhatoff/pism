#include "gpism/multigrid.h"

#include <algorithm>
#include <vector>

#include "gpism/ssa_operator.h"

namespace gpism {
namespace {

int coarsen_dim(int n) { return (n + 1) / 2; }

bool is_dirichlet(const SSABoundaryCondition* bc, int i, int j, int comp) {
  if (!bc || !bc->mask) {
    return false;
  }
  return (*bc->mask)(i, j, comp) != 0;
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

void jacobi_smooth(const Grid2D& grid, const FieldStag2D<double>& nuH,
                   const FieldStag2D<double>& beta, const FieldStag2D<double>& b,
                   FieldStag2D<double>& x, int iterations, double omega,
                   const SSABoundaryCondition* bc) {
  if (iterations <= 0) {
    return;
  }

  FieldStag2D<double> diag(grid.local_mx(), grid.local_my(), grid.ghost_width());
  FieldStag2D<double> Ax(grid.local_mx(), grid.local_my(), grid.ghost_width());

  compute_jacobi_diag(grid, nuH, beta, diag, bc);

  for (int iter = 0; iter < iterations; ++iter) {
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

}  // namespace gpism
