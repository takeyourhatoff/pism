#pragma once

#include <vector>

#include "gpism/field_stag2d.h"
#include "gpism/grid2d.h"

namespace gpism {

class Context;
struct SSABoundaryCondition;

struct MGLevel {
  Grid2D grid;
  FieldStag2D<double> u;
  FieldStag2D<double> rhs;
  FieldStag2D<double> r;
  FieldStag2D<double> nuH;
  FieldStag2D<double> beta;
  FieldStag2D<double> diag;
  FieldStag2D<double> Ax;
  FieldStag2D<double> corr;

  explicit MGLevel(const Grid2D& grid_in)
      : grid(grid_in),
        u(grid_in.local_mx(), grid_in.local_my(), grid_in.ghost_width()),
        rhs(grid_in.local_mx(), grid_in.local_my(), grid_in.ghost_width()),
        r(grid_in.local_mx(), grid_in.local_my(), grid_in.ghost_width()),
        nuH(grid_in.local_mx(), grid_in.local_my(), grid_in.ghost_width()),
        beta(grid_in.local_mx(), grid_in.local_my(), grid_in.ghost_width()),
        diag(grid_in.local_mx(), grid_in.local_my(), grid_in.ghost_width()),
        Ax(grid_in.local_mx(), grid_in.local_my(), grid_in.ghost_width()),
        corr(grid_in.local_mx(), grid_in.local_my(), grid_in.ghost_width()) {}
};

class MultigridHierarchy {
public:
  explicit MultigridHierarchy(const Grid2D& fine_grid, int min_size = 4);

  int num_levels() const { return static_cast<int>(levels_.size()); }

  const MGLevel& level(int idx) const { return levels_.at(idx); }
  MGLevel& level(int idx) { return levels_.at(idx); }

private:
  std::vector<MGLevel> levels_;
};

void restrict_stag(const FieldStag2D<double>& fine, FieldStag2D<double>& coarse);
void prolong_stag(const FieldStag2D<double>& coarse, FieldStag2D<double>& fine);
void compute_residual(const Grid2D& grid, const FieldStag2D<double>& nuH,
                      const FieldStag2D<double>& beta,
                      const FieldStag2D<double>& b,
                      const FieldStag2D<double>& x, FieldStag2D<double>& r,
                      FieldStag2D<double>& Ax,
                      const SSABoundaryCondition* bc = nullptr,
                      const Context* context = nullptr);
void jacobi_smooth(const Grid2D& grid, const FieldStag2D<double>& nuH,
                   const FieldStag2D<double>& beta, const FieldStag2D<double>& b,
                   FieldStag2D<double>& x, int iterations, double omega,
                   FieldStag2D<double>& diag, FieldStag2D<double>& Ax,
                   const SSABoundaryCondition* bc = nullptr,
                   const Context* context = nullptr);
void chebyshev_jacobi_smooth(const Grid2D& grid, const FieldStag2D<double>& nuH,
                             const FieldStag2D<double>& beta,
                             const FieldStag2D<double>& b, FieldStag2D<double>& x,
                             int iterations, double lambda_min,
                             double lambda_max,
                             const SSABoundaryCondition* bc = nullptr);
void v_cycle(MultigridHierarchy& mg, int pre_iters, int post_iters,
             int coarse_iters, double omega,
             const SSABoundaryCondition* bc = nullptr,
             const Context* context = nullptr);

}  // namespace gpism
