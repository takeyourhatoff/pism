#pragma once

#include <vector>

#include "gpism/field_stag2d.h"
#include "gpism/grid2d.h"

namespace gpism {

struct SSABoundaryCondition;

struct MGLevel {
  Grid2D grid;
  FieldStag2D<double> u;
  FieldStag2D<double> r;
  FieldStag2D<double> nuH;

  explicit MGLevel(const Grid2D& grid_in)
      : grid(grid_in),
        u(grid_in.local_mx(), grid_in.local_my(), grid_in.ghost_width()),
        r(grid_in.local_mx(), grid_in.local_my(), grid_in.ghost_width()),
        nuH(grid_in.local_mx(), grid_in.local_my(), grid_in.ghost_width()) {}
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
void jacobi_smooth(const Grid2D& grid, const FieldStag2D<double>& nuH,
                   const FieldStag2D<double>& beta, const FieldStag2D<double>& b,
                   FieldStag2D<double>& x, int iterations, double omega,
                   const SSABoundaryCondition* bc = nullptr);

}  // namespace gpism
