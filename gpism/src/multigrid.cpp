#include "gpism/multigrid.h"

#include <algorithm>
#include <vector>

namespace gpism {
namespace {

int coarsen_dim(int n) { return (n + 1) / 2; }

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

}  // namespace gpism
