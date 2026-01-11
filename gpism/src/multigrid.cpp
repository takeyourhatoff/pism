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

}  // namespace gpism
