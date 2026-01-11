#include "gpism/geometry.h"

namespace gpism {

void GeometryDiagnostics::compute_usurf_cpu(const Grid2D& grid,
                                            const Field2D<double>& thk,
                                            const Field2D<double>& topg,
                                            Field2D<double>& usurf) {
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      usurf(i, j) = topg(i, j) + thk(i, j);
    }
  }
}

void GeometryDiagnostics::compute_surface_slopes_cpu(const Grid2D& grid,
                                                     const Field2D<double>& usurf,
                                                     Field2D<double>& dhdx,
                                                     Field2D<double>& dhdy) {
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  const double inv_dx = 1.0 / grid.dx();
  const double inv_dy = 1.0 / grid.dy();

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const double left = (i == 0) ? usurf(i, j) : usurf(i - 1, j);
      const double right = (i == mx - 1) ? usurf(i, j) : usurf(i + 1, j);
      const double down = (j == 0) ? usurf(i, j) : usurf(i, j - 1);
      const double up = (j == my - 1) ? usurf(i, j) : usurf(i, j + 1);
      dhdx(i, j) = 0.5 * (right - left) * inv_dx;
      dhdy(i, j) = 0.5 * (up - down) * inv_dy;
    }
  }
}

void GeometryDiagnostics::compute_usurf(const Grid2D& grid, const Field2D<double>& thk,
                                        const Field2D<double>& topg,
                                        Field2D<double>& usurf) {
  compute_usurf_cpu(grid, thk, topg, usurf);
}

void GeometryDiagnostics::compute_surface_slopes(const Grid2D& grid,
                                                 const Field2D<double>& usurf,
                                                 Field2D<double>& dhdx,
                                                 Field2D<double>& dhdy) {
  compute_surface_slopes_cpu(grid, usurf, dhdx, dhdy);
}

}  // namespace gpism
