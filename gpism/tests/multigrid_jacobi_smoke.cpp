#include "gpism/multigrid.h"

#include <cmath>
#include <iostream>

namespace {

bool nearly_equal(double a, double b, double tol = 1e-10) {
  return std::abs(a - b) <= tol;
}

}  // namespace

int main() {
  const int mx = 4;
  const int my = 3;
  const int gw = 1;
  gpism::Grid2D grid(mx, my, 2.0, 3.0, gw, 0, 1);

  gpism::FieldStag2D<double> nuH(mx, my, gw);
  gpism::FieldStag2D<double> beta(mx, my, gw);
  gpism::FieldStag2D<double> b(mx, my, gw);
  gpism::FieldStag2D<double> x(mx, my, gw);

  nuH.fill(0.0);
  beta.fill(2.0);
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      b(i, j, 0) = static_cast<double>(i + j + 1);
      b(i, j, 1) = static_cast<double>(2 * i - j);
      x(i, j, 0) = 0.0;
      x(i, j, 1) = 0.0;
    }
  }

  gpism::jacobi_smooth(grid, nuH, beta, b, x, 1, 1.0);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      if (!nearly_equal(x(i, j, 0), 0.5 * b(i, j, 0)) ||
          !nearly_equal(x(i, j, 1), 0.5 * b(i, j, 1))) {
        std::cerr << "jacobi update mismatch at " << i << "," << j << "\n";
        return 1;
      }
    }
  }

  std::cout << "multigrid_jacobi_smoke passed\n";
  return 0;
}
