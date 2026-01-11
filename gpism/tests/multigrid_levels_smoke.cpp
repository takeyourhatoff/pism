#include "gpism/multigrid.h"

#include <cmath>
#include <iostream>

namespace {

bool nearly_equal(double a, double b, double tol = 1e-12) {
  return std::abs(a - b) <= tol;
}

bool check_level(const gpism::MGLevel& level, int mx, int my, double dx,
                 double dy) {
  if (level.grid.global_mx() != mx || level.grid.global_my() != my) {
    std::cerr << "grid size mismatch: " << level.grid.global_mx() << "x"
              << level.grid.global_my() << " vs " << mx << "x" << my << "\n";
    return false;
  }
  if (!nearly_equal(level.grid.dx(), dx) || !nearly_equal(level.grid.dy(), dy)) {
    std::cerr << "grid spacing mismatch: " << level.grid.dx() << ","
              << level.grid.dy() << " vs " << dx << "," << dy << "\n";
    return false;
  }
  if (level.u.local_mx() != mx || level.u.local_my() != my) {
    std::cerr << "field size mismatch for level\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  const int gw = 1;
  gpism::Grid2D grid(17, 9, 1.0, 2.0, gw, 0, 1);
  gpism::MultigridHierarchy mg(grid, 4);

  if (mg.num_levels() != 3) {
    std::cerr << "unexpected level count: " << mg.num_levels() << "\n";
    return 1;
  }

  if (!check_level(mg.level(0), 17, 9, 1.0, 2.0)) {
    return 1;
  }
  if (!check_level(mg.level(1), 9, 5, 2.0, 4.0)) {
    return 1;
  }
  if (!check_level(mg.level(2), 5, 3, 4.0, 8.0)) {
    return 1;
  }

  std::cout << "multigrid_levels_smoke passed\n";
  return 0;
}
