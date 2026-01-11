#include "gpism/multigrid.h"

#include <cmath>
#include <iostream>

namespace {

double norm2_host(const gpism::FieldStag2D<double>& r) {
  double sum = 0.0;
  for (int j = 0; j < r.local_my(); ++j) {
    for (int i = 0; i < r.local_mx(); ++i) {
      sum += r(i, j, 0) * r(i, j, 0) + r(i, j, 1) * r(i, j, 1);
    }
  }
  return std::sqrt(sum);
}

bool run_case(bool variable) {
  const int gw = 1;
  gpism::Grid2D grid(8, 8, 1.0, 1.0, gw, 0, 1);
  gpism::MultigridHierarchy mg(grid, 2);

  for (int level = 0; level < mg.num_levels(); ++level) {
    gpism::MGLevel& lvl = mg.level(level);
    for (int j = 0; j < lvl.grid.local_my(); ++j) {
      for (int i = 0; i < lvl.grid.local_mx(); ++i) {
        const double base = variable ? (0.2 + 0.01 * i + 0.02 * j + 0.05 * level)
                                     : 0.5;
        const double drag = variable ? (1.0 + 0.005 * i - 0.003 * j)
                                     : 2.0;
        lvl.nuH(i, j, 0) = base;
        lvl.nuH(i, j, 1) = base;
        lvl.beta(i, j, 0) = drag;
        lvl.beta(i, j, 1) = drag;
      }
    }
    lvl.u.fill(0.0);
    lvl.rhs.fill(0.0);
  }

  gpism::MGLevel& fine = mg.level(0);
  for (int j = 0; j < fine.grid.local_my(); ++j) {
    for (int i = 0; i < fine.grid.local_mx(); ++i) {
      fine.rhs(i, j, 0) = static_cast<double>(i + j + 1);
      fine.rhs(i, j, 1) = static_cast<double>(2 * i - j + 1);
    }
  }

  gpism::compute_residual(fine.grid, fine.nuH, fine.beta, fine.rhs, fine.u, fine.r);
  const double r0 = norm2_host(fine.r);

  gpism::v_cycle(mg, 2, 2, 4, 0.8);

  gpism::compute_residual(fine.grid, fine.nuH, fine.beta, fine.rhs, fine.u, fine.r);
  const double r1 = norm2_host(fine.r);

  if (r1 >= r0) {
    std::cerr << "residual did not decrease (variable=" << variable << ")\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!run_case(false)) {
    return 1;
  }
  if (!run_case(true)) {
    return 1;
  }

  std::cout << "multigrid_coeff_smoke passed\n";
  return 0;
}
