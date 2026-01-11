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

double run_case(int n) {
  const int gw = 1;
  gpism::Grid2D grid(n, n, 1.0, 1.0, gw, 0, 1);
  gpism::MultigridHierarchy mg(grid, 2);

  for (int level = 0; level < mg.num_levels(); ++level) {
    gpism::MGLevel& lvl = mg.level(level);
    lvl.nuH.fill(0.5);
    lvl.beta.fill(2.0);
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

  for (int cycle = 0; cycle < 3; ++cycle) {
    gpism::v_cycle(mg, 2, 2, 4, 0.8);
  }

  gpism::compute_residual(fine.grid, fine.nuH, fine.beta, fine.rhs, fine.u, fine.r);
  const double r1 = norm2_host(fine.r);

  return (r0 > 0.0) ? (r1 / r0) : 0.0;
}

}  // namespace

int main() {
  const double ratio8 = run_case(8);
  const double ratio16 = run_case(16);

  if (ratio8 <= 0.0 || ratio16 <= 0.0) {
    std::cerr << "invalid residual ratios\n";
    return 1;
  }

  if (ratio16 > 2.0 * ratio8) {
    std::cerr << "residual reduction degraded with resolution: " << ratio8
              << " vs " << ratio16 << "\n";
    return 1;
  }

  std::cout << "multigrid_scaling_smoke passed\n";
  return 0;
}
