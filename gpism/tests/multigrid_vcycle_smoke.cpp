#include "gpism/multigrid.h"
#include "gpism/field_sync.h"

#include <cmath>
#include <iostream>

namespace {

bool nearly_equal(double a, double b, double tol = 1e-10) {
  return std::abs(a - b) <= tol;
}

}  // namespace

int main() {
  const int gw = 1;
  gpism::Grid2D grid(4, 4, 1.0, 1.0, gw, 0, 1);
  gpism::MultigridHierarchy mg(grid, 2);

  if (mg.num_levels() < 2) {
    std::cerr << "expected at least 2 levels\n";
    return 1;
  }

  gpism::MGLevel& fine = mg.level(0);
  gpism::MGLevel& coarse = mg.level(1);

  fine.nuH.fill(0.0);
  fine.beta.fill(2.0);
  fine.u.fill(0.0);
  for (int j = 0; j < fine.grid.local_my(); ++j) {
    for (int i = 0; i < fine.grid.local_mx(); ++i) {
      fine.rhs(i, j, 0) = static_cast<double>(i + j + 1);
      fine.rhs(i, j, 1) = static_cast<double>(2 * i - j + 1);
    }
  }

  coarse.nuH.fill(0.0);
  coarse.beta.fill(2.0);
  coarse.u.fill(0.0);
  coarse.rhs.fill(0.0);

  gpism::sync_host_to_device(fine.nuH);
  gpism::sync_host_to_device(fine.beta);
  gpism::sync_host_to_device(fine.u);
  gpism::sync_host_to_device(fine.rhs);
  gpism::sync_host_to_device(coarse.nuH);
  gpism::sync_host_to_device(coarse.beta);
  gpism::sync_host_to_device(coarse.u);
  gpism::sync_host_to_device(coarse.rhs);

  gpism::v_cycle(mg, 1, 0, 1, 1.0);
  gpism::sync_device_to_host(fine.u);

  for (int j = 0; j < fine.grid.local_my(); ++j) {
    for (int i = 0; i < fine.grid.local_mx(); ++i) {
      if (!nearly_equal(fine.u(i, j, 0), 0.5 * fine.rhs(i, j, 0)) ||
          !nearly_equal(fine.u(i, j, 1), 0.5 * fine.rhs(i, j, 1))) {
        std::cerr << "v-cycle mismatch at " << i << "," << j << "\n";
        return 1;
      }
    }
  }

  std::cout << "multigrid_vcycle_smoke passed\n";
  return 0;
}
