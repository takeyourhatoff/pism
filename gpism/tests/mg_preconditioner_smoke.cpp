#include "gpism/mg_preconditioner.h"
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

  for (int level = 0; level < mg.num_levels(); ++level) {
    gpism::MGLevel& lvl = mg.level(level);
    lvl.nuH.fill(0.0);
    lvl.beta.fill(2.0);
    lvl.u.fill(0.0);
    lvl.rhs.fill(0.0);
    gpism::sync_host_to_device(lvl.nuH);
    gpism::sync_host_to_device(lvl.beta);
    gpism::sync_host_to_device(lvl.u);
    gpism::sync_host_to_device(lvl.rhs);
  }

  gpism::FieldStag2D<double> x(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> y(grid.local_mx(), grid.local_my(), gw);
  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      x(i, j, 0) = static_cast<double>(i + j + 1);
      x(i, j, 1) = static_cast<double>(2 * i - j + 1);
    }
  }

  gpism::MultigridPreconditioner pc(mg, 1, 0, 1, 1.0);
  gpism::sync_host_to_device(x);
  pc.apply(x, y);
  gpism::sync_device_to_host(y);

  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      if (!nearly_equal(y(i, j, 0), 0.5 * x(i, j, 0)) ||
          !nearly_equal(y(i, j, 1), 0.5 * x(i, j, 1))) {
        std::cerr << "preconditioner mismatch at " << i << "," << j << "\n";
        return 1;
      }
    }
  }

  std::cout << "mg_preconditioner_smoke passed\n";
  return 0;
}
