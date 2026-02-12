#include "gpism/config.h"
#include "gpism/field_sync.h"
#include "gpism/gmres.h"
#include "gpism/linear_algebra.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

class IdentityOperator final : public gpism::LinearOperator {
public:
  void apply(const gpism::FieldStag2D<double>& x,
             gpism::FieldStag2D<double>& y) const override {
    gpism::copy(x, y);
  }
};

}  // namespace

int main() {
  const int mx = 8;
  const int my = 8;
  const int gw = 1;
  gpism::FieldStag2D<double> b(mx, my, gw);
  gpism::FieldStag2D<double> x(mx, my, gw);
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      b(i, j, 0) = 1.0 + 0.01 * i;
      b(i, j, 1) = -2.0 + 0.02 * j;
      x(i, j, 0) = 0.0;
      x(i, j, 1) = 0.0;
    }
  }
  gpism::sync_host_to_device(b);
  gpism::sync_host_to_device(x);

  IdentityOperator op;
  gpism::GMRESOptions opts;
  opts.restart = 10;
  opts.max_iter = 20;
  opts.tol = 1e-12;
  opts.tol_relative = false;
  opts.residual_check_interval = 1;
  gpism::GMRESResult result = gpism::gmres_solve(op, b, x, opts);
  if (!result.converged) {
    std::cerr << "GMRES did not converge on identity operator\n";
    return 1;
  }

  gpism::sync_device_to_host(x);
  double max_err = 0.0;
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      max_err = std::max(max_err, std::abs(x(i, j, 0) - b(i, j, 0)));
      max_err = std::max(max_err, std::abs(x(i, j, 1) - b(i, j, 1)));
    }
  }
  if (max_err > 1e-10) {
    std::cerr << "solution mismatch after GMRES (max_err=" << max_err << ")\n";
    return 1;
  }

  std::cout << "gmres_device_orthogonalization_smoke passed\n";
  return 0;
}
