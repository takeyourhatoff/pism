#include "gpism/gmres.h"

#include <cmath>
#include <iostream>

#include "gpism/field_sync.h"
#include "gpism/grid2d.h"
#include "gpism/linear_algebra.h"
#include "gpism/ssa_operator.h"

namespace {

bool nearly_equal(double a, double b, double tol = 1e-12) {
  return std::abs(a - b) <= tol;
}

struct SSAOperatorWrapper : public gpism::LinearOperator {
  SSAOperatorWrapper(const gpism::SSAOperator& op, const gpism::Grid2D& grid,
                     const gpism::FieldStag2D<double>& nuH,
                     const gpism::FieldStag2D<double>& beta)
      : op_(op), grid_(grid), nuH_(nuH), beta_(beta) {}

  void apply(const gpism::FieldStag2D<double>& x,
             gpism::FieldStag2D<double>& y) const override {
    op_.apply(grid_, nuH_, beta_, x, y, nullptr);
  }

  const gpism::SSAOperator& op_;
  const gpism::Grid2D& grid_;
  const gpism::FieldStag2D<double>& nuH_;
  const gpism::FieldStag2D<double>& beta_;
};

}  // namespace

int main() {
  const int mx = 1;
  const int my = 1;
  const int gw = 1;

  gpism::Grid2D grid(mx, my, 1000.0, 1000.0, gw, 0, 1);
  gpism::SSAOperator ssa(910.0, 9.81);

  gpism::FieldStag2D<double> nuH(mx, my, gw);
  gpism::FieldStag2D<double> beta(mx, my, gw);
  gpism::FieldStag2D<double> b(mx, my, gw);
  gpism::FieldStag2D<double> x(mx, my, gw);

  gpism::set(0.0, nuH);
  beta.fill(0.0);
  beta(0, 0, 0) = 2.0;
  beta(0, 0, 1) = 3.0;

  b.fill(0.0);
  b(0, 0, 0) = 4.0;
  b(0, 0, 1) = 9.0;

  gpism::sync_host_to_device(nuH);
  gpism::sync_host_to_device(beta);
  gpism::sync_host_to_device(b);

  gpism::set(0.0, x);

  SSAOperatorWrapper op(ssa, grid, nuH, beta);
  gpism::GMRESOptions opts;
  opts.restart = 2;
  opts.max_iter = 2;
  opts.tol = 1e-12;

  gpism::GMRESResult res = gpism::gmres_solve(op, b, x, opts);
  if (!res.converged) {
    std::cerr << "GMRES did not converge on a 2-eigenvalue diagonal system\n";
    return 1;
  }

  gpism::sync_device_to_host(x);
  if (!nearly_equal(x(0, 0, 0), 2.0, 1e-10) ||
      !nearly_equal(x(0, 0, 1), 3.0, 1e-10)) {
    std::cerr << "GMRES solution mismatch: got (" << x(0, 0, 0) << ", "
              << x(0, 0, 1) << "), expected (2, 3)\n";
    return 1;
  }

  std::cout << "gmres_givens_smoke passed\n";
  return 0;
}

