#include "gpism/gmres.h"
#include "gpism/linear_algebra.h"
#include "gpism/mg_preconditioner.h"
#include "gpism/ssa_operator.h"

#include <cmath>
#include <iostream>

namespace {

struct SSAOperatorWrapper : public gpism::LinearOperator {
  SSAOperatorWrapper(const gpism::SSAOperator& op, const gpism::Grid2D& grid,
                     const gpism::FieldStag2D<double>& nuH,
                     const gpism::FieldStag2D<double>& beta,
                     const gpism::SSABoundaryCondition* bc)
      : op_(op), grid_(grid), nuH_(nuH), beta_(beta), bc_(bc) {}

  void apply(const gpism::FieldStag2D<double>& x,
             gpism::FieldStag2D<double>& y) const override {
    op_.apply(grid_, nuH_, beta_, x, y, bc_);
  }

  const gpism::SSAOperator& op_;
  const gpism::Grid2D& grid_;
  const gpism::FieldStag2D<double>& nuH_;
  const gpism::FieldStag2D<double>& beta_;
  const gpism::SSABoundaryCondition* bc_;
};

struct SolveStats {
  int gmres_iters = 0;
  int mg_iters = 0;
  double gmres_res = 0.0;
  double mg_res = 0.0;
};

void fill_pattern(gpism::FieldStag2D<double>& x) {
  const int mx = x.local_mx();
  const int my = x.local_my();
  const int gw = x.ghost_width();
  for (int j = -gw; j < my + gw; ++j) {
    for (int i = -gw; i < mx + gw; ++i) {
      const double val =
          std::sin(0.2 * static_cast<double>(i)) +
          std::cos(0.3 * static_cast<double>(j));
      x(i, j, 0) = val;
      x(i, j, 1) = 0.5 * val;
    }
  }
}

SolveStats run_case(int mx, int my) {
  const int gw = 1;
  gpism::Grid2D grid(mx, my, 1000.0, 1000.0, gw, 0, 1);
  gpism::SSAOperator ssa(910.0, 9.81, 100.0);

  gpism::Field2D<double> tauc(mx, my, gw);
  tauc.fill(5.0);
  gpism::FieldStag2D<double> beta(mx, my, gw);
  ssa.compute_basal_drag(grid, tauc, beta);

  gpism::FieldStag2D<double> nuH(mx, my, gw);
  for (int j = -gw; j < my + gw; ++j) {
    for (int i = -gw; i < mx + gw; ++i) {
      const double bump =
          1.0 + 0.5 * std::sin(0.1 * static_cast<double>(i)) *
                    std::cos(0.1 * static_cast<double>(j));
      nuH(i, j, 0) = 5.0 * bump;
      nuH(i, j, 1) = 5.0 * bump;
    }
  }

  gpism::FieldStag2D<double> x_true(mx, my, gw);
  gpism::FieldStag2D<double> b(mx, my, gw);
  fill_pattern(x_true);
  ssa.apply(grid, nuH, beta, x_true, b);

  SSAOperatorWrapper op(ssa, grid, nuH, beta, nullptr);
  gpism::GMRESOptions opts;
  opts.restart = 30;
  opts.max_iter = 120;
  opts.tol = 1e-10;

  gpism::FieldStag2D<double> x(mx, my, gw);
  gpism::set(0.0, x);
  gpism::GMRESResult gmres_res = gpism::gmres_solve(op, b, x, opts);

  gpism::MultigridHierarchy mg(grid);
  gpism::copy(nuH, mg.level(0).nuH);
  gpism::copy(beta, mg.level(0).beta);
  for (int level = 1; level < mg.num_levels(); ++level) {
    gpism::restrict_stag(mg.level(level - 1).nuH, mg.level(level).nuH);
    gpism::restrict_stag(mg.level(level - 1).beta, mg.level(level).beta);
  }

  gpism::MultigridPreconditioner precond(
      mg, 3, 3, 10, 0.8, gpism::MGSmoother::Jacobi, 0.1, 2.0);
  gpism::set(0.0, x);
  gpism::GMRESResult mg_res = gpism::gmres_solve(op, b, x, opts, &precond);

  SolveStats stats;
  stats.gmres_iters = gmres_res.iterations;
  stats.mg_iters = mg_res.iterations;
  stats.gmres_res = gmres_res.residual;
  stats.mg_res = mg_res.residual;
  return stats;
}

}  // namespace

int main() {
  const SolveStats small = run_case(16, 16);
  const SolveStats large = run_case(32, 32);

  if (small.gmres_iters == 0 || large.gmres_iters == 0) {
    std::cerr << "GMRES iterations not recorded\n";
    return 1;
  }
  if (small.mg_iters == 0 || large.mg_iters == 0) {
    std::cerr << "MG-GMRES iterations not recorded\n";
    return 1;
  }
  if (small.mg_iters > small.gmres_iters ||
      large.mg_iters > large.gmres_iters) {
    std::cerr << "MG preconditioner did not improve GMRES iterations (small: "
              << small.gmres_iters << " vs " << small.mg_iters
              << ", large: " << large.gmres_iters << " vs " << large.mg_iters
              << ")\n";
    return 1;
  }

  if (large.mg_iters > small.mg_iters + 4) {
    std::cerr << "MG iterations grew too fast with refinement: "
              << small.mg_iters << " -> " << large.mg_iters << "\n";
    return 1;
  }

  if (!std::isfinite(small.mg_res) || !std::isfinite(large.mg_res)) {
    std::cerr << "MG residual not finite\n";
    return 1;
  }

  std::cout << "gmres_mg_compare_smoke passed\n";
  return 0;
}
