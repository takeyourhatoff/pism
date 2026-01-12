#include "gpism/device_policy.h"
#include "gpism/field_sync.h"
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

struct Effectiveness {
  double r_norm = 0.0;
  double post_norm = 0.0;
};

void fill_pattern(gpism::FieldStag2D<double>& x) {
  const int mx = x.local_mx();
  const int my = x.local_my();
  const int gw = x.ghost_width();
  for (int j = -gw; j < my + gw; ++j) {
    for (int i = -gw; i < mx + gw; ++i) {
      const double val =
          std::sin(0.17 * static_cast<double>(i)) +
          std::cos(0.11 * static_cast<double>(j));
      x(i, j, 0) = val;
      x(i, j, 1) = 0.5 * val;
    }
  }
}

Effectiveness measure_effectiveness(bool use_device) {
  const bool prev_device = gpism::device_enabled();
  gpism::set_device_enabled(use_device);

  const int gw = 1;
  gpism::Grid2D grid(32, 32, 1000.0, 1000.0, gw, 0, 1);
  gpism::SSAOperator ssa(910.0, 9.81, 100.0);

  gpism::Field2D<double> tauc(grid.local_mx(), grid.local_my(), gw);
  tauc.fill(5.0);
  gpism::FieldStag2D<double> beta(grid.local_mx(), grid.local_my(), gw);

  gpism::FieldStag2D<double> nuH(grid.local_mx(), grid.local_my(), gw);
  for (int j = -gw; j < grid.local_my() + gw; ++j) {
    for (int i = -gw; i < grid.local_mx() + gw; ++i) {
      const double bump =
          1.0 + 0.4 * std::sin(0.09 * static_cast<double>(i)) *
                    std::cos(0.07 * static_cast<double>(j));
      nuH(i, j, 0) = 5.0 * bump;
      nuH(i, j, 1) = 5.0 * bump;
    }
  }

  gpism::FieldStag2D<double> x_true(grid.local_mx(), grid.local_my(), gw);
  fill_pattern(x_true);

  if (use_device) {
    gpism::sync_host_to_device(tauc);
    gpism::sync_host_to_device(nuH);
    gpism::sync_host_to_device(x_true);
  }

  ssa.compute_basal_drag(grid, tauc, beta);

  SSAOperatorWrapper op(ssa, grid, nuH, beta, nullptr);

  gpism::FieldStag2D<double> b(grid.local_mx(), grid.local_my(), gw);
  op.apply(x_true, b);

  gpism::MultigridHierarchy mg(grid, 4);
  gpism::copy(nuH, mg.level(0).nuH);
  gpism::copy(beta, mg.level(0).beta);
  for (int level = 1; level < mg.num_levels(); ++level) {
    gpism::restrict_stag(mg.level(level - 1).nuH, mg.level(level).nuH);
    gpism::restrict_stag(mg.level(level - 1).beta, mg.level(level).beta);
  }

  gpism::MultigridPreconditioner precond(
      mg, 2, 2, 10, 0.8, gpism::MGSmoother::Jacobi, 0.1, 2.0);

  gpism::FieldStag2D<double> x0(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> Ax(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> r(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> z(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> Az(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> r2(grid.local_mx(), grid.local_my(), gw);

  gpism::set(0.0, x0);
  op.apply(x0, Ax);
  gpism::copy(b, r);
  gpism::axpy(-1.0, Ax, r);

  precond.apply(r, z);
  op.apply(z, Az);
  gpism::copy(r, r2);
  gpism::axpy(-1.0, Az, r2);

  Effectiveness eff;
  eff.r_norm = gpism::norm2(r);
  eff.post_norm = gpism::norm2(r2);

  gpism::set_device_enabled(prev_device);
  return eff;
}

}  // namespace

int main() {
  const Effectiveness cpu = measure_effectiveness(false);
  std::cout << "MG effectiveness CPU: ||r||=" << cpu.r_norm
            << " ||r-Az||=" << cpu.post_norm
            << " ratio=" << (cpu.r_norm > 0.0 ? cpu.post_norm / cpu.r_norm : 0.0)
            << "\n";

  if (!(cpu.post_norm < cpu.r_norm)) {
    std::cerr << "MG did not reduce residual on CPU\n";
    return 1;
  }

#if GPISM_HAVE_CUDA
  const Effectiveness gpu = measure_effectiveness(true);
  std::cout << "MG effectiveness GPU: ||r||=" << gpu.r_norm
            << " ||r-Az||=" << gpu.post_norm
            << " ratio=" << (gpu.r_norm > 0.0 ? gpu.post_norm / gpu.r_norm : 0.0)
            << "\n";

  if (!(gpu.post_norm < gpu.r_norm)) {
    std::cerr << "MG did not reduce residual on GPU\n";
    return 1;
  }

  const double ratio_cpu = cpu.post_norm / cpu.r_norm;
  const double ratio_gpu = gpu.post_norm / gpu.r_norm;
  const double rel =
      (ratio_cpu != 0.0) ? std::abs(ratio_cpu - ratio_gpu) / ratio_cpu : 0.0;
  if (rel > 1e-3) {
    std::cerr << "MG CPU/GPU ratio mismatch: " << ratio_cpu << " vs "
              << ratio_gpu << "\n";
    return 1;
  }
#endif

  std::cout << "mg_preconditioner_effectiveness_smoke passed\n";
  return 0;
}
