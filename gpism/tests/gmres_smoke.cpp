#include "gpism/gmres.h"

#include <cmath>
#include <cstring>
#include <iostream>

#include "gpism/linear_algebra.h"
#include "gpism/ssa_operator.h"

namespace {

bool nearly_equal(double a, double b, double tol = 1e-10) {
  return std::abs(a - b) <= tol;
}

template <typename T>
void sync_host_to_device(gpism::Field2D<T>& field) {
#if GPISM_HAVE_CUDA
  if (!field.host_staging_data() || field.elements() == 0) {
    return;
  }
  std::memcpy(field.host_staging_data(), field.data(),
              field.elements() * sizeof(T));
  field.copy_host_to_device();
#else
  (void)field;
#endif
}

template <typename T>
void sync_device_to_host(gpism::Field2D<T>& field) {
#if GPISM_HAVE_CUDA
  if (!field.host_staging_data() || field.elements() == 0) {
    return;
  }
  field.copy_device_to_host();
  std::memcpy(field.data(), field.host_staging_data(),
              field.elements() * sizeof(T));
#else
  (void)field;
#endif
}

template <typename T>
void sync_host_to_device(gpism::FieldStag2D<T>& field) {
  sync_host_to_device(field.component(0));
  sync_host_to_device(field.component(1));
}

template <typename T>
void sync_device_to_host(gpism::FieldStag2D<T>& field) {
  sync_device_to_host(field.component(0));
  sync_device_to_host(field.component(1));
}

struct IdentityOperator : public gpism::LinearOperator {
  void apply(const gpism::FieldStag2D<double>& x,
             gpism::FieldStag2D<double>& y) const override {
    gpism::copy(x, y);
  }
};

struct ScaleOperator : public gpism::LinearOperator {
  explicit ScaleOperator(double scale) : scale_(scale) {}

  void apply(const gpism::FieldStag2D<double>& x,
             gpism::FieldStag2D<double>& y) const override {
    gpism::copy(x, y);
    gpism::scal(scale_, y);
  }

  double scale_;
};

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

bool check_solution(const gpism::FieldStag2D<double>& x,
                    const gpism::FieldStag2D<double>& expected,
                    const char* name) {
  for (int j = 0; j < x.local_my(); ++j) {
    for (int i = 0; i < x.local_mx(); ++i) {
      for (int comp = 0; comp < 2; ++comp) {
        if (!nearly_equal(x(i, j, comp), expected(i, j, comp), 1e-8)) {
          std::cerr << name << " mismatch at " << i << "," << j << "," << comp
                    << " (" << x(i, j, comp) << " vs " << expected(i, j, comp)
                    << ")\n";
          return false;
        }
      }
    }
  }
  return true;
}

}  // namespace

int main() {
  const int mx = 5;
  const int my = 4;
  const int gw = 1;

  gpism::FieldStag2D<double> b(mx, my, gw);
  gpism::FieldStag2D<double> x(mx, my, gw);
  gpism::FieldStag2D<double> expected(mx, my, gw);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      b(i, j, 0) = static_cast<double>(i + 2 * j + 1);
      b(i, j, 1) = static_cast<double>(-3 + 2 * i - j);
    }
  }

  sync_host_to_device(b);
  gpism::set(0.0, x);

  gpism::GMRESOptions opts;
  opts.restart = 10;
  opts.max_iter = 30;
  opts.tol = 1e-10;

  IdentityOperator I;
  gpism::GMRESResult res = gpism::gmres_solve(I, b, x, opts);
  if (!res.converged) {
    std::cerr << "GMRES identity solve did not converge\n";
    return 1;
  }

  sync_device_to_host(x);
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      expected(i, j, 0) = b(i, j, 0);
      expected(i, j, 1) = b(i, j, 1);
    }
  }
  if (!check_solution(x, expected, "identity")) {
    return 1;
  }

  gpism::set(0.0, x);
  const double scale = 2.5;
  ScaleOperator A(scale);
  res = gpism::gmres_solve(A, b, x, opts);
  if (!res.converged) {
    std::cerr << "GMRES scaled identity solve did not converge\n";
    return 1;
  }

  sync_device_to_host(x);
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      expected(i, j, 0) = b(i, j, 0) / scale;
      expected(i, j, 1) = b(i, j, 1) / scale;
    }
  }
  if (!check_solution(x, expected, "scaled")) {
    return 1;
  }

  if (res.residuals.size() < 2 ||
      res.residuals.back() > res.residuals.front()) {
    std::cerr << "GMRES residual history did not decrease\n";
    return 1;
  }

  gpism::Grid2D grid(mx, my, 2.0, 3.0, gw, 0, 1);
  gpism::SSAOperator ssa(910.0, 9.81, 100.0);

  gpism::Field2D<double> tauc(mx, my, gw);
  gpism::Field2D<double> thk(mx, my, gw);
  gpism::Field2D<double> dhdx(mx, my, gw);
  gpism::Field2D<double> dhdy(mx, my, gw);
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      tauc(i, j) = 100.0;
      thk(i, j) = 2.0 + 0.1 * i;
      dhdx(i, j) = 0.05 + 0.01 * j;
      dhdy(i, j) = -0.02 + 0.005 * i;
    }
  }

  gpism::FieldStag2D<double> beta(mx, my, gw);
  gpism::FieldStag2D<double> nuH(mx, my, gw);
  gpism::FieldStag2D<double> rhs(mx, my, gw);
  gpism::FieldStag2D<double> x_ssa(mx, my, gw);

  sync_host_to_device(tauc);
  sync_host_to_device(thk);
  sync_host_to_device(dhdx);
  sync_host_to_device(dhdy);

  gpism::set(0.0, nuH);
  ssa.compute_basal_drag(grid, tauc, beta);
  ssa.assemble_rhs(grid, thk, dhdx, dhdy, rhs);

  SSAOperatorWrapper ssa_op(ssa, grid, nuH, beta, nullptr);
  gpism::set(0.0, x_ssa);
  res = gpism::gmres_solve(ssa_op, rhs, x_ssa, opts);
  if (!res.converged) {
    std::cerr << "GMRES SSA smoke solve did not converge\n";
    return 1;
  }

  sync_device_to_host(x_ssa);
  sync_device_to_host(rhs);
  if (!check_solution(x_ssa, rhs, "ssa")) {
    return 1;
  }

  gpism::Field2D<double> tauc2(mx, my, gw);
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      tauc2(i, j) = 200.0;
    }
  }

  gpism::FieldStag2D<double> beta2(mx, my, gw);
  gpism::FieldStag2D<double> nuH2(mx, my, gw);
  gpism::FieldStag2D<double> x_true(mx, my, gw);
  gpism::FieldStag2D<double> b_op(mx, my, gw);
  gpism::FieldStag2D<double> x_guess(mx, my, gw);
  gpism::FieldStag2D<double> err(mx, my, gw);

  for (int j = -gw; j < my + gw; ++j) {
    for (int i = -gw; i < mx + gw; ++i) {
      x_true(i, j, 0) = static_cast<double>(i + 1.5 * j);
      x_true(i, j, 1) = static_cast<double>(-2.0 + 0.5 * i - j);
    }
  }

  sync_host_to_device(tauc2);
  sync_host_to_device(x_true);

  gpism::set(0.5, nuH2);
  ssa.compute_basal_drag(grid, tauc2, beta2);

  SSAOperatorWrapper ssa_op2(ssa, grid, nuH2, beta2, nullptr);
  ssa_op2.apply(x_true, b_op);

  gpism::GMRESOptions opts_ssa = opts;
  opts_ssa.max_iter = 80;
  opts_ssa.tol = 1e-8;
  gpism::set(0.0, x_guess);
  res = gpism::gmres_solve(ssa_op2, b_op, x_guess, opts_ssa);
  if (!res.converged) {
    std::cerr << "GMRES SSA non-trivial solve did not converge\n";
    return 1;
  }

  gpism::copy(x_guess, err);
  gpism::axpy(-1.0, x_true, err);
  const double rel_err = gpism::norm2(err) / gpism::norm2(x_true);
  if (rel_err > 1e-6) {
    std::cerr << "GMRES SSA non-trivial relative error too high: " << rel_err
              << "\n";
    return 1;
  }

  std::cout << "gmres_smoke passed\n";
  return 0;
}
