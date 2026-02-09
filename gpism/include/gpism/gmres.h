#pragma once

#include <vector>

#include "gpism/field_stag2d.h"

namespace gpism {

class Context;

class LinearOperator {
public:
  virtual ~LinearOperator() = default;
  virtual void apply(const FieldStag2D<double>& x,
                     FieldStag2D<double>& y) const = 0;
};

class Preconditioner {
public:
  virtual ~Preconditioner() = default;
  virtual void apply(const FieldStag2D<double>& x,
                     FieldStag2D<double>& y) const = 0;
};

class IdentityPreconditioner final : public Preconditioner {
public:
  void apply(const FieldStag2D<double>& x,
             FieldStag2D<double>& y) const override;
};

struct GMRESOptions {
  int restart = 30;
  int max_iter = 200;
  double tol = 1e-8;
  bool tol_relative = false;
  // If true, scale the relative tolerance using max(||r0||, ||b||) instead of
  // only ||r0||. This avoids over-tightening the solve when the initial guess
  // is already close (||r0|| << ||b||), which can stall GMRES.
  bool tol_relative_to_rhs = false;
  bool verbose = false;
  bool precond_diagnostic = false;
  const Context* context = nullptr;
};

struct GMRESResult {
  int iterations = 0;
  // Residual norm estimate tracked by the Arnoldi process. For GMRES with a
  // left preconditioner this corresponds to ||M^{-1}(b - A x)||.
  double residual = 0.0;
  // True residual norm ||b - A x|| (unpreconditioned). This is computed at
  // the start and after each restart update for diagnostics.
  double true_residual = 0.0;
  bool converged = false;
  std::vector<double> residuals;
};

GMRESResult gmres_solve(const LinearOperator& op,
                        const FieldStag2D<double>& b,
                        FieldStag2D<double>& x,
                        const GMRESOptions& options,
                        const Preconditioner* precond = nullptr);

}  // namespace gpism
