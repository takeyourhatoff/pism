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
  bool verbose = false;
  bool precond_diagnostic = false;
  bool device_full = false;
  const Context* context = nullptr;
};

struct GMRESResult {
  int iterations = 0;
  double residual = 0.0;
  bool converged = false;
  std::vector<double> residuals;
};

GMRESResult gmres_solve(const LinearOperator& op,
                        const FieldStag2D<double>& b,
                        FieldStag2D<double>& x,
                        const GMRESOptions& options,
                        const Preconditioner* precond = nullptr);

}  // namespace gpism
