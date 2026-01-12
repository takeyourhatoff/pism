#pragma once

#include "gpism/gmres.h"
#include "gpism/multigrid.h"

namespace gpism {

class MultigridPreconditioner final : public Preconditioner {
public:
  MultigridPreconditioner(MultigridHierarchy& mg, int pre_iters,
                          int post_iters, int coarse_iters, double omega,
                          MGSmoother smoother = MGSmoother::Jacobi,
                          double cheby_lambda_min = 0.1,
                          double cheby_lambda_max = 2.0,
                          const SSABoundaryCondition* bc = nullptr,
                          const Context* context = nullptr)
      : mg_(mg),
        pre_iters_(pre_iters),
        post_iters_(post_iters),
        coarse_iters_(coarse_iters),
        omega_(omega),
        smoother_(smoother),
        cheby_lambda_min_(cheby_lambda_min),
        cheby_lambda_max_(cheby_lambda_max),
        bc_(bc),
        context_(context) {}

  void apply(const FieldStag2D<double>& x,
             FieldStag2D<double>& y) const override;

private:
  MultigridHierarchy& mg_;
  int pre_iters_;
  int post_iters_;
  int coarse_iters_;
  double omega_;
  MGSmoother smoother_;
  double cheby_lambda_min_;
  double cheby_lambda_max_;
  const SSABoundaryCondition* bc_;
  const Context* context_;
};

}  // namespace gpism
