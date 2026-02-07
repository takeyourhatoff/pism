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
                          bool cheby_estimate = false,
                          int cheby_estimate_iters = 5,
                          double cheby_estimate_min_factor = 0.1,
                          double cheby_estimate_max_factor = 1.1,
                          const SSABoundaryCondition* bc = nullptr,
                          const Context* context = nullptr,
                          bool diagnostic = false,
                          double beta_ice_free_bedrock = 0.0)
      : mg_(mg),
        pre_iters_(pre_iters),
        post_iters_(post_iters),
        coarse_iters_(coarse_iters),
        omega_(omega),
        smoother_(smoother),
        cheby_lambda_min_(cheby_lambda_min),
        cheby_lambda_max_(cheby_lambda_max),
        cheby_estimate_(cheby_estimate),
        cheby_estimate_iters_(cheby_estimate_iters),
        cheby_estimate_min_factor_(cheby_estimate_min_factor),
        cheby_estimate_max_factor_(cheby_estimate_max_factor),
        beta_ice_free_bedrock_(beta_ice_free_bedrock),
        bc_(bc),
        context_(context),
        diagnostic_(diagnostic) {}

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
  bool cheby_estimate_;
  int cheby_estimate_iters_;
  double cheby_estimate_min_factor_;
  double cheby_estimate_max_factor_;
  mutable std::vector<ChebyBounds> cheby_bounds_cache_;
  mutable bool cheby_bounds_cached_ = false;
  const double beta_ice_free_bedrock_ = 0.0;
  mutable bool beta_guarded_ = false;
  mutable bool bc_levels_cached_ = false;
  mutable std::vector<SSABoundaryCondition> bc_levels_;
  const SSABoundaryCondition* bc_;
  const Context* context_;
  const bool diagnostic_ = false;
  mutable bool diagnostic_printed_ = false;
};

}  // namespace gpism
