#include "gpism/mg_preconditioner.h"

#include "gpism/linear_algebra.h"

namespace gpism {

void MultigridPreconditioner::apply(const FieldStag2D<double>& x,
                                    FieldStag2D<double>& y) const {
  if (mg_.num_levels() == 0) {
    return;
  }

  MGLevel& fine = mg_.level(0);
  copy(x, fine.rhs);
  set(0.0, fine.u);

  const std::vector<ChebyBounds>* bounds_ptr = nullptr;
  if (smoother_ == MGSmoother::Chebyshev) {
    if (!cheby_bounds_cached_ ||
        static_cast<int>(cheby_bounds_cache_.size()) != mg_.num_levels()) {
      cheby_bounds_cache_ = estimate_cheby_bounds(
          mg_, cheby_lambda_min_, cheby_lambda_max_, cheby_estimate_,
          cheby_estimate_iters_, cheby_estimate_min_factor_,
          cheby_estimate_max_factor_, bc_, context_);
      cheby_bounds_cached_ = true;
    }
    bounds_ptr = &cheby_bounds_cache_;
  }

  v_cycle(mg_, pre_iters_, post_iters_, coarse_iters_, omega_, smoother_,
          cheby_lambda_min_, cheby_lambda_max_, cheby_estimate_,
          cheby_estimate_iters_, cheby_estimate_min_factor_,
          cheby_estimate_max_factor_, bc_, context_, bounds_ptr);

  copy(fine.u, y);
}

}  // namespace gpism
