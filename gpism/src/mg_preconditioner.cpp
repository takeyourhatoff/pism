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

  v_cycle(mg_, pre_iters_, post_iters_, coarse_iters_, omega_, smoother_,
          cheby_lambda_min_, cheby_lambda_max_, bc_, context_);

  copy(fine.u, y);
}

}  // namespace gpism
