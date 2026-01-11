#include "gpism/mg_preconditioner.h"

namespace gpism {

void MultigridPreconditioner::apply(const FieldStag2D<double>& x,
                                    FieldStag2D<double>& y) const {
  if (mg_.num_levels() == 0) {
    return;
  }

  MGLevel& fine = mg_.level(0);
  for (int j = 0; j < fine.grid.local_my(); ++j) {
    for (int i = 0; i < fine.grid.local_mx(); ++i) {
      fine.rhs(i, j, 0) = x(i, j, 0);
      fine.rhs(i, j, 1) = x(i, j, 1);
    }
  }
  fine.u.fill(0.0);

  v_cycle(mg_, pre_iters_, post_iters_, coarse_iters_, omega_);

  for (int j = 0; j < fine.grid.local_my(); ++j) {
    for (int i = 0; i < fine.grid.local_mx(); ++i) {
      y(i, j, 0) = fine.u(i, j, 0);
      y(i, j, 1) = fine.u(i, j, 1);
    }
  }
}

}  // namespace gpism
