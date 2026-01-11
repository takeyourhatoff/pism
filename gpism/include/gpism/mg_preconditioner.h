#pragma once

#include "gpism/gmres.h"
#include "gpism/multigrid.h"

namespace gpism {

class MultigridPreconditioner final : public Preconditioner {
public:
  MultigridPreconditioner(MultigridHierarchy& mg, int pre_iters,
                          int post_iters, int coarse_iters, double omega)
      : mg_(mg),
        pre_iters_(pre_iters),
        post_iters_(post_iters),
        coarse_iters_(coarse_iters),
        omega_(omega) {}

  void apply(const FieldStag2D<double>& x,
             FieldStag2D<double>& y) const override;

private:
  MultigridHierarchy& mg_;
  int pre_iters_;
  int post_iters_;
  int coarse_iters_;
  double omega_;
};

}  // namespace gpism
