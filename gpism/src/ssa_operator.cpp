#include "gpism/ssa_operator.h"

#include <algorithm>

namespace gpism {
namespace {

double avg2(double a, double b) { return 0.5 * (a + b); }

bool is_dirichlet(const SSABoundaryCondition* bc, int i, int j, int comp) {
  if (!bc || !bc->mask) {
    return false;
  }
  return (*bc->mask)(i, j, comp) != 0;
}

}  // namespace

SSAOperator::SSAOperator(double rho, double g, double u_threshold)
    : rho_(rho), g_(g), u_threshold_(u_threshold) {}

void SSAOperator::compute_basal_drag(const Grid2D& grid,
                                     const Field2D<double>& tauc,
                                     FieldStag2D<double>& beta) const {
  const double denom = std::max(u_threshold_, 1e-6);
  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      const double tauc_u = avg2(tauc(i, j), tauc(i + 1, j));
      const double tauc_v = avg2(tauc(i, j), tauc(i, j + 1));
      beta(i, j, 0) = tauc_u / denom;
      beta(i, j, 1) = tauc_v / denom;
    }
  }
}

void SSAOperator::assemble_rhs(const Grid2D& grid, const Field2D<double>& thk,
                               const Field2D<double>& dhdx,
                               const Field2D<double>& dhdy,
                               FieldStag2D<double>& rhs,
                               const SSABoundaryCondition* bc) const {
  const double scale = rho_ * g_;
  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      const double H_u = avg2(thk(i, j), thk(i + 1, j));
      const double H_v = avg2(thk(i, j), thk(i, j + 1));
      const double slope_x = avg2(dhdx(i, j), dhdx(i + 1, j));
      const double slope_y = avg2(dhdy(i, j), dhdy(i, j + 1));

      rhs(i, j, 0) = scale * H_u * slope_x;
      rhs(i, j, 1) = scale * H_v * slope_y;

      if (is_dirichlet(bc, i, j, 0) && bc->values) {
        rhs(i, j, 0) = (*bc->values)(i, j, 0);
      }
      if (is_dirichlet(bc, i, j, 1) && bc->values) {
        rhs(i, j, 1) = (*bc->values)(i, j, 1);
      }
    }
  }
}

void SSAOperator::apply(const Grid2D& grid, const FieldStag2D<double>& nuH,
                        const FieldStag2D<double>& beta,
                        const FieldStag2D<double>& vel,
                        FieldStag2D<double>& out,
                        const SSABoundaryCondition* bc) const {
  const double dx = grid.dx();
  const double dy = grid.dy();
  const double inv_dx2 = 1.0 / (dx * dx);
  const double inv_dy2 = 1.0 / (dy * dy);
  const double inv_2dx = 1.0 / (2.0 * dx);
  const double inv_2dy = 1.0 / (2.0 * dy);

  const Field2D<double>& u = vel.component(0);
  const Field2D<double>& v = vel.component(1);
  const Field2D<double>& nu_u = nuH.component(0);
  const Field2D<double>& nu_v = nuH.component(1);
  const Field2D<double>& beta_u = beta.component(0);
  const Field2D<double>& beta_v = beta.component(1);

  auto shear = [&](int i, int j) {
    const double du_dy = (u(i, j + 1) - u(i, j - 1)) * inv_2dy;
    const double dv_dx = (v(i + 1, j) - v(i - 1, j)) * inv_2dx;
    return du_dy + dv_dx;
  };

  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      if (is_dirichlet(bc, i, j, 0)) {
        out(i, j, 0) = u(i, j);
      } else {
        const double u_c = u(i, j);
        const double flux_x =
            nu_u(i + 1, j) * (u(i + 1, j) - u_c) -
            nu_u(i - 1, j) * (u_c - u(i - 1, j));
        const double flux_y =
            nu_u(i, j + 1) * (u(i, j + 1) - u_c) -
            nu_u(i, j - 1) * (u_c - u(i, j - 1));
        double coupling = 0.0;
        if (j >= 1 && j <= grid.local_my() - 2) {
          const double shear_p = shear(i, j + 1);
          const double shear_m = shear(i, j - 1);
          coupling = nu_u(i, j) * (shear_p - shear_m) * inv_2dy;
        }
        out(i, j, 0) = flux_x * inv_dx2 + flux_y * inv_dy2 + coupling +
                       beta_u(i, j) * u_c;
      }

      if (is_dirichlet(bc, i, j, 1)) {
        out(i, j, 1) = v(i, j);
      } else {
        const double v_c = v(i, j);
        const double flux_x =
            nu_v(i + 1, j) * (v(i + 1, j) - v_c) -
            nu_v(i - 1, j) * (v_c - v(i - 1, j));
        const double flux_y =
            nu_v(i, j + 1) * (v(i, j + 1) - v_c) -
            nu_v(i, j - 1) * (v_c - v(i, j - 1));
        double coupling = 0.0;
        if (i >= 1 && i <= grid.local_mx() - 2) {
          const double shear_p = shear(i + 1, j);
          const double shear_m = shear(i - 1, j);
          coupling = nu_v(i, j) * (shear_p - shear_m) * inv_2dx;
        }
        out(i, j, 1) = flux_x * inv_dx2 + flux_y * inv_dy2 + coupling +
                       beta_v(i, j) * v_c;
      }
    }
  }
}

}  // namespace gpism
