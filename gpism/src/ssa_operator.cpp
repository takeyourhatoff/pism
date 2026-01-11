#include "gpism/ssa_operator.h"

#include <algorithm>

#include "gpism/config.h"

#if GPISM_HAVE_CUDA
namespace gpism {
void ssa_compute_basal_drag_cuda(int mx, int my, int gw, int stride,
                                 const double* tauc, double* beta_u,
                                 double* beta_v, double denom);
void ssa_assemble_rhs_cuda(int mx, int my, int gw, int stride_thk,
                           int stride_dhdx, int stride_dhdy, int stride_rhs,
                           const double* thk, const double* dhdx,
                           const double* dhdy, double* rhs_u, double* rhs_v,
                           double scale, const int* mask_u,
                           const int* mask_v, const double* bc_u,
                           const double* bc_v, int has_bc);
void ssa_apply_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                    int stride_nu_u, int stride_nu_v, int stride_beta_u,
                    int stride_beta_v, int stride_out_u, int stride_out_v,
                    const double* u, const double* v, const double* nu_u,
                    const double* nu_v, const double* beta_u,
                    const double* beta_v, double* out_u, double* out_v,
                    double inv_dx2, double inv_dy2, double inv_2dx,
                    double inv_2dy, const int* mask_u, const int* mask_v,
                    int has_bc);
}  // namespace gpism
#endif

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
#if GPISM_HAVE_CUDA
  if (tauc.has_device_data() && beta.component(0).has_device_data() &&
      beta.component(1).has_device_data()) {
    const double denom = std::max(u_threshold_, 1e-6);
    ssa_compute_basal_drag_cuda(grid.local_mx(), grid.local_my(),
                                tauc.ghost_width(), tauc.stride(),
                                tauc.device_data(),
                                beta.component(0).device_data(),
                                beta.component(1).device_data(), denom);
    return;
  }
#endif
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
#if GPISM_HAVE_CUDA
  const bool has_bc = bc && bc->mask && bc->values;
  if (thk.has_device_data() && dhdx.has_device_data() && dhdy.has_device_data() &&
      rhs.component(0).has_device_data() && rhs.component(1).has_device_data() &&
      (!has_bc ||
       (bc->mask->component(0).has_device_data() &&
        bc->mask->component(1).has_device_data() &&
        bc->values->component(0).has_device_data() &&
        bc->values->component(1).has_device_data()))) {
    ssa_assemble_rhs_cuda(
        grid.local_mx(), grid.local_my(), thk.ghost_width(), thk.stride(),
        dhdx.stride(), dhdy.stride(), rhs.component(0).stride(),
        thk.device_data(), dhdx.device_data(), dhdy.device_data(),
        rhs.component(0).device_data(), rhs.component(1).device_data(),
        rho_ * g_,
        has_bc ? bc->mask->component(0).device_data() : nullptr,
        has_bc ? bc->mask->component(1).device_data() : nullptr,
        has_bc ? bc->values->component(0).device_data() : nullptr,
        has_bc ? bc->values->component(1).device_data() : nullptr,
        has_bc ? 1 : 0);
    return;
  }
#endif
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

#if GPISM_HAVE_CUDA
  const bool has_bc = bc && bc->mask && bc->values;
  if (nuH.component(0).has_device_data() && nuH.component(1).has_device_data() &&
      beta.component(0).has_device_data() && beta.component(1).has_device_data() &&
      vel.component(0).has_device_data() && vel.component(1).has_device_data() &&
      out.component(0).has_device_data() && out.component(1).has_device_data() &&
      (!has_bc ||
       (bc->mask->component(0).has_device_data() &&
        bc->mask->component(1).has_device_data() &&
        bc->values->component(0).has_device_data() &&
        bc->values->component(1).has_device_data()))) {
    ssa_apply_cuda(
        grid.local_mx(), grid.local_my(), vel.component(0).ghost_width(),
        vel.component(0).stride(), vel.component(1).stride(),
        nuH.component(0).stride(), nuH.component(1).stride(),
        beta.component(0).stride(), beta.component(1).stride(),
        out.component(0).stride(), out.component(1).stride(),
        vel.component(0).device_data(), vel.component(1).device_data(),
        nuH.component(0).device_data(), nuH.component(1).device_data(),
        beta.component(0).device_data(), beta.component(1).device_data(),
        out.component(0).device_data(), out.component(1).device_data(),
        inv_dx2, inv_dy2, inv_2dx, inv_2dy,
        has_bc ? bc->mask->component(0).device_data() : nullptr,
        has_bc ? bc->mask->component(1).device_data() : nullptr,
        has_bc ? 1 : 0);
    return;
  }
#endif

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
