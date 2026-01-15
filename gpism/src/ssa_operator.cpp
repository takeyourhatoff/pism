#include "gpism/ssa_operator.h"

#include "gpism/field_sync.h"
#include "gpism/linear_algebra.h"

#include <algorithm>
#include <cmath>

#include "gpism/config.h"
#include "gpism/geometry.h"

#if GPISM_HAVE_CUDA
namespace gpism {
void ssa_compute_basal_drag_cuda(int mx, int my, int gw, int stride_tauc,
                                 int stride_u, int stride_v,
                                 int stride_mask, const double* tauc,
                                 const double* u_center,
                                 const double* v_center,
                                 const int* cell_type, double* beta_u,
                                 double* beta_v, double q,
                                 double u_threshold,
                                 double plastic_regularization,
                                 double sliding_scale_factor,
                                 double beta_ice_free_bedrock,
                                 int pseudo_plastic);
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
                    double inv_2dy, int stride_mask_u, int stride_mask_v,
                    const int* mask_u, const int* mask_v, int has_bc);
void ssa_apply_region_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                           int stride_nu_u, int stride_nu_v, int stride_beta_u,
                           int stride_beta_v, int stride_out_u, int stride_out_v,
                           const double* u, const double* v, const double* nu_u,
                           const double* nu_v, const double* beta_u,
                           const double* beta_v, double* out_u, double* out_v,
                           double inv_dx2, double inv_dy2, double inv_2dx,
                           double inv_2dy, int stride_mask_u, int stride_mask_v,
                           const int* mask_u, const int* mask_v, int has_bc,
                           int i_start, int i_end, int j_start, int j_end);
void ssa_replace_zero_diagonal_entries_cuda(
    int mx, int my, int gw, int stride_nu_u, int stride_nu_v,
    int stride_beta_u, int stride_beta_v, double* beta_u, double* beta_v,
    const double* nu_u, const double* nu_v, double inv_dx2, double inv_dy2,
    int stride_mask_u, int stride_mask_v, const int* mask_u,
    const int* mask_v, int has_bc, double beta_ice_free_bedrock);
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

SSAOperator::SSAOperator(double rho, double g) : rho_(rho), g_(g) {}

void SSAOperator::compute_basal_drag(const Grid2D& grid,
                                     const Field2D<double>& tauc,
                                     const Field2D<double>& u_center,
                                     const Field2D<double>& v_center,
                                     const Field2D<int>& cell_type,
                                     FieldStag2D<double>& beta,
                                     const BasalResistanceParams& params) const {
#if GPISM_HAVE_CUDA
  if (tauc.has_device_data() && u_center.has_device_data() &&
      v_center.has_device_data() && cell_type.has_device_data() &&
      beta.component(0).has_device_data() &&
      beta.component(1).has_device_data()) {
    ssa_compute_basal_drag_cuda(
        grid.local_mx(), grid.local_my(), tauc.ghost_width(), tauc.stride(),
        u_center.stride(), v_center.stride(), cell_type.stride(),
        tauc.device_data(), u_center.device_data(), v_center.device_data(),
        cell_type.device_data(), beta.component(0).device_data(),
        beta.component(1).device_data(), params.q, params.u_threshold,
        params.plastic_regularization, params.sliding_scale_factor,
        params.beta_ice_free_bedrock,
        params.law == BasalResistanceLaw::PseudoPlastic ? 1 : 0);
    return;
  }
#endif

  const double reg = params.plastic_regularization;
  const double q = params.q;
  const double u_threshold = std::max(params.u_threshold, 1e-6);
  const double u_threshold_factor = std::pow(u_threshold, -q);
  const double Aq =
      (params.sliding_scale_factor > 0.0)
          ? std::pow(params.sliding_scale_factor, q)
          : 1.0;

  auto beta_center = [&](int i, int j) {
    const int mask = cell_type(i, j);
    if (mask == IceFreeBedrock) {
      return params.beta_ice_free_bedrock;
    }
    if (mask == IceFreeOcean || mask == FloatingIce) {
      return 0.0;
    }
    const double u = u_center(i, j);
    const double v = v_center(i, j);
    const double mag2 = reg * reg + u * u + v * v;
    if (params.law == BasalResistanceLaw::PseudoPlastic) {
      return (tauc(i, j) / Aq) * std::pow(mag2, 0.5 * (q - 1.0)) *
             u_threshold_factor;
    }
    return tauc(i, j) / std::sqrt(mag2);
  };

  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      const int ie = (i == grid.local_mx() - 1) ? i : i + 1;
      const int jn = (j == grid.local_my() - 1) ? j : j + 1;

      const int mask_c = cell_type(i, j);
      const int mask_e = cell_type(ie, j);
      const int mask_n = cell_type(i, jn);

      const double beta_c = beta_center(i, j);
      const double beta_e = beta_center(ie, j);
      const double beta_n = beta_center(i, jn);

      if (mask_c == IceFreeBedrock || mask_e == IceFreeBedrock) {
        beta(i, j, 0) = params.beta_ice_free_bedrock;
      } else {
        beta(i, j, 0) = 0.5 * (beta_c + beta_e);
      }

      if (mask_c == IceFreeBedrock || mask_n == IceFreeBedrock) {
        beta(i, j, 1) = params.beta_ice_free_bedrock;
      } else {
        beta(i, j, 1) = 0.5 * (beta_c + beta_n);
      }
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
        -rho_ * g_,
        has_bc ? bc->mask->component(0).device_data() : nullptr,
        has_bc ? bc->mask->component(1).device_data() : nullptr,
        has_bc ? bc->values->component(0).device_data() : nullptr,
        has_bc ? bc->values->component(1).device_data() : nullptr,
        has_bc ? 1 : 0);
    return;
  }
#endif
  const double scale = -rho_ * g_;
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

void SSAOperator::apply_region(const Grid2D& grid,
                               const FieldStag2D<double>& nuH,
                               const FieldStag2D<double>& beta,
                               const FieldStag2D<double>& vel,
                               FieldStag2D<double>& out, int i_start,
                               int i_end, int j_start, int j_end,
                               const SSABoundaryCondition* bc) const {
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  const int i0 = std::max(0, i_start);
  const int i1 = std::min(mx, i_end);
  const int j0 = std::max(0, j_start);
  const int j1 = std::min(my, j_end);
  if (i0 >= i1 || j0 >= j1) {
    return;
  }
  const double dx = grid.dx();
  const double dy = grid.dy();
  const double inv_dx2 = 1.0 / (dx * dx);
  const double inv_dy2 = 1.0 / (dy * dy);
  const double inv_2dx = 1.0 / (2.0 * dx);
  const double inv_2dy = 1.0 / (2.0 * dy);
  const double inv_d4 = 1.0 / (4.0 * dx * dy);
  const double inv_d2 = 2.0 * inv_d4;

#if GPISM_HAVE_CUDA
  const bool has_bc = bc && bc->mask && bc->values;
  const bool deterministic = deterministic_reductions_enabled();
  if (deterministic) {
    if (vel.component(0).has_device_data() && vel.component(1).has_device_data()) {
      sync_device_to_host(const_cast<FieldStag2D<double>&>(vel));
    }
    if (nuH.component(0).has_device_data() && nuH.component(1).has_device_data()) {
      sync_device_to_host(const_cast<FieldStag2D<double>&>(nuH));
    }
    if (beta.component(0).has_device_data() && beta.component(1).has_device_data()) {
      sync_device_to_host(const_cast<FieldStag2D<double>&>(beta));
    }
  }
  if (!deterministic && nuH.component(0).has_device_data() &&
      nuH.component(1).has_device_data() &&
      beta.component(0).has_device_data() && beta.component(1).has_device_data() &&
      vel.component(0).has_device_data() && vel.component(1).has_device_data() &&
      out.component(0).has_device_data() && out.component(1).has_device_data() &&
      (!has_bc ||
       (bc->mask->component(0).has_device_data() &&
        bc->mask->component(1).has_device_data() &&
        bc->values->component(0).has_device_data() &&
        bc->values->component(1).has_device_data()))) {
    ssa_apply_region_cuda(
        mx, my, vel.component(0).ghost_width(),
        vel.component(0).stride(), vel.component(1).stride(),
        nuH.component(0).stride(), nuH.component(1).stride(),
        beta.component(0).stride(), beta.component(1).stride(),
        out.component(0).stride(), out.component(1).stride(),
        vel.component(0).device_data(), vel.component(1).device_data(),
        nuH.component(0).device_data(), nuH.component(1).device_data(),
        beta.component(0).device_data(), beta.component(1).device_data(),
        out.component(0).device_data(), out.component(1).device_data(),
        inv_dx2, inv_dy2, inv_2dx, inv_2dy,
        has_bc ? bc->mask->component(0).stride() : 0,
        has_bc ? bc->mask->component(1).stride() : 0,
        has_bc ? bc->mask->component(0).device_data() : nullptr,
        has_bc ? bc->mask->component(1).device_data() : nullptr,
        has_bc ? 1 : 0, i0, i1, j0, j1);
    return;
  }
#endif

  const Field2D<double>& u = vel.component(0);
  const Field2D<double>& v = vel.component(1);
  const Field2D<double>& nu_u = nuH.component(0);
  const Field2D<double>& nu_v = nuH.component(1);
  const Field2D<double>& beta_u = beta.component(0);
  const Field2D<double>& beta_v = beta.component(1);

  for (int j = j0; j < j1; ++j) {
    for (int i = i0; i < i1; ++i) {
      const int im1 = (i == 0) ? i : i - 1;
      const int ip1 = (i == mx - 1) ? i : i + 1;
      const int jm1 = (j == 0) ? j : j - 1;
      const int jp1 = (j == my - 1) ? j : j + 1;
      if (is_dirichlet(bc, i, j, 0)) {
        out(i, j, 0) = u(i, j);
      } else {
        const double u_c = u(i, j);
        const double c_n = nu_v(i, j);
        const double c_s = nu_v(i, jm1);
        const double c_e = nu_u(i, j);
        const double c_w = nu_u(im1, j);
        double sum =
            (-c_n * u(i, jp1) - c_s * u(i, jm1) +
             (c_n + c_s) * u_c) *
                inv_dy2 +
            (-4.0 * c_e * u(ip1, j) - 4.0 * c_w * u(im1, j) +
             4.0 * (c_e + c_w) * u_c) *
                inv_dx2;
        sum += (c_w * inv_d2 + c_n * inv_d4) * v(im1, jp1);
        sum += (c_w - c_e) * inv_d2 * v(i, jp1);
        sum += (-c_e * inv_d2 - c_n * inv_d4) * v(ip1, jp1);
        sum += (c_n - c_s) * inv_d4 * v(im1, j);
        sum += (c_s - c_n) * inv_d4 * v(ip1, j);
        sum += (-c_w * inv_d2 - c_s * inv_d4) * v(im1, jm1);
        sum += (c_e - c_w) * inv_d2 * v(i, jm1);
        sum += (c_e * inv_d2 + c_s * inv_d4) * v(ip1, jm1);
        out(i, j, 0) = sum + beta_u(i, j) * u_c;
      }

      if (is_dirichlet(bc, i, j, 1)) {
        out(i, j, 1) = v(i, j);
      } else {
        const double v_c = v(i, j);
        const double c_n = nu_v(i, j);
        const double c_s = nu_v(i, jm1);
        const double c_e = nu_u(i, j);
        const double c_w = nu_u(im1, j);
        double sum =
            (-4.0 * c_n * v(i, jp1) - 4.0 * c_s * v(i, jm1) +
             4.0 * (c_n + c_s) * v_c) *
                inv_dy2 +
            (-c_e * v(ip1, j) - c_w * v(im1, j) +
             (c_e + c_w) * v_c) *
                inv_dx2;
        sum += (c_w * inv_d4 + c_n * inv_d2) * u(im1, jp1);
        sum += (c_w - c_e) * inv_d4 * u(i, jp1);
        sum += (-c_e * inv_d4 - c_n * inv_d2) * u(ip1, jp1);
        sum += (c_n - c_s) * inv_d2 * u(im1, j);
        sum += (c_s - c_n) * inv_d2 * u(ip1, j);
        sum += (-c_w * inv_d4 - c_s * inv_d2) * u(im1, jm1);
        sum += (c_e - c_w) * inv_d4 * u(i, jm1);
        sum += (c_e * inv_d4 + c_s * inv_d2) * u(ip1, jm1);
        out(i, j, 1) = sum + beta_v(i, j) * v_c;
      }
    }
  }

#if GPISM_HAVE_CUDA
  if (deterministic && out.component(0).has_device_data() &&
      out.component(1).has_device_data()) {
    sync_host_to_device(out);
  }
#endif
}

void SSAOperator::apply(const Grid2D& grid, const FieldStag2D<double>& nuH,
                        const FieldStag2D<double>& beta,
                        const FieldStag2D<double>& vel,
                        FieldStag2D<double>& out,
                        const SSABoundaryCondition* bc) const {
  apply_region(grid, nuH, beta, vel, out, 0, grid.local_mx(), 0,
               grid.local_my(), bc);
}

void SSAOperator::replace_zero_diagonal_entries(
    const Grid2D& grid, const FieldStag2D<double>& nuH,
    FieldStag2D<double>& beta, const SSABoundaryCondition* bc,
    double beta_ice_free_bedrock) const {
  if (beta_ice_free_bedrock <= 0.0) {
    return;
  }
#if GPISM_HAVE_CUDA
  const bool has_bc = bc && bc->mask;
  if (!deterministic_reductions_enabled() &&
      nuH.component(0).has_device_data() &&
      nuH.component(1).has_device_data() &&
      beta.component(0).has_device_data() &&
      beta.component(1).has_device_data() &&
      (!has_bc ||
       (bc->mask->component(0).has_device_data() &&
        bc->mask->component(1).has_device_data()))) {
    ssa_replace_zero_diagonal_entries_cuda(
        grid.local_mx(), grid.local_my(), grid.ghost_width(),
        nuH.component(0).stride(), nuH.component(1).stride(),
        beta.component(0).stride(), beta.component(1).stride(),
        beta.component(0).device_data(), beta.component(1).device_data(),
        nuH.component(0).device_data(), nuH.component(1).device_data(),
        1.0 / (grid.dx() * grid.dx()), 1.0 / (grid.dy() * grid.dy()),
        has_bc ? bc->mask->component(0).stride() : 0,
        has_bc ? bc->mask->component(1).stride() : 0,
        has_bc ? bc->mask->component(0).device_data() : nullptr,
        has_bc ? bc->mask->component(1).device_data() : nullptr,
        has_bc ? 1 : 0, beta_ice_free_bedrock);
    return;
  }
#endif

  const double inv_dx2 = 1.0 / (grid.dx() * grid.dx());
  const double inv_dy2 = 1.0 / (grid.dy() * grid.dy());
  const double eps = 1e-16;
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  for (int j = 0; j < my; ++j) {
    const int jm1 = (j == 0) ? j : j - 1;
    for (int i = 0; i < mx; ++i) {
      const int im1 = (i == 0) ? i : i - 1;
      const double c_n = nuH(i, j, 1);
      const double c_s = nuH(i, jm1, 1);
      const double c_e = nuH(i, j, 0);
      const double c_w = nuH(im1, j, 0);
      const double diag_u =
          beta(i, j, 0) + (c_n + c_s) * inv_dy2 + 4.0 * (c_e + c_w) * inv_dx2;
      const double diag_v =
          beta(i, j, 1) + 4.0 * (c_n + c_s) * inv_dy2 + (c_e + c_w) * inv_dx2;
      if (!is_dirichlet(bc, i, j, 0) && std::abs(diag_u) < eps) {
        beta(i, j, 0) = beta_ice_free_bedrock;
      }
      if (!is_dirichlet(bc, i, j, 1) && std::abs(diag_v) < eps) {
        beta(i, j, 1) = beta_ice_free_bedrock;
      }
    }
  }
}

}  // namespace gpism
