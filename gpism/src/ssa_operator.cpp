#include "gpism/ssa_operator.h"

#include <algorithm>
#include <stdexcept>

#include "gpism/geometry.h"

namespace gpism {
void ssa_compute_basal_drag_cuda(int mx, int my, int gw, int stride_tauc,
                                 int stride_u, int stride_v,
                                 int stride_topg, int stride_usurf,
                                 int stride_mask, const double* tauc,
                                 const double* u_center,
                                 const double* v_center,
                                 const double* topg,
                                 const double* usurf,
                                 const int* cell_type, double* beta_u,
                                 double* beta_v, double q,
                                 double u_threshold,
                                 double plastic_regularization,
                                 double sliding_scale_factor,
                                 double beta_ice_free_bedrock,
                                 double beta_lateral_margin,
                                 int pseudo_plastic,
                                 int periodic);
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
                    const int* mask_u, const int* mask_v, int has_bc,
                    int periodic);
void ssa_apply_region_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                           int stride_nu_u, int stride_nu_v, int stride_beta_u,
                           int stride_beta_v, int stride_out_u, int stride_out_v,
                           const double* u, const double* v, const double* nu_u,
                           const double* nu_v, const double* beta_u,
                           const double* beta_v, double* out_u, double* out_v,
                           double inv_dx2, double inv_dy2, double inv_2dx,
                           double inv_2dy, int stride_mask_u, int stride_mask_v,
                           const int* mask_u, const int* mask_v, int has_bc,
                           int i_start, int i_end, int j_start, int j_end,
                           int periodic);
void ssa_replace_zero_diagonal_entries_cuda(
    int mx, int my, int gw, int stride_nu_u, int stride_nu_v,
    int stride_beta_u, int stride_beta_v, double* beta_u, double* beta_v,
    const double* nu_u, const double* nu_v, double inv_dx2, double inv_dy2,
    int stride_mask_u, int stride_mask_v, const int* mask_u,
    const int* mask_v, int has_bc, double beta_ice_free_bedrock,
    int periodic);
}  // namespace gpism

namespace gpism {

SSAOperator::SSAOperator(double rho, double g) : rho_(rho), g_(g) {}

void SSAOperator::compute_basal_drag(const Grid2D& grid,
                                     const Field2D<double>& tauc,
                                     const Field2D<double>& u_center,
                                     const Field2D<double>& v_center,
                                     const Field2D<double>& topg,
                                     const Field2D<double>& usurf,
                                     const Field2D<int>& cell_type,
                                     FieldStag2D<double>& beta,
                                     const BasalResistanceParams& params) const {
  if (!(tauc.has_device_data() && u_center.has_device_data() &&
        v_center.has_device_data() && topg.has_device_data() &&
        usurf.has_device_data() && cell_type.has_device_data() &&
        beta.component(0).has_device_data() &&
        beta.component(1).has_device_data())) {
    throw std::runtime_error(
        "SSAOperator::compute_basal_drag requires device-resident fields");
  }

  const int periodic = (grid.dims_x() == 1 && grid.dims_y() == 1) ? 1 : 0;
  ssa_compute_basal_drag_cuda(
      grid.local_mx(), grid.local_my(), tauc.ghost_width(), tauc.stride(),
      u_center.stride(), v_center.stride(), topg.stride(), usurf.stride(),
      cell_type.stride(), tauc.device_data(), u_center.device_data(),
      v_center.device_data(), topg.device_data(), usurf.device_data(),
      cell_type.device_data(), beta.component(0).device_data(),
      beta.component(1).device_data(), params.q, params.u_threshold,
      params.plastic_regularization, params.sliding_scale_factor,
      params.beta_ice_free_bedrock, params.beta_lateral_margin,
      params.law == BasalResistanceLaw::PseudoPlastic ? 1 : 0, periodic);
}

void SSAOperator::assemble_rhs(const Grid2D& grid, const Field2D<double>& thk,
                               const Field2D<double>& dhdx,
                               const Field2D<double>& dhdy,
                               FieldStag2D<double>& rhs,
                               const SSABoundaryCondition* bc) const {
  const bool has_bc = bc && bc->mask && bc->values;
  if (!(thk.has_device_data() && dhdx.has_device_data() &&
        dhdy.has_device_data() && rhs.component(0).has_device_data() &&
        rhs.component(1).has_device_data())) {
    throw std::runtime_error(
        "SSAOperator::assemble_rhs requires device-resident fields");
  }
  if (has_bc && !(bc->mask->component(0).has_device_data() &&
                  bc->mask->component(1).has_device_data() &&
                  bc->values->component(0).has_device_data() &&
                  bc->values->component(1).has_device_data())) {
    throw std::runtime_error(
        "SSAOperator::assemble_rhs requires device-resident BC fields");
  }

  ssa_assemble_rhs_cuda(
      grid.local_mx(), grid.local_my(), thk.ghost_width(), thk.stride(),
      dhdx.stride(), dhdy.stride(), rhs.component(0).stride(), thk.device_data(),
      dhdx.device_data(), dhdy.device_data(), rhs.component(0).device_data(),
      rhs.component(1).device_data(), -rho_ * g_,
      has_bc ? bc->mask->component(0).device_data() : nullptr,
      has_bc ? bc->mask->component(1).device_data() : nullptr,
      has_bc ? bc->values->component(0).device_data() : nullptr,
      has_bc ? bc->values->component(1).device_data() : nullptr,
      has_bc ? 1 : 0);
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
  const bool has_bc = bc && bc->mask && bc->values;
  if (!(nuH.component(0).has_device_data() &&
        nuH.component(1).has_device_data() &&
        beta.component(0).has_device_data() &&
        beta.component(1).has_device_data() &&
        vel.component(0).has_device_data() && vel.component(1).has_device_data() &&
        out.component(0).has_device_data() &&
        out.component(1).has_device_data())) {
    throw std::runtime_error(
        "SSAOperator::apply_region requires device-resident fields");
  }
  if (has_bc && !(bc->mask->component(0).has_device_data() &&
                  bc->mask->component(1).has_device_data() &&
                  bc->values->component(0).has_device_data() &&
                  bc->values->component(1).has_device_data())) {
    throw std::runtime_error(
        "SSAOperator::apply_region requires device-resident BC fields");
  }

  const int periodic = (grid.dims_x() == 1 && grid.dims_y() == 1) ? 1 : 0;
  ssa_apply_region_cuda(
      mx, my, vel.component(0).ghost_width(), vel.component(0).stride(),
      vel.component(1).stride(), nuH.component(0).stride(),
      nuH.component(1).stride(), beta.component(0).stride(),
      beta.component(1).stride(), out.component(0).stride(),
      out.component(1).stride(), vel.component(0).device_data(),
      vel.component(1).device_data(), nuH.component(0).device_data(),
      nuH.component(1).device_data(), beta.component(0).device_data(),
      beta.component(1).device_data(), out.component(0).device_data(),
      out.component(1).device_data(), inv_dx2, inv_dy2, inv_2dx, inv_2dy,
      has_bc ? bc->mask->component(0).stride() : 0,
      has_bc ? bc->mask->component(1).stride() : 0,
      has_bc ? bc->mask->component(0).device_data() : nullptr,
      has_bc ? bc->mask->component(1).device_data() : nullptr,
      has_bc ? 1 : 0, i0, i1, j0, j1, periodic);
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
  const bool has_bc = bc && bc->mask;
  if (!(nuH.component(0).has_device_data() &&
        nuH.component(1).has_device_data() &&
        beta.component(0).has_device_data() &&
        beta.component(1).has_device_data())) {
    throw std::runtime_error(
        "SSAOperator::replace_zero_diagonal_entries requires "
        "device-resident fields");
  }
  if (has_bc && !(bc->mask->component(0).has_device_data() &&
                  bc->mask->component(1).has_device_data())) {
    throw std::runtime_error(
        "SSAOperator::replace_zero_diagonal_entries requires "
        "device-resident BC mask");
  }

  const int periodic = (grid.dims_x() == 1 && grid.dims_y() == 1) ? 1 : 0;
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
      has_bc ? 1 : 0, beta_ice_free_bedrock, periodic);
}

}  // namespace gpism
