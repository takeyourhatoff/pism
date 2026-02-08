#include "gpism/viscosity.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "gpism/config.h"

#if GPISM_HAVE_CUDA
namespace gpism {
void viscosity_compute_nuH_cuda(int mx, int my, int gw, int stride_thk,
                                int stride_u, int stride_v, int stride_nuH_u,
                                int stride_nuH_v, const double* thk,
                                const double* u, const double* v, double* nuH_u,
                                double* nuH_v, double nuH_regularization,
                                double strength_extension_nu,
                                double strength_extension_min_thickness,
                                const double* enthalpy, int nz, int enthalpy_gw,
                                int enthalpy_stride, double enthalpy_gamma,
                                double enthalpy_ref, double B, double n_eff,
                                double eps0, double inv_dx, double inv_dy);
}  // namespace gpism
#endif

namespace gpism {
namespace {

double clamp_positive(double value, double floor) {
  return value < floor ? floor : value;
}

double center_from_faces(double left, double right) {
  return 0.5 * (left + right);
}

}  // namespace

ViscosityModel::ViscosityModel(double A, double n, double eps0,
                               double enhancement)
    : A_(A), n_(n), eps0_(eps0), enhancement_(enhancement) {}

void ViscosityModel::compute_nuH(const Grid2D& grid, const Field2D<double>& thk,
                                 const FieldStag2D<double>& vel,
                                 FieldStag2D<double>& nuH,
                                 double nuH_regularization,
                                 double strength_extension_nu,
                                 double strength_extension_min_thickness,
                                 const Field3D<double>* enthalpy,
                                 double enthalpy_gamma,
                                 double enthalpy_ref) const {
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  const double dx = grid.dx();
  const double dy = grid.dy();
  const double inv_dx = 1.0 / dx;
  const double inv_dy = 1.0 / dy;

#if GPISM_HAVE_CUDA
  if (thk.has_device_data() && vel.component(0).has_device_data() &&
      vel.component(1).has_device_data() && nuH.component(0).has_device_data() &&
      nuH.component(1).has_device_data()) {
    const double enhancement = clamp_positive(enhancement_, 1.0);
    const double A_eff = clamp_positive(A_ * enhancement, 1e-20);
    const double n_eff = clamp_positive(n_, 1.0);
    const double B = std::pow(2.0 * A_eff, -1.0 / n_eff);
    const bool use_temp = enthalpy && enthalpy->has_device_data() &&
                          enthalpy_gamma != 0.0;
    const double* enthalpy_ptr = use_temp ? enthalpy->device_data() : nullptr;
    const int nz = use_temp ? enthalpy->local_mz() : 0;
    const int enthalpy_gw = use_temp ? enthalpy->ghost_width() : 0;
    const int enthalpy_stride = use_temp ? enthalpy->stride() : 0;
    viscosity_compute_nuH_cuda(
        mx, my, thk.ghost_width(), thk.stride(), vel.component(0).stride(),
        vel.component(1).stride(), nuH.component(0).stride(),
        nuH.component(1).stride(), thk.device_data(),
        vel.component(0).device_data(), vel.component(1).device_data(),
        nuH.component(0).device_data(), nuH.component(1).device_data(),
        nuH_regularization, strength_extension_nu,
        strength_extension_min_thickness, enthalpy_ptr, nz, enthalpy_gw,
        enthalpy_stride, enthalpy_gamma, enthalpy_ref, B, n_eff, eps0_, inv_dx,
        inv_dy);
    return;
  }
#endif

  std::vector<double> u_center(static_cast<std::size_t>(mx) * my);
  std::vector<double> v_center(static_cast<std::size_t>(mx) * my);
  std::vector<double> nu_center(static_cast<std::size_t>(mx) * my);
  std::vector<double> temp_avg;
  if (enthalpy && enthalpy_gamma != 0.0) {
    temp_avg.assign(static_cast<std::size_t>(mx) * my, 0.0);
  }

  auto idx = [mx](int i, int j) { return static_cast<std::size_t>(j * mx + i); };

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const double u_left = (i == 0) ? vel(i, j, 0) : vel(i - 1, j, 0);
      const double u_right = vel(i, j, 0);
      const double v_down = (j == 0) ? vel(i, j, 1) : vel(i, j - 1, 1);
      const double v_up = vel(i, j, 1);
      u_center[idx(i, j)] = center_from_faces(u_left, u_right);
      v_center[idx(i, j)] = center_from_faces(v_down, v_up);
      if (!temp_avg.empty()) {
        double sum = 0.0;
        for (int k = 0; k < enthalpy->local_mz(); ++k) {
          sum += (*enthalpy)(i, j, k);
        }
        temp_avg[idx(i, j)] = sum / static_cast<double>(enthalpy->local_mz());
      }
    }
  }

  const double enhancement = clamp_positive(enhancement_, 1.0);
  const double A_eff = clamp_positive(A_ * enhancement, 1e-20);
  const double n_eff = clamp_positive(n_, 1.0);
  const double B = std::pow(2.0 * A_eff, -1.0 / n_eff);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const int il = (i == 0) ? i : i - 1;
      const int ir = (i == mx - 1) ? i : i + 1;
      const int jd = (j == 0) ? j : j - 1;
      const int ju = (j == my - 1) ? j : j + 1;

      const double du_dx = (u_center[idx(ir, j)] - u_center[idx(il, j)]) *
                           ((i == 0 || i == mx - 1) ? inv_dx : 0.5 * inv_dx);
      const double dv_dy = (v_center[idx(i, ju)] - v_center[idx(i, jd)]) *
                           ((j == 0 || j == my - 1) ? inv_dy : 0.5 * inv_dy);
      const double du_dy = (u_center[idx(i, ju)] - u_center[idx(i, jd)]) *
                           ((j == 0 || j == my - 1) ? inv_dy : 0.5 * inv_dy);
      const double dv_dx = (v_center[idx(ir, j)] - v_center[idx(il, j)]) *
                           ((i == 0 || i == mx - 1) ? inv_dx : 0.5 * inv_dx);

      const double eps_xx = du_dx;
      const double eps_yy = dv_dy;
      const double eps_xy = 0.5 * (du_dy + dv_dx);
      // Second invariant of the strain rate tensor in 2D SSA:
      // eps_II^2 = eps_xx^2 + eps_yy^2 + eps_xx*eps_yy + eps_xy^2
      const double eps2 = std::max(
          0.0, eps_xx * eps_xx + eps_yy * eps_yy + eps_xx * eps_yy +
                   eps_xy * eps_xy);
      const double eps_e = std::sqrt(eps2 + eps0_ * eps0_);

      double nu = 0.5 * B * std::pow(eps_e, (1.0 / n_eff) - 1.0);
      if (!temp_avg.empty()) {
        const double temp = temp_avg[idx(i, j)];
        const double scale = std::exp(-enthalpy_gamma * (temp - enthalpy_ref));
        nu *= scale;
      }
      nu_center[idx(i, j)] = nu;
    }
  }

  const double nu_reg = std::max(0.0, nuH_regularization);
  const double nu_ext = strength_extension_nu;
  const double H_ext_min = std::max(0.0, strength_extension_min_thickness);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const int ir = (i == mx - 1) ? i : i + 1;
      const int ju = (j == my - 1) ? j : j + 1;

      // nuH(i,j,0) is located on the u-staggered grid interface between
      // cell centers (i,j) and (i+1,j).
      const double nu_left = nu_center[idx(i, j)];
      const double nu_right = nu_center[idx(ir, j)];
      const double H_left = thk(i, j);
      const double H_right = thk(ir, j);
      double H_face = 0.5 * (H_left + H_right);
      double nu_face = 0.5 * (nu_left + nu_right);
      if (nu_ext > 0.0 && H_face < H_ext_min) {
        H_face = std::max(H_face, H_ext_min);
        nu_face = nu_ext;
      }
      nuH(i, j, 0) = nu_face * H_face + nu_reg;

      // nuH(i,j,1) is located on the v-staggered grid interface between
      // cell centers (i,j) and (i,j+1).
      const double nu_down = nu_center[idx(i, j)];
      const double nu_up = nu_center[idx(i, ju)];
      const double H_down = thk(i, j);
      const double H_up = thk(i, ju);
      H_face = 0.5 * (H_down + H_up);
      nu_face = 0.5 * (nu_down + nu_up);
      if (nu_ext > 0.0 && H_face < H_ext_min) {
        H_face = std::max(H_face, H_ext_min);
        nu_face = nu_ext;
      }
      nuH(i, j, 1) = nu_face * H_face + nu_reg;
    }
  }
}

}  // namespace gpism
