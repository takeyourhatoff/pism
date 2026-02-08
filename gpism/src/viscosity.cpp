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

  const double nu_reg = std::max(0.0, nuH_regularization);
  const double nu_ext = strength_extension_nu;
  const double H_ext_min = std::max(0.0, strength_extension_min_thickness);

  auto clamp_i = [mx](int i) { return std::max(0, std::min(mx - 1, i)); };
  auto clamp_j = [my](int j) { return std::max(0, std::min(my - 1, j)); };

  auto U = [&](int i, int j) { return u_center[idx(clamp_i(i), clamp_j(j))]; };
  auto V = [&](int i, int j) { return v_center[idx(clamp_i(i), clamp_j(j))]; };

  auto apply_temp_scale = [&](int i, int j, double nu) {
    if (temp_avg.empty()) {
      return nu;
    }
    const double temp = temp_avg[idx(clamp_i(i), clamp_j(j))];
    return nu * std::exp(-enthalpy_gamma * (temp - enthalpy_ref));
  };

  auto viscosity_from_derivs = [&](double u_x, double u_y, double v_x,
                                   double v_y) {
    const double eps_xx = u_x;
    const double eps_yy = v_y;
    const double eps_xy = 0.5 * (u_y + v_x);
    const double eps2 = std::max(
        0.0, eps_xx * eps_xx + eps_yy * eps_yy + eps_xx * eps_yy +
                 eps_xy * eps_xy);
    const double eps_e = std::sqrt(eps2 + eps0_ * eps0_);
    return 0.5 * B * std::pow(eps_e, (1.0 / n_eff) - 1.0);
  };

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const int ip1 = (i == mx - 1) ? i : i + 1;
      const int im1 = (i == 0) ? i : i - 1;
      const int jp1 = (j == my - 1) ? j : j + 1;
      const int jm1 = (j == 0) ? j : j - 1;

      // PISM-style nuH computation on the staggered grid, using cell-centered
      // velocities U,V derived from staggered u/v.
      //
      // o=0 (u-staggered): between (i,j) and (i+1,j)
      {
        const double u_x = (U(ip1, j) - U(i, j)) * inv_dx;
        const double v_x = (V(ip1, j) - V(i, j)) * inv_dx;
        const double u_y = (U(i, jp1) + U(ip1, jp1) - U(i, jm1) - U(ip1, jm1)) *
                           (0.25 * inv_dy);
        const double v_y = (V(i, jp1) + V(ip1, jp1) - V(i, jm1) - V(ip1, jm1)) *
                           (0.25 * inv_dy);

        double nu = apply_temp_scale(i, j, viscosity_from_derivs(u_x, u_y, v_x, v_y));

        double H_face = 0.5 * (thk(i, j) + thk(ip1, j));
        if (nu_ext > 0.0 && H_face < H_ext_min) {
          H_face = std::max(H_face, H_ext_min);
          nu = nu_ext;
        }
        nuH(i, j, 0) = nu * H_face + nu_reg;
      }

      // o=1 (v-staggered): between (i,j) and (i,j+1)
      {
        const double u_y = (U(i, jp1) - U(i, j)) * inv_dy;
        const double v_y = (V(i, jp1) - V(i, j)) * inv_dy;
        const double u_x =
            (U(ip1, j) + U(ip1, jp1) - U(im1, j) - U(im1, jp1)) * (0.25 * inv_dx);
        const double v_x =
            (V(ip1, j) + V(ip1, jp1) - V(im1, j) - V(im1, jp1)) * (0.25 * inv_dx);

        double nu = apply_temp_scale(i, j, viscosity_from_derivs(u_x, u_y, v_x, v_y));

        double H_face = 0.5 * (thk(i, j) + thk(i, jp1));
        if (nu_ext > 0.0 && H_face < H_ext_min) {
          H_face = std::max(H_face, H_ext_min);
          nu = nu_ext;
        }
        nuH(i, j, 1) = nu * H_face + nu_reg;
      }
    }
  }
}

}  // namespace gpism
