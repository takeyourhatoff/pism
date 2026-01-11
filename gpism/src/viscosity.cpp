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
                                double* nuH_v, double B, double n_eff,
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

ViscosityModel::ViscosityModel(double A, double n, double eps0)
    : A_(A), n_(n), eps0_(eps0) {}

void ViscosityModel::compute_nuH(const Grid2D& grid, const Field2D<double>& thk,
                                 const FieldStag2D<double>& vel,
                                 FieldStag2D<double>& nuH) const {
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
    const double A_eff = clamp_positive(A_, 1e-20);
    const double n_eff = clamp_positive(n_, 1.0);
    const double B = std::pow(2.0 * A_eff, -1.0 / n_eff);
    viscosity_compute_nuH_cuda(mx, my, thk.ghost_width(), thk.stride(),
                               vel.component(0).stride(),
                               vel.component(1).stride(),
                               nuH.component(0).stride(),
                               nuH.component(1).stride(), thk.device_data(),
                               vel.component(0).device_data(),
                               vel.component(1).device_data(),
                               nuH.component(0).device_data(),
                               nuH.component(1).device_data(), B, n_eff, eps0_,
                               inv_dx, inv_dy);
    return;
  }
#endif

  std::vector<double> u_center(static_cast<std::size_t>(mx) * my);
  std::vector<double> v_center(static_cast<std::size_t>(mx) * my);
  std::vector<double> nu_center(static_cast<std::size_t>(mx) * my);

  auto idx = [mx](int i, int j) { return static_cast<std::size_t>(j * mx + i); };

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const double u_left = (i == 0) ? vel(i, j, 0) : vel(i - 1, j, 0);
      const double u_right = vel(i, j, 0);
      const double v_down = (j == 0) ? vel(i, j, 1) : vel(i, j - 1, 1);
      const double v_up = vel(i, j, 1);
      u_center[idx(i, j)] = center_from_faces(u_left, u_right);
      v_center[idx(i, j)] = center_from_faces(v_down, v_up);
    }
  }

  const double A_eff = clamp_positive(A_, 1e-20);
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
      const double eps2 =
          0.5 * (eps_xx * eps_xx + eps_yy * eps_yy) + eps_xy * eps_xy;
      const double eps_e = std::sqrt(eps2 + eps0_ * eps0_);

      const double nu = 0.5 * B * std::pow(eps_e, (1.0 / n_eff) - 1.0);
      nu_center[idx(i, j)] = nu;
    }
  }

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const int il = (i == 0) ? i : i - 1;
      const int jd = (j == 0) ? j : j - 1;

      const double nu_left = nu_center[idx(il, j)];
      const double nu_right = nu_center[idx(i, j)];
      const double H_left = thk(il, j);
      const double H_right = thk(i, j);
      nuH(i, j, 0) = 0.5 * (nu_left + nu_right) * 0.5 * (H_left + H_right);

      const double nu_down = nu_center[idx(i, jd)];
      const double nu_up = nu_center[idx(i, j)];
      const double H_down = thk(i, jd);
      const double H_up = thk(i, j);
      nuH(i, j, 1) = 0.5 * (nu_down + nu_up) * 0.5 * (H_down + H_up);
    }
  }
}

}  // namespace gpism
