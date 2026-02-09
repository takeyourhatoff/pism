#include "gpism/thickness.h"

#include <algorithm>

#include "gpism/config.h"

#if GPISM_HAVE_CUDA
namespace gpism {
void compute_face_fluxes_cuda(int mx, int my, int gw, int stride_thk,
                              int stride_vel_u, int stride_vel_v,
                              int stride_flux_u, int stride_flux_v,
                              const double* thk, const double* vel_u,
                              const double* vel_v, double* flux_u,
                              double* flux_v);
void update_thickness_cuda(int mx, int my, int gw, int stride_thk,
                           int stride_flux_u, int stride_flux_v,
                           int stride_smb, const double* flux_u,
                           const double* flux_v, const double* smb, double dt,
                           double inv_dx, double inv_dy,
                           int enforce_nonnegative, double* thk);
void update_mask_cuda(int mx, int my, int gw, int stride_thk, int stride_mask,
                      const double* thk, int* mask);
void compute_face_velocity_from_center_cuda(
    int mx, int my, int gw, int stride_u_center, int stride_v_center,
    int stride_u_face, int stride_v_face, const double* u_center,
    const double* v_center, double* u_face, double* v_face);
void compute_cell_center_velocity_cuda(int mx, int my, int gw, int stride_u,
                                       int stride_v, int stride_uvel,
                                       int stride_vvel, const double* u_face,
                                       const double* v_face, double* uvel,
                                       double* vvel);
}  // namespace gpism
#endif

namespace gpism {

void compute_face_fluxes(const Grid2D& grid, const Field2D<double>& thk,
                         const FieldStag2D<double>& vel,
                         FieldStag2D<double>& flux) {
#if GPISM_HAVE_CUDA
  if (thk.has_device_data() && vel.component(0).has_device_data() &&
      vel.component(1).has_device_data() && flux.component(0).has_device_data() &&
      flux.component(1).has_device_data()) {
    compute_face_fluxes_cuda(grid.local_mx(), grid.local_my(),
                             thk.ghost_width(), thk.stride(),
                             vel.component(0).stride(),
                             vel.component(1).stride(),
                             flux.component(0).stride(),
                             flux.component(1).stride(),
                             thk.device_data(), vel.component(0).device_data(),
                             vel.component(1).device_data(),
                             flux.component(0).device_data(),
                             flux.component(1).device_data());
    return;
  }
#endif
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  const int gw = grid.ghost_width();

  for (int j = -gw; j < my + gw; ++j) {
    for (int i = -gw; i < mx; ++i) {
      const int i0 = std::clamp(i, 0, mx - 1);
      const int i1 = std::clamp(i + 1, 0, mx - 1);
      const int j0 = std::clamp(j, 0, my - 1);
      const double H_left = thk(i0, j0);
      const double H_right = thk(i1, j0);
      const double H_face = 0.5 * (H_left + H_right);
      flux(i, j, 0) = H_face * vel(i0, j0, 0);
    }
  }

  for (int j = -gw; j < my; ++j) {
    for (int i = -gw; i < mx + gw; ++i) {
      const int i0 = std::clamp(i, 0, mx - 1);
      const int j0 = std::clamp(j, 0, my - 1);
      const int j1 = std::clamp(j + 1, 0, my - 1);
      const double H_down = thk(i0, j0);
      const double H_up = thk(i0, j1);
      const double H_face = 0.5 * (H_down + H_up);
      flux(i, j, 1) = H_face * vel(i0, j0, 1);
    }
  }
}

void update_thickness(const Grid2D& grid, const FieldStag2D<double>& flux,
                      const Field2D<double>& smb, double dt,
                      const ThicknessUpdateOptions& options,
                      Field2D<double>& thk) {
#if GPISM_HAVE_CUDA
  if (thk.has_device_data() && smb.has_device_data() &&
      flux.component(0).has_device_data() &&
      flux.component(1).has_device_data()) {
    update_thickness_cuda(grid.local_mx(), grid.local_my(), thk.ghost_width(),
                          thk.stride(), flux.component(0).stride(),
                          flux.component(1).stride(), smb.stride(),
                          flux.component(0).device_data(),
                          flux.component(1).device_data(), smb.device_data(),
                          dt, 1.0 / grid.dx(), 1.0 / grid.dy(),
                          options.enforce_nonnegative ? 1 : 0,
                          thk.device_data());
    return;
  }
#endif
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  const double inv_dx = 1.0 / grid.dx();
  const double inv_dy = 1.0 / grid.dy();

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const double div_x =
          (flux(i, j, 0) - flux(i - 1, j, 0)) * inv_dx;
      const double div_y =
          (flux(i, j, 1) - flux(i, j - 1, 1)) * inv_dy;
      double updated = thk(i, j) + dt * (smb(i, j) - (div_x + div_y));
      if (options.enforce_nonnegative) {
        updated = std::max(0.0, updated);
      }
      thk(i, j) = updated;
    }
  }
}

void update_mask(const Grid2D& grid, const Field2D<double>& thk,
                 Field2D<int>& mask) {
#if GPISM_HAVE_CUDA
  if (thk.has_device_data() && mask.has_device_data()) {
    update_mask_cuda(grid.local_mx(), grid.local_my(), thk.ghost_width(),
                     thk.stride(), mask.stride(), thk.device_data(),
                     mask.device_data());
    return;
  }
#endif
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      mask(i, j) = (thk(i, j) > 0.0) ? 1 : 0;
    }
  }
}

void compute_face_velocity_from_center(const Grid2D& grid,
                                       const FieldStag2D<double>& vel_center,
                                       FieldStag2D<double>& vel_face) {
#if GPISM_HAVE_CUDA
  if (vel_center.component(0).has_device_data() &&
      vel_center.component(1).has_device_data() &&
      vel_face.component(0).has_device_data() &&
      vel_face.component(1).has_device_data()) {
    compute_face_velocity_from_center_cuda(
        grid.local_mx(), grid.local_my(), vel_center.component(0).ghost_width(),
        vel_center.component(0).stride(), vel_center.component(1).stride(),
        vel_face.component(0).stride(), vel_face.component(1).stride(),
        vel_center.component(0).device_data(),
        vel_center.component(1).device_data(),
        vel_face.component(0).device_data(), vel_face.component(1).device_data());
    return;
  }
#endif
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  const int gw = grid.ghost_width();
  auto clamp_i = [mx](int i) { return std::clamp(i, 0, mx - 1); };
  auto clamp_j = [my](int j) { return std::clamp(j, 0, my - 1); };

  const Field2D<double>& u_center = vel_center.component(0);
  const Field2D<double>& v_center = vel_center.component(1);

  for (int j = -gw; j < my + gw; ++j) {
    const int jc = clamp_j(j);
    for (int i = -gw; i < mx + gw; ++i) {
      const int i0 = clamp_i(i);
      const int i1 = clamp_i(i + 1);
      vel_face(i, j, 0) = 0.5 * (u_center(i0, jc) + u_center(i1, jc));
    }
  }

  for (int j = -gw; j < my + gw; ++j) {
    const int j0 = clamp_j(j);
    const int j1 = clamp_j(j + 1);
    for (int i = -gw; i < mx + gw; ++i) {
      const int ic = clamp_i(i);
      vel_face(i, j, 1) = 0.5 * (v_center(ic, j0) + v_center(ic, j1));
    }
  }
}

void compute_cell_center_velocity(const Grid2D& grid,
                                  const FieldStag2D<double>& vel,
                                  Field2D<double>& uvel,
                                  Field2D<double>& vvel) {
#if GPISM_HAVE_CUDA
  if (vel.component(0).has_device_data() && vel.component(1).has_device_data() &&
      uvel.has_device_data() && vvel.has_device_data()) {
    compute_cell_center_velocity_cuda(
        grid.local_mx(), grid.local_my(), uvel.ghost_width(),
        vel.component(0).stride(), vel.component(1).stride(), uvel.stride(),
        vvel.stride(), vel.component(0).device_data(),
        vel.component(1).device_data(), uvel.device_data(), vvel.device_data());
    return;
  }
#endif
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const int il = (i == 0) ? i : i - 1;
      const int jd = (j == 0) ? j : j - 1;
      uvel(i, j) = 0.5 * (vel(i, j, 0) + vel(il, j, 0));
      vvel(i, j) = 0.5 * (vel(i, j, 1) + vel(i, jd, 1));
    }
  }
}

}  // namespace gpism
