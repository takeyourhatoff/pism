#include "gpism/thickness.h"

#include <algorithm>
#include <stdexcept>

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

namespace gpism {

void compute_face_fluxes(const Grid2D& grid, const Field2D<double>& thk,
                         const FieldStag2D<double>& vel,
                         FieldStag2D<double>& flux) {
  if (!(thk.has_device_data() && vel.component(0).has_device_data() &&
        vel.component(1).has_device_data() &&
        flux.component(0).has_device_data() &&
        flux.component(1).has_device_data())) {
    throw std::runtime_error(
        "compute_face_fluxes requires device-resident thk/vel/flux");
  }
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
}

void update_thickness(const Grid2D& grid, const FieldStag2D<double>& flux,
                      const Field2D<double>& smb, double dt,
                      const ThicknessUpdateOptions& options,
                      Field2D<double>& thk) {
  if (!(thk.has_device_data() && smb.has_device_data() &&
        flux.component(0).has_device_data() &&
        flux.component(1).has_device_data())) {
    throw std::runtime_error(
        "update_thickness requires device-resident thk/smb/flux");
  }
  update_thickness_cuda(grid.local_mx(), grid.local_my(), thk.ghost_width(),
                        thk.stride(), flux.component(0).stride(),
                        flux.component(1).stride(), smb.stride(),
                        flux.component(0).device_data(),
                        flux.component(1).device_data(), smb.device_data(), dt,
                        1.0 / grid.dx(), 1.0 / grid.dy(),
                        options.enforce_nonnegative ? 1 : 0, thk.device_data());
}

void update_mask(const Grid2D& grid, const Field2D<double>& thk,
                 Field2D<int>& mask) {
  if (!(thk.has_device_data() && mask.has_device_data())) {
    throw std::runtime_error(
        "update_mask requires device-resident thk/mask");
  }
  update_mask_cuda(grid.local_mx(), grid.local_my(), thk.ghost_width(),
                   thk.stride(), mask.stride(), thk.device_data(),
                   mask.device_data());
}

void compute_face_velocity_from_center(const Grid2D& grid,
                                       const FieldStag2D<double>& vel_center,
                                       FieldStag2D<double>& vel_face) {
  if (!(vel_center.component(0).has_device_data() &&
        vel_center.component(1).has_device_data() &&
        vel_face.component(0).has_device_data() &&
        vel_face.component(1).has_device_data())) {
    throw std::runtime_error(
        "compute_face_velocity_from_center requires device-resident "
        "vel_center/vel_face");
  }
  compute_face_velocity_from_center_cuda(
      grid.local_mx(), grid.local_my(), vel_center.component(0).ghost_width(),
      vel_center.component(0).stride(), vel_center.component(1).stride(),
      vel_face.component(0).stride(), vel_face.component(1).stride(),
      vel_center.component(0).device_data(), vel_center.component(1).device_data(),
      vel_face.component(0).device_data(), vel_face.component(1).device_data());
}

void compute_cell_center_velocity(const Grid2D& grid,
                                  const FieldStag2D<double>& vel,
                                  Field2D<double>& uvel,
                                  Field2D<double>& vvel) {
  if (!(vel.component(0).has_device_data() &&
        vel.component(1).has_device_data() && uvel.has_device_data() &&
        vvel.has_device_data())) {
    throw std::runtime_error(
        "compute_cell_center_velocity requires device-resident vel/uvel/vvel");
  }
  compute_cell_center_velocity_cuda(
      grid.local_mx(), grid.local_my(), uvel.ghost_width(),
      vel.component(0).stride(), vel.component(1).stride(), uvel.stride(),
      vvel.stride(), vel.component(0).device_data(), vel.component(1).device_data(),
      uvel.device_data(), vvel.device_data());
}

}  // namespace gpism
