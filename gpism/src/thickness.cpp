#include "gpism/thickness.h"

#include <algorithm>

namespace gpism {

void compute_face_fluxes(const Grid2D& grid, const Field2D<double>& thk,
                         const FieldStag2D<double>& vel,
                         FieldStag2D<double>& flux) {
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
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      mask(i, j) = (thk(i, j) > 0.0) ? 1 : 0;
    }
  }
}

void compute_cell_center_velocity(const Grid2D& grid,
                                  const FieldStag2D<double>& vel,
                                  Field2D<double>& uvel,
                                  Field2D<double>& vvel) {
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
