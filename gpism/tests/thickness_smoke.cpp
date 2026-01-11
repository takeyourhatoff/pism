#include "gpism/grid2d.h"
#include "gpism/thickness.h"

#include <cmath>
#include <iostream>

int main() {
  const int mx = 4;
  const int my = 4;
  const int gw = 1;
  gpism::Grid2D grid(mx, my, 1.0, 1.0, gw, 0, 1);

  gpism::Field2D<double> thk(mx, my, gw);
  gpism::Field2D<double> smb(mx, my, gw);
  gpism::FieldStag2D<double> vel(mx, my, gw);
  gpism::FieldStag2D<double> flux(mx, my, gw);

  thk.fill(1.0);
  smb.fill(0.1);
  vel.fill(2.0);

  gpism::compute_face_fluxes(grid, thk, vel, flux);
  gpism::ThicknessUpdateOptions options;
  gpism::update_thickness(grid, flux, smb, 1.0, options, thk);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      if (std::abs(thk(i, j) - 1.1) > 1e-8) {
        std::cerr << "thickness update mismatch: " << thk(i, j) << "\n";
        return 1;
      }
    }
  }

  smb.fill(-10.0);
  gpism::compute_face_fluxes(grid, thk, vel, flux);
  gpism::update_thickness(grid, flux, smb, 1.0, options, thk);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      if (thk(i, j) < 0.0) {
        std::cerr << "negative thickness encountered\n";
        return 1;
      }
    }
  }

  std::cout << "thickness_smoke passed\n";
  return 0;
}
