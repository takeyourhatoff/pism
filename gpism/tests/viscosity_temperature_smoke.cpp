#include "gpism/device_policy.h"
#include "gpism/field_sync.h"
#include "gpism/viscosity.h"

#include <cmath>
#include <iostream>

int main() {
  const int mx = 3;
  const int my = 3;
  const int nz = 6;
  const int gw = 1;
  gpism::Grid2D grid(mx, my, 1.0, 1.0, gw, 0, 1);

  gpism::Field2D<double> thk(mx, my, gw);
  gpism::FieldStag2D<double> vel(mx, my, gw);
  gpism::Field3D<double> enthalpy(mx, my, nz, gw);
  gpism::FieldStag2D<double> nuH_base(mx, my, gw);
  gpism::FieldStag2D<double> nuH_temp(mx, my, gw);

  thk.fill(1000.0);
  vel.fill(5.0);
  enthalpy.fill(10.0);

  gpism::ViscosityModel viscosity(1e-16, 3.0, 1.0);
  const double gamma = 0.1;
  const double ref = 0.0;

  gpism::set_device_enabled(false);
  viscosity.compute_nuH(grid, thk, vel, nuH_base);
  viscosity.compute_nuH(grid, thk, vel, nuH_temp, 0.0, 0.0, 0.0, &enthalpy,
                        gamma, ref);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      for (int comp = 0; comp < 2; ++comp) {
        if (!(nuH_temp(i, j, comp) < nuH_base(i, j, comp))) {
          std::cerr << "temperature coupling did not reduce viscosity\n";
          return 1;
        }
      }
    }
  }

  gpism::set_device_enabled(true);
  gpism::Field2D<double> thk_gpu(mx, my, gw);
  gpism::FieldStag2D<double> vel_gpu(mx, my, gw);
  gpism::Field3D<double> enthalpy_gpu(mx, my, nz, gw);
  gpism::FieldStag2D<double> nuH_gpu(mx, my, gw);

  thk_gpu.fill(1000.0);
  vel_gpu.fill(5.0);
  enthalpy_gpu.fill(10.0);
  nuH_gpu.fill(0.0);

  gpism::sync_host_to_device(thk_gpu);
  gpism::sync_host_to_device(vel_gpu);
  gpism::sync_host_to_device(enthalpy_gpu);
  gpism::sync_host_to_device(nuH_gpu);

  viscosity.compute_nuH(grid, thk_gpu, vel_gpu, nuH_gpu, 0.0, 0.0, 0.0,
                        &enthalpy_gpu, gamma, ref);
  gpism::sync_device_to_host(nuH_gpu);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      for (int comp = 0; comp < 2; ++comp) {
        const double diff = std::abs(nuH_gpu(i, j, comp) - nuH_temp(i, j, comp));
        if (!std::isfinite(diff) || diff > 5e-7) {
          std::cerr << "CPU/GPU viscosity mismatch: " << diff << "\n";
          return 1;
        }
      }
    }
  }

  std::cout << "viscosity_temperature_smoke passed\n";
  return 0;
}
