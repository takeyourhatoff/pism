#include "gpism/device_policy.h"
#include "gpism/field_sync.h"
#include "gpism/thermodynamics.h"

#include <cmath>
#include <iostream>

int main() {
  const int mx = 3;
  const int my = 2;
  const int nz = 8;
  const int gw = 1;
  const double dz = 1.0;
  const double dt = 0.1;

  gpism::VerticalDiffusionOptions options;
  options.kappa = 1.0;
  options.surface_value = -5.0;
  options.basal_value = 10.0;
  options.dirichlet = true;

  gpism::set_device_enabled(false);
  gpism::Field3D<double> enthalpy_cpu(mx, my, nz, gw);
  gpism::Field3D<double> enthalpy_cpu_out(mx, my, nz, gw);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      for (int k = 0; k < nz; ++k) {
        enthalpy_cpu(i, j, k) = static_cast<double>(k);
      }
    }
  }

  gpism::vertical_diffusion_step(enthalpy_cpu, nz, dz, dt, options,
                                 enthalpy_cpu_out);

  gpism::set_device_enabled(true);
  gpism::Field3D<double> enthalpy_gpu(mx, my, nz, gw);
  gpism::Field3D<double> enthalpy_gpu_out(mx, my, nz, gw);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      for (int k = 0; k < nz; ++k) {
        enthalpy_gpu(i, j, k) = static_cast<double>(k);
      }
    }
  }

  gpism::sync_host_to_device(enthalpy_gpu);
  gpism::sync_host_to_device(enthalpy_gpu_out);

  gpism::vertical_diffusion_step(enthalpy_gpu, nz, dz, dt, options,
                                 enthalpy_gpu_out);
  gpism::sync_device_to_host(enthalpy_gpu_out);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      if (std::abs(enthalpy_cpu_out(i, j, 0) - options.surface_value) > 1e-12 ||
          std::abs(enthalpy_gpu_out(i, j, 0) - options.surface_value) > 1e-12) {
        std::cerr << "surface boundary mismatch\n";
        return 1;
      }
      if (std::abs(enthalpy_cpu_out(i, j, nz - 1) - options.basal_value) > 1e-12 ||
          std::abs(enthalpy_gpu_out(i, j, nz - 1) - options.basal_value) > 1e-12) {
        std::cerr << "basal boundary mismatch\n";
        return 1;
      }
    }
  }

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      for (int k = 0; k < nz; ++k) {
        const double diff = std::abs(enthalpy_cpu_out(i, j, k) -
                                     enthalpy_gpu_out(i, j, k));
        if (!std::isfinite(diff) || diff > 5e-7) {
          std::cerr << "CPU/GPU diffusion mismatch at (" << i << "," << j
                    << "," << k << "): " << diff << "\n";
          return 1;
        }
      }
    }
  }

  std::cout << "vertical_diffusion_smoke passed\n";
  return 0;
}
