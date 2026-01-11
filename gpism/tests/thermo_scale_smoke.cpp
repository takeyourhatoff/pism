#include "gpism/config.h"
#include "gpism/device_policy.h"
#include "gpism/field_sync.h"
#include "gpism/thermodynamics.h"

#include <cmath>
#include <iostream>
#include <utility>

int main() {
#if !GPISM_HAVE_CUDA
  std::cout << "thermo_scale_smoke skipped (CUDA disabled)\n";
  return 0;
#else
  gpism::set_device_enabled(true);

  const int mx = 128;
  const int my = 128;
  const int nz = 64;
  const int gw = 1;
  const double dz = 1.0;
  const double dt = 0.1;
  const int steps = 20;

  gpism::Field3D<double> enthalpy(mx, my, nz, gw);
  gpism::Field3D<double> enthalpy_next(mx, my, nz, gw);

  gpism::VerticalDiffusionOptions options;
  options.kappa = 1.0;
  options.surface_value = -5.0;
  options.basal_value = 5.0;
  options.dirichlet = true;

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      for (int k = 0; k < nz; ++k) {
        const double t = (nz > 1) ? static_cast<double>(k) / (nz - 1) : 0.0;
        enthalpy(i, j, k) =
            (1.0 - t) * options.surface_value + t * options.basal_value;
      }
    }
  }

  gpism::sync_host_to_device(enthalpy);
  gpism::sync_host_to_device(enthalpy_next);

  for (int step = 0; step < steps; ++step) {
    gpism::vertical_diffusion_step(enthalpy, nz, dz, dt, options,
                                   enthalpy_next);
    std::swap(enthalpy, enthalpy_next);
  }

  gpism::sync_device_to_host(enthalpy);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const double top = enthalpy(i, j, 0);
      const double bottom = enthalpy(i, j, nz - 1);
      if (std::abs(top - options.surface_value) > 1e-10 ||
          std::abs(bottom - options.basal_value) > 1e-10) {
        std::cerr << "boundary mismatch after diffusion\n";
        return 1;
      }
    }
  }

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      for (int k = 0; k < nz; ++k) {
        const double value = enthalpy(i, j, k);
        if (!std::isfinite(value)) {
          std::cerr << "non-finite enthalpy value detected\n";
          return 1;
        }
      }
    }
  }

  std::cout << "thermo_scale_smoke passed\n";
  return 0;
#endif
}
