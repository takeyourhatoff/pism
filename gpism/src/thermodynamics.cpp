#include "gpism/thermodynamics.h"

#include <algorithm>

#include "gpism/config.h"

#if GPISM_HAVE_CUDA
namespace gpism {
void vertical_diffusion_step_cuda(int mx, int my, int gw, int nz, int stride,
                                  const double* enthalpy_in, double* enthalpy_out,
                                  double kappa, double dz, double dt,
                                  double surface_value, double basal_value,
                                  int dirichlet);
}  // namespace gpism
#endif

namespace gpism {
namespace {

int idx(int i, int j, int k, int gw, int stride, int nz) {
  return ((j + gw) * stride + (i + gw)) * nz + k;
}

}  // namespace

void vertical_diffusion_step(const Field3D<double>& enthalpy_in, int nz,
                             double dz, double dt,
                             const VerticalDiffusionOptions& options,
                             Field3D<double>& enthalpy_out) {
  const int mx = enthalpy_in.local_mx();
  const int my = enthalpy_in.local_my();
  const int gw = enthalpy_in.ghost_width();
  const int stride = enthalpy_in.stride();

#if GPISM_HAVE_CUDA
  if (enthalpy_in.has_device_data() && enthalpy_out.has_device_data()) {
    vertical_diffusion_step_cuda(mx, my, gw, nz, stride,
                                 enthalpy_in.device_data(),
                                 enthalpy_out.device_data(), options.kappa,
                                 dz, dt, options.surface_value,
                                 options.basal_value, options.dirichlet ? 1 : 0);
    return;
  }
#endif

  Field3D<double> a(mx, my, nz, gw);
  Field3D<double> b(mx, my, nz, gw);
  Field3D<double> c(mx, my, nz, gw);
  Field3D<double> d(mx, my, nz, gw);

  const double r = options.kappa * dt / (dz * dz);
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      for (int k = 0; k < nz; ++k) {
        if (options.dirichlet && (k == 0 || k == nz - 1)) {
          a(i, j, k) = 0.0;
          b(i, j, k) = 1.0;
          c(i, j, k) = 0.0;
          d(i, j, k) = (k == 0) ? options.surface_value : options.basal_value;
        } else {
          a(i, j, k) = -r;
          b(i, j, k) = 1.0 + 2.0 * r;
          c(i, j, k) = -r;
          d(i, j, k) = enthalpy_in(i, j, k);
        }
      }

      // Forward elimination (overwrite c and d).
      double c_prev = c(i, j, 0) / b(i, j, 0);
      double d_prev = d(i, j, 0) / b(i, j, 0);
      c(i, j, 0) = c_prev;
      d(i, j, 0) = d_prev;

      for (int k = 1; k < nz; ++k) {
        const double denom = b(i, j, k) - a(i, j, k) * c_prev;
        c_prev = c(i, j, k) / denom;
        d_prev = (d(i, j, k) - a(i, j, k) * d_prev) / denom;
        c(i, j, k) = c_prev;
        d(i, j, k) = d_prev;
      }

      // Back substitution.
      enthalpy_out(i, j, nz - 1) = d(i, j, nz - 1);
      for (int k = nz - 2; k >= 0; --k) {
        enthalpy_out(i, j, k) = d(i, j, k) - c(i, j, k) * enthalpy_out(i, j, k + 1);
      }
    }
  }

  (void)idx;
}

}  // namespace gpism
