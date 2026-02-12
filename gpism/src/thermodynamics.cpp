#include "gpism/thermodynamics.h"

#include <stdexcept>

namespace gpism {
void vertical_diffusion_step_cuda(int mx, int my, int gw, int nz, int stride,
                                  const double* enthalpy_in, double* enthalpy_out,
                                  double kappa, double dz, double dt,
                                  double surface_value, double basal_value,
                                  int dirichlet);
}  // namespace gpism

namespace gpism {

void vertical_diffusion_step(const Field3D<double>& enthalpy_in, int nz,
                             double dz, double dt,
                             const VerticalDiffusionOptions& options,
                             Field3D<double>& enthalpy_out) {
  const int mx = enthalpy_in.local_mx();
  const int my = enthalpy_in.local_my();
  const int gw = enthalpy_in.ghost_width();
  const int stride = enthalpy_in.stride();

  if (!(enthalpy_in.has_device_data() && enthalpy_out.has_device_data())) {
    throw std::runtime_error(
        "vertical_diffusion_step requires device-resident enthalpy fields");
  }
  vertical_diffusion_step_cuda(mx, my, gw, nz, stride, enthalpy_in.device_data(),
                               enthalpy_out.device_data(), options.kappa, dz, dt,
                               options.surface_value, options.basal_value,
                               options.dirichlet ? 1 : 0);
}

}  // namespace gpism
