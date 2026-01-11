#pragma once

#include "gpism/field3d.h"

namespace gpism {

struct VerticalDiffusionOptions {
  double kappa = 1.0;
  double surface_value = 0.0;
  double basal_value = 0.0;
  bool dirichlet = true;
};

void vertical_diffusion_step(const Field3D<double>& enthalpy_in, int nz,
                             double dz, double dt,
                             const VerticalDiffusionOptions& options,
                             Field3D<double>& enthalpy_out);

}  // namespace gpism
