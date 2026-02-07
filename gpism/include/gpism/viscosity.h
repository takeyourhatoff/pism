#pragma once

#include "gpism/field2d.h"
#include "gpism/field3d.h"
#include "gpism/field_stag2d.h"
#include "gpism/grid2d.h"

namespace gpism {

class ViscosityModel {
public:
  ViscosityModel(double A, double n, double eps0, double enhancement = 1.0);

  void compute_nuH(const Grid2D& grid, const Field2D<double>& thk,
                   const FieldStag2D<double>& vel, FieldStag2D<double>& nuH,
                   double nuH_regularization = 0.0,
                   double strength_extension_nu = 0.0,
                   double strength_extension_min_thickness = 0.0,
                   const Field3D<double>* enthalpy = nullptr,
                   double enthalpy_gamma = 0.0,
                   double enthalpy_ref = 0.0) const;

private:
  double A_;
  double n_;
  double eps0_;
  double enhancement_;
};

}  // namespace gpism
