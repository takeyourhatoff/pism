#pragma once

#include "gpism/field2d.h"
#include "gpism/field_stag2d.h"
#include "gpism/grid2d.h"

namespace gpism {

class ViscosityModel {
public:
  ViscosityModel(double A, double n, double eps0);

  void compute_nuH(const Grid2D& grid, const Field2D<double>& thk,
                   const FieldStag2D<double>& vel, FieldStag2D<double>& nuH) const;

private:
  double A_;
  double n_;
  double eps0_;
};

}  // namespace gpism
