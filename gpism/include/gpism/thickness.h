#pragma once

#include "gpism/field2d.h"
#include "gpism/field_stag2d.h"
#include "gpism/grid2d.h"

namespace gpism {

struct ThicknessUpdateOptions {
  bool enforce_nonnegative = true;
};

void compute_face_fluxes(const Grid2D& grid, const Field2D<double>& thk,
                         const FieldStag2D<double>& vel,
                         FieldStag2D<double>& flux);

void update_thickness(const Grid2D& grid, const FieldStag2D<double>& flux,
                      const Field2D<double>& smb, double dt,
                      const ThicknessUpdateOptions& options,
                      Field2D<double>& thk);

void update_mask(const Grid2D& grid, const Field2D<double>& thk,
                 Field2D<int>& mask);

void compute_cell_center_velocity(const Grid2D& grid,
                                  const FieldStag2D<double>& vel,
                                  Field2D<double>& uvel,
                                  Field2D<double>& vvel);

}  // namespace gpism
