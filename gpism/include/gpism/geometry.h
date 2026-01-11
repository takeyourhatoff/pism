#pragma once

#include "gpism/field2d.h"
#include "gpism/grid2d.h"

namespace gpism {

class GeometryDiagnostics {
public:
  void compute_usurf(const Grid2D& grid, const Field2D<double>& thk,
                     const Field2D<double>& topg, Field2D<double>& usurf);
  void compute_surface_slopes(const Grid2D& grid, const Field2D<double>& usurf,
                              Field2D<double>& dhdx, Field2D<double>& dhdy);

  static void compute_usurf_cpu(const Grid2D& grid, const Field2D<double>& thk,
                                const Field2D<double>& topg, Field2D<double>& usurf);
  static void compute_surface_slopes_cpu(const Grid2D& grid,
                                         const Field2D<double>& usurf,
                                         Field2D<double>& dhdx,
                                         Field2D<double>& dhdy);
};

}  // namespace gpism
