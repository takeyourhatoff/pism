#pragma once

#include "gpism/field2d.h"
#include "gpism/grid2d.h"

namespace gpism {

enum CellType : int {
  IceFreeBedrock = 0,
  GroundedIce = 1,
  FloatingIce = 2,
  IceFreeOcean = 3
};

void compute_cell_type(const Grid2D& grid, const Field2D<double>& thk,
                       const Field2D<double>& topg, double sea_level,
                       double rho_ice, double rho_water,
                       Field2D<int>& cell_type);

void compute_usurf_flotation(const Grid2D& grid, const Field2D<double>& thk,
                             const Field2D<double>& topg,
                             const Field2D<int>& cell_type,
                             double sea_level, double rho_ice,
                             double rho_water, Field2D<double>& usurf);

void compute_surface_slopes_pism(const Grid2D& grid,
                                 const Field2D<double>& usurf,
                                 const Field2D<int>& cell_type,
                                 Field2D<double>& dhdx,
                                 Field2D<double>& dhdy,
                                 bool surface_gradient_inward,
                                 bool uphill,
                                 bool use_cfbc);

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
