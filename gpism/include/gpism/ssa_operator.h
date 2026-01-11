#pragma once

#include "gpism/field2d.h"
#include "gpism/field_stag2d.h"
#include "gpism/grid2d.h"

namespace gpism {

struct SSABoundaryCondition {
  const FieldStag2D<int>* mask = nullptr;
  const FieldStag2D<double>* values = nullptr;
};

class SSAOperator {
public:
  SSAOperator(double rho, double g, double u_threshold);

  void compute_basal_drag(const Grid2D& grid, const Field2D<double>& tauc,
                          FieldStag2D<double>& beta) const;

  void assemble_rhs(const Grid2D& grid, const Field2D<double>& thk,
                    const Field2D<double>& dhdx, const Field2D<double>& dhdy,
                    FieldStag2D<double>& rhs,
                    const SSABoundaryCondition* bc = nullptr) const;

  void apply(const Grid2D& grid, const FieldStag2D<double>& nuH,
             const FieldStag2D<double>& beta, const FieldStag2D<double>& vel,
             FieldStag2D<double>& out,
             const SSABoundaryCondition* bc = nullptr) const;
  void apply_region(const Grid2D& grid, const FieldStag2D<double>& nuH,
                    const FieldStag2D<double>& beta,
                    const FieldStag2D<double>& vel,
                    FieldStag2D<double>& out, int i_start, int i_end,
                    int j_start, int j_end,
                    const SSABoundaryCondition* bc = nullptr) const;

private:
  double rho_;
  double g_;
  double u_threshold_;
};

}  // namespace gpism
