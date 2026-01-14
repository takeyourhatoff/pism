#pragma once

#include "gpism/field2d.h"
#include "gpism/field_stag2d.h"
#include "gpism/grid2d.h"

namespace gpism {

struct SSABoundaryCondition {
  const FieldStag2D<int>* mask = nullptr;
  const FieldStag2D<double>* values = nullptr;
};

enum class BasalResistanceLaw { Plastic, PseudoPlastic };

struct BasalResistanceParams {
  BasalResistanceLaw law = BasalResistanceLaw::Plastic;
  double q = 0.25;
  double u_threshold = 100.0;
  double plastic_regularization = 0.01;
  double sliding_scale_factor = -1.0;
  double beta_ice_free_bedrock = 0.0;
};

class SSAOperator {
public:
  SSAOperator(double rho, double g);

  void compute_basal_drag(const Grid2D& grid, const Field2D<double>& tauc,
                          const Field2D<double>& u_center,
                          const Field2D<double>& v_center,
                          const Field2D<int>& cell_type,
                          FieldStag2D<double>& beta,
                          const BasalResistanceParams& params) const;

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
};

}  // namespace gpism
