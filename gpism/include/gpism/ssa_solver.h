#pragma once

#include "gpism/field2d.h"
#include "gpism/field_stag2d.h"
#include "gpism/grid2d.h"
#include "gpism/ssa_operator.h"
#include "gpism/viscosity.h"

namespace gpism {

struct SSASolverOptions {
  int max_picard = 10;
  double tol_nuH = 1e-6;
  double tol_vel = 1e-6;

  double vel_relax = 1.0;
  double nuH_relax = 1.0;
  double nuH_min = 0.0;
  double nuH_max = 0.0;

  int gmres_restart = 30;
  int gmres_max_iter = 200;
  double gmres_tol = 1e-8;

  bool use_bc = false;
};

struct SSASolverResult {
  int picard_iters = 0;
  bool converged = false;
  double nuH_change = 0.0;
  double vel_change = 0.0;
  int linear_iters = 0;
  double linear_residual = 0.0;
};

class SSASolver {
public:
  SSASolver(const Grid2D& grid, double rho, double g, double u_threshold,
            const ViscosityModel& viscosity_model);

  SSASolverResult solve(const Field2D<double>& thk, const Field2D<double>& topg,
                        const Field2D<double>& tauc,
                        const Field2D<double>* u_bc,
                        const Field2D<double>* v_bc,
                        const Field2D<int>* vel_bc_mask,
                        FieldStag2D<double>& vel,
                        const SSASolverOptions& options);

private:
  const Grid2D& grid_;
  SSAOperator ssa_;
  GeometryDiagnostics geometry_;
  ViscosityModel viscosity_;
};

}  // namespace gpism
