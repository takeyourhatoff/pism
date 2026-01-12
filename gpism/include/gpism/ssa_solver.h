#pragma once

#include <memory>

#include "gpism/field2d.h"
#include "gpism/field3d.h"
#include "gpism/field_stag2d.h"
#include "gpism/geometry.h"
#include "gpism/grid2d.h"
#include "gpism/multigrid.h"
#include "gpism/ssa_operator.h"
#include "gpism/viscosity.h"

namespace gpism {

class Context;

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

  bool use_mg_precond = false;
  int mg_pre_iters = 2;
  int mg_post_iters = 2;
  int mg_coarse_iters = 10;
  double mg_omega = 0.8;
  int mg_min_size = 4;
  MGSmoother mg_smoother = MGSmoother::Jacobi;
  double mg_cheby_lambda_min = 0.1;
  double mg_cheby_lambda_max = 2.0;
  bool mg_diagnostic = false;
  bool gmres_precond_diagnostic = false;

  bool use_bc = false;
  const Field3D<double>* enthalpy = nullptr;
  double enthalpy_gamma = 0.0;
  double enthalpy_ref = 0.0;
  const Context* context = nullptr;
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
  ViscosityModel viscosity_;
  struct Workspace {
    int mx = 0;
    int my = 0;
    int gw = 0;
    bool initialized = false;
    Field2D<double> usurf;
    Field2D<double> dhdx;
    Field2D<double> dhdy;
    FieldStag2D<double> beta;
    FieldStag2D<double> rhs;
    FieldStag2D<double> nuH;
    FieldStag2D<double> nuH_prev;
    FieldStag2D<double> vel_prev;
    FieldStag2D<int> bc_mask;
    FieldStag2D<double> bc_values;
    std::unique_ptr<MultigridHierarchy> mg;
    int mg_min_size = 0;

    void ensure(const Grid2D& grid) {
      const int mx_new = grid.local_mx();
      const int my_new = grid.local_my();
      const int gw_new = grid.ghost_width();
      const bool dims_changed =
          (!initialized || mx != mx_new || my != my_new || gw != gw_new);
      mx = mx_new;
      my = my_new;
      gw = gw_new;
      if (!initialized) {
        usurf.resize(mx, my, gw);
        dhdx.resize(mx, my, gw);
        dhdy.resize(mx, my, gw);
        beta.resize(mx, my, gw);
        rhs.resize(mx, my, gw);
        nuH.resize(mx, my, gw);
        nuH_prev.resize(mx, my, gw);
        vel_prev.resize(mx, my, gw);
        bc_mask.resize(mx, my, gw);
        bc_values.resize(mx, my, gw);
      } else if (dims_changed) {
        usurf.resize(mx, my, gw);
        dhdx.resize(mx, my, gw);
        dhdy.resize(mx, my, gw);
        beta.resize(mx, my, gw);
        rhs.resize(mx, my, gw);
        nuH.resize(mx, my, gw);
        nuH_prev.resize(mx, my, gw);
        vel_prev.resize(mx, my, gw);
        bc_mask.resize(mx, my, gw);
        bc_values.resize(mx, my, gw);
      }
      initialized = true;
    }
  };

  Workspace workspace_;
};

}  // namespace gpism
