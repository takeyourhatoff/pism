#pragma once

#include <memory>
#include <string>

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

enum class SSAPrecondPrecision { FP64, FP32 };
enum class SSAHaloMode { Device };

struct SSASolverOptions {
  int max_picard = 10;
  int picard_convergence_check_interval = 1;
  bool device_metrics_batch = true;
  double tol_nuH = 1e-6;
  double tol_vel = 1e-6;

  double vel_relax = 1.0;
  double nuH_relax = 1.0;
  double nuH_regularization = 0.0;
  double nuH_min = 0.0;
  double nuH_max = 0.0;
  double strength_extension_nu = 0.0;
  double strength_extension_min_thickness = 0.0;

  int gmres_restart = 30;
  int gmres_max_iter = 200;
  double gmres_tol = 1e-8;
  bool gmres_tol_relative = true;
  bool gmres_tol_relative_to_rhs = false;
  int gmres_residual_check_interval = 1;
  bool gmres_verbose = false;

  bool use_mg_precond = false;
  int mg_pre_iters = 2;
  int mg_post_iters = 2;
  int mg_coarse_iters = 10;
  double mg_omega = 0.8;
  int mg_min_size = 4;
  int mg_jacobi_sweeps_per_launch = 1;
  MGSmoother mg_smoother = MGSmoother::Jacobi;
  double mg_cheby_lambda_min = 0.1;
  double mg_cheby_lambda_max = 2.0;
  bool mg_cheby_estimate = false;
  int mg_cheby_estimate_iters = 5;
  double mg_cheby_estimate_min_factor = 0.1;
  double mg_cheby_estimate_max_factor = 1.1;
  bool mg_diagnostic = false;
  bool gmres_precond_diagnostic = false;
  bool diagnostic = false;
  SSAPrecondPrecision precond_precision = SSAPrecondPrecision::FP64;
  bool fail_fast = true;
  bool fail_fast_require_converged = false;
  double fail_fast_residual_max = 0.0;
  std::string fail_fast_dump_prefix;
  std::string config_override_path;

  BasalResistanceParams basal_params;
  double sea_level = 0.0;
  double rho_ice = 910.0;
  double rho_water = 1028.0;
  double ice_free_thickness_standard = 0.0;
  bool surface_gradient_inward = false;
  bool surface_slope_uphill = true;
  bool extrapolate_at_margins = false;
  bool use_cfbc = false;
  bool replace_zero_diagonal_entries = true;

  bool use_bc = false;
  SSAHaloMode halo_mode = SSAHaloMode::Device;
  bool require_cuda_aware_mpi = false;
  // PISM only enforces "ice-free velocity = 0" Dirichlet conditions when using
  // calving-front stress boundary conditions (CFBC). When CFBC is disabled,
  // enforcing Dirichlet conditions in ice-free areas can clamp the solution
  // along margins and lead to large divergences.
  bool enforce_ice_free_bc = false;
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
    Field2D<int> cell_type;
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
        cell_type.resize(mx, my, gw);
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
        cell_type.resize(mx, my, gw);
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
