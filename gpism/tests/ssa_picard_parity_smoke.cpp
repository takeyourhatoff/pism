#include "gpism/device_policy.h"
#include "gpism/field_sync.h"
#include "gpism/geometry.h"
#include "gpism/gmres.h"
#include "gpism/linear_algebra.h"
#include "gpism/mg_preconditioner.h"
#include "gpism/multigrid.h"
#include "gpism/ssa_solver.h"
#include "gpism/thickness.h"

#include <cmath>
#include <iostream>

namespace {

double vel_norm2(const gpism::FieldStag2D<double>& vel) {
  double sum = 0.0;
  for (int j = 0; j < vel.local_my(); ++j) {
    for (int i = 0; i < vel.local_mx(); ++i) {
      sum += vel(i, j, 0) * vel(i, j, 0) + vel(i, j, 1) * vel(i, j, 1);
    }
  }
  return sum;
}

double vel_diff_norm2(const gpism::FieldStag2D<double>& a,
                      const gpism::FieldStag2D<double>& b) {
  double sum = 0.0;
  for (int j = 0; j < a.local_my(); ++j) {
    for (int i = 0; i < a.local_mx(); ++i) {
      const double du = a(i, j, 0) - b(i, j, 0);
      const double dv = a(i, j, 1) - b(i, j, 1);
      sum += du * du + dv * dv;
    }
  }
  return sum;
}

bool compare_field(const gpism::Field2D<double>& a,
                   const gpism::Field2D<double>& b, double tol,
                   const char* name) {
  double max_diff = 0.0;
  for (int j = 0; j < a.local_my(); ++j) {
    for (int i = 0; i < a.local_mx(); ++i) {
      const double diff = std::abs(a(i, j) - b(i, j));
      if (diff > max_diff) {
        max_diff = diff;
      }
    }
  }
  if (!std::isfinite(max_diff) || max_diff > tol) {
    std::cerr << name << " CPU/GPU mismatch max_diff=" << max_diff << "\n";
    return false;
  }
  return true;
}

bool compare_field(const gpism::Field2D<int>& a,
                   const gpism::Field2D<int>& b, const char* name) {
  for (int j = 0; j < a.local_my(); ++j) {
    for (int i = 0; i < a.local_mx(); ++i) {
      if (a(i, j) != b(i, j)) {
        std::cerr << name << " CPU/GPU mismatch at " << i << "," << j
                  << " (" << a(i, j) << " vs " << b(i, j) << ")\n";
        return false;
      }
    }
  }
  return true;
}

bool compare_stag(const gpism::FieldStag2D<double>& a,
                  const gpism::FieldStag2D<double>& b, double tol,
                  const char* name) {
  double max_diff = 0.0;
  int max_i = -1;
  int max_j = -1;
  int max_comp = -1;
  double max_a = 0.0;
  double max_b = 0.0;
  for (int j = 0; j < a.local_my(); ++j) {
    for (int i = 0; i < a.local_mx(); ++i) {
      for (int comp = 0; comp < 2; ++comp) {
        const double diff = std::abs(a(i, j, comp) - b(i, j, comp));
        if (diff > max_diff) {
          max_diff = diff;
          max_i = i;
          max_j = j;
          max_comp = comp;
          max_a = a(i, j, comp);
          max_b = b(i, j, comp);
        }
      }
    }
  }
  if (!std::isfinite(max_diff) || max_diff > tol) {
    std::cerr << name << " CPU/GPU mismatch max_diff=" << max_diff
              << " at " << max_i << "," << max_j << "," << max_comp
              << " (" << max_a << " vs " << max_b << ")\n";
    return false;
  }
  return true;
}

struct SSAOp : public gpism::LinearOperator {
  SSAOp(const gpism::SSAOperator& op, const gpism::Grid2D& grid,
        const gpism::FieldStag2D<double>& nuH,
        const gpism::FieldStag2D<double>& beta)
      : op_(op), grid_(grid), nuH_(nuH), beta_(beta) {}

  void apply(const gpism::FieldStag2D<double>& x,
             gpism::FieldStag2D<double>& y) const override {
    op_.apply(grid_, nuH_, beta_, x, y, nullptr);
  }

  const gpism::SSAOperator& op_;
  const gpism::Grid2D& grid_;
  const gpism::FieldStag2D<double>& nuH_;
  const gpism::FieldStag2D<double>& beta_;
};

}  // namespace

int main() {
  const int mx = 4;
  const int my = 4;
  const int gw = 1;

  gpism::SSASolverOptions options;
  options.max_picard = 20;
  options.tol_nuH = 0.7;
  options.tol_vel = 0.7;
  options.gmres_max_iter = 150;
  options.gmres_tol = 1e-7;
  options.use_bc = false;
  options.enforce_ice_free_bc = false;
  options.use_mg_precond = true;
  options.force_host_convergence = true;
  options.diagnostic = false;

  gpism::set_deterministic_reductions(true);

  gpism::set_device_enabled(false);
  gpism::Grid2D grid_cpu(mx, my, 1.0, 1.0, gw, 0, 1);
  gpism::Field2D<double> thk_cpu(mx, my, gw);
  gpism::Field2D<double> topg_cpu(mx, my, gw);
  gpism::Field2D<double> tauc_cpu(mx, my, gw);
  gpism::Field2D<int> cell_cpu(mx, my, gw);
  gpism::Field2D<double> usurf_cpu(mx, my, gw);
  gpism::Field2D<double> dhdx_cpu(mx, my, gw);
  gpism::Field2D<double> dhdy_cpu(mx, my, gw);
  gpism::Field2D<double> u_center_cpu(mx, my, gw);
  gpism::Field2D<double> v_center_cpu(mx, my, gw);
  gpism::FieldStag2D<double> beta_cpu(mx, my, gw);
  gpism::FieldStag2D<double> nuH_cpu(mx, my, gw);
  gpism::FieldStag2D<double> rhs_cpu(mx, my, gw);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      thk_cpu(i, j) = 2.0;
      topg_cpu(i, j) = 0.1 * i - 0.05 * j;
      tauc_cpu(i, j) = 120.0;
    }
  }

  gpism::FieldStag2D<double> vel_cpu(mx, my, gw);
  vel_cpu.fill(0.0);
  gpism::ViscosityModel viscosity_cpu(1e-16, 3.0, 1.0);

  gpism::compute_cell_type(grid_cpu, thk_cpu, topg_cpu, options.sea_level,
                           options.rho_ice, options.rho_water, cell_cpu);
  gpism::compute_usurf_flotation(grid_cpu, thk_cpu, topg_cpu, cell_cpu,
                                 options.sea_level, options.rho_ice,
                                 options.rho_water, usurf_cpu);
  gpism::compute_surface_slopes_pism(grid_cpu, usurf_cpu, cell_cpu, dhdx_cpu,
                                     dhdy_cpu, options.surface_gradient_inward,
                                     options.surface_slope_uphill,
                                     options.use_cfbc);
  gpism::compute_cell_center_velocity(grid_cpu, vel_cpu, u_center_cpu,
                                      v_center_cpu);
  gpism::SSAOperator ssa_cpu(910.0, 9.81);
  ssa_cpu.compute_basal_drag(grid_cpu, tauc_cpu, u_center_cpu, v_center_cpu,
                             cell_cpu, beta_cpu, options.basal_params);
  viscosity_cpu.compute_nuH(grid_cpu, thk_cpu, vel_cpu, nuH_cpu,
                            options.nuH_regularization,
                            options.strength_extension_nu,
                            options.strength_extension_min_thickness,
                            options.enthalpy, options.enthalpy_gamma,
                            options.enthalpy_ref);
  ssa_cpu.assemble_rhs(grid_cpu, thk_cpu, dhdx_cpu, dhdy_cpu, rhs_cpu);

  gpism::SSASolver solver_cpu(grid_cpu, 910.0, 9.81, 100.0, viscosity_cpu);

  gpism::set_device_enabled(true);
  gpism::Grid2D grid_gpu(mx, my, 1.0, 1.0, gw, 0, 1);
  gpism::Field2D<double> thk_gpu(mx, my, gw);
  gpism::Field2D<double> topg_gpu(mx, my, gw);
  gpism::Field2D<double> tauc_gpu(mx, my, gw);
  gpism::Field2D<int> cell_gpu(mx, my, gw);
  gpism::Field2D<double> usurf_gpu(mx, my, gw);
  gpism::Field2D<double> dhdx_gpu(mx, my, gw);
  gpism::Field2D<double> dhdy_gpu(mx, my, gw);
  gpism::Field2D<double> u_center_gpu(mx, my, gw);
  gpism::Field2D<double> v_center_gpu(mx, my, gw);
  gpism::FieldStag2D<double> beta_gpu(mx, my, gw);
  gpism::FieldStag2D<double> nuH_gpu(mx, my, gw);
  gpism::FieldStag2D<double> rhs_gpu(mx, my, gw);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      thk_gpu(i, j) = 2.0;
      topg_gpu(i, j) = 0.1 * i - 0.05 * j;
      tauc_gpu(i, j) = 120.0;
    }
  }

  gpism::ViscosityModel viscosity_gpu(1e-16, 3.0, 1.0);
  gpism::SSASolver solver_gpu(grid_gpu, 910.0, 9.81, 100.0, viscosity_gpu);
  gpism::FieldStag2D<double> vel_gpu(mx, my, gw);
  vel_gpu.fill(0.0);
  gpism::sync_host_to_device(thk_gpu);
  gpism::sync_host_to_device(topg_gpu);
  gpism::sync_host_to_device(tauc_gpu);
  gpism::sync_host_to_device(vel_gpu);
  gpism::sync_host_to_device(cell_gpu);
  gpism::sync_host_to_device(usurf_gpu);
  gpism::sync_host_to_device(dhdx_gpu);
  gpism::sync_host_to_device(dhdy_gpu);
  gpism::sync_host_to_device(u_center_gpu);
  gpism::sync_host_to_device(v_center_gpu);
  gpism::sync_host_to_device(beta_gpu);
  gpism::sync_host_to_device(nuH_gpu);
  gpism::sync_host_to_device(rhs_gpu);

  gpism::compute_cell_type(grid_gpu, thk_gpu, topg_gpu, options.sea_level,
                           options.rho_ice, options.rho_water, cell_gpu);
  gpism::compute_usurf_flotation(grid_gpu, thk_gpu, topg_gpu, cell_gpu,
                                 options.sea_level, options.rho_ice,
                                 options.rho_water, usurf_gpu);
  gpism::compute_surface_slopes_pism(grid_gpu, usurf_gpu, cell_gpu, dhdx_gpu,
                                     dhdy_gpu, options.surface_gradient_inward,
                                     options.surface_slope_uphill,
                                     options.use_cfbc);
  gpism::compute_cell_center_velocity(grid_gpu, vel_gpu, u_center_gpu,
                                      v_center_gpu);
  gpism::SSAOperator ssa_gpu(910.0, 9.81);
  ssa_gpu.compute_basal_drag(grid_gpu, tauc_gpu, u_center_gpu, v_center_gpu,
                             cell_gpu, beta_gpu, options.basal_params);
  viscosity_gpu.compute_nuH(grid_gpu, thk_gpu, vel_gpu, nuH_gpu,
                            options.nuH_regularization,
                            options.strength_extension_nu,
                            options.strength_extension_min_thickness,
                            options.enthalpy, options.enthalpy_gamma,
                            options.enthalpy_ref);
  ssa_gpu.assemble_rhs(grid_gpu, thk_gpu, dhdx_gpu, dhdy_gpu, rhs_gpu);

  gpism::sync_device_to_host(cell_gpu);
  gpism::sync_device_to_host(usurf_gpu);
  gpism::sync_device_to_host(dhdx_gpu);
  gpism::sync_device_to_host(dhdy_gpu);
  gpism::sync_device_to_host(u_center_gpu);
  gpism::sync_device_to_host(v_center_gpu);
  gpism::sync_device_to_host(beta_gpu);
  gpism::sync_device_to_host(nuH_gpu);
  gpism::sync_device_to_host(rhs_gpu);

  if (!compare_field(cell_cpu, cell_gpu, "cell_type")) {
    return 1;
  }
  if (!compare_field(usurf_cpu, usurf_gpu, 1e-10, "usurf")) {
    return 1;
  }
  if (!compare_field(dhdx_cpu, dhdx_gpu, 1e-10, "dhdx")) {
    return 1;
  }
  if (!compare_field(dhdy_cpu, dhdy_gpu, 1e-10, "dhdy")) {
    return 1;
  }
  if (!compare_field(u_center_cpu, u_center_gpu, 1e-10, "u_center")) {
    return 1;
  }
  if (!compare_field(v_center_cpu, v_center_gpu, 1e-10, "v_center")) {
    return 1;
  }
  if (!compare_stag(beta_cpu, beta_gpu, 1e-10, "beta")) {
    return 1;
  }
  if (!compare_stag(nuH_cpu, nuH_gpu, 1e-10, "nuH")) {
    return 1;
  }
  if (!compare_stag(rhs_cpu, rhs_gpu, 1e-10, "rhs")) {
    return 1;
  }

  gpism::set_device_enabled(false);
  gpism::MultigridHierarchy mg_cpu(grid_cpu, options.mg_min_size);
  copy(nuH_cpu, mg_cpu.level(0).nuH);
  copy(beta_cpu, mg_cpu.level(0).beta);
  for (int level = 1; level < mg_cpu.num_levels(); ++level) {
    restrict_stag(mg_cpu.level(level - 1).nuH, mg_cpu.level(level).nuH);
    restrict_stag(mg_cpu.level(level - 1).beta, mg_cpu.level(level).beta);
  }
  gpism::FieldStag2D<double> mg_z_cpu(mx, my, gw);
  gpism::MultigridPreconditioner mg_prec_cpu(
      mg_cpu, options.mg_pre_iters, options.mg_post_iters,
      options.mg_coarse_iters, options.mg_omega, options.mg_smoother,
      options.mg_cheby_lambda_min, options.mg_cheby_lambda_max,
      options.mg_cheby_estimate, options.mg_cheby_estimate_iters,
      options.mg_cheby_estimate_min_factor, options.mg_cheby_estimate_max_factor,
      nullptr, nullptr, false, options.basal_params.beta_ice_free_bedrock);
  mg_prec_cpu.apply(rhs_cpu, mg_z_cpu);

  gpism::set_device_enabled(true);
  gpism::MultigridHierarchy mg_gpu(grid_gpu, options.mg_min_size);
  copy(nuH_gpu, mg_gpu.level(0).nuH);
  copy(beta_gpu, mg_gpu.level(0).beta);
  for (int level = 1; level < mg_gpu.num_levels(); ++level) {
    restrict_stag(mg_gpu.level(level - 1).nuH, mg_gpu.level(level).nuH);
    restrict_stag(mg_gpu.level(level - 1).beta, mg_gpu.level(level).beta);
  }
  gpism::FieldStag2D<double> mg_z_gpu(mx, my, gw);
  gpism::MultigridPreconditioner mg_prec_gpu(
      mg_gpu, options.mg_pre_iters, options.mg_post_iters,
      options.mg_coarse_iters, options.mg_omega, options.mg_smoother,
      options.mg_cheby_lambda_min, options.mg_cheby_lambda_max,
      options.mg_cheby_estimate, options.mg_cheby_estimate_iters,
      options.mg_cheby_estimate_min_factor, options.mg_cheby_estimate_max_factor,
      nullptr, nullptr, false, options.basal_params.beta_ice_free_bedrock);
  mg_prec_gpu.apply(rhs_gpu, mg_z_gpu);
  gpism::sync_device_to_host(mg_z_gpu);

  if (!compare_stag(mg_z_cpu, mg_z_gpu, 1e-6, "MG preconditioner")) {
    return 1;
  }

  gpism::FieldStag2D<double> Ax_cpu(mx, my, gw);
  gpism::FieldStag2D<double> Ax_gpu(mx, my, gw);
  gpism::set_device_enabled(false);
  ssa_cpu.apply(grid_cpu, nuH_cpu, beta_cpu, vel_cpu, Ax_cpu, nullptr);
  const double rhs_dot_cpu = gpism::dot(rhs_cpu, rhs_cpu);
  gpism::set_device_enabled(true);
  gpism::sync_host_to_device(vel_gpu);
  ssa_gpu.apply(grid_gpu, nuH_gpu, beta_gpu, vel_gpu, Ax_gpu, nullptr);
  gpism::sync_device_to_host(Ax_gpu);
  const double rhs_dot_gpu = gpism::dot(rhs_gpu, rhs_gpu);

  if (!compare_stag(Ax_cpu, Ax_gpu, 1e-10, "Ax")) {
    return 1;
  }
  const double rhs_dot_diff = std::abs(rhs_dot_cpu - rhs_dot_gpu);
  if (!std::isfinite(rhs_dot_diff) ||
      rhs_dot_diff > 1e-10 * std::max(1.0, std::abs(rhs_dot_cpu))) {
    std::cerr << "rhs dot CPU/GPU mismatch: " << rhs_dot_cpu << " vs "
              << rhs_dot_gpu << "\n";
    return 1;
  }

  gpism::GMRESOptions gmres_opts;
  gmres_opts.restart = options.gmres_restart;
  gmres_opts.max_iter = options.gmres_max_iter;
  gmres_opts.tol = options.gmres_tol;
  gmres_opts.tol_relative = options.gmres_tol_relative;
  gmres_opts.verbose = false;
  gmres_opts.context = nullptr;

  gpism::set_device_enabled(false);
  gpism::FieldStag2D<double> vel_lin_cpu(mx, my, gw);
  vel_lin_cpu.fill(0.0);
  SSAOp op_lin_cpu(ssa_cpu, grid_cpu, nuH_cpu, beta_cpu);
  gpism::GMRESResult gmres_cpu =
      gpism::gmres_solve(op_lin_cpu, rhs_cpu, vel_lin_cpu, gmres_opts,
                         &mg_prec_cpu);

  gpism::set_device_enabled(true);
  gpism::FieldStag2D<double> vel_lin_gpu(mx, my, gw);
  vel_lin_gpu.fill(0.0);
  gpism::sync_host_to_device(vel_lin_gpu);
  SSAOp op_lin_gpu(ssa_gpu, grid_gpu, nuH_gpu, beta_gpu);
  gpism::GMRESResult gmres_gpu =
      gpism::gmres_solve(op_lin_gpu, rhs_gpu, vel_lin_gpu, gmres_opts,
                         &mg_prec_gpu);
  gpism::sync_device_to_host(vel_lin_gpu);

  const double denom_lin = std::max(1e-12, vel_norm2(vel_lin_cpu));
  const double rel_err_lin =
      std::sqrt(vel_diff_norm2(vel_lin_cpu, vel_lin_gpu) / denom_lin);
  if (!std::isfinite(rel_err_lin) || rel_err_lin > 5e-2) {
    std::cerr << "GMRES (MG precond) CPU/GPU parity error: " << rel_err_lin
              << "\n";
    std::cerr << "cpu_res=" << gmres_cpu.residual
              << " gpu_res=" << gmres_gpu.residual << "\n";
    return 1;
  }

  gpism::FieldStag2D<double> Ax_lin_cpu(mx, my, gw);
  gpism::FieldStag2D<double> Ax_lin_gpu(mx, my, gw);
  gpism::set_device_enabled(false);
  ssa_cpu.apply(grid_cpu, nuH_cpu, beta_cpu, vel_lin_cpu, Ax_lin_cpu, nullptr);
  gpism::set_device_enabled(true);
  gpism::FieldStag2D<double> vel_lin_match_gpu(mx, my, gw);
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      vel_lin_match_gpu(i, j, 0) = vel_lin_cpu(i, j, 0);
      vel_lin_match_gpu(i, j, 1) = vel_lin_cpu(i, j, 1);
    }
  }
  gpism::sync_host_to_device(vel_lin_match_gpu);
  ssa_gpu.apply(grid_gpu, nuH_gpu, beta_gpu, vel_lin_match_gpu, Ax_lin_gpu,
                nullptr);
  gpism::sync_device_to_host(Ax_lin_gpu);
  if (!compare_stag(Ax_lin_cpu, Ax_lin_gpu, 1e-6, "Ax (linear)")) {
    return 1;
  }

  gpism::Field2D<double> u_center_lin_cpu(mx, my, gw);
  gpism::Field2D<double> v_center_lin_cpu(mx, my, gw);
  gpism::Field2D<double> u_center_lin_gpu(mx, my, gw);
  gpism::Field2D<double> v_center_lin_gpu(mx, my, gw);
  gpism::FieldStag2D<double> beta_lin_cpu(mx, my, gw);
  gpism::FieldStag2D<double> beta_lin_gpu(mx, my, gw);
  gpism::FieldStag2D<double> nuH_lin_cpu(mx, my, gw);
  gpism::FieldStag2D<double> nuH_lin_gpu(mx, my, gw);

  gpism::set_device_enabled(false);
  gpism::compute_cell_center_velocity(grid_cpu, vel_lin_cpu, u_center_lin_cpu,
                                      v_center_lin_cpu);
  ssa_cpu.compute_basal_drag(grid_cpu, tauc_cpu, u_center_lin_cpu,
                             v_center_lin_cpu, cell_cpu, beta_lin_cpu,
                             options.basal_params);
  viscosity_cpu.compute_nuH(grid_cpu, thk_cpu, vel_lin_cpu, nuH_lin_cpu,
                            options.nuH_regularization,
                            options.strength_extension_nu,
                            options.strength_extension_min_thickness,
                            options.enthalpy, options.enthalpy_gamma,
                            options.enthalpy_ref);

  gpism::set_device_enabled(true);
  gpism::compute_cell_center_velocity(grid_gpu, vel_lin_gpu, u_center_lin_gpu,
                                      v_center_lin_gpu);
  ssa_gpu.compute_basal_drag(grid_gpu, tauc_gpu, u_center_lin_gpu,
                             v_center_lin_gpu, cell_gpu, beta_lin_gpu,
                             options.basal_params);
  viscosity_gpu.compute_nuH(grid_gpu, thk_gpu, vel_lin_gpu, nuH_lin_gpu,
                            options.nuH_regularization,
                            options.strength_extension_nu,
                            options.strength_extension_min_thickness,
                            options.enthalpy, options.enthalpy_gamma,
                            options.enthalpy_ref);
  gpism::sync_device_to_host(u_center_lin_gpu);
  gpism::sync_device_to_host(v_center_lin_gpu);
  gpism::sync_device_to_host(beta_lin_gpu);
  gpism::sync_device_to_host(nuH_lin_gpu);

  if (!compare_field(u_center_lin_cpu, u_center_lin_gpu, 1e-6,
                     "u_center (linear)")) {
    return 1;
  }
  if (!compare_field(v_center_lin_cpu, v_center_lin_gpu, 1e-6,
                     "v_center (linear)")) {
    return 1;
  }
  if (!compare_stag(beta_lin_cpu, beta_lin_gpu, 1e-6, "beta (linear)")) {
    return 1;
  }
  if (!compare_stag(nuH_lin_cpu, nuH_lin_gpu, 1e-6, "nuH (linear)")) {
    return 1;
  }

  gpism::set_device_enabled(false);
  gpism::SSASolverResult cpu_result =
      solver_cpu.solve(thk_cpu, topg_cpu, tauc_cpu, nullptr, nullptr, nullptr,
                       vel_cpu, options);
  gpism::set_device_enabled(true);
  gpism::SSASolverResult gpu_result =
      solver_gpu.solve(thk_gpu, topg_gpu, tauc_gpu, nullptr, nullptr, nullptr,
                       vel_gpu, options);
  gpism::sync_device_to_host(vel_gpu);

  if (cpu_result.picard_iters != gpu_result.picard_iters) {
    std::cerr << "Picard iteration count mismatch: cpu_iters="
              << cpu_result.picard_iters << " gpu_iters="
              << gpu_result.picard_iters << "\n";
  }

  if (!cpu_result.converged || !gpu_result.converged) {
    std::cerr << "Picard solver did not converge for CPU/GPU parity test\n";
    std::cerr << "cpu_iters=" << cpu_result.picard_iters
              << " cpu_res=" << cpu_result.linear_residual
              << " gpu_iters=" << gpu_result.picard_iters
              << " gpu_res=" << gpu_result.linear_residual << "\n";
    return 1;
  }

  const double denom = std::max(1e-12, vel_norm2(vel_cpu));
  const double rel_err = std::sqrt(vel_diff_norm2(vel_cpu, vel_gpu) / denom);

  if (!std::isfinite(rel_err) || rel_err > 5e-2) {
    std::cerr << "CPU/GPU parity relative error too large: " << rel_err << "\n";
    std::cerr << "cpu_norm=" << std::sqrt(vel_norm2(vel_cpu))
              << " gpu_norm=" << std::sqrt(vel_norm2(vel_gpu)) << "\n";
    std::cerr << "cpu_res=" << cpu_result.linear_residual
              << " gpu_res=" << gpu_result.linear_residual << "\n";
    return 1;
  }

  std::cout << "ssa_picard_parity_smoke passed\n";
  return 0;
}
