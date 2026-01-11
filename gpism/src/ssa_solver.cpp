#include "gpism/ssa_solver.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "gpism/gmres.h"

namespace gpism {
namespace {

template <typename T>
void sync_host_to_device(Field2D<T>& field) {
#if GPISM_HAVE_CUDA
  if (!field.host_staging_data() || field.elements() == 0) {
    return;
  }
  std::memcpy(field.host_staging_data(), field.data(),
              field.elements() * sizeof(T));
  field.copy_host_to_device();
#else
  (void)field;
#endif
}

template <typename T>
void sync_device_to_host(Field2D<T>& field) {
#if GPISM_HAVE_CUDA
  if (!field.host_staging_data() || field.elements() == 0) {
    return;
  }
  field.copy_device_to_host();
  std::memcpy(field.data(), field.host_staging_data(),
              field.elements() * sizeof(T));
#else
  (void)field;
#endif
}

template <typename T>
void sync_host_to_device(FieldStag2D<T>& field) {
  sync_host_to_device(field.component(0));
  sync_host_to_device(field.component(1));
}

template <typename T>
void sync_device_to_host(FieldStag2D<T>& field) {
  sync_device_to_host(field.component(0));
  sync_device_to_host(field.component(1));
}

double norm1(const FieldStag2D<double>& a) {
  double sum = 0.0;
  for (int j = 0; j < a.local_my(); ++j) {
    for (int i = 0; i < a.local_mx(); ++i) {
      sum += std::abs(a(i, j, 0)) + std::abs(a(i, j, 1));
    }
  }
  return sum;
}

double norm2(const FieldStag2D<double>& a) {
  double sum = 0.0;
  for (int j = 0; j < a.local_my(); ++j) {
    for (int i = 0; i < a.local_mx(); ++i) {
      sum += a(i, j, 0) * a(i, j, 0) + a(i, j, 1) * a(i, j, 1);
    }
  }
  return std::sqrt(sum);
}

double diff_norm1(const FieldStag2D<double>& a,
                  const FieldStag2D<double>& b) {
  double sum = 0.0;
  for (int j = 0; j < a.local_my(); ++j) {
    for (int i = 0; i < a.local_mx(); ++i) {
      sum += std::abs(a(i, j, 0) - b(i, j, 0)) +
             std::abs(a(i, j, 1) - b(i, j, 1));
    }
  }
  return sum;
}

double clamp_value(double value, double min_value, double max_value) {
  if (max_value > 0.0 && value > max_value) {
    value = max_value;
  }
  if (min_value > 0.0 && value < min_value) {
    value = min_value;
  }
  return value;
}

class SSAApplyOperator : public LinearOperator {
public:
  SSAApplyOperator(const SSAOperator& op, const Grid2D& grid,
                   const FieldStag2D<double>& nuH,
                   const FieldStag2D<double>& beta,
                   const SSABoundaryCondition* bc)
      : op_(op), grid_(grid), nuH_(nuH), beta_(beta), bc_(bc) {}

  void apply(const FieldStag2D<double>& x,
             FieldStag2D<double>& y) const override {
    op_.apply(grid_, nuH_, beta_, x, y, bc_);
  }

private:
  const SSAOperator& op_;
  const Grid2D& grid_;
  const FieldStag2D<double>& nuH_;
  const FieldStag2D<double>& beta_;
  const SSABoundaryCondition* bc_;
};

void build_bc_stag(const Grid2D& grid, const Field2D<double>* u_bc,
                   const Field2D<double>* v_bc,
                   const Field2D<int>* vel_bc_mask,
                   FieldStag2D<int>& mask_stag,
                   FieldStag2D<double>& values_stag) {
  mask_stag.fill(0);
  values_stag.fill(0.0);
  if (!vel_bc_mask || !u_bc || !v_bc) {
    return;
  }
  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      const int mask = (*vel_bc_mask)(i, j);
      mask_stag(i, j, 0) = mask;
      mask_stag(i, j, 1) = mask;
      values_stag(i, j, 0) = (*u_bc)(i, j);
      values_stag(i, j, 1) = (*v_bc)(i, j);
    }
  }
}

void copy_field(const Field2D<double>& src, Field2D<double>& dst) {
  for (int j = 0; j < src.local_my(); ++j) {
    for (int i = 0; i < src.local_mx(); ++i) {
      dst(i, j) = src(i, j);
    }
  }
}

}  // namespace

SSASolver::SSASolver(const Grid2D& grid, double rho, double g, double u_threshold,
                     const ViscosityModel& viscosity_model)
    : grid_(grid), ssa_(rho, g, u_threshold), viscosity_(viscosity_model) {}

SSASolverResult SSASolver::solve(const Field2D<double>& thk,
                                 const Field2D<double>& topg,
                                 const Field2D<double>& tauc,
                                 const Field2D<double>* u_bc,
                                 const Field2D<double>* v_bc,
                                 const Field2D<int>* vel_bc_mask,
                                 FieldStag2D<double>& vel,
                                 const SSASolverOptions& options) {
  SSASolverResult result{};

  Field2D<double> usurf(grid_.local_mx(), grid_.local_my(), grid_.ghost_width());
  Field2D<double> dhdx(grid_.local_mx(), grid_.local_my(), grid_.ghost_width());
  Field2D<double> dhdy(grid_.local_mx(), grid_.local_my(), grid_.ghost_width());
  FieldStag2D<double> beta(grid_.local_mx(), grid_.local_my(),
                           grid_.ghost_width());
  FieldStag2D<double> rhs(grid_.local_mx(), grid_.local_my(),
                          grid_.ghost_width());
  FieldStag2D<double> nuH(grid_.local_mx(), grid_.local_my(),
                          grid_.ghost_width());
  FieldStag2D<double> nuH_prev(grid_.local_mx(), grid_.local_my(),
                               grid_.ghost_width());
  FieldStag2D<double> vel_prev(grid_.local_mx(), grid_.local_my(),
                               grid_.ghost_width());

  FieldStag2D<int> bc_mask(grid_.local_mx(), grid_.local_my(), grid_.ghost_width());
  FieldStag2D<double> bc_values(grid_.local_mx(), grid_.local_my(),
                                grid_.ghost_width());
  SSABoundaryCondition bc;
  if (options.use_bc) {
    build_bc_stag(grid_, u_bc, v_bc, vel_bc_mask, bc_mask, bc_values);
    bc.mask = &bc_mask;
    bc.values = &bc_values;
    sync_host_to_device(bc_mask);
    sync_host_to_device(bc_values);
  }

  GeometryDiagnostics::compute_usurf_cpu(grid_, thk, topg, usurf);
  GeometryDiagnostics::compute_surface_slopes_cpu(grid_, usurf, dhdx, dhdy);

  Field2D<double> thk_dev(grid_.local_mx(), grid_.local_my(), grid_.ghost_width());
  Field2D<double> tauc_dev(grid_.local_mx(), grid_.local_my(), grid_.ghost_width());
  Field2D<double> dhdx_dev(grid_.local_mx(), grid_.local_my(), grid_.ghost_width());
  Field2D<double> dhdy_dev(grid_.local_mx(), grid_.local_my(), grid_.ghost_width());
  copy_field(thk, thk_dev);
  copy_field(tauc, tauc_dev);
  copy_field(dhdx, dhdx_dev);
  copy_field(dhdy, dhdy_dev);

  sync_host_to_device(thk_dev);
  sync_host_to_device(tauc_dev);
  sync_host_to_device(dhdx_dev);
  sync_host_to_device(dhdy_dev);

  ssa_.compute_basal_drag(grid_, tauc_dev, beta);
  sync_device_to_host(beta);
  sync_host_to_device(beta);

  ssa_.assemble_rhs(grid_, thk_dev, dhdx_dev, dhdy_dev, rhs,
                    options.use_bc ? &bc : nullptr);
  sync_device_to_host(rhs);
  sync_host_to_device(rhs);

  for (int iter = 0; iter < options.max_picard; ++iter) {
    for (int j = 0; j < grid_.local_my(); ++j) {
      for (int i = 0; i < grid_.local_mx(); ++i) {
        nuH_prev(i, j, 0) = nuH(i, j, 0);
        nuH_prev(i, j, 1) = nuH(i, j, 1);
        vel_prev(i, j, 0) = vel(i, j, 0);
        vel_prev(i, j, 1) = vel(i, j, 1);
      }
    }

    viscosity_.compute_nuH(grid_, thk, vel, nuH);
    if (options.nuH_min > 0.0 || options.nuH_max > 0.0 ||
        options.nuH_relax < 1.0) {
      for (int j = 0; j < grid_.local_my(); ++j) {
        for (int i = 0; i < grid_.local_mx(); ++i) {
          for (int comp = 0; comp < 2; ++comp) {
            double value = nuH(i, j, comp);
            value = clamp_value(value, options.nuH_min, options.nuH_max);
            if (options.nuH_relax < 1.0) {
              value = options.nuH_relax * value +
                      (1.0 - options.nuH_relax) * nuH_prev(i, j, comp);
            }
            nuH(i, j, comp) = value;
          }
        }
      }
    }
    sync_host_to_device(nuH);
    sync_host_to_device(vel);

    GMRESOptions gmres_opts;
    gmres_opts.restart = options.gmres_restart;
    gmres_opts.max_iter = options.gmres_max_iter;
    gmres_opts.tol = options.gmres_tol;

    SSAApplyOperator op(ssa_, grid_, nuH, beta, options.use_bc ? &bc : nullptr);
    GMRESResult gmres_result = gmres_solve(op, rhs, vel, gmres_opts);
    result.linear_iters = gmres_result.iterations;
    result.linear_residual = gmres_result.residual;

    sync_device_to_host(vel);
    sync_device_to_host(nuH);

    if (options.vel_relax < 1.0) {
      for (int j = 0; j < grid_.local_my(); ++j) {
        for (int i = 0; i < grid_.local_mx(); ++i) {
          vel(i, j, 0) = options.vel_relax * vel(i, j, 0) +
                         (1.0 - options.vel_relax) * vel_prev(i, j, 0);
          vel(i, j, 1) = options.vel_relax * vel(i, j, 1) +
                         (1.0 - options.vel_relax) * vel_prev(i, j, 1);
        }
      }
    }

    const double nuH_diff = diff_norm1(nuH, nuH_prev);
    const double nuH_norm = std::max(norm1(nuH_prev), 1e-12);
    result.nuH_change = nuH_diff / nuH_norm;

    const double vel_diff = diff_norm1(vel, vel_prev);
    const double vel_norm = std::max(norm1(vel_prev), 1e-12);
    result.vel_change = vel_diff / vel_norm;

    result.picard_iters = iter + 1;
    if (result.nuH_change <= options.tol_nuH &&
        result.vel_change <= options.tol_vel) {
      result.converged = true;
      break;
    }
  }

  return result;
}

}  // namespace gpism
