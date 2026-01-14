#include "gpism/context.h"
#include "gpism/device_policy.h"
#include "gpism/field_sync.h"
#include "gpism/gmres.h"
#include "gpism/halo_exchange.h"
#include "gpism/linear_algebra.h"
#include "gpism/mg_preconditioner.h"
#include "gpism/geometry.h"
#include "gpism/ssa_operator.h"

#include <cmath>
#include <iostream>

#if GPISM_HAVE_MPI
#include <mpi.h>
#endif

namespace {

struct SSAOperatorWrapper : public gpism::LinearOperator {
  SSAOperatorWrapper(const gpism::SSAOperator& op, const gpism::Grid2D& grid,
                     const gpism::FieldStag2D<double>& nuH,
                     const gpism::FieldStag2D<double>& beta,
                     const gpism::SSABoundaryCondition* bc)
      : op_(op), grid_(grid), nuH_(nuH), beta_(beta), bc_(bc) {}

  void apply(const gpism::FieldStag2D<double>& x,
             gpism::FieldStag2D<double>& y) const override {
    op_.apply(grid_, nuH_, beta_, x, y, bc_);
  }

  const gpism::SSAOperator& op_;
  const gpism::Grid2D& grid_;
  const gpism::FieldStag2D<double>& nuH_;
  const gpism::FieldStag2D<double>& beta_;
  const gpism::SSABoundaryCondition* bc_;
};

template <typename T>
void exchange_for_device(gpism::FieldStag2D<T>& field, const gpism::Grid2D& grid,
                         const gpism::Context& context) {
  if (!context.mpi_enabled()) {
    return;
  }
  gpism::HaloExchange2D exchange;
#if GPISM_HAVE_CUDA
  const bool can_device = context.cuda_aware_mpi() &&
                          field.component(0).device_data() != nullptr &&
                          field.component(1).device_data() != nullptr;
  if (can_device) {
    exchange.exchange(field, grid, context, gpism::HaloExchange2D::Mode::Device);
    return;
  }
#endif
  gpism::sync_device_to_host(field);
  exchange.exchange(field, grid, context, gpism::HaloExchange2D::Mode::Host);
  gpism::sync_host_to_device(field);
}

double global_norm2(const gpism::FieldStag2D<double>& field,
                    const gpism::Context& context) {
  const double local = gpism::dot(field, field);
#if GPISM_HAVE_MPI
  if (context.mpi_enabled()) {
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return std::sqrt(global);
  }
#endif
  return std::sqrt(local);
}

struct Effectiveness {
  double r_norm = 0.0;
  double post_norm = 0.0;
};

void fill_pattern(gpism::FieldStag2D<double>& x) {
  const int mx = x.local_mx();
  const int my = x.local_my();
  const int gw = x.ghost_width();
  for (int j = -gw; j < my + gw; ++j) {
    for (int i = -gw; i < mx + gw; ++i) {
      const double val =
          std::sin(0.19 * static_cast<double>(i)) +
          std::cos(0.13 * static_cast<double>(j));
      x(i, j, 0) = val;
      x(i, j, 1) = 0.5 * val;
    }
  }
}

void build_bc(const gpism::Grid2D& grid, const gpism::FieldStag2D<double>& x_true,
              gpism::FieldStag2D<int>& mask, gpism::FieldStag2D<double>& values) {
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  const int xs = grid.xs();
  const int ys = grid.ys();
  const int gmx = grid.global_mx();
  const int gmy = grid.global_my();
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const int gi = xs + i;
      const int gj = ys + j;
      const bool is_boundary =
          (gi == 0 || gi == gmx - 1 || gj == 0 || gj == gmy - 1);
      for (int comp = 0; comp < 2; ++comp) {
        mask(i, j, comp) = is_boundary ? 1 : 0;
        values(i, j, comp) = is_boundary ? x_true(i, j, comp) : 0.0;
      }
    }
  }
}

Effectiveness measure_effectiveness(const gpism::Context& context,
                                    bool use_device) {
  const bool prev_device = gpism::device_enabled();
  gpism::set_device_enabled(use_device);

  const int gw = 1;
  gpism::Grid2D grid(32, 32, 1000.0, 1000.0, gw, context.rank(), context.size());
  gpism::SSAOperator ssa(910.0, 9.81);

  gpism::Field2D<double> tauc(grid.local_mx(), grid.local_my(), gw);
  tauc.fill(5.0);
  gpism::Field2D<double> u_center(grid.local_mx(), grid.local_my(), gw);
  gpism::Field2D<double> v_center(grid.local_mx(), grid.local_my(), gw);
  gpism::Field2D<int> cell_type(grid.local_mx(), grid.local_my(), gw);
  u_center.fill(0.0);
  v_center.fill(0.0);
  cell_type.fill(gpism::GroundedIce);
  gpism::FieldStag2D<double> beta(grid.local_mx(), grid.local_my(), gw);

  gpism::FieldStag2D<double> nuH(grid.local_mx(), grid.local_my(), gw);
  for (int j = -gw; j < grid.local_my() + gw; ++j) {
    for (int i = -gw; i < grid.local_mx() + gw; ++i) {
      const double bump =
          1.0 + 0.4 * std::sin(0.09 * static_cast<double>(i)) *
                    std::cos(0.07 * static_cast<double>(j));
      nuH(i, j, 0) = 5.0 * bump;
      nuH(i, j, 1) = 5.0 * bump;
    }
  }

  gpism::FieldStag2D<double> x_true(grid.local_mx(), grid.local_my(), gw);
  fill_pattern(x_true);

  gpism::FieldStag2D<int> bc_mask(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> bc_values(grid.local_mx(), grid.local_my(), gw);
  build_bc(grid, x_true, bc_mask, bc_values);
  gpism::SSABoundaryCondition bc;
  bc.mask = &bc_mask;
  bc.values = &bc_values;

  if (use_device) {
    gpism::sync_host_to_device(tauc);
    gpism::sync_host_to_device(u_center);
    gpism::sync_host_to_device(v_center);
    gpism::sync_host_to_device(cell_type);
    gpism::sync_host_to_device(nuH);
    gpism::sync_host_to_device(x_true);
    gpism::sync_host_to_device(bc_mask);
    gpism::sync_host_to_device(bc_values);
  }

  gpism::BasalResistanceParams basal_params;
  ssa.compute_basal_drag(grid, tauc, u_center, v_center, cell_type, beta,
                         basal_params);
  if (context.mpi_enabled()) {
    exchange_for_device(beta, grid, context);
  }

  SSAOperatorWrapper op(ssa, grid, nuH, beta, &bc);

  gpism::FieldStag2D<double> b(grid.local_mx(), grid.local_my(), gw);
  if (context.mpi_enabled()) {
    exchange_for_device(x_true, grid, context);
  }
  op.apply(x_true, b);
  if (context.mpi_enabled()) {
    exchange_for_device(b, grid, context);
  }

  gpism::MultigridHierarchy mg(grid, 4);
  gpism::copy(nuH, mg.level(0).nuH);
  gpism::copy(beta, mg.level(0).beta);
  for (int level = 1; level < mg.num_levels(); ++level) {
    gpism::restrict_stag(mg.level(level - 1).nuH, mg.level(level).nuH);
    gpism::restrict_stag(mg.level(level - 1).beta, mg.level(level).beta);
  }

  gpism::MultigridPreconditioner precond(
      mg, 2, 2, 10, 0.8, gpism::MGSmoother::Jacobi, 0.1, 2.0, false, 5, 0.1,
      1.1, &bc, &context);

  gpism::FieldStag2D<double> x0(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> Ax(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> r(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> z(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> Az(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> r2(grid.local_mx(), grid.local_my(), gw);

  gpism::set(0.0, x0);
  if (context.mpi_enabled()) {
    exchange_for_device(x0, grid, context);
  }
  op.apply(x0, Ax);
  gpism::copy(b, r);
  gpism::axpy(-1.0, Ax, r);

  precond.apply(r, z);
  if (context.mpi_enabled()) {
    exchange_for_device(z, grid, context);
  }
  op.apply(z, Az);
  gpism::copy(r, r2);
  gpism::axpy(-1.0, Az, r2);

  Effectiveness eff;
  eff.r_norm = global_norm2(r, context);
  eff.post_norm = global_norm2(r2, context);

  gpism::set_device_enabled(prev_device);
  return eff;
}

}  // namespace

int main(int argc, char** argv) {
  gpism::Context context(&argc, &argv);

  const Effectiveness cpu = measure_effectiveness(context, false);
  if (context.rank() == 0) {
    std::cout << "MG effectiveness BC MPI CPU: ||r||=" << cpu.r_norm
              << " ||r-Az||=" << cpu.post_norm
              << " ratio="
              << (cpu.r_norm > 0.0 ? cpu.post_norm / cpu.r_norm : 0.0) << "\n";
  }

  if (!(cpu.post_norm < cpu.r_norm)) {
    std::cerr << "MG did not reduce residual on CPU\n";
    return 1;
  }

#if GPISM_HAVE_CUDA
  const Effectiveness gpu = measure_effectiveness(context, true);
  if (context.rank() == 0) {
    std::cout << "MG effectiveness BC MPI GPU: ||r||=" << gpu.r_norm
              << " ||r-Az||=" << gpu.post_norm
              << " ratio="
              << (gpu.r_norm > 0.0 ? gpu.post_norm / gpu.r_norm : 0.0) << "\n";
  }

  if (!(gpu.post_norm < gpu.r_norm)) {
    std::cerr << "MG did not reduce residual on GPU\n";
    return 1;
  }

  const double ratio_cpu = cpu.post_norm / cpu.r_norm;
  const double ratio_gpu = gpu.post_norm / gpu.r_norm;
  const double rel =
      (ratio_cpu != 0.0) ? std::abs(ratio_cpu - ratio_gpu) / ratio_cpu : 0.0;
  if (rel > 1e-3) {
    std::cerr << "MG CPU/GPU ratio mismatch: " << ratio_cpu << " vs "
              << ratio_gpu << "\n";
    return 1;
  }
#endif

  if (context.rank() == 0) {
    std::cout << "mg_preconditioner_effectiveness_bc_mpi_smoke passed\n";
  }
  return 0;
}
