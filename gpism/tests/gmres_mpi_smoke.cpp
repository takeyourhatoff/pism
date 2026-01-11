#include "gpism/config.h"
#include "gpism/context.h"
#include "gpism/gmres.h"
#include "gpism/linear_algebra.h"
#include "gpism/ssa_operator.h"

#include <cmath>
#include <cstring>
#include <iostream>

#if GPISM_HAVE_CUDA
#include <cuda_runtime.h>
#endif

#if GPISM_HAVE_MPI
#include <mpi.h>
#endif

namespace {

template <typename T>
void sync_host_to_device(gpism::Field2D<T>& field) {
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
void sync_device_to_host(gpism::Field2D<T>& field) {
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
void sync_host_to_device(gpism::FieldStag2D<T>& field) {
  sync_host_to_device(field.component(0));
  sync_host_to_device(field.component(1));
}

template <typename T>
void sync_device_to_host(gpism::FieldStag2D<T>& field) {
  sync_device_to_host(field.component(0));
  sync_device_to_host(field.component(1));
}

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

}  // namespace

int main(int argc, char** argv) {
  gpism::Context context(&argc, &argv);

#if GPISM_HAVE_CUDA
  cudaSetDevice(context.device_id());
#endif

  const int global_mx = 8;
  const int global_my = 6;
  const int gw = 1;
  gpism::Grid2D grid(global_mx, global_my, 1000.0, 1000.0, gw, context.rank(),
                     context.size());
  gpism::SSAOperator ssa(910.0, 9.81, 100.0);

  gpism::Field2D<double> tauc(grid.local_mx(), grid.local_my(), gw);
  gpism::Field2D<double> thk(grid.local_mx(), grid.local_my(), gw);
  gpism::Field2D<double> dhdx(grid.local_mx(), grid.local_my(), gw);
  gpism::Field2D<double> dhdy(grid.local_mx(), grid.local_my(), gw);

  tauc.fill(100.0);
  thk.fill(1000.0);
  dhdx.fill(0.01);
  dhdy.fill(-0.02);

  sync_host_to_device(tauc);
  sync_host_to_device(thk);
  sync_host_to_device(dhdx);
  sync_host_to_device(dhdy);

  gpism::FieldStag2D<double> beta(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> nuH(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> rhs(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> x(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> Ax(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> r(grid.local_mx(), grid.local_my(), gw);

  ssa.compute_basal_drag(grid, tauc, beta);
  ssa.assemble_rhs(grid, thk, dhdx, dhdy, rhs);

  sync_device_to_host(beta);
  sync_device_to_host(rhs);

  nuH.fill(0.0);
  sync_host_to_device(nuH);
  sync_host_to_device(rhs);

  gpism::set(0.0, x);
  SSAOperatorWrapper op(ssa, grid, nuH, beta, nullptr);

  gpism::GMRESOptions opts;
  opts.restart = 5;
  opts.max_iter = 10;
  opts.tol = 1e-12;

  gpism::GMRESResult res = gpism::gmres_solve(op, rhs, x, opts);
  if (!res.converged) {
    std::cerr << "GMRES MPI SSA solve did not converge\n";
    return 1;
  }

  op.apply(x, Ax);
  gpism::copy(rhs, r);
  gpism::axpy(-1.0, Ax, r);
  const double res_norm = gpism::norm2(r);
  const double rhs_norm = gpism::norm2(rhs);
  const double tol = 1e-8 * std::max(1.0, rhs_norm);
  if (!(std::isfinite(res_norm) && res_norm <= tol)) {
    std::cerr << "Residual norm too large: " << res_norm << " > " << tol
              << "\n";
    return 1;
  }

#if GPISM_HAVE_MPI
  int min_iter = res.iterations;
  int max_iter = res.iterations;
  MPI_Allreduce(&res.iterations, &min_iter, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&res.iterations, &max_iter, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (max_iter - min_iter > 0) {
    std::cerr << "Iteration counts differ across ranks: " << min_iter << " vs "
              << max_iter << "\n";
    return 1;
  }
#endif

  if (res.iterations > 5) {
    std::cerr << "Iteration count too large for diagonal SSA case: "
              << res.iterations << "\n";
    return 1;
  }

  std::cout << "gmres_mpi_smoke passed\n";
  return 0;
}
