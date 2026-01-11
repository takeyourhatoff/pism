#include "gpism/config.h"
#include "gpism/context.h"
#include "gpism/linear_algebra.h"
#include "gpism/ssa_operator.h"

#include <cmath>
#include <cstring>
#include <iostream>

#if GPISM_HAVE_CUDA
#include <cuda_runtime.h>
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

bool all_finite(const gpism::FieldStag2D<double>& field) {
  for (int j = 0; j < field.local_my(); ++j) {
    for (int i = 0; i < field.local_mx(); ++i) {
      for (int comp = 0; comp < 2; ++comp) {
        if (!std::isfinite(field(i, j, comp))) {
          return false;
        }
      }
    }
  }
  return true;
}

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
  gpism::SSAOperator op(910.0, 9.81, 100.0);

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
  gpism::FieldStag2D<double> rhs(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> nuH(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> vel(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> out(grid.local_mx(), grid.local_my(), gw);
  gpism::FieldStag2D<double> residual(grid.local_mx(), grid.local_my(), gw);

  op.compute_basal_drag(grid, tauc, beta);
  op.assemble_rhs(grid, thk, dhdx, dhdy, rhs);

  sync_device_to_host(beta);
  sync_device_to_host(rhs);

  nuH.fill(0.0);

  for (int j = -gw; j < grid.local_my() + gw; ++j) {
    for (int i = -gw; i < grid.local_mx() + gw; ++i) {
      for (int comp = 0; comp < 2; ++comp) {
        const double denom = beta(i, j, comp);
        vel(i, j, comp) = (denom != 0.0) ? rhs(i, j, comp) / denom : 0.0;
      }
    }
  }

  sync_host_to_device(nuH);
  sync_host_to_device(vel);

  op.apply(grid, nuH, beta, vel, out);

  gpism::copy(out, residual);
  gpism::axpy(-1.0, rhs, residual);
  const double res_norm = gpism::norm2(residual);
  const double rhs_norm = gpism::norm2(rhs);
  const double tol = 1e-8 * std::max(1.0, rhs_norm);

  sync_device_to_host(out);
  sync_device_to_host(residual);

  if (!all_finite(out) || !all_finite(residual)) {
    std::cerr << "NaN detected in SSA operator MPI smoke test\n";
    return 1;
  }

  if (!(std::isfinite(res_norm) && res_norm <= tol)) {
    std::cerr << "Residual norm too large: " << res_norm << " > " << tol
              << "\n";
    return 1;
  }

  std::cout << "ssa_operator_mpi_smoke passed\n";
  return 0;
}
