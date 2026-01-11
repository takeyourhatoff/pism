#include "gpism/geometry.h"

#include "gpism/profile.h"

#include <cuda_runtime.h>

namespace gpism {
namespace {

__global__ void usurf_kernel(int mx, int my, const double* thk, const double* topg,
                             double* usurf, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  int idx = (j + 1) * stride + (i + 1);
  usurf[idx] = topg[idx] + thk[idx];
}

__global__ void slope_kernel(int mx, int my, const double* usurf, double* dhdx,
                             double* dhdy, int stride, double inv_dx,
                             double inv_dy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  int idx = (j + 1) * stride + (i + 1);
  int idx_left = (j + 1) * stride + (i);
  int idx_right = (j + 1) * stride + (i + 2);
  int idx_down = (j)*stride + (i + 1);
  int idx_up = (j + 2) * stride + (i + 1);

  double left = usurf[idx_left];
  double right = usurf[idx_right];
  double down = usurf[idx_down];
  double up = usurf[idx_up];

  dhdx[idx] = 0.5 * (right - left) * inv_dx;
  dhdy[idx] = 0.5 * (up - down) * inv_dy;
}

}  // namespace

void GeometryDiagnostics::compute_usurf(const Grid2D& grid, const Field2D<double>& thk,
                                        const Field2D<double>& topg,
                                        Field2D<double>& usurf) {
  if (thk.has_device_data() && topg.has_device_data() && usurf.has_device_data()) {
    CudaEventTimer timer("geometry_usurf");
    dim3 block(16, 16);
    dim3 grid_dim((grid.local_mx() + block.x - 1) / block.x,
                  (grid.local_my() + block.y - 1) / block.y);
    usurf_kernel<<<grid_dim, block>>>(grid.local_mx(), grid.local_my(),
                                      thk.device_data(), topg.device_data(),
                                      usurf.device_data(), thk.stride());
    return;
  }
  GeometryDiagnostics::compute_usurf_cpu(grid, thk, topg, usurf);
}

void GeometryDiagnostics::compute_surface_slopes(const Grid2D& grid,
                                                 const Field2D<double>& usurf,
                                                 Field2D<double>& dhdx,
                                                 Field2D<double>& dhdy) {
  if (usurf.has_device_data() && dhdx.has_device_data() && dhdy.has_device_data()) {
    CudaEventTimer timer("geometry_slopes");
    dim3 block(16, 16);
    dim3 grid_dim((grid.local_mx() + block.x - 1) / block.x,
                  (grid.local_my() + block.y - 1) / block.y);
    slope_kernel<<<grid_dim, block>>>(grid.local_mx(), grid.local_my(),
                                      usurf.device_data(), dhdx.device_data(),
                                      dhdy.device_data(), usurf.stride(),
                                      1.0 / grid.dx(), 1.0 / grid.dy());
    return;
  }
  GeometryDiagnostics::compute_surface_slopes_cpu(grid, usurf, dhdx, dhdy);
}

}  // namespace gpism
