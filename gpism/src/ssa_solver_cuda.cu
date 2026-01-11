#include "gpism/ssa_solver.h"

#include "gpism/profile.h"

#include <cuda_runtime.h>

namespace gpism {
namespace {

__device__ inline int idx(int i, int j, int gw, int stride) {
  return (j + gw) * stride + (i + gw);
}

__global__ void relax_nuH_kernel(int mx, int my, int gw, int stride,
                                 const double* prev, double* cur,
                                 double min_value, double max_value,
                                 double relax) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int id = idx(i, j, gw, stride);
  double value = cur[id];
  if (max_value > 0.0 && value > max_value) {
    value = max_value;
  }
  if (min_value > 0.0 && value < min_value) {
    value = min_value;
  }
  if (relax < 1.0) {
    value = relax * value + (1.0 - relax) * prev[id];
  }
  cur[id] = value;
}

__global__ void relax_vel_kernel(int mx, int my, int gw, int stride,
                                 const double* prev, double* cur,
                                 double relax) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int id = idx(i, j, gw, stride);
  cur[id] = relax * cur[id] + (1.0 - relax) * prev[id];
}

}  // namespace

void ssa_relax_nuH_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                        const double* nuH_prev_u, const double* nuH_prev_v,
                        double* nuH_u, double* nuH_v, double nuH_min,
                        double nuH_max, double nuH_relax) {
  CudaEventTimer timer("ssa_relax_nuH");
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  relax_nuH_kernel<<<grid, block>>>(mx, my, gw, stride_u, nuH_prev_u, nuH_u,
                                    nuH_min, nuH_max, nuH_relax);
  relax_nuH_kernel<<<grid, block>>>(mx, my, gw, stride_v, nuH_prev_v, nuH_v,
                                    nuH_min, nuH_max, nuH_relax);
}

void ssa_relax_vel_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                        const double* vel_prev_u, const double* vel_prev_v,
                        double* vel_u, double* vel_v, double vel_relax) {
  CudaEventTimer timer("ssa_relax_vel");
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  relax_vel_kernel<<<grid, block>>>(mx, my, gw, stride_u, vel_prev_u, vel_u,
                                    vel_relax);
  relax_vel_kernel<<<grid, block>>>(mx, my, gw, stride_v, vel_prev_v, vel_v,
                                    vel_relax);
}

}  // namespace gpism
