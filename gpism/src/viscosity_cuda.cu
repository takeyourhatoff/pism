#include "gpism/viscosity.h"

#include <cmath>

#include "gpism/profile.h"

#include <cuda_runtime.h>

namespace gpism {
namespace {

__device__ inline int idx(int i, int j, int stride) {
  return j * stride + i;
}

__device__ inline int idx(int i, int j, int gw, int stride) {
  return (j + gw) * stride + (i + gw);
}

__global__ void center_velocity_kernel(int mx, int my, int gw, int stride_u,
                                       int stride_v, int stride_center,
                                       const double* u_face,
                                       const double* v_face, double* u_center,
                                       double* v_center) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int il = (i == 0) ? i : i - 1;
  const int jd = (j == 0) ? j : j - 1;
  u_center[idx(i, j, stride_center)] =
      0.5 * (u_face[idx(il, j, gw, stride_u)] +
             u_face[idx(i, j, gw, stride_u)]);
  v_center[idx(i, j, stride_center)] =
      0.5 * (v_face[idx(i, jd, gw, stride_v)] +
             v_face[idx(i, j, gw, stride_v)]);
}

__global__ void nu_center_kernel(int mx, int my, int stride_center,
                                 const double* u_center,
                                 const double* v_center,
                                 const double* temp_avg, double* nu_center,
                                 double gamma, double ref, double B,
                                 double n_eff, double eps0, double inv_dx,
                                 double inv_dy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int il = (i == 0) ? i : i - 1;
  const int ir = (i == mx - 1) ? i : i + 1;
  const int jd = (j == 0) ? j : j - 1;
  const int ju = (j == my - 1) ? j : j + 1;

  const double du_dx =
      (u_center[idx(ir, j, stride_center)] -
       u_center[idx(il, j, stride_center)]) *
      ((i == 0 || i == mx - 1) ? inv_dx : 0.5 * inv_dx);
  const double dv_dy =
      (v_center[idx(i, ju, stride_center)] -
       v_center[idx(i, jd, stride_center)]) *
      ((j == 0 || j == my - 1) ? inv_dy : 0.5 * inv_dy);
  const double du_dy =
      (u_center[idx(i, ju, stride_center)] -
       u_center[idx(i, jd, stride_center)]) *
      ((j == 0 || j == my - 1) ? inv_dy : 0.5 * inv_dy);
  const double dv_dx =
      (v_center[idx(ir, j, stride_center)] -
       v_center[idx(il, j, stride_center)]) *
      ((i == 0 || i == mx - 1) ? inv_dx : 0.5 * inv_dx);

  const double eps_xx = du_dx;
  const double eps_yy = dv_dy;
  const double eps_xy = 0.5 * (du_dy + dv_dx);
  const double eps2 =
      0.5 * (eps_xx * eps_xx + eps_yy * eps_yy) + eps_xy * eps_xy;
  const double eps_e = sqrt(eps2 + eps0 * eps0);

  double nu = 0.5 * B * pow(eps_e, (1.0 / n_eff) - 1.0);
  if (temp_avg && gamma != 0.0) {
    const double temp = temp_avg[idx(i, j, stride_center)];
    nu *= exp(-gamma * (temp - ref));
  }
  nu_center[idx(i, j, stride_center)] = nu;
}

__global__ void column_avg_kernel(int mx, int my, int gw, int nz, int stride,
                                  const double* enthalpy, double* temp_avg) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  double sum = 0.0;
  const int base = (j + gw) * stride + (i + gw);
  const int offset = base * nz;
  for (int k = 0; k < nz; ++k) {
    sum += enthalpy[offset + k];
  }
  temp_avg[idx(i, j, mx)] = sum / static_cast<double>(nz);
}

__global__ void nuH_kernel(int mx, int my, int gw, int stride_thk,
                           int stride_center, int stride_u, int stride_v,
                           const double* thk, const double* nu_center,
                           double* nuH_u, double* nuH_v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int il = (i == 0) ? i : i - 1;
  const int jd = (j == 0) ? j : j - 1;

  const double nu_left = nu_center[idx(il, j, stride_center)];
  const double nu_right = nu_center[idx(i, j, stride_center)];
  const double H_left = thk[idx(il, j, gw, stride_thk)];
  const double H_right = thk[idx(i, j, gw, stride_thk)];
  nuH_u[idx(i, j, gw, stride_u)] =
      0.5 * (nu_left + nu_right) * 0.5 * (H_left + H_right);

  const double nu_down = nu_center[idx(i, jd, stride_center)];
  const double nu_up = nu_center[idx(i, j, stride_center)];
  const double H_down = thk[idx(i, jd, gw, stride_thk)];
  const double H_up = thk[idx(i, j, gw, stride_thk)];
  nuH_v[idx(i, j, gw, stride_v)] =
      0.5 * (nu_down + nu_up) * 0.5 * (H_down + H_up);
}

}  // namespace

void viscosity_compute_nuH_cuda(int mx, int my, int gw, int stride_thk,
                                int stride_u, int stride_v, int stride_nuH_u,
                                int stride_nuH_v, const double* thk,
                                const double* u, const double* v, double* nuH_u,
                                double* nuH_v, const double* enthalpy, int nz,
                                int enthalpy_gw, int enthalpy_stride,
                                double enthalpy_gamma, double enthalpy_ref,
                                double B, double n_eff, double eps0,
                                double inv_dx, double inv_dy) {
  CudaEventTimer timer("viscosity_nuH");
  const std::size_t count = static_cast<std::size_t>(mx) * my;
  double* u_center = nullptr;
  double* v_center = nullptr;
  double* nu_center = nullptr;
  double* temp_avg = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&u_center), count * sizeof(double));
  cudaMalloc(reinterpret_cast<void**>(&v_center), count * sizeof(double));
  cudaMalloc(reinterpret_cast<void**>(&nu_center), count * sizeof(double));

  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);

  if (enthalpy && enthalpy_gamma != 0.0) {
    cudaMalloc(reinterpret_cast<void**>(&temp_avg), count * sizeof(double));
    column_avg_kernel<<<grid, block>>>(mx, my, enthalpy_gw, nz,
                                       enthalpy_stride, enthalpy, temp_avg);
  }

  center_velocity_kernel<<<grid, block>>>(mx, my, gw, stride_u, stride_v, mx, u,
                                          v, u_center, v_center);
  nu_center_kernel<<<grid, block>>>(mx, my, mx, u_center, v_center, temp_avg,
                                    nu_center, enthalpy_gamma, enthalpy_ref, B,
                                    n_eff, eps0, inv_dx, inv_dy);
  nuH_kernel<<<grid, block>>>(mx, my, gw, stride_thk, mx, stride_nuH_u,
                              stride_nuH_v, thk, nu_center, nuH_u, nuH_v);

  if (temp_avg) {
    cudaFree(temp_avg);
  }
  cudaFree(u_center);
  cudaFree(v_center);
  cudaFree(nu_center);
}

}  // namespace gpism
