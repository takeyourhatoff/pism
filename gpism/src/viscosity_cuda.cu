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

__device__ inline int clamp_index(int idx, int max_idx) {
  return idx < 0 ? 0 : (idx >= max_idx ? (max_idx - 1) : idx);
}

__device__ inline double viscosity_from_derivs(double u_x, double u_y, double v_x,
                                               double v_y, double B, double n_eff,
                                               double eps0) {
  const double eps_xx = u_x;
  const double eps_yy = v_y;
  const double eps_xy = 0.5 * (u_y + v_x);
  const double eps2 = fmax(0.0, eps_xx * eps_xx + eps_yy * eps_yy +
                                    eps_xx * eps_yy + eps_xy * eps_xy);
  const double eps_e = sqrt(eps2 + eps0 * eps0);
  return 0.5 * B * pow(eps_e, (1.0 / n_eff) - 1.0);
}

__device__ inline double temp_scale(const double* temp_avg, int i, int j,
                                    int stride_center, double gamma, double ref) {
  if (!temp_avg || gamma == 0.0) {
    return 1.0;
  }
  const double temp = temp_avg[idx(i, j, stride_center)];
  return exp(-gamma * (temp - ref));
}

__device__ inline double center_value(const double* center, int ii, int jj, int mx,
                                      int my, int stride_center) {
  const int ci = clamp_index(ii, mx);
  const int cj = clamp_index(jj, my);
  return center[idx(ci, cj, stride_center)];
}

__global__ void nuH_kernel(int mx, int my, int gw, int stride_thk,
                           int stride_center, int stride_u, int stride_v,
                           const double* thk, const double* u_center,
                           const double* v_center, const double* temp_avg,
                           double* nuH_u, double* nuH_v,
                           double nuH_regularization,
                           double strength_extension_nu,
                           double strength_extension_min_thickness,
                           double gamma, double ref, double B, double n_eff,
                           double eps0, double inv_dx, double inv_dy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }

  const int ip1 = (i == mx - 1) ? i : i + 1;
  const int im1 = (i == 0) ? i : i - 1;
  const int jp1 = (j == my - 1) ? j : j + 1;
  const int jm1 = (j == 0) ? j : j - 1;

  const double tscale = temp_scale(temp_avg, i, j, stride_center, gamma, ref);

  // u-staggered: between (i,j) and (i+1,j)
  {
    const double u_x =
        (center_value(u_center, ip1, j, mx, my, stride_center) -
         center_value(u_center, i, j, mx, my, stride_center)) *
        inv_dx;
    const double v_x =
        (center_value(v_center, ip1, j, mx, my, stride_center) -
         center_value(v_center, i, j, mx, my, stride_center)) *
        inv_dx;
    const double u_y =
        (center_value(u_center, i, jp1, mx, my, stride_center) +
         center_value(u_center, ip1, jp1, mx, my, stride_center) -
         center_value(u_center, i, jm1, mx, my, stride_center) -
         center_value(u_center, ip1, jm1, mx, my, stride_center)) *
        (0.25 * inv_dy);
    const double v_y =
        (center_value(v_center, i, jp1, mx, my, stride_center) +
         center_value(v_center, ip1, jp1, mx, my, stride_center) -
         center_value(v_center, i, jm1, mx, my, stride_center) -
         center_value(v_center, ip1, jm1, mx, my, stride_center)) *
        (0.25 * inv_dy);

    double nu = viscosity_from_derivs(u_x, u_y, v_x, v_y, B, n_eff, eps0) * tscale;
    double H_face =
        0.5 * (thk[idx(i, j, gw, stride_thk)] + thk[idx(ip1, j, gw, stride_thk)]);
    if (strength_extension_nu > 0.0 && H_face < strength_extension_min_thickness) {
      H_face = fmax(H_face, strength_extension_min_thickness);
      nu = strength_extension_nu;
    }
    nuH_u[idx(i, j, gw, stride_u)] = nu * H_face + nuH_regularization;
  }

  // v-staggered: between (i,j) and (i,j+1)
  {
    const double u_y =
        (center_value(u_center, i, jp1, mx, my, stride_center) -
         center_value(u_center, i, j, mx, my, stride_center)) *
        inv_dy;
    const double v_y =
        (center_value(v_center, i, jp1, mx, my, stride_center) -
         center_value(v_center, i, j, mx, my, stride_center)) *
        inv_dy;
    const double u_x =
        (center_value(u_center, ip1, j, mx, my, stride_center) +
         center_value(u_center, ip1, jp1, mx, my, stride_center) -
         center_value(u_center, im1, j, mx, my, stride_center) -
         center_value(u_center, im1, jp1, mx, my, stride_center)) *
        (0.25 * inv_dx);
    const double v_x =
        (center_value(v_center, ip1, j, mx, my, stride_center) +
         center_value(v_center, ip1, jp1, mx, my, stride_center) -
         center_value(v_center, im1, j, mx, my, stride_center) -
         center_value(v_center, im1, jp1, mx, my, stride_center)) *
        (0.25 * inv_dx);

    double nu = viscosity_from_derivs(u_x, u_y, v_x, v_y, B, n_eff, eps0) * tscale;
    double H_face =
        0.5 * (thk[idx(i, j, gw, stride_thk)] + thk[idx(i, jp1, gw, stride_thk)]);
    if (strength_extension_nu > 0.0 && H_face < strength_extension_min_thickness) {
      H_face = fmax(H_face, strength_extension_min_thickness);
      nu = strength_extension_nu;
    }
    nuH_v[idx(i, j, gw, stride_v)] = nu * H_face + nuH_regularization;
  }
}

}  // namespace

void viscosity_compute_nuH_cuda(int mx, int my, int gw, int stride_thk,
                                int stride_u, int stride_v, int stride_nuH_u,
                                int stride_nuH_v, const double* thk,
                                const double* u, const double* v, double* nuH_u,
                                double* nuH_v, double nuH_regularization,
                                double strength_extension_nu,
                                double strength_extension_min_thickness,
                                const double* enthalpy, int nz, int enthalpy_gw,
                                int enthalpy_stride, double enthalpy_gamma,
                                double enthalpy_ref, double B, double n_eff,
                                double eps0, double inv_dx, double inv_dy) {
  CudaEventTimer timer("viscosity_nuH");
  const std::size_t count = static_cast<std::size_t>(mx) * my;
  double* u_center = nullptr;
  double* v_center = nullptr;
  double* temp_avg = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&u_center), count * sizeof(double));
  cudaMalloc(reinterpret_cast<void**>(&v_center), count * sizeof(double));

  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);

  if (enthalpy && enthalpy_gamma != 0.0) {
    cudaMalloc(reinterpret_cast<void**>(&temp_avg), count * sizeof(double));
    column_avg_kernel<<<grid, block>>>(mx, my, enthalpy_gw, nz,
                                       enthalpy_stride, enthalpy, temp_avg);
  }

  center_velocity_kernel<<<grid, block>>>(mx, my, gw, stride_u, stride_v, mx, u,
                                          v, u_center, v_center);
  nuH_kernel<<<grid, block>>>(mx, my, gw, stride_thk, mx, stride_nuH_u,
                              stride_nuH_v, thk, u_center, v_center, temp_avg,
                              nuH_u, nuH_v, nuH_regularization,
                              strength_extension_nu,
                              strength_extension_min_thickness, enthalpy_gamma,
                              enthalpy_ref, B, n_eff, eps0, inv_dx, inv_dy);

  if (temp_avg) {
    cudaFree(temp_avg);
  }
  cudaFree(u_center);
  cudaFree(v_center);
}

}  // namespace gpism
