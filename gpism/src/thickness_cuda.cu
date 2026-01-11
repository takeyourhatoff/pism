#include "gpism/thickness.h"

#include <cuda_runtime.h>

namespace gpism {
namespace {

__device__ inline int idx(int i, int j, int gw, int stride) {
  return (j + gw) * stride + (i + gw);
}

__global__ void flux_u_kernel(int mx, int my, int gw, int stride_thk,
                              int stride_vel, int stride_flux,
                              const double* thk, const double* vel_u,
                              double* flux_u) {
  const int nx = mx + gw;
  const int ny = my + 2 * gw;
  int ti = blockIdx.x * blockDim.x + threadIdx.x;
  int tj = blockIdx.y * blockDim.y + threadIdx.y;
  if (ti >= nx || tj >= ny) {
    return;
  }
  const int i = ti - gw;
  const int j = tj - gw;
  int i0 = i < 0 ? 0 : (i >= mx ? mx - 1 : i);
  int i1 = i + 1;
  i1 = i1 < 0 ? 0 : (i1 >= mx ? mx - 1 : i1);
  int j0 = j < 0 ? 0 : (j >= my ? my - 1 : j);

  const double H_left = thk[idx(i0, j0, gw, stride_thk)];
  const double H_right = thk[idx(i1, j0, gw, stride_thk)];
  const double H_face = 0.5 * (H_left + H_right);
  flux_u[idx(i, j, gw, stride_flux)] =
      H_face * vel_u[idx(i0, j0, gw, stride_vel)];
}

__global__ void flux_v_kernel(int mx, int my, int gw, int stride_thk,
                              int stride_vel, int stride_flux,
                              const double* thk, const double* vel_v,
                              double* flux_v) {
  const int nx = mx + 2 * gw;
  const int ny = my + gw;
  int ti = blockIdx.x * blockDim.x + threadIdx.x;
  int tj = blockIdx.y * blockDim.y + threadIdx.y;
  if (ti >= nx || tj >= ny) {
    return;
  }
  const int i = ti - gw;
  const int j = tj - gw;
  int i0 = i < 0 ? 0 : (i >= mx ? mx - 1 : i);
  int j0 = j < 0 ? 0 : (j >= my ? my - 1 : j);
  int j1 = j + 1;
  j1 = j1 < 0 ? 0 : (j1 >= my ? my - 1 : j1);

  const double H_down = thk[idx(i0, j0, gw, stride_thk)];
  const double H_up = thk[idx(i0, j1, gw, stride_thk)];
  const double H_face = 0.5 * (H_down + H_up);
  flux_v[idx(i, j, gw, stride_flux)] =
      H_face * vel_v[idx(i0, j0, gw, stride_vel)];
}

__global__ void update_thickness_kernel(int mx, int my, int gw,
                                        int stride_thk, int stride_flux_u,
                                        int stride_flux_v, int stride_smb,
                                        const double* flux_u,
                                        const double* flux_v,
                                        const double* smb, double dt,
                                        double inv_dx, double inv_dy,
                                        int enforce_nonnegative,
                                        double* thk) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const double div_x =
      (flux_u[idx(i, j, gw, stride_flux_u)] -
       flux_u[idx(i - 1, j, gw, stride_flux_u)]) *
      inv_dx;
  const double div_y =
      (flux_v[idx(i, j, gw, stride_flux_v)] -
       flux_v[idx(i, j - 1, gw, stride_flux_v)]) *
      inv_dy;
  const int id = idx(i, j, gw, stride_thk);
  double updated = thk[id] + dt * (smb[idx(i, j, gw, stride_smb)] -
                                   (div_x + div_y));
  if (enforce_nonnegative && updated < 0.0) {
    updated = 0.0;
  }
  thk[id] = updated;
}

__global__ void update_mask_kernel(int mx, int my, int gw, int stride_thk,
                                   int stride_mask, const double* thk,
                                   int* mask) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int id = idx(i, j, gw, stride_thk);
  mask[idx(i, j, gw, stride_mask)] = (thk[id] > 0.0) ? 1 : 0;
}

__global__ void cell_center_velocity_kernel(int mx, int my, int gw,
                                            int stride_u, int stride_v,
                                            int stride_uvel, int stride_vvel,
                                            const double* u_face,
                                            const double* v_face, double* uvel,
                                            double* vvel) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int il = (i == 0) ? i : i - 1;
  const int jd = (j == 0) ? j : j - 1;
  uvel[idx(i, j, gw, stride_uvel)] =
      0.5 * (u_face[idx(i, j, gw, stride_u)] +
             u_face[idx(il, j, gw, stride_u)]);
  vvel[idx(i, j, gw, stride_vvel)] =
      0.5 * (v_face[idx(i, j, gw, stride_v)] +
             v_face[idx(i, jd, gw, stride_v)]);
}

}  // namespace

void compute_face_fluxes_cuda(int mx, int my, int gw, int stride_thk,
                              int stride_vel_u, int stride_vel_v,
                              int stride_flux_u, int stride_flux_v,
                              const double* thk, const double* vel_u,
                              const double* vel_v, double* flux_u,
                              double* flux_v) {
  dim3 block(16, 16);
  dim3 grid_u((mx + gw + block.x - 1) / block.x,
              (my + 2 * gw + block.y - 1) / block.y);
  flux_u_kernel<<<grid_u, block>>>(mx, my, gw, stride_thk, stride_vel_u,
                                   stride_flux_u, thk, vel_u, flux_u);

  dim3 grid_v((mx + 2 * gw + block.x - 1) / block.x,
              (my + gw + block.y - 1) / block.y);
  flux_v_kernel<<<grid_v, block>>>(mx, my, gw, stride_thk, stride_vel_v,
                                   stride_flux_v, thk, vel_v, flux_v);
}

void update_thickness_cuda(int mx, int my, int gw, int stride_thk,
                           int stride_flux_u, int stride_flux_v,
                           int stride_smb, const double* flux_u,
                           const double* flux_v, const double* smb, double dt,
                           double inv_dx, double inv_dy,
                           int enforce_nonnegative, double* thk) {
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  update_thickness_kernel<<<grid, block>>>(
      mx, my, gw, stride_thk, stride_flux_u, stride_flux_v, stride_smb, flux_u,
      flux_v, smb, dt, inv_dx, inv_dy, enforce_nonnegative, thk);
}

void update_mask_cuda(int mx, int my, int gw, int stride_thk, int stride_mask,
                      const double* thk, int* mask) {
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  update_mask_kernel<<<grid, block>>>(mx, my, gw, stride_thk, stride_mask, thk,
                                      mask);
}

void compute_cell_center_velocity_cuda(int mx, int my, int gw, int stride_u,
                                       int stride_v, int stride_uvel,
                                       int stride_vvel, const double* u_face,
                                       const double* v_face, double* uvel,
                                       double* vvel) {
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  cell_center_velocity_kernel<<<grid, block>>>(
      mx, my, gw, stride_u, stride_v, stride_uvel, stride_vvel, u_face, v_face,
      uvel, vvel);
}

}  // namespace gpism
