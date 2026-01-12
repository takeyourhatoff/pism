#include "gpism/config.h"

#if GPISM_HAVE_CUDA
#include <cuda_runtime.h>

namespace gpism {
namespace {

__device__ int idx(int i, int j, int gw, int stride) {
  return (j + gw) * stride + (i + gw);
}

__device__ int clamp_int(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void restrict_stag_kernel(int coarse_mx, int coarse_my, int fine_gw,
                                     int fine_stride_u, int fine_stride_v,
                                     int coarse_gw, int coarse_stride_u,
                                     int coarse_stride_v, const double* fine_u,
                                     const double* fine_v, double* coarse_u,
                                     double* coarse_v) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= coarse_mx || j >= coarse_my) {
    return;
  }
  const int fi = 2 * i;
  const int fj = 2 * j;
  const int f00 = idx(fi, fj, fine_gw, fine_stride_u);
  const int f10 = idx(fi + 1, fj, fine_gw, fine_stride_u);
  const int f01 = idx(fi, fj + 1, fine_gw, fine_stride_u);
  const int f11 = idx(fi + 1, fj + 1, fine_gw, fine_stride_u);
  const int f00v = idx(fi, fj, fine_gw, fine_stride_v);
  const int f10v = idx(fi + 1, fj, fine_gw, fine_stride_v);
  const int f01v = idx(fi, fj + 1, fine_gw, fine_stride_v);
  const int f11v = idx(fi + 1, fj + 1, fine_gw, fine_stride_v);
  const int cidx_u = idx(i, j, coarse_gw, coarse_stride_u);
  const int cidx_v = idx(i, j, coarse_gw, coarse_stride_v);
  coarse_u[cidx_u] =
      0.25 * (fine_u[f00] + fine_u[f10] + fine_u[f01] + fine_u[f11]);
  coarse_v[cidx_v] =
      0.25 * (fine_v[f00v] + fine_v[f10v] + fine_v[f01v] + fine_v[f11v]);
}

__global__ void prolong_stag_kernel(int fine_mx, int fine_my, int fine_gw,
                                    int fine_stride_u, int fine_stride_v,
                                    int coarse_mx, int coarse_my, int coarse_gw,
                                    int coarse_stride_u, int coarse_stride_v,
                                    const double* coarse_u,
                                    const double* coarse_v, double* fine_u,
                                    double* fine_v) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= fine_mx || j >= fine_my) {
    return;
  }
  const int ic = i >> 1;
  const int jc = j >> 1;
  const int di = i & 1;
  const int dj = j & 1;
  const int ic0 = clamp_int(ic, 0, coarse_mx - 1);
  const int jc0 = clamp_int(jc, 0, coarse_my - 1);
  const int ic1 = clamp_int(ic + 1, 0, coarse_mx - 1);
  const int jc1 = clamp_int(jc + 1, 0, coarse_my - 1);
  const int c00_u = idx(ic0, jc0, coarse_gw, coarse_stride_u);
  const int c10_u = idx(ic1, jc0, coarse_gw, coarse_stride_u);
  const int c01_u = idx(ic0, jc1, coarse_gw, coarse_stride_u);
  const int c11_u = idx(ic1, jc1, coarse_gw, coarse_stride_u);
  const int c00_v = idx(ic0, jc0, coarse_gw, coarse_stride_v);
  const int c10_v = idx(ic1, jc0, coarse_gw, coarse_stride_v);
  const int c01_v = idx(ic0, jc1, coarse_gw, coarse_stride_v);
  const int c11_v = idx(ic1, jc1, coarse_gw, coarse_stride_v);
  const int fidx_u = idx(i, j, fine_gw, fine_stride_u);
  const int fidx_v = idx(i, j, fine_gw, fine_stride_v);

  if (di == 0 && dj == 0) {
    fine_u[fidx_u] = coarse_u[c00_u];
    fine_v[fidx_v] = coarse_v[c00_v];
  } else if (di == 1 && dj == 0) {
    fine_u[fidx_u] = 0.5 * (coarse_u[c00_u] + coarse_u[c10_u]);
    fine_v[fidx_v] = 0.5 * (coarse_v[c00_v] + coarse_v[c10_v]);
  } else if (di == 0 && dj == 1) {
    fine_u[fidx_u] = 0.5 * (coarse_u[c00_u] + coarse_u[c01_u]);
    fine_v[fidx_v] = 0.5 * (coarse_v[c00_v] + coarse_v[c01_v]);
  } else {
    fine_u[fidx_u] = 0.25 * (coarse_u[c00_u] + coarse_u[c10_u] +
                             coarse_u[c01_u] + coarse_u[c11_u]);
    fine_v[fidx_v] = 0.25 * (coarse_v[c00_v] + coarse_v[c10_v] +
                             coarse_v[c01_v] + coarse_v[c11_v]);
  }
}

__global__ void compute_diag_kernel(int mx, int my, int gw, int stride_u,
                                    int stride_v, const double* nu_u,
                                    const double* nu_v, const double* beta_u,
                                    const double* beta_v, double* diag_u,
                                    double* diag_v, double inv_dx2,
                                    double inv_dy2, int stride_mask_u,
                                    int stride_mask_v, const int* mask_u,
                                    const int* mask_v, int has_bc) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int idx_u = idx(i, j, gw, stride_u);
  const int idx_v = idx(i, j, gw, stride_v);
  const int mask_idx_u = idx(i, j, gw, stride_mask_u);
  const int mask_idx_v = idx(i, j, gw, stride_mask_v);
  if (has_bc && mask_u[mask_idx_u] != 0) {
    diag_u[idx_u] = 1.0;
  } else {
    const double dxx =
        (nu_u[idx(i + 1, j, gw, stride_u)] +
         nu_u[idx(i - 1, j, gw, stride_u)]) *
        inv_dx2;
    const double dyy =
        (nu_u[idx(i, j + 1, gw, stride_u)] +
         nu_u[idx(i, j - 1, gw, stride_u)]) *
        inv_dy2;
    diag_u[idx_u] = beta_u[idx_u] + dxx + dyy;
  }

  if (has_bc && mask_v[mask_idx_v] != 0) {
    diag_v[idx_v] = 1.0;
  } else {
    const double dxx =
        (nu_v[idx(i + 1, j, gw, stride_v)] +
         nu_v[idx(i - 1, j, gw, stride_v)]) *
        inv_dx2;
    const double dyy =
        (nu_v[idx(i, j + 1, gw, stride_v)] +
         nu_v[idx(i, j - 1, gw, stride_v)]) *
        inv_dy2;
    diag_v[idx_v] = beta_v[idx_v] + dxx + dyy;
  }
}

__global__ void residual_kernel(int mx, int my, int gw, int stride_u,
                                int stride_v, const double* b_u,
                                const double* b_v, const double* Ax_u,
                                const double* Ax_v, double* r_u, double* r_v,
                                int stride_mask_u, int stride_mask_v,
                                const int* mask_u, const int* mask_v,
                                int has_bc) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int idx_u = idx(i, j, gw, stride_u);
  const int idx_v = idx(i, j, gw, stride_v);
  const int mask_idx_u = idx(i, j, gw, stride_mask_u);
  const int mask_idx_v = idx(i, j, gw, stride_mask_v);
  if (has_bc && mask_u[mask_idx_u] != 0) {
    r_u[idx_u] = 0.0;
  } else {
    r_u[idx_u] = b_u[idx_u] - Ax_u[idx_u];
  }
  if (has_bc && mask_v[mask_idx_v] != 0) {
    r_v[idx_v] = 0.0;
  } else {
    r_v[idx_v] = b_v[idx_v] - Ax_v[idx_v];
  }
}

__global__ void jacobi_update_kernel(
    int mx, int my, int gw, int stride_u, int stride_v, const double* b_u,
    const double* b_v, const double* Ax_u, const double* Ax_v,
    const double* diag_u, const double* diag_v, double* x_u, double* x_v,
    double omega, int stride_mask_u, int stride_mask_v, const int* mask_u,
    const int* mask_v, int stride_bc_u, int stride_bc_v, const double* bc_u,
    const double* bc_v, int has_bc, int has_values) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int idx_u = idx(i, j, gw, stride_u);
  const int idx_v = idx(i, j, gw, stride_v);
  const int mask_idx_u = idx(i, j, gw, stride_mask_u);
  const int mask_idx_v = idx(i, j, gw, stride_mask_v);
  const int bc_idx_u = idx(i, j, gw, stride_bc_u);
  const int bc_idx_v = idx(i, j, gw, stride_bc_v);
  if (has_bc && mask_u[mask_idx_u] != 0) {
    if (has_values) {
      x_u[idx_u] = bc_u[bc_idx_u];
    }
  } else {
    const double r = b_u[idx_u] - Ax_u[idx_u];
    const double d = diag_u[idx_u];
    if (d != 0.0) {
      x_u[idx_u] += omega * r / d;
    }
  }
  if (has_bc && mask_v[mask_idx_v] != 0) {
    if (has_values) {
      x_v[idx_v] = bc_v[bc_idx_v];
    }
  } else {
    const double r = b_v[idx_v] - Ax_v[idx_v];
    const double d = diag_v[idx_v];
    if (d != 0.0) {
      x_v[idx_v] += omega * r / d;
    }
  }
}

}  // namespace

void mg_restrict_stag_cuda(int fine_mx, int fine_my, int fine_gw,
                           int fine_stride_u, int fine_stride_v, int coarse_mx,
                           int coarse_my, int coarse_gw, int coarse_stride_u,
                           int coarse_stride_v, const double* fine_u,
                           const double* fine_v, double* coarse_u,
                           double* coarse_v) {
  const dim3 block(16, 16);
  const dim3 grid((coarse_mx + block.x - 1) / block.x,
                  (coarse_my + block.y - 1) / block.y);
  restrict_stag_kernel<<<grid, block>>>(
      coarse_mx, coarse_my, fine_gw, fine_stride_u, fine_stride_v, coarse_gw,
      coarse_stride_u, coarse_stride_v, fine_u, fine_v, coarse_u, coarse_v);
}

void mg_prolong_stag_cuda(int coarse_mx, int coarse_my, int coarse_gw,
                          int coarse_stride_u, int coarse_stride_v, int fine_mx,
                          int fine_my, int fine_gw, int fine_stride_u,
                          int fine_stride_v, const double* coarse_u,
                          const double* coarse_v, double* fine_u,
                          double* fine_v) {
  const dim3 block(16, 16);
  const dim3 grid((fine_mx + block.x - 1) / block.x,
                  (fine_my + block.y - 1) / block.y);
  prolong_stag_kernel<<<grid, block>>>(
      fine_mx, fine_my, fine_gw, fine_stride_u, fine_stride_v, coarse_mx,
      coarse_my, coarse_gw, coarse_stride_u, coarse_stride_v, coarse_u,
      coarse_v, fine_u, fine_v);
}

void mg_compute_diag_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                          const double* nu_u, const double* nu_v,
                          const double* beta_u, const double* beta_v,
                          double* diag_u, double* diag_v, double inv_dx2,
                          double inv_dy2, int stride_mask_u, int stride_mask_v,
                          const int* mask_u, const int* mask_v, int has_bc) {
  const dim3 block(16, 16);
  const dim3 grid((mx + block.x - 1) / block.x,
                  (my + block.y - 1) / block.y);
  compute_diag_kernel<<<grid, block>>>(
      mx, my, gw, stride_u, stride_v, nu_u, nu_v, beta_u, beta_v, diag_u,
      diag_v, inv_dx2, inv_dy2, stride_mask_u, stride_mask_v, mask_u, mask_v,
      has_bc);
}

void mg_residual_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                      const double* b_u, const double* b_v, const double* Ax_u,
                      const double* Ax_v, double* r_u, double* r_v,
                      int stride_mask_u, int stride_mask_v, const int* mask_u,
                      const int* mask_v, int has_bc) {
  const dim3 block(16, 16);
  const dim3 grid((mx + block.x - 1) / block.x,
                  (my + block.y - 1) / block.y);
  residual_kernel<<<grid, block>>>(
      mx, my, gw, stride_u, stride_v, b_u, b_v, Ax_u, Ax_v, r_u, r_v,
      stride_mask_u, stride_mask_v, mask_u, mask_v, has_bc);
}

void mg_jacobi_update_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                           const double* b_u, const double* b_v,
                           const double* Ax_u, const double* Ax_v,
                           const double* diag_u, const double* diag_v,
                           double* x_u, double* x_v, double omega,
                           int stride_mask_u, int stride_mask_v,
                           const int* mask_u, const int* mask_v,
                           int stride_bc_u, int stride_bc_v, const double* bc_u,
                           const double* bc_v, int has_bc, int has_values) {
  const dim3 block(16, 16);
  const dim3 grid((mx + block.x - 1) / block.x,
                  (my + block.y - 1) / block.y);
  jacobi_update_kernel<<<grid, block>>>(
      mx, my, gw, stride_u, stride_v, b_u, b_v, Ax_u, Ax_v, diag_u, diag_v,
      x_u, x_v, omega, stride_mask_u, stride_mask_v, mask_u, mask_v,
      stride_bc_u, stride_bc_v, bc_u, bc_v, has_bc, has_values);
}

}  // namespace gpism

#endif
