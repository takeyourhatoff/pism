#include "gpism/ssa_operator.h"

#include "gpism/geometry.h"
#include "gpism/profile.h"

#include <cuda_runtime.h>

namespace gpism {
namespace {

__device__ inline int idx(int i, int j, int gw, int stride) {
  return (j + gw) * stride + (i + gw);
}

__device__ inline double beta_center(int i, int j, int gw, int stride_tauc,
                                     int stride_u, int stride_v,
                                     int stride_mask, const double* tauc,
                                     const double* u_center,
                                     const double* v_center,
                                     const int* cell_type, double q,
                                     double u_threshold, double reg,
                                     double sliding_scale_factor,
                                     double beta_ice_free_bedrock,
                                     int pseudo_plastic) {
  const int c = idx(i, j, gw, stride_tauc);
  const int mask = cell_type[c];
  if (mask == IceFreeBedrock) {
    return beta_ice_free_bedrock;
  }
  if (mask == IceFreeOcean || mask == FloatingIce) {
    return 0.0;
  }
  const double u = u_center[idx(i, j, gw, stride_u)];
  const double v = v_center[idx(i, j, gw, stride_v)];
  const double mag2 = reg * reg + u * u + v * v;
  if (pseudo_plastic) {
    const double u_thresh = fmax(u_threshold, 1.0e-6);
    const double u_thresh_factor = pow(u_thresh, -q);
    const double Aq =
        (sliding_scale_factor > 0.0) ? pow(sliding_scale_factor, q) : 1.0;
    return (tauc[c] / Aq) * pow(mag2, 0.5 * (q - 1.0)) * u_thresh_factor;
  }
  return tauc[c] / sqrt(mag2);
}

__global__ void basal_drag_kernel(
    int mx, int my, int gw, int stride_tauc, int stride_u, int stride_v,
    int stride_mask, const double* tauc, const double* u_center,
    const double* v_center, const int* cell_type, double* beta_u,
    double* beta_v, double q, double u_threshold, double reg,
    double sliding_scale_factor, double beta_ice_free_bedrock,
    int pseudo_plastic) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int ie = (i == mx - 1) ? i : (i + 1);
  const int jn = (j == my - 1) ? j : (j + 1);
  const int c = idx(i, j, gw, stride_tauc);
  const int e = idx(ie, j, gw, stride_tauc);
  const int n = idx(i, jn, gw, stride_tauc);

  const int mask_c = cell_type[c];
  const int mask_e = cell_type[e];
  const int mask_n = cell_type[n];

  const double beta_c = beta_center(i, j, gw, stride_tauc, stride_u, stride_v,
                                    stride_mask, tauc, u_center, v_center,
                                    cell_type, q, u_threshold, reg,
                                    sliding_scale_factor,
                                    beta_ice_free_bedrock, pseudo_plastic);
  const double beta_e = beta_center(ie, j, gw, stride_tauc, stride_u, stride_v,
                                    stride_mask, tauc, u_center, v_center,
                                    cell_type, q, u_threshold, reg,
                                    sliding_scale_factor,
                                    beta_ice_free_bedrock, pseudo_plastic);
  const double beta_n = beta_center(i, jn, gw, stride_tauc, stride_u, stride_v,
                                    stride_mask, tauc, u_center, v_center,
                                    cell_type, q, u_threshold, reg,
                                    sliding_scale_factor,
                                    beta_ice_free_bedrock, pseudo_plastic);

  if (mask_c == IceFreeBedrock || mask_e == IceFreeBedrock) {
    beta_u[c] = beta_ice_free_bedrock;
  } else {
    beta_u[c] = 0.5 * (beta_c + beta_e);
  }

  if (mask_c == IceFreeBedrock || mask_n == IceFreeBedrock) {
    beta_v[c] = beta_ice_free_bedrock;
  } else {
    beta_v[c] = 0.5 * (beta_c + beta_n);
  }
}

__global__ void rhs_kernel(int mx, int my, int gw, int stride_thk, int stride_dhdx,
                           int stride_dhdy, int stride_rhs, const double* thk,
                           const double* dhdx, const double* dhdy, double* rhs_u,
                           double* rhs_v, double scale, const int* mask_u,
                           const int* mask_v, const double* bc_u,
                           const double* bc_v, int has_bc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int c = idx(i, j, gw, stride_thk);
  const int e = idx(i + 1, j, gw, stride_thk);
  const int n = idx(i, j + 1, gw, stride_thk);
  const int c_rhs = idx(i, j, gw, stride_rhs);
  const int e_dhdx = idx(i + 1, j, gw, stride_dhdx);
  const int n_dhdy = idx(i, j + 1, gw, stride_dhdy);

  const double H_u = 0.5 * (thk[c] + thk[e]);
  const double H_v = 0.5 * (thk[c] + thk[n]);
  const double slope_x = 0.5 * (dhdx[c] + dhdx[e_dhdx]);
  const double slope_y = 0.5 * (dhdy[c] + dhdy[n_dhdy]);

  double rhs0 = scale * H_u * slope_x;
  double rhs1 = scale * H_v * slope_y;

  if (has_bc && mask_u && bc_u && mask_u[c_rhs] != 0) {
    rhs0 = bc_u[c_rhs];
  }
  if (has_bc && mask_v && bc_v && mask_v[c_rhs] != 0) {
    rhs1 = bc_v[c_rhs];
  }

  rhs_u[c_rhs] = rhs0;
  rhs_v[c_rhs] = rhs1;
}

__device__ inline double shear(int i, int j, int gw, int stride_u, int stride_v,
                               const double* u, const double* v, double inv_2dx,
                               double inv_2dy) {
  const int c = idx(i, j, gw, stride_u);
  const int n = idx(i, j + 1, gw, stride_u);
  const int s = idx(i, j - 1, gw, stride_u);
  const int e = idx(i + 1, j, gw, stride_v);
  const int w = idx(i - 1, j, gw, stride_v);
  const double du_dy = (u[n] - u[s]) * inv_2dy;
  const double dv_dx = (v[e] - v[w]) * inv_2dx;
  return du_dy + dv_dx;
}

__global__ void apply_kernel(int mx, int my, int gw, int stride_u, int stride_v,
                             int stride_nu_u, int stride_nu_v, int stride_beta_u,
                             int stride_beta_v, int stride_out_u,
                             int stride_out_v, const double* u, const double* v,
                             const double* nu_u, const double* nu_v,
                             const double* beta_u, const double* beta_v,
                             double* out_u, double* out_v, double inv_dx2,
                             double inv_dy2, double inv_2dx, double inv_2dy,
                             int stride_mask_u, int stride_mask_v,
                             const int* mask_u, const int* mask_v, int has_bc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }

  const int c_u = idx(i, j, gw, stride_u);
  const int c_v = idx(i, j, gw, stride_v);
  const int mask_u_idx = idx(i, j, gw, stride_mask_u);
  const int mask_v_idx = idx(i, j, gw, stride_mask_v);
  if (has_bc && mask_u && mask_u[mask_u_idx] != 0) {
    out_u[c_u] = u[c_u];
  } else {
    const int e = idx(i + 1, j, gw, stride_u);
    const int w = idx(i - 1, j, gw, stride_u);
    const int n = idx(i, j + 1, gw, stride_u);
    const int s = idx(i, j - 1, gw, stride_u);
    const int e_nu = idx(i + 1, j, gw, stride_nu_u);
    const int w_nu = idx(i - 1, j, gw, stride_nu_u);
    const int n_nu = idx(i, j + 1, gw, stride_nu_u);
    const int s_nu = idx(i, j - 1, gw, stride_nu_u);

    const double flux_x = nu_u[e_nu] * (u[e] - u[c_u]) -
                          nu_u[w_nu] * (u[c_u] - u[w]);
    const double flux_y = nu_u[n_nu] * (u[n] - u[c_u]) -
                          nu_u[s_nu] * (u[c_u] - u[s]);
    double coupling = 0.0;
    if (j >= 1 && j <= my - 2) {
      const double shear_p = shear(i, j + 1, gw, stride_u, stride_v, u, v,
                                   inv_2dx, inv_2dy);
      const double shear_m = shear(i, j - 1, gw, stride_u, stride_v, u, v,
                                   inv_2dx, inv_2dy);
      coupling = nu_u[c_u] * (shear_p - shear_m) * inv_2dy;
    }
    out_u[c_u] = flux_x * inv_dx2 + flux_y * inv_dy2 + coupling +
                 beta_u[idx(i, j, gw, stride_beta_u)] * u[c_u];
  }

  if (has_bc && mask_v && mask_v[mask_v_idx] != 0) {
    out_v[c_v] = v[c_v];
  } else {
    const int e = idx(i + 1, j, gw, stride_v);
    const int w = idx(i - 1, j, gw, stride_v);
    const int n = idx(i, j + 1, gw, stride_v);
    const int s = idx(i, j - 1, gw, stride_v);
    const int e_nu = idx(i + 1, j, gw, stride_nu_v);
    const int w_nu = idx(i - 1, j, gw, stride_nu_v);
    const int n_nu = idx(i, j + 1, gw, stride_nu_v);
    const int s_nu = idx(i, j - 1, gw, stride_nu_v);

    const double flux_x = nu_v[e_nu] * (v[e] - v[c_v]) -
                          nu_v[w_nu] * (v[c_v] - v[w]);
    const double flux_y = nu_v[n_nu] * (v[n] - v[c_v]) -
                          nu_v[s_nu] * (v[c_v] - v[s]);
    double coupling = 0.0;
    if (i >= 1 && i <= mx - 2) {
      const double shear_p = shear(i + 1, j, gw, stride_u, stride_v, u, v,
                                   inv_2dx, inv_2dy);
      const double shear_m = shear(i - 1, j, gw, stride_u, stride_v, u, v,
                                   inv_2dx, inv_2dy);
      coupling = nu_v[c_v] * (shear_p - shear_m) * inv_2dx;
    }
    out_v[c_v] = flux_x * inv_dx2 + flux_y * inv_dy2 + coupling +
                 beta_v[idx(i, j, gw, stride_beta_v)] * v[c_v];
  }
}

__global__ void apply_region_kernel(
    int mx, int my, int gw, int stride_u, int stride_v, int stride_nu_u,
    int stride_nu_v, int stride_beta_u, int stride_beta_v, int stride_out_u,
    int stride_out_v, const double* u, const double* v, const double* nu_u,
    const double* nu_v, const double* beta_u, const double* beta_v,
    double* out_u, double* out_v, double inv_dx2, double inv_dy2,
    double inv_2dx, double inv_2dy, const int* mask_u, const int* mask_v,
    int stride_mask_u, int stride_mask_v, int has_bc, int i_start, int i_end,
    int j_start, int j_end) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + i_start;
  int j = blockIdx.y * blockDim.y + threadIdx.y + j_start;
  if (i >= i_end || j >= j_end) {
    return;
  }

  const int c_u = idx(i, j, gw, stride_u);
  const int c_v = idx(i, j, gw, stride_v);
  const int mask_u_idx = idx(i, j, gw, stride_mask_u);
  const int mask_v_idx = idx(i, j, gw, stride_mask_v);
  if (has_bc && mask_u && mask_u[mask_u_idx] != 0) {
    out_u[c_u] = u[c_u];
  } else {
    const int e = idx(i + 1, j, gw, stride_u);
    const int w = idx(i - 1, j, gw, stride_u);
    const int n = idx(i, j + 1, gw, stride_u);
    const int s = idx(i, j - 1, gw, stride_u);
    const int e_nu = idx(i + 1, j, gw, stride_nu_u);
    const int w_nu = idx(i - 1, j, gw, stride_nu_u);
    const int n_nu = idx(i, j + 1, gw, stride_nu_u);
    const int s_nu = idx(i, j - 1, gw, stride_nu_u);

    const double flux_x = nu_u[e_nu] * (u[e] - u[c_u]) -
                          nu_u[w_nu] * (u[c_u] - u[w]);
    const double flux_y = nu_u[n_nu] * (u[n] - u[c_u]) -
                          nu_u[s_nu] * (u[c_u] - u[s]);
    double coupling = 0.0;
    if (j >= 1 && j <= my - 2) {
      const double shear_p = shear(i, j + 1, gw, stride_u, stride_v, u, v,
                                   inv_2dx, inv_2dy);
      const double shear_m = shear(i, j - 1, gw, stride_u, stride_v, u, v,
                                   inv_2dx, inv_2dy);
      coupling = nu_u[c_u] * (shear_p - shear_m) * inv_2dy;
    }
    out_u[c_u] = flux_x * inv_dx2 + flux_y * inv_dy2 + coupling +
                 beta_u[idx(i, j, gw, stride_beta_u)] * u[c_u];
  }

  if (has_bc && mask_v && mask_v[mask_v_idx] != 0) {
    out_v[c_v] = v[c_v];
  } else {
    const int e = idx(i + 1, j, gw, stride_v);
    const int w = idx(i - 1, j, gw, stride_v);
    const int n = idx(i, j + 1, gw, stride_v);
    const int s = idx(i, j - 1, gw, stride_v);
    const int e_nu = idx(i + 1, j, gw, stride_nu_v);
    const int w_nu = idx(i - 1, j, gw, stride_nu_v);
    const int n_nu = idx(i, j + 1, gw, stride_nu_v);
    const int s_nu = idx(i, j - 1, gw, stride_nu_v);

    const double flux_x = nu_v[e_nu] * (v[e] - v[c_v]) -
                          nu_v[w_nu] * (v[c_v] - v[w]);
    const double flux_y = nu_v[n_nu] * (v[n] - v[c_v]) -
                          nu_v[s_nu] * (v[c_v] - v[s]);
    double coupling = 0.0;
    if (i >= 1 && i <= mx - 2) {
      const double shear_p = shear(i + 1, j, gw, stride_u, stride_v, u, v,
                                   inv_2dx, inv_2dy);
      const double shear_m = shear(i - 1, j, gw, stride_u, stride_v, u, v,
                                   inv_2dx, inv_2dy);
      coupling = nu_v[c_v] * (shear_p - shear_m) * inv_2dx;
    }
    out_v[c_v] = flux_x * inv_dx2 + flux_y * inv_dy2 + coupling +
                 beta_v[idx(i, j, gw, stride_beta_v)] * v[c_v];
  }
}

}  // namespace

void ssa_compute_basal_drag_cuda(int mx, int my, int gw, int stride_tauc,
                                 int stride_u, int stride_v,
                                 int stride_mask, const double* tauc,
                                 const double* u_center,
                                 const double* v_center,
                                 const int* cell_type, double* beta_u,
                                 double* beta_v, double q,
                                 double u_threshold,
                                 double plastic_regularization,
                                 double sliding_scale_factor,
                                 double beta_ice_free_bedrock,
                                 int pseudo_plastic) {
  CudaEventTimer timer("ssa_basal_drag");
  dim3 block(16, 16);
  dim3 grid_dim((mx + block.x - 1) / block.x,
                (my + block.y - 1) / block.y);
  basal_drag_kernel<<<grid_dim, block>>>(
      mx, my, gw, stride_tauc, stride_u, stride_v, stride_mask, tauc, u_center,
      v_center, cell_type, beta_u, beta_v, q, u_threshold,
      plastic_regularization, sliding_scale_factor, beta_ice_free_bedrock,
      pseudo_plastic);
}

void ssa_assemble_rhs_cuda(int mx, int my, int gw, int stride_thk,
                           int stride_dhdx, int stride_dhdy, int stride_rhs,
                           const double* thk, const double* dhdx,
                           const double* dhdy, double* rhs_u, double* rhs_v,
                           double scale, const int* mask_u,
                           const int* mask_v, const double* bc_u,
                           const double* bc_v, int has_bc) {
  CudaEventTimer timer("ssa_rhs");
  dim3 block(16, 16);
  dim3 grid_dim((mx + block.x - 1) / block.x,
                (my + block.y - 1) / block.y);
  rhs_kernel<<<grid_dim, block>>>(mx, my, gw, stride_thk, stride_dhdx, stride_dhdy,
                                  stride_rhs, thk, dhdx, dhdy, rhs_u, rhs_v,
                                  scale, mask_u, mask_v, bc_u, bc_v, has_bc);
}

void ssa_apply_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                    int stride_nu_u, int stride_nu_v, int stride_beta_u,
                    int stride_beta_v, int stride_out_u, int stride_out_v,
                    const double* u, const double* v, const double* nu_u,
                    const double* nu_v, const double* beta_u,
                    const double* beta_v, double* out_u, double* out_v,
                    double inv_dx2, double inv_dy2, double inv_2dx,
                    double inv_2dy, int stride_mask_u, int stride_mask_v,
                    const int* mask_u, const int* mask_v, int has_bc) {
  CudaEventTimer timer("ssa_apply");
  dim3 block(16, 16);
  dim3 grid_dim((mx + block.x - 1) / block.x,
                (my + block.y - 1) / block.y);
  apply_kernel<<<grid_dim, block>>>(mx, my, gw, stride_u, stride_v, stride_nu_u,
                                    stride_nu_v, stride_beta_u, stride_beta_v,
                                    stride_out_u, stride_out_v, u, v, nu_u,
                                    nu_v, beta_u, beta_v, out_u, out_v, inv_dx2,
                                    inv_dy2, inv_2dx, inv_2dy, stride_mask_u,
                                    stride_mask_v, mask_u, mask_v, has_bc);
}

void ssa_apply_region_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                           int stride_nu_u, int stride_nu_v, int stride_beta_u,
                           int stride_beta_v, int stride_out_u,
                           int stride_out_v, const double* u, const double* v,
                           const double* nu_u, const double* nu_v,
                           const double* beta_u, const double* beta_v,
                           double* out_u, double* out_v, double inv_dx2,
                           double inv_dy2, double inv_2dx, double inv_2dy,
                           int stride_mask_u, int stride_mask_v,
                           const int* mask_u, const int* mask_v, int has_bc,
                           int i_start, int i_end, int j_start, int j_end) {
  CudaEventTimer timer("ssa_apply");
  if (i_start >= i_end || j_start >= j_end) {
    return;
  }
  dim3 block(16, 16);
  dim3 grid_dim((i_end - i_start + block.x - 1) / block.x,
                (j_end - j_start + block.y - 1) / block.y);
  apply_region_kernel<<<grid_dim, block>>>(
      mx, my, gw, stride_u, stride_v, stride_nu_u, stride_nu_v, stride_beta_u,
      stride_beta_v, stride_out_u, stride_out_v, u, v, nu_u, nu_v, beta_u,
      beta_v, out_u, out_v, inv_dx2, inv_dy2, inv_2dx, inv_2dy, mask_u, mask_v,
      stride_mask_u, stride_mask_v, has_bc, i_start, i_end, j_start, j_end);
}

}  // namespace gpism
