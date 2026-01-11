#include "gpism/linear_algebra.h"

#include <cmath>

#include "gpism/profile.h"

#include <cuda_runtime.h>

namespace gpism {
namespace {

__device__ inline int idx(int i, int j, int gw, int stride) {
  return (j + gw) * stride + (i + gw);
}

double* scalar_device_buffer() {
  static double* buffer = nullptr;
  if (!buffer) {
    cudaMalloc(reinterpret_cast<void**>(&buffer), sizeof(double));
  }
  return buffer;
}

__global__ void axpy_kernel(int mx, int my, int gw, int stride_x, int stride_y,
                            const double* x, double* y, double alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int ix = idx(i, j, gw, stride_x);
  const int iy = idx(i, j, gw, stride_y);
  y[iy] += alpha * x[ix];
}

__global__ void scal_kernel(int mx, int my, int gw, int stride, double* x,
                            double alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int ix = idx(i, j, gw, stride);
  x[ix] *= alpha;
}

__global__ void copy_kernel(int mx, int my, int gw, int stride_x, int stride_y,
                            const double* x, double* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int ix = idx(i, j, gw, stride_x);
  const int iy = idx(i, j, gw, stride_y);
  y[iy] = x[ix];
}

__global__ void set_kernel(int mx, int my, int gw, int stride, double* x,
                           double value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int ix = idx(i, j, gw, stride);
  x[ix] = value;
}

__global__ void dot_kernel(int mx, int my, int gw, int stride_a, int stride_b,
                           const double* a, const double* b, double* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int ia = idx(i, j, gw, stride_a);
  const int ib = idx(i, j, gw, stride_b);
  atomicAdd(out, a[ia] * b[ib]);
}

__global__ void dot_stag_kernel(int mx, int my, int gw, int stride_u, int stride_v,
                                const double* a_u, const double* a_v,
                                const double* b_u, const double* b_v,
                                double* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int iu = idx(i, j, gw, stride_u);
  const int iv = idx(i, j, gw, stride_v);
  atomicAdd(out, a_u[iu] * b_u[iu] + a_v[iv] * b_v[iv]);
}

__global__ void dot_stag_batch_kernel(int mx, int my, int gw, int stride_u,
                                      int stride_v, const double* w_u,
                                      const double* w_v, const double** V_u,
                                      const double** V_v, int count,
                                      double* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int iu = idx(i, j, gw, stride_u);
  const int iv = idx(i, j, gw, stride_v);
  const double wu = w_u[iu];
  const double wv = w_v[iv];
  for (int k = 0; k < count; ++k) {
    atomicAdd(&out[k], wu * V_u[k][iu] + wv * V_v[k][iv]);
  }
}

__global__ void norm1_kernel(int mx, int my, int gw, int stride, const double* a,
                             double* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int ia = idx(i, j, gw, stride);
  atomicAdd(out, fabs(a[ia]));
}

__global__ void norm1_stag_kernel(int mx, int my, int gw, int stride_u, int stride_v,
                                  const double* a_u, const double* a_v,
                                  double* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int iu = idx(i, j, gw, stride_u);
  const int iv = idx(i, j, gw, stride_v);
  atomicAdd(out, fabs(a_u[iu]) + fabs(a_v[iv]));
}

__global__ void diff_norm1_kernel(int mx, int my, int gw, int stride_a,
                                  int stride_b, const double* a,
                                  const double* b, double* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int ia = idx(i, j, gw, stride_a);
  const int ib = idx(i, j, gw, stride_b);
  atomicAdd(out, fabs(a[ia] - b[ib]));
}

__global__ void diff_norm1_stag_kernel(int mx, int my, int gw, int stride_u,
                                       int stride_v, const double* a_u,
                                       const double* a_v, const double* b_u,
                                       const double* b_v, double* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int iu = idx(i, j, gw, stride_u);
  const int iv = idx(i, j, gw, stride_v);
  atomicAdd(out, fabs(a_u[iu] - b_u[iu]) + fabs(a_v[iv] - b_v[iv]));
}

__global__ void orthogonalize_stag_kernel(int mx, int my, int gw, int stride_u,
                                          int stride_v, const double** V_u,
                                          const double** V_v,
                                          const double* hij, int count,
                                          double* w_u, double* w_v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int iu = idx(i, j, gw, stride_u);
  const int iv = idx(i, j, gw, stride_v);
  double wu = w_u[iu];
  double wv = w_v[iv];
  for (int k = 0; k < count; ++k) {
    const double coeff = hij[k];
    wu -= coeff * V_u[k][iu];
    wv -= coeff * V_v[k][iv];
  }
  w_u[iu] = wu;
  w_v[iv] = wv;
}

__global__ void normalize_stag_kernel(int mx, int my, int gw, int stride_u,
                                      int stride_v, const double* w_u,
                                      const double* w_v, double* out_u,
                                      double* out_v, const double* norm_in) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int iu = idx(i, j, gw, stride_u);
  const int iv = idx(i, j, gw, stride_v);
  const double norm = sqrt(fmax(norm_in[0], 0.0));
  if (norm == 0.0) {
    out_u[iu] = 0.0;
    out_v[iv] = 0.0;
    return;
  }
  const double inv = 1.0 / norm;
  out_u[iu] = w_u[iu] * inv;
  out_v[iv] = w_v[iv] * inv;
}

__global__ void gmres_update_kernel(int restart, int j, double* H, double* cs,
                                    double* sn, double* g,
                                    const double* hij) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int stride = restart + 1;
  for (int i = 0; i <= j; ++i) {
    H[i + stride * j] = hij[i];
  }
  double h_next = sqrt(fmax(hij[j + 1], 0.0));
  H[j + 1 + stride * j] = h_next;

  for (int i = 0; i < j; ++i) {
    double h0 = H[i + stride * j];
    double h1 = H[i + 1 + stride * j];
    const double c = cs[i];
    const double s = sn[i];
    const double temp = c * h0 + s * h1;
    h1 = -s * h0 + c * h1;
    h0 = temp;
    H[i + stride * j] = h0;
    H[i + 1 + stride * j] = h1;
  }

  double h00 = H[j + stride * j];
  double h10 = H[j + 1 + stride * j];
  double c = 1.0;
  double s = 0.0;
  if (h10 != 0.0) {
    if (fabs(h10) > fabs(h00)) {
      const double tau = -h00 / h10;
      s = 1.0 / sqrt(1.0 + tau * tau);
      c = s * tau;
    } else {
      const double tau = -h10 / h00;
      c = 1.0 / sqrt(1.0 + tau * tau);
      s = c * tau;
    }
  }
  cs[j] = c;
  sn[j] = s;
  const double temp = c * h00 + s * h10;
  h10 = -s * h00 + c * h10;
  h00 = temp;
  H[j + stride * j] = h00;
  H[j + 1 + stride * j] = h10;

  double g0 = g[j];
  double g1 = g[j + 1];
  const double gtemp = c * g0 + s * g1;
  g1 = -s * g0 + c * g1;
  g0 = gtemp;
  g[j] = g0;
  g[j + 1] = g1;
}

}  // namespace

void axpy_cuda(int mx, int my, int gw, int stride_x, int stride_y,
              const double* x, double* y, double alpha) {
  CudaEventTimer timer("la_axpy");
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  axpy_kernel<<<grid, block>>>(mx, my, gw, stride_x, stride_y, x, y, alpha);
}

void scal_cuda(int mx, int my, int gw, int stride, double* x, double alpha) {
  CudaEventTimer timer("la_scal");
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  scal_kernel<<<grid, block>>>(mx, my, gw, stride, x, alpha);
}

void copy_cuda(int mx, int my, int gw, int stride_x, int stride_y,
               const double* x, double* y) {
  CudaEventTimer timer("la_copy");
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  copy_kernel<<<grid, block>>>(mx, my, gw, stride_x, stride_y, x, y);
}

void set_cuda(int mx, int my, int gw, int stride, double* x, double value) {
  CudaEventTimer timer("la_set");
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  set_kernel<<<grid, block>>>(mx, my, gw, stride, x, value);
}

double dot_cuda(int mx, int my, int gw, int stride_a, int stride_b,
                const double* a, const double* b) {
  CudaEventTimer timer("la_dot");
  double* d_out = scalar_device_buffer();
  cudaMemset(d_out, 0, sizeof(double));

  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  dot_kernel<<<grid, block>>>(mx, my, gw, stride_a, stride_b, a, b, d_out);

  double result = 0.0;
  cudaMemcpy(&result, d_out, sizeof(double), cudaMemcpyDeviceToHost);
  return result;
}

double dot_stag_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                     const double* a_u, const double* a_v,
                     const double* b_u, const double* b_v) {
  CudaEventTimer timer("la_dot_stag");
  double* d_out = scalar_device_buffer();
  cudaMemset(d_out, 0, sizeof(double));

  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  dot_stag_kernel<<<grid, block>>>(mx, my, gw, stride_u, stride_v, a_u, a_v,
                                   b_u, b_v, d_out);

  double result = 0.0;
  cudaMemcpy(&result, d_out, sizeof(double), cudaMemcpyDeviceToHost);
  return result;
}

void dot_stag_batch_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                         const double* w_u, const double* w_v,
                         const double** V_u, const double** V_v, int count,
                         double* hij) {
  CudaEventTimer timer("la_dot_stag_batch");
  if (count <= 0) {
    return;
  }
  cudaMemset(hij, 0, static_cast<std::size_t>(count) * sizeof(double));
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  dot_stag_batch_kernel<<<grid, block>>>(mx, my, gw, stride_u, stride_v, w_u,
                                         w_v, V_u, V_v, count, hij);
}

void dot_stag_self_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                        const double* w_u, const double* w_v, double* out) {
  CudaEventTimer timer("la_dot_stag_self");
  cudaMemset(out, 0, sizeof(double));
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  dot_stag_kernel<<<grid, block>>>(mx, my, gw, stride_u, stride_v, w_u, w_v,
                                   w_u, w_v, out);
}

void orthogonalize_stag_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                             double* w_u, double* w_v, const double** V_u,
                             const double** V_v, int count,
                             const double* hij) {
  CudaEventTimer timer("la_orthogonalize_stag");
  if (count <= 0) {
    return;
  }
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  orthogonalize_stag_kernel<<<grid, block>>>(mx, my, gw, stride_u, stride_v,
                                             V_u, V_v, hij, count, w_u, w_v);
}

void normalize_stag_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                         const double* w_u, const double* w_v, double* out_u,
                         double* out_v, const double* norm_in) {
  CudaEventTimer timer("la_normalize_stag");
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  normalize_stag_kernel<<<grid, block>>>(mx, my, gw, stride_u, stride_v, w_u,
                                         w_v, out_u, out_v, norm_in);
}

void gmres_update_cuda(int restart, int j, double* H, double* cs, double* sn,
                       double* g, const double* hij) {
  CudaEventTimer timer("gmres_update");
  gmres_update_kernel<<<1, 1>>>(restart, j, H, cs, sn, g, hij);
}

double norm1_cuda(int mx, int my, int gw, int stride, const double* a) {
  CudaEventTimer timer("la_norm1");
  double* d_out = scalar_device_buffer();
  cudaMemset(d_out, 0, sizeof(double));

  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  norm1_kernel<<<grid, block>>>(mx, my, gw, stride, a, d_out);

  double result = 0.0;
  cudaMemcpy(&result, d_out, sizeof(double), cudaMemcpyDeviceToHost);
  return result;
}

double norm1_stag_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                       const double* a_u, const double* a_v) {
  CudaEventTimer timer("la_norm1_stag");
  double* d_out = scalar_device_buffer();
  cudaMemset(d_out, 0, sizeof(double));

  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  norm1_stag_kernel<<<grid, block>>>(mx, my, gw, stride_u, stride_v, a_u, a_v,
                                     d_out);

  double result = 0.0;
  cudaMemcpy(&result, d_out, sizeof(double), cudaMemcpyDeviceToHost);
  return result;
}

double diff_norm1_cuda(int mx, int my, int gw, int stride_a, int stride_b,
                       const double* a, const double* b) {
  CudaEventTimer timer("la_diff_norm1");
  double* d_out = scalar_device_buffer();
  cudaMemset(d_out, 0, sizeof(double));

  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  diff_norm1_kernel<<<grid, block>>>(mx, my, gw, stride_a, stride_b, a, b,
                                     d_out);

  double result = 0.0;
  cudaMemcpy(&result, d_out, sizeof(double), cudaMemcpyDeviceToHost);
  return result;
}

double diff_norm1_stag_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                            const double* a_u, const double* a_v,
                            const double* b_u, const double* b_v) {
  CudaEventTimer timer("la_diff_norm1_stag");
  double* d_out = scalar_device_buffer();
  cudaMemset(d_out, 0, sizeof(double));

  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  diff_norm1_stag_kernel<<<grid, block>>>(mx, my, gw, stride_u, stride_v, a_u,
                                          a_v, b_u, b_v, d_out);

  double result = 0.0;
  cudaMemcpy(&result, d_out, sizeof(double), cudaMemcpyDeviceToHost);
  return result;
}

}  // namespace gpism
