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
