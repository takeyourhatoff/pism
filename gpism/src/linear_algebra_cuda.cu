#include "gpism/linear_algebra.h"

#include <cmath>

#include <cuda_runtime.h>

namespace gpism {
namespace {

__device__ inline int idx(int i, int j, int gw, int stride) {
  return (j + gw) * stride + (i + gw);
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

}  // namespace

void axpy_cuda(int mx, int my, int gw, int stride_x, int stride_y,
              const double* x, double* y, double alpha) {
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  axpy_kernel<<<grid, block>>>(mx, my, gw, stride_x, stride_y, x, y, alpha);
  cudaDeviceSynchronize();
}

void scal_cuda(int mx, int my, int gw, int stride, double* x, double alpha) {
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  scal_kernel<<<grid, block>>>(mx, my, gw, stride, x, alpha);
  cudaDeviceSynchronize();
}

void copy_cuda(int mx, int my, int gw, int stride_x, int stride_y,
               const double* x, double* y) {
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  copy_kernel<<<grid, block>>>(mx, my, gw, stride_x, stride_y, x, y);
  cudaDeviceSynchronize();
}

void set_cuda(int mx, int my, int gw, int stride, double* x, double value) {
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  set_kernel<<<grid, block>>>(mx, my, gw, stride, x, value);
  cudaDeviceSynchronize();
}

double dot_cuda(int mx, int my, int gw, int stride_a, int stride_b,
                const double* a, const double* b) {
  double* d_out = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(double));
  cudaMemset(d_out, 0, sizeof(double));

  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  dot_kernel<<<grid, block>>>(mx, my, gw, stride_a, stride_b, a, b, d_out);
  cudaDeviceSynchronize();

  double result = 0.0;
  cudaMemcpy(&result, d_out, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  return result;
}

double norm1_cuda(int mx, int my, int gw, int stride, const double* a) {
  double* d_out = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(double));
  cudaMemset(d_out, 0, sizeof(double));

  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  norm1_kernel<<<grid, block>>>(mx, my, gw, stride, a, d_out);
  cudaDeviceSynchronize();

  double result = 0.0;
  cudaMemcpy(&result, d_out, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  return result;
}

}  // namespace gpism
