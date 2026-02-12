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

double* scalar_host_buffer() {
  static double* buffer = nullptr;
  if (!buffer) {
    cudaHostAlloc(reinterpret_cast<void**>(&buffer),
                  sizeof(double), cudaHostAllocDefault);
  }
  return buffer;
}

double read_scalar(double* d_out) {
  double* h_out = scalar_host_buffer();
  // Use a blocking copy instead of async+stream synchronize to avoid an extra
  // API sync call per scalar reduction.
  cudaMemcpy(h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost);
  return *h_out;
}

cudaStream_t resolve_stream(cudaStream_t stream) { return stream; }

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

__global__ void axpy_stag_kernel(int mx, int my, int gw, int stride_x_u,
                                 int stride_x_v, int stride_y_u,
                                 int stride_y_v, const double* x_u,
                                 const double* x_v, double* y_u, double* y_v,
                                 double alpha) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int ix_u = idx(i, j, gw, stride_x_u);
  const int ix_v = idx(i, j, gw, stride_x_v);
  const int iy_u = idx(i, j, gw, stride_y_u);
  const int iy_v = idx(i, j, gw, stride_y_v);
  y_u[iy_u] += alpha * x_u[ix_u];
  y_v[iy_v] += alpha * x_v[ix_v];
}

__global__ void scal_stag_kernel(int mx, int my, int gw, int stride_u,
                                 int stride_v, double* x_u, double* x_v,
                                 double alpha) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int iu = idx(i, j, gw, stride_u);
  const int iv = idx(i, j, gw, stride_v);
  x_u[iu] *= alpha;
  x_v[iv] *= alpha;
}

__global__ void copy_stag_kernel(int mx, int my, int gw, int stride_x_u,
                                 int stride_x_v, int stride_y_u,
                                 int stride_y_v, const double* x_u,
                                 const double* x_v, double* y_u, double* y_v) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int ix_u = idx(i, j, gw, stride_x_u);
  const int ix_v = idx(i, j, gw, stride_x_v);
  const int iy_u = idx(i, j, gw, stride_y_u);
  const int iy_v = idx(i, j, gw, stride_y_v);
  y_u[iy_u] = x_u[ix_u];
  y_v[iy_v] = x_v[ix_v];
}

__global__ void set_stag_kernel(int mx, int my, int gw, int stride_u,
                                int stride_v, double* x_u, double* x_v,
                                double value) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int iu = idx(i, j, gw, stride_u);
  const int iv = idx(i, j, gw, stride_v);
  x_u[iu] = value;
  x_v[iv] = value;
}

__global__ void dot_kernel(int mx, int my, int gw, int stride_a, int stride_b,
                           const double* a, const double* b, double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const bool active = (i < mx && j < my);
  double val = 0.0;
  if (active) {
    const int ia = idx(i, j, gw, stride_a);
    const int ib = idx(i, j, gw, stride_b);
    val = a[ia] * b[ib];
  }
  extern __shared__ double sdata[];
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_threads = blockDim.x * blockDim.y;
  sdata[tid] = val;
  __syncthreads();
  for (int s = block_threads / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(out, sdata[0]);
  }
}

__global__ void dot_stag_kernel(int mx, int my, int gw, int stride_u, int stride_v,
                                const double* a_u, const double* a_v,
                                const double* b_u, const double* b_v,
                                double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const bool active = (i < mx && j < my);
  double val = 0.0;
  if (active) {
    const int iu = idx(i, j, gw, stride_u);
    const int iv = idx(i, j, gw, stride_v);
    val = a_u[iu] * b_u[iu] + a_v[iv] * b_v[iv];
  }

  extern __shared__ double sdata[];
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_threads = blockDim.x * blockDim.y;
  sdata[tid] = val;
  __syncthreads();

  for (int s = block_threads / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(out, sdata[0]);
  }
}

__global__ void dot_stag_batch_kernel(int mx, int my, int gw, int stride_u,
                                      int stride_v, const double* w_u,
                                      const double* w_v, const double** V_u,
                                      const double** V_v, int count,
                                      double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const bool active = (i < mx && j < my);
  int iu = 0;
  int iv = 0;
  double wu = 0.0;
  double wv = 0.0;
  if (active) {
    iu = idx(i, j, gw, stride_u);
    iv = idx(i, j, gw, stride_v);
    wu = w_u[iu];
    wv = w_v[iv];
  }

  extern __shared__ double sdata[];
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_threads = blockDim.x * blockDim.y;

  for (int k = 0; k < count; ++k) {
    double val = 0.0;
    if (active) {
      val = wu * V_u[k][iu] + wv * V_v[k][iv];
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = block_threads / 2; s > 0; s >>= 1) {
      if (tid < s) {
        sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
    }

    if (tid == 0) {
      atomicAdd(&out[k], sdata[0]);
    }
    __syncthreads();
  }
}

__global__ void norm1_kernel(int mx, int my, int gw, int stride, const double* a,
                             double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const bool active = (i < mx && j < my);
  double val = 0.0;
  if (active) {
    const int ia = idx(i, j, gw, stride);
    val = fabs(a[ia]);
  }
  extern __shared__ double sdata[];
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_threads = blockDim.x * blockDim.y;
  sdata[tid] = val;
  __syncthreads();
  for (int s = block_threads / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(out, sdata[0]);
  }
}

__global__ void norm1_stag_kernel(int mx, int my, int gw, int stride_u, int stride_v,
                                  const double* a_u, const double* a_v,
                                  double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const bool active = (i < mx && j < my);
  double val = 0.0;
  if (active) {
    const int iu = idx(i, j, gw, stride_u);
    const int iv = idx(i, j, gw, stride_v);
    val = fabs(a_u[iu]) + fabs(a_v[iv]);
  }
  extern __shared__ double sdata[];
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_threads = blockDim.x * blockDim.y;
  sdata[tid] = val;
  __syncthreads();
  for (int s = block_threads / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(out, sdata[0]);
  }
}

__global__ void diff_norm1_kernel(int mx, int my, int gw, int stride_a,
                                  int stride_b, const double* a,
                                  const double* b, double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const bool active = (i < mx && j < my);
  double val = 0.0;
  if (active) {
    const int ia = idx(i, j, gw, stride_a);
    const int ib = idx(i, j, gw, stride_b);
    val = fabs(a[ia] - b[ib]);
  }
  extern __shared__ double sdata[];
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_threads = blockDim.x * blockDim.y;
  sdata[tid] = val;
  __syncthreads();
  for (int s = block_threads / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(out, sdata[0]);
  }
}

__global__ void diff_norm1_stag_kernel(int mx, int my, int gw, int stride_u,
                                       int stride_v, const double* a_u,
                                       const double* a_v, const double* b_u,
                                       const double* b_v, double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const bool active = (i < mx && j < my);
  double val = 0.0;
  if (active) {
    const int iu = idx(i, j, gw, stride_u);
    const int iv = idx(i, j, gw, stride_v);
    val = fabs(a_u[iu] - b_u[iu]) + fabs(a_v[iv] - b_v[iv]);
  }
  extern __shared__ double sdata[];
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_threads = blockDim.x * blockDim.y;
  sdata[tid] = val;
  __syncthreads();
  for (int s = block_threads / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(out, sdata[0]);
  }
}

__global__ void orthogonalize_stag_kernel(int mx, int my, int gw, int stride_u,
                                          int stride_v, const double** V_u,
                                          const double** V_v,
                                          const double* hij, int count,
                                          double* w_u, double* w_v,
                                          double* w_norm2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  const bool active = (i < mx && j < my);
  double val = 0.0;
  if (active) {
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
    val = wu * wu + wv * wv;
  }

  extern __shared__ double sdata[];
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_threads = blockDim.x * blockDim.y;
  sdata[tid] = val;
  __syncthreads();

  for (int s = block_threads / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(w_norm2, sdata[0]);
  }
}

__global__ void normalize_stag_from_norm2_kernel(
    int mx, int my, int gw, int stride_u, int stride_v, const double* src_u,
    const double* src_v, double* dst_u, double* dst_v, const double* norm2_ptr) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int iu = idx(i, j, gw, stride_u);
  const int iv = idx(i, j, gw, stride_v);
  const double norm2 = norm2_ptr[0];
  if (norm2 > 0.0) {
    const double inv = 1.0 / sqrt(norm2);
    dst_u[iu] = src_u[iu] * inv;
    dst_v[iv] = src_v[iv] * inv;
  } else {
    dst_u[iu] = 0.0;
    dst_v[iv] = 0.0;
  }
}

__global__ void gmres_apply_givens_kernel(double* hessenberg, int restart,
                                          int j, double* cs, double* sn,
                                          double* g) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  const int lda = restart + 1;
  const int col = j * lda;
  for (int i = 0; i < j; ++i) {
    const double c = cs[i];
    const double s = sn[i];
    const double h0 = hessenberg[col + i];
    const double h1 = hessenberg[col + i + 1];
    hessenberg[col + i] = c * h0 + s * h1;
    hessenberg[col + i + 1] = -s * h0 + c * h1;
  }

  double h00 = hessenberg[col + j];
  double h10 = hessenberg[col + j + 1];
  h10 = (h10 > 0.0) ? sqrt(h10) : 0.0;
  hessenberg[col + j + 1] = h10;
  double c = 1.0;
  double s = 0.0;
  if (h10 != 0.0) {
    if (fabs(h10) > fabs(h00)) {
      const double tau = h00 / h10;
      s = 1.0 / sqrt(1.0 + tau * tau);
      c = s * tau;
    } else {
      const double tau = h10 / h00;
      c = 1.0 / sqrt(1.0 + tau * tau);
      s = c * tau;
    }
  }
  cs[j] = c;
  sn[j] = s;

  hessenberg[col + j] = c * h00 + s * h10;
  hessenberg[col + j + 1] = -s * h00 + c * h10;

  const double g0 = g[j];
  const double g1 = g[j + 1];
  g[j] = c * g0 + s * g1;
  g[j + 1] = -s * g0 + c * g1;
}

__global__ void gmres_backsolve_kernel(const double* hessenberg, int restart,
                                       int k, const double* g, double* y) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  const int lda = restart + 1;
  for (int i = 0; i < k; ++i) {
    y[i] = 0.0;
  }
  for (int i = k - 1; i >= 0; --i) {
    double sum = g[i];
    for (int j = i + 1; j < k; ++j) {
      sum -= hessenberg[j * lda + i] * y[j];
    }
    const double h_ii = hessenberg[i * lda + i];
    y[i] = (h_ii == 0.0) ? 0.0 : (sum / h_ii);
  }
}

__global__ void gmres_update_solution_stag_kernel(
    int mx, int my, int gw, int stride_u, int stride_v, double* x_u,
    double* x_v, const double** V_u, const double** V_v, const double* y, int k) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int iu = idx(i, j, gw, stride_u);
  const int iv = idx(i, j, gw, stride_v);
  double du = 0.0;
  double dv = 0.0;
  for (int n = 0; n < k; ++n) {
    const double coeff = y[n];
    du += coeff * V_u[n][iu];
    dv += coeff * V_v[n][iv];
  }
  x_u[iu] += du;
  x_v[iv] += dv;
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

void axpy_stag_cuda(int mx, int my, int gw, int stride_x_u, int stride_x_v,
                    int stride_y_u, int stride_y_v, const double* x_u,
                    const double* x_v, double* y_u, double* y_v,
                    double alpha) {
  CudaEventTimer timer("la_axpy_stag");
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  axpy_stag_kernel<<<grid, block>>>(mx, my, gw, stride_x_u, stride_x_v,
                                    stride_y_u, stride_y_v, x_u, x_v, y_u, y_v,
                                    alpha);
}

void scal_stag_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                    double* x_u, double* x_v, double alpha) {
  CudaEventTimer timer("la_scal_stag");
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  scal_stag_kernel<<<grid, block>>>(mx, my, gw, stride_u, stride_v, x_u, x_v,
                                    alpha);
}

void copy_stag_cuda(int mx, int my, int gw, int stride_x_u, int stride_x_v,
                    int stride_y_u, int stride_y_v, const double* x_u,
                    const double* x_v, double* y_u, double* y_v) {
  CudaEventTimer timer("la_copy_stag");
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  copy_stag_kernel<<<grid, block>>>(mx, my, gw, stride_x_u, stride_x_v,
                                    stride_y_u, stride_y_v, x_u, x_v, y_u, y_v);
}

void set_stag_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                   double* x_u, double* x_v, double value) {
  CudaEventTimer timer("la_set_stag");
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  set_stag_kernel<<<grid, block>>>(mx, my, gw, stride_u, stride_v, x_u, x_v,
                                   value);
}

double dot_cuda(int mx, int my, int gw, int stride_a, int stride_b,
                const double* a, const double* b) {
  CudaEventTimer timer("la_dot");
  double* d_out = scalar_device_buffer();
  cudaMemset(d_out, 0, sizeof(double));

  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  const std::size_t shared_bytes =
      static_cast<std::size_t>(block.x) * static_cast<std::size_t>(block.y) *
      sizeof(double);
  dot_kernel<<<grid, block, shared_bytes>>>(mx, my, gw, stride_a, stride_b, a,
                                            b, d_out);

  return read_scalar(d_out);
}

double dot_stag_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                     const double* a_u, const double* a_v,
                     const double* b_u, const double* b_v) {
  CudaEventTimer timer("la_dot_stag");
  double* d_out = scalar_device_buffer();
  cudaMemset(d_out, 0, sizeof(double));

  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  const std::size_t shared_bytes =
      static_cast<std::size_t>(block.x) * static_cast<std::size_t>(block.y) *
      sizeof(double);
  dot_stag_kernel<<<grid, block, shared_bytes>>>(
      mx, my, gw, stride_u, stride_v, a_u, a_v, b_u, b_v, d_out);

  return read_scalar(d_out);
}

void dot_stag_cuda_async(int mx, int my, int gw, int stride_u, int stride_v,
                         const double* a_u, const double* a_v,
                         const double* b_u, const double* b_v, double* out,
                         int out_index, cudaStream_t stream) {
  CudaEventTimer timer("la_dot_stag_async");
  cudaStream_t exec_stream = resolve_stream(stream);
  double* out_ptr = out + out_index;
  cudaMemsetAsync(out_ptr, 0, sizeof(double), exec_stream);
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  const std::size_t shared_bytes =
      static_cast<std::size_t>(block.x) * static_cast<std::size_t>(block.y) *
      sizeof(double);
  dot_stag_kernel<<<grid, block, shared_bytes, exec_stream>>>(
      mx, my, gw, stride_u, stride_v, a_u, a_v, b_u, b_v, out_ptr);
}

void dot_stag_batch_cuda_async(int mx, int my, int gw, int stride_u,
                               int stride_v, const double* w_u,
                               const double* w_v, const double** V_u,
                               const double** V_v, int count, double* out,
                               cudaStream_t stream) {
  CudaEventTimer timer("la_dot_stag_batch_async");
  if (count <= 0) {
    return;
  }
  cudaStream_t exec_stream = resolve_stream(stream);
  cudaMemsetAsync(out, 0, static_cast<std::size_t>(count) * sizeof(double),
                  exec_stream);
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  const std::size_t shared_bytes =
      static_cast<std::size_t>(block.x) * static_cast<std::size_t>(block.y) *
      sizeof(double);
  dot_stag_batch_kernel<<<grid, block, shared_bytes, exec_stream>>>(
      mx, my, gw, stride_u, stride_v, w_u, w_v, V_u, V_v, count, out);
}

void norm1_cuda_async(int mx, int my, int gw, int stride, const double* a,
                      double* out, int out_index, cudaStream_t stream) {
  CudaEventTimer timer("la_norm1_async");
  cudaStream_t exec_stream = resolve_stream(stream);
  double* out_ptr = out + out_index;
  cudaMemsetAsync(out_ptr, 0, sizeof(double), exec_stream);
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  const std::size_t shared_bytes =
      static_cast<std::size_t>(block.x) * static_cast<std::size_t>(block.y) *
      sizeof(double);
  norm1_kernel<<<grid, block, shared_bytes, exec_stream>>>(mx, my, gw, stride,
                                                            a, out_ptr);
}

void diff_norm1_cuda_async(int mx, int my, int gw, int stride_a, int stride_b,
                           const double* a, const double* b, double* out,
                           int out_index, cudaStream_t stream) {
  CudaEventTimer timer("la_diff_norm1_async");
  cudaStream_t exec_stream = resolve_stream(stream);
  double* out_ptr = out + out_index;
  cudaMemsetAsync(out_ptr, 0, sizeof(double), exec_stream);
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  const std::size_t shared_bytes =
      static_cast<std::size_t>(block.x) * static_cast<std::size_t>(block.y) *
      sizeof(double);
  diff_norm1_kernel<<<grid, block, shared_bytes, exec_stream>>>(
      mx, my, gw, stride_a, stride_b, a, b, out_ptr);
}

void device_scalars_copy_to_host(const double* d_src, double* h_dst, int count) {
  if (count <= 0) {
    return;
  }
  cudaMemcpy(h_dst, d_src, static_cast<std::size_t>(count) * sizeof(double),
             cudaMemcpyDeviceToHost);
}

void device_scalars_copy_to_host_async(const double* d_src, double* h_dst,
                                       int count, cudaStream_t stream) {
  if (count <= 0) {
    return;
  }
  cudaMemcpyAsync(h_dst, d_src, static_cast<std::size_t>(count) * sizeof(double),
                  cudaMemcpyDeviceToHost, resolve_stream(stream));
}

void cuda_stream_synchronize(cudaStream_t stream) {
  cudaStreamSynchronize(resolve_stream(stream));
}

void orthogonalize_stag_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                             double* w_u, double* w_v, const double** V_u,
                             const double** V_v, int count, double* hij) {
  CudaEventTimer timer("la_orthogonalize_stag");
  if (count <= 0) {
    return;
  }
  cudaMemset(hij, 0, static_cast<std::size_t>(count + 1) * sizeof(double));
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  const std::size_t shared_bytes =
      static_cast<std::size_t>(block.x) * static_cast<std::size_t>(block.y) *
      sizeof(double);
  dot_stag_batch_kernel<<<grid, block, shared_bytes>>>(
      mx, my, gw, stride_u, stride_v, w_u, w_v, V_u, V_v, count, hij);
  orthogonalize_stag_kernel<<<grid, block, shared_bytes>>>(
      mx, my, gw, stride_u, stride_v, V_u, V_v, hij, count, w_u, w_v,
      hij + count);
}

void normalize_stag_from_norm2_cuda(int mx, int my, int gw, int stride_u,
                                    int stride_v, const double* src_u,
                                    const double* src_v, double* dst_u,
                                    double* dst_v, const double* norm2,
                                    cudaStream_t stream) {
  CudaEventTimer timer("la_normalize_stag_from_norm2");
  cudaStream_t exec_stream = resolve_stream(stream);
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  normalize_stag_from_norm2_kernel<<<grid, block, 0, exec_stream>>>(
      mx, my, gw, stride_u, stride_v, src_u, src_v, dst_u, dst_v, norm2);
}

void gmres_apply_givens_cuda(double* hessenberg, int restart, int j,
                             double* cs, double* sn, double* g,
                             cudaStream_t stream) {
  CudaEventTimer timer("la_gmres_apply_givens");
  gmres_apply_givens_kernel<<<1, 1, 0, resolve_stream(stream)>>>(
      hessenberg, restart, j, cs, sn, g);
}

void gmres_backsolve_cuda(const double* hessenberg, int restart, int k,
                          const double* g, double* y, cudaStream_t stream) {
  CudaEventTimer timer("la_gmres_backsolve");
  gmres_backsolve_kernel<<<1, 1, 0, resolve_stream(stream)>>>(
      hessenberg, restart, k, g, y);
}

void gmres_update_solution_stag_cuda(
    int mx, int my, int gw, int stride_u, int stride_v, double* x_u,
    double* x_v, const double** V_u, const double** V_v, const double* y, int k,
    cudaStream_t stream) {
  CudaEventTimer timer("la_gmres_update_solution");
  if (k <= 0) {
    return;
  }
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  gmres_update_solution_stag_kernel<<<grid, block, 0, resolve_stream(stream)>>>(
      mx, my, gw, stride_u, stride_v, x_u, x_v, V_u, V_v, y, k);
}

double norm1_cuda(int mx, int my, int gw, int stride, const double* a) {
  CudaEventTimer timer("la_norm1");
  double* d_out = scalar_device_buffer();
  cudaMemset(d_out, 0, sizeof(double));

  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  const std::size_t shared_bytes =
      static_cast<std::size_t>(block.x) * static_cast<std::size_t>(block.y) *
      sizeof(double);
  norm1_kernel<<<grid, block, shared_bytes>>>(mx, my, gw, stride, a, d_out);

  return read_scalar(d_out);
}

double norm1_stag_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                       const double* a_u, const double* a_v) {
  CudaEventTimer timer("la_norm1_stag");
  double* d_out = scalar_device_buffer();
  cudaMemset(d_out, 0, sizeof(double));

  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  const std::size_t shared_bytes =
      static_cast<std::size_t>(block.x) * static_cast<std::size_t>(block.y) *
      sizeof(double);
  norm1_stag_kernel<<<grid, block, shared_bytes>>>(
      mx, my, gw, stride_u, stride_v, a_u, a_v, d_out);

  return read_scalar(d_out);
}

double diff_norm1_cuda(int mx, int my, int gw, int stride_a, int stride_b,
                       const double* a, const double* b) {
  CudaEventTimer timer("la_diff_norm1");
  double* d_out = scalar_device_buffer();
  cudaMemset(d_out, 0, sizeof(double));

  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  const std::size_t shared_bytes =
      static_cast<std::size_t>(block.x) * static_cast<std::size_t>(block.y) *
      sizeof(double);
  diff_norm1_kernel<<<grid, block, shared_bytes>>>(
      mx, my, gw, stride_a, stride_b, a, b, d_out);

  return read_scalar(d_out);
}

double diff_norm1_stag_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                            const double* a_u, const double* a_v,
                            const double* b_u, const double* b_v) {
  CudaEventTimer timer("la_diff_norm1_stag");
  double* d_out = scalar_device_buffer();
  cudaMemset(d_out, 0, sizeof(double));

  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  const std::size_t shared_bytes =
      static_cast<std::size_t>(block.x) * static_cast<std::size_t>(block.y) *
      sizeof(double);
  diff_norm1_stag_kernel<<<grid, block, shared_bytes>>>(
      mx, my, gw, stride_u, stride_v, a_u, a_v, b_u, b_v, d_out);

  return read_scalar(d_out);
}

}  // namespace gpism
