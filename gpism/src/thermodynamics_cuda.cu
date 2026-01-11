#include "gpism/thermodynamics.h"

#include <cuda_runtime.h>

namespace gpism {
namespace {

__device__ inline int idx(int i, int j, int k, int gw, int stride, int nz) {
  return ((j + gw) * stride + (i + gw)) * nz + k;
}

struct DiffusionWorkspace {
  std::size_t capacity = 0;
  double* a = nullptr;
  double* b = nullptr;
  double* c = nullptr;
  double* d = nullptr;
};

DiffusionWorkspace& diffusion_workspace() {
  static DiffusionWorkspace workspace;
  return workspace;
}

__global__ void assemble_kernel(int mx, int my, int gw, int nz, int stride,
                                const double* enthalpy, double* a, double* b,
                                double* c, double* d, double r,
                                double surface_value, double basal_value,
                                int dirichlet) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  for (int k = 0; k < nz; ++k) {
    const int id = idx(i, j, k, gw, stride, nz);
    if (dirichlet && (k == 0 || k == nz - 1)) {
      a[id] = 0.0;
      b[id] = 1.0;
      c[id] = 0.0;
      d[id] = (k == 0) ? surface_value : basal_value;
    } else {
      a[id] = -r;
      b[id] = 1.0 + 2.0 * r;
      c[id] = -r;
      d[id] = enthalpy[id];
    }
  }
}

__global__ void solve_kernel(int mx, int my, int gw, int nz, int stride,
                             const double* a, const double* b, double* c,
                             double* d, double* enthalpy_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }

  int base = idx(i, j, 0, gw, stride, nz);
  double c_prev = c[base] / b[base];
  double d_prev = d[base] / b[base];
  c[base] = c_prev;
  d[base] = d_prev;

  for (int k = 1; k < nz; ++k) {
    const int id = base + k;
    const double denom = b[id] - a[id] * c_prev;
    c_prev = c[id] / denom;
    d_prev = (d[id] - a[id] * d_prev) / denom;
    c[id] = c_prev;
    d[id] = d_prev;
  }

  int last = base + (nz - 1);
  enthalpy_out[last] = d[last];
  for (int k = nz - 2; k >= 0; --k) {
    const int id = base + k;
    enthalpy_out[id] = d[id] - c[id] * enthalpy_out[id + 1];
  }
}

}  // namespace

void vertical_diffusion_step_cuda(int mx, int my, int gw, int nz, int stride,
                                  const double* enthalpy_in, double* enthalpy_out,
                                  double kappa, double dz, double dt,
                                  double surface_value, double basal_value,
                                  int dirichlet) {
  const std::size_t total = static_cast<std::size_t>(stride) *
                            static_cast<std::size_t>(my + 2 * gw) *
                            static_cast<std::size_t>(nz);
  DiffusionWorkspace& workspace = diffusion_workspace();
  if (total > workspace.capacity) {
    if (workspace.a) {
      cudaFree(workspace.a);
      cudaFree(workspace.b);
      cudaFree(workspace.c);
      cudaFree(workspace.d);
    }
    cudaMalloc(reinterpret_cast<void**>(&workspace.a), total * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&workspace.b), total * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&workspace.c), total * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&workspace.d), total * sizeof(double));
    workspace.capacity = total;
  }

  const double r = kappa * dt / (dz * dz);
  dim3 block(16, 16);
  dim3 grid((mx + block.x - 1) / block.x, (my + block.y - 1) / block.y);
  assemble_kernel<<<grid, block>>>(mx, my, gw, nz, stride, enthalpy_in,
                                   workspace.a, workspace.b, workspace.c,
                                   workspace.d, r, surface_value, basal_value,
                                   dirichlet);
  solve_kernel<<<grid, block>>>(mx, my, gw, nz, stride, workspace.a,
                                workspace.b, workspace.c, workspace.d,
                                enthalpy_out);
}

}  // namespace gpism
