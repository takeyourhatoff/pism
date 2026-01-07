// Copyright (C) 2024 PISM Authors
//
// This file is part of PISM.
//
// PISM is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 3 of the License, or (at your option) any later
// version.
//
// PISM is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License
// along with PISM; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

#include "pism/stressbalance/timestepping_cuda.hh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <cerrno>
#include <petscconf.h>
#include <petscdmda.h>
#include <petscvec.h>
#include <algorithm>
#include <vector>

#include "pism/stressbalance/sia/SIAFD_stencils.hh"
#include "pism/util/petscwrappers/DM.hh"
#include "pism/util/petscwrappers/Vec.hh"
#include "pism/util/error_handling.hh"

#if defined(PETSC_HAVE_CUDA)

namespace pism {
namespace stressbalance {
namespace cuda {

namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 16;

double *g_z_device = nullptr;
int g_z_size = 0;

struct ReduceBuffers {
  double *u = nullptr;
  double *v = nullptr;
  double *w = nullptr;
  double *dt = nullptr;
  int size = 0;
};

ReduceBuffers g_cfl3d_buffers;
ReduceBuffers g_cfl2d_buffers;

int read_env_int(const char *name, int fallback, int min_value, int max_value) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0') {
    return fallback;
  }
  errno = 0;
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0') {
    return fallback;
  }
  if (parsed < min_value || parsed > max_value) {
    return fallback;
  }
  return static_cast<int>(parsed);
}

dim3 cuda_block_dims() {
  const int max_dim = 32;
  int bx = read_env_int("PISM_SIAFD_CUDA_BLOCK_X", kBlockX, 4, max_dim);
  int by = read_env_int("PISM_SIAFD_CUDA_BLOCK_Y", kBlockY, 4, max_dim);
  return dim3(bx, by);
}

void check_cuda(cudaError_t err, const char *name) {
  if (err != cudaSuccess) {
    throw RuntimeError::formatted(PISM_ERROR_LOCATION, "CUDA error in %s: %s", name,
                                  cudaGetErrorString(err));
  }
}

template <typename View>
void fill_layout(View &view, const array::Array &array) {
  ::DM da = array.dm()->get();
  int gxs = 0, gys = 0, gxm = 0, gym = 0;
  PetscErrorCode ierr = 0;
  if (array.stencil_width() > 0) {
    ierr = DMDAGetGhostCorners(da, &gxs, &gys, NULL, &gxm, &gym, NULL);
    PISM_CHK(ierr, "DMDAGetGhostCorners");
  } else {
    ierr = DMDAGetCorners(da, &gxs, &gys, NULL, &gxm, &gym, NULL);
    PISM_CHK(ierr, "DMDAGetCorners");
  }
  int dof = 1;
  ierr = DMDAGetInfo(da, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                     &dof, NULL, NULL, NULL, NULL, NULL);
  PISM_CHK(ierr, "DMDAGetInfo");
  view.gxs = gxs;
  view.gys = gys;
  view.gxm = gxm;
  view.dof = dof;
}

struct CudaArrayConstView {
  sia_kernels::DeviceArray2DConstView view;
  ::Vec vec;
  explicit CudaArrayConstView(const array::Array &array) : vec(array.vec().get()) {
    fill_layout(view, array);
    PetscErrorCode ierr = VecCUDAGetArrayRead(vec, &view.data);
    PISM_CHK(ierr, "VecCUDAGetArrayRead");
  }
  ~CudaArrayConstView() {
    PetscErrorCode ierr = VecCUDARestoreArrayRead(vec, &view.data);
    CHKERRCONTINUE(ierr);
  }
};

struct CudaMaskView {
  sia_kernels::DeviceMaskView view;
  ::Vec vec;
  explicit CudaMaskView(const array::Array &array) : vec(array.vec().get()) {
    fill_layout(view, array);
    PetscErrorCode ierr = VecCUDAGetArrayRead(vec, &view.data);
    PISM_CHK(ierr, "VecCUDAGetArrayRead");
  }
  ~CudaMaskView() {
    PetscErrorCode ierr = VecCUDARestoreArrayRead(vec, &view.data);
    CHKERRCONTINUE(ierr);
  }
};

void update_vertical_grid(const double *z, int Mz) {
  if (Mz <= 0) {
    return;
  }
  if (g_z_size != Mz) {
    if (g_z_device) {
      check_cuda(cudaFree(g_z_device), "cudaFree(g_z_device)");
      g_z_device = nullptr;
      g_z_size = 0;
    }
    check_cuda(cudaMalloc(&g_z_device, Mz * sizeof(double)), "cudaMalloc(g_z_device)");
    g_z_size = Mz;
  }
  check_cuda(cudaMemcpy(g_z_device, z, Mz * sizeof(double), cudaMemcpyHostToDevice),
             "cudaMemcpy(g_z_device)");
}

PISM_HOST_DEVICE inline int k_below_height(const double *z, int Mz, double height) {
  if (height <= z[0]) {
    return 0;
  }
  if (height >= z[Mz - 1]) {
    return Mz - 1;
  }
  int k = 0;
  for (; k < Mz - 1; ++k) {
    if (height < z[k + 1]) {
      break;
    }
  }
  return k;
}

void ensure_buffers(ReduceBuffers &buffers, int count) {
  if (count <= buffers.size) {
    return;
  }
  if (buffers.u) {
    check_cuda(cudaFree(buffers.u), "cudaFree(cfl_u)");
    buffers.u = nullptr;
  }
  if (buffers.v) {
    check_cuda(cudaFree(buffers.v), "cudaFree(cfl_v)");
    buffers.v = nullptr;
  }
  if (buffers.w) {
    check_cuda(cudaFree(buffers.w), "cudaFree(cfl_w)");
    buffers.w = nullptr;
  }
  if (buffers.dt) {
    check_cuda(cudaFree(buffers.dt), "cudaFree(cfl_dt)");
    buffers.dt = nullptr;
  }

  check_cuda(cudaMalloc(&buffers.u, sizeof(double) * count), "cudaMalloc(cfl_u)");
  check_cuda(cudaMalloc(&buffers.v, sizeof(double) * count), "cudaMalloc(cfl_v)");
  check_cuda(cudaMalloc(&buffers.w, sizeof(double) * count), "cudaMalloc(cfl_w)");
  check_cuda(cudaMalloc(&buffers.dt, sizeof(double) * count), "cudaMalloc(cfl_dt)");
  buffers.size = count;
}

__global__ void cfl3d_reduce_kernel(sia_kernels::DeviceMaskView mask,
                                    sia_kernels::DeviceArray2DConstView thk,
                                    sia_kernels::DeviceArray2DConstView u,
                                    sia_kernels::DeviceArray2DConstView v,
                                    sia_kernels::DeviceArray2DConstView w,
                                    int xs, int ys, int xm, int ym, int Mz,
                                    const double *z,
                                    double one_over_dx,
                                    double one_over_dy,
                                    double dt_max_default,
                                    double *out_u,
                                    double *out_v,
                                    double *out_w,
                                    double *out_dt) {
  const int i = xs + blockIdx.x * blockDim.x + threadIdx.x;
  const int j = ys + blockIdx.y * blockDim.y + threadIdx.y;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_size = blockDim.x * blockDim.y;

  extern __shared__ double shared[];
  double *s_u = shared;
  double *s_v = s_u + block_size;
  double *s_w = s_v + block_size;
  double *s_dt = s_w + block_size;

  double u_max = 0.0;
  double v_max = 0.0;
  double w_max = 0.0;
  double dt_min = dt_max_default;

  if (i < xs + xm && j < ys + ym) {
    if (sia_kernels::MaskOps::icy(mask.value(i, j))) {
      const double H = thk(i, j);
      const int ks = k_below_height(z, Mz, H);

      for (int k = 0; k <= ks; ++k) {
        const double u_abs = fabs(u(i, j, k));
        const double v_abs = fabs(v(i, j, k));
        u_max = fmax(u_max, u_abs);
        v_max = fmax(v_max, v_abs);
        const double denom = u_abs * one_over_dx + v_abs * one_over_dy;
        if (denom > 0.0) {
          dt_min = fmin(dt_min, 1.0 / denom);
        }
      }

      for (int k = 0; k <= ks; ++k) {
        w_max = fmax(w_max, fabs(w(i, j, k)));
      }
    }
  }

  s_u[tid] = u_max;
  s_v[tid] = v_max;
  s_w[tid] = w_max;
  s_dt[tid] = dt_min;
  __syncthreads();

  for (int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_u[tid] = fmax(s_u[tid], s_u[tid + stride]);
      s_v[tid] = fmax(s_v[tid], s_v[tid + stride]);
      s_w[tid] = fmax(s_w[tid], s_w[tid + stride]);
      s_dt[tid] = fmin(s_dt[tid], s_dt[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    const int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    out_u[block_id] = s_u[0];
    out_v[block_id] = s_v[0];
    out_w[block_id] = s_w[0];
    out_dt[block_id] = s_dt[0];
  }
}

__global__ void cfl2d_reduce_kernel(sia_kernels::DeviceMaskView mask,
                                    sia_kernels::DeviceArray2DConstView velocity,
                                    int xs, int ys, int xm, int ym,
                                    double dx, double dy,
                                    double dt_max_default,
                                    double *out_u,
                                    double *out_v,
                                    double *out_dt) {
  const int i = xs + blockIdx.x * blockDim.x + threadIdx.x;
  const int j = ys + blockIdx.y * blockDim.y + threadIdx.y;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_size = blockDim.x * blockDim.y;

  extern __shared__ double shared[];
  double *s_u = shared;
  double *s_v = s_u + block_size;
  double *s_dt = s_v + block_size;

  double u_max = 0.0;
  double v_max = 0.0;
  double dt_min = dt_max_default;

  if (i < xs + xm && j < ys + ym) {
    if (sia_kernels::MaskOps::icy(mask.value(i, j))) {
      const double u_abs = fabs(velocity(i, j, 0));
      const double v_abs = fabs(velocity(i, j, 1));
      u_max = fmax(u_max, u_abs);
      v_max = fmax(v_max, v_abs);
      const double denom = u_abs / dx + v_abs / dy;
      if (denom > 0.0) {
        dt_min = fmin(dt_min, 1.0 / denom);
      }
    }
  }

  s_u[tid] = u_max;
  s_v[tid] = v_max;
  s_dt[tid] = dt_min;
  __syncthreads();

  for (int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_u[tid] = fmax(s_u[tid], s_u[tid + stride]);
      s_v[tid] = fmax(s_v[tid], s_v[tid + stride]);
      s_dt[tid] = fmin(s_dt[tid], s_dt[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    const int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    out_u[block_id] = s_u[0];
    out_v[block_id] = s_v[0];
    out_dt[block_id] = s_dt[0];
  }
}

} // namespace

void max_timestep_cfl_3d_local(const array::Scalar &ice_thickness,
                               const array::CellType &cell_type,
                               const array::Array3D &u3,
                               const array::Array3D &v3,
                               const array::Array3D &w3,
                               int xs, int ys, int xm, int ym, int Mz,
                               const double *z,
                               double one_over_dx,
                               double one_over_dy,
                               double dt_max_default,
                               double *u_max,
                               double *v_max,
                               double *w_max,
                               double *dt_min) {
  update_vertical_grid(z, Mz);

  CudaMaskView mask_view(cell_type);
  CudaArrayConstView thk_view(ice_thickness);
  CudaArrayConstView u_view(u3);
  CudaArrayConstView v_view(v3);
  CudaArrayConstView w_view(w3);

  dim3 block = cuda_block_dims();
  dim3 grid((xm + block.x - 1) / block.x,
            (ym + block.y - 1) / block.y);
  const int num_blocks = static_cast<int>(grid.x * grid.y);
  ensure_buffers(g_cfl3d_buffers, num_blocks);

  const int block_size = block.x * block.y;
  const size_t shared_bytes = sizeof(double) * block_size * 4;

  cfl3d_reduce_kernel<<<grid, block, shared_bytes>>>(
    mask_view.view, thk_view.view, u_view.view, v_view.view, w_view.view,
    xs, ys, xm, ym, Mz, g_z_device, one_over_dx, one_over_dy, dt_max_default,
    g_cfl3d_buffers.u, g_cfl3d_buffers.v, g_cfl3d_buffers.w, g_cfl3d_buffers.dt);
  check_cuda(cudaGetLastError(), "cfl3d_reduce_kernel");

  std::vector<double> host_u(num_blocks);
  std::vector<double> host_v(num_blocks);
  std::vector<double> host_w(num_blocks);
  std::vector<double> host_dt(num_blocks);
  check_cuda(cudaMemcpy(host_u.data(), g_cfl3d_buffers.u, sizeof(double) * num_blocks,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(cfl3d_u)");
  check_cuda(cudaMemcpy(host_v.data(), g_cfl3d_buffers.v, sizeof(double) * num_blocks,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(cfl3d_v)");
  check_cuda(cudaMemcpy(host_w.data(), g_cfl3d_buffers.w, sizeof(double) * num_blocks,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(cfl3d_w)");
  check_cuda(cudaMemcpy(host_dt.data(), g_cfl3d_buffers.dt, sizeof(double) * num_blocks,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(cfl3d_dt)");

  double u_max_local = 0.0;
  double v_max_local = 0.0;
  double w_max_local = 0.0;
  double dt_min_local = dt_max_default;
  for (int i = 0; i < num_blocks; ++i) {
    u_max_local = std::max(u_max_local, host_u[i]);
    v_max_local = std::max(v_max_local, host_v[i]);
    w_max_local = std::max(w_max_local, host_w[i]);
    dt_min_local = std::min(dt_min_local, host_dt[i]);
  }

  *u_max = u_max_local;
  *v_max = v_max_local;
  *w_max = w_max_local;
  *dt_min = dt_min_local;
}

void max_timestep_cfl_2d_local(const array::CellType &cell_type,
                               const array::Vector &velocity,
                               int xs, int ys, int xm, int ym,
                               double dx, double dy,
                               double dt_max_default,
                               double *u_max,
                               double *v_max,
                               double *dt_min) {
  CudaMaskView mask_view(cell_type);
  CudaArrayConstView vel_view(velocity);

  dim3 block = cuda_block_dims();
  dim3 grid((xm + block.x - 1) / block.x,
            (ym + block.y - 1) / block.y);
  const int num_blocks = static_cast<int>(grid.x * grid.y);
  ensure_buffers(g_cfl2d_buffers, num_blocks);

  const int block_size = block.x * block.y;
  const size_t shared_bytes = sizeof(double) * block_size * 3;

  cfl2d_reduce_kernel<<<grid, block, shared_bytes>>>(
    mask_view.view, vel_view.view,
    xs, ys, xm, ym, dx, dy, dt_max_default,
    g_cfl2d_buffers.u, g_cfl2d_buffers.v, g_cfl2d_buffers.dt);
  check_cuda(cudaGetLastError(), "cfl2d_reduce_kernel");

  std::vector<double> host_u(num_blocks);
  std::vector<double> host_v(num_blocks);
  std::vector<double> host_dt(num_blocks);
  check_cuda(cudaMemcpy(host_u.data(), g_cfl2d_buffers.u, sizeof(double) * num_blocks,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(cfl2d_u)");
  check_cuda(cudaMemcpy(host_v.data(), g_cfl2d_buffers.v, sizeof(double) * num_blocks,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(cfl2d_v)");
  check_cuda(cudaMemcpy(host_dt.data(), g_cfl2d_buffers.dt, sizeof(double) * num_blocks,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(cfl2d_dt)");

  double u_max_local = 0.0;
  double v_max_local = 0.0;
  double dt_min_local = dt_max_default;
  for (int i = 0; i < num_blocks; ++i) {
    u_max_local = std::max(u_max_local, host_u[i]);
    v_max_local = std::max(v_max_local, host_v[i]);
    dt_min_local = std::min(dt_min_local, host_dt[i]);
  }

  *u_max = u_max_local;
  *v_max = v_max_local;
  *dt_min = dt_min_local;
}

} // namespace cuda
} // namespace stressbalance
} // namespace pism

#else

namespace pism {
namespace stressbalance {
namespace cuda {

void max_timestep_cfl_3d_local(const array::Scalar &, const array::CellType &,
                               const array::Array3D &, const array::Array3D &,
                               const array::Array3D &,
                               int, int, int, int, int,
                               const double *, double, double, double,
                               double *, double *, double *, double *) {}

void max_timestep_cfl_2d_local(const array::CellType &, const array::Vector &,
                               int, int, int, int,
                               double, double, double,
                               double *, double *, double *) {}

} // namespace cuda
} // namespace stressbalance
} // namespace pism

#endif
