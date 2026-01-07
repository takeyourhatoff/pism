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

#include "pism/stressbalance/StressBalance_cuda.hh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <cerrno>
#include <petscconf.h>
#include <petscdmda.h>
#include <petscvec.h>

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

struct CudaArrayView {
  sia_kernels::DeviceArray2DView view;
  ::Vec vec;
  explicit CudaArrayView(array::Array &array) : vec(array.vec().get()) {
    fill_layout(view, array);
    PetscErrorCode ierr = VecCUDAGetArray(vec, &view.data);
    PISM_CHK(ierr, "VecCUDAGetArray");
  }
  ~CudaArrayView() {
    PetscErrorCode ierr = VecCUDARestoreArray(vec, &view.data);
    CHKERRCONTINUE(ierr);
  }
};

struct CudaArrayWriteView {
  sia_kernels::DeviceArray2DView view;
  ::Vec vec;
  explicit CudaArrayWriteView(array::Array &array) : vec(array.vec().get()) {
    fill_layout(view, array);
    PetscErrorCode ierr = VecCUDAGetArrayWrite(vec, &view.data);
    PISM_CHK(ierr, "VecCUDAGetArrayWrite");
  }
  ~CudaArrayWriteView() {
    PetscErrorCode ierr = VecCUDARestoreArrayWrite(vec, &view.data);
    CHKERRCONTINUE(ierr);
  }
};

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

template <typename Functor>
__global__ void kernel_2d(Functor functor, int xs, int ys, int xm, int ym, int W) {
  const int i = xs - W + blockIdx.x * blockDim.x + threadIdx.x;
  const int j = ys - W + blockIdx.y * blockDim.y + threadIdx.y;
  if (i <= xs + xm + W - 1 && j <= ys + ym + W - 1) {
    functor(i, j);
  }
}

template <typename Functor>
void launch_2d(Functor functor, int xs, int ys, int xm, int ym, int W) {
  const int nx = xm + 2 * W;
  const int ny = ym + 2 * W;
  dim3 block = cuda_block_dims();
  dim3 grid((nx + block.x - 1) / block.x,
            (ny + block.y - 1) / block.y);
  kernel_2d<<<grid, block>>>(functor, xs, ys, xm, ym, W);
  check_cuda(cudaGetLastError(), "kernel_2d");
}

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

__device__ inline double pb_softness(double T, double A,
                                     double Q, double gas_const) {
  return A * exp(-Q / (gas_const * T));
}

__device__ inline double pb_cold_temperature(double E, double P,
                                             double T_melting, double beta,
                                             double c_i, double T_0) {
  const double T_m = T_melting - beta * P;
  const double E_s = c_i * (T_m - T_0);
  if (E < E_s) {
    return (E / c_i) + T_0;
  }
  return T_m;
}

__device__ inline double strain_heating_invariant(double u_x, double u_y, double u_z,
                                                  double v_x, double v_y, double v_z) {
  const double ux_vy = u_x + v_y;
  const double uy_vx = u_y + v_x;
  return 0.5 * (ux_vy * ux_vy + u_x * u_x + v_y * v_y +
                0.5 * (uy_vx * uy_vx + u_z * u_z + v_z * v_z));
}

struct VerticalVelocityNoBasal {
  sia_kernels::DeviceMaskView mask;
  sia_kernels::DeviceArray2DConstView u;
  sia_kernels::DeviceArray2DConstView v;
  sia_kernels::DeviceArray2DView w;
  const double *z;
  int Mz;
  double dx;
  double dy;
  int use_upstream_fd;

  __device__ inline void operator()(int i, int j) const {
    double west = 1.0;
    double east = 1.0;
    double south = 1.0;
    double north = 1.0;
    double D_x = 0.0;
    double D_y = 0.0;

    if (use_upstream_fd) {
      const double uw = 0.5 * (u(i - 1, j, 0) + u(i, j, 0));
      const double ue = 0.5 * (u(i, j, 0) + u(i + 1, j, 0));
      if (uw > 0.0 && ue >= 0.0) {
        west = 1.0;
        east = 0.0;
      } else if (uw <= 0.0 && ue < 0.0) {
        west = 0.0;
        east = 1.0;
      } else {
        west = 1.0;
        east = 1.0;
      }
    }

    const int mask_c = mask.value(i, j);
    if ((sia_kernels::MaskOps::icy(mask_c) && sia_kernels::MaskOps::ice_free(mask.value(i + 1, j))) ||
        (sia_kernels::MaskOps::ice_free(mask_c) && sia_kernels::MaskOps::icy(mask.value(i + 1, j)))) {
      east = 0.0;
    }
    if ((sia_kernels::MaskOps::icy(mask_c) && sia_kernels::MaskOps::ice_free(mask.value(i - 1, j))) ||
        (sia_kernels::MaskOps::ice_free(mask_c) && sia_kernels::MaskOps::icy(mask.value(i - 1, j)))) {
      west = 0.0;
    }
    if (east + west > 0.0) {
      D_x = 1.0 / (dx * (east + west));
    }

    if (use_upstream_fd) {
      const double vs = 0.5 * (v(i, j - 1, 0) + v(i, j, 0));
      const double vn = 0.5 * (v(i, j, 0) + v(i, j + 1, 0));
      if (vs > 0.0 && vn >= 0.0) {
        south = 1.0;
        north = 0.0;
      } else if (vs <= 0.0 && vn < 0.0) {
        south = 0.0;
        north = 1.0;
      } else {
        south = 1.0;
        north = 1.0;
      }
    }

    if ((sia_kernels::MaskOps::icy(mask_c) && sia_kernels::MaskOps::ice_free(mask.value(i, j + 1))) ||
        (sia_kernels::MaskOps::ice_free(mask_c) && sia_kernels::MaskOps::icy(mask.value(i, j + 1)))) {
      north = 0.0;
    }
    if ((sia_kernels::MaskOps::icy(mask_c) && sia_kernels::MaskOps::ice_free(mask.value(i, j - 1))) ||
        (sia_kernels::MaskOps::ice_free(mask_c) && sia_kernels::MaskOps::icy(mask.value(i, j - 1)))) {
      south = 0.0;
    }
    if (north + south > 0.0) {
      D_y = 1.0 / (dy * (north + south));
    }

    w(i, j, 0) = 0.0;
    double prev = 0.0;
    for (int k = 0; k < Mz; ++k) {
      const double u_x = D_x * (west * (u(i, j, k) - u(i - 1, j, k)) +
                                east * (u(i + 1, j, k) - u(i, j, k)));
      const double v_y = D_y * (south * (v(i, j, k) - v(i, j - 1, k)) +
                                north * (v(i, j + 1, k) - v(i, j, k)));
      const double current = u_x + v_y;
      if (k > 0) {
        const double dz = z[k] - z[k - 1];
        w(i, j, k) = w(i, j, k - 1) - 0.5 * dz * (current + prev);
      }
      prev = current;
    }
  }
};

struct VerticalVelocityBasal {
  sia_kernels::DeviceMaskView mask;
  sia_kernels::DeviceArray2DConstView u;
  sia_kernels::DeviceArray2DConstView v;
  sia_kernels::DeviceArray2DConstView basal;
  sia_kernels::DeviceArray2DView w;
  const double *z;
  int Mz;
  double dx;
  double dy;
  int use_upstream_fd;

  __device__ inline void operator()(int i, int j) const {
    double west = 1.0;
    double east = 1.0;
    double south = 1.0;
    double north = 1.0;
    double D_x = 0.0;
    double D_y = 0.0;

    if (use_upstream_fd) {
      const double uw = 0.5 * (u(i - 1, j, 0) + u(i, j, 0));
      const double ue = 0.5 * (u(i, j, 0) + u(i + 1, j, 0));
      if (uw > 0.0 && ue >= 0.0) {
        west = 1.0;
        east = 0.0;
      } else if (uw <= 0.0 && ue < 0.0) {
        west = 0.0;
        east = 1.0;
      } else {
        west = 1.0;
        east = 1.0;
      }
    }

    const int mask_c = mask.value(i, j);
    if ((sia_kernels::MaskOps::icy(mask_c) && sia_kernels::MaskOps::ice_free(mask.value(i + 1, j))) ||
        (sia_kernels::MaskOps::ice_free(mask_c) && sia_kernels::MaskOps::icy(mask.value(i + 1, j)))) {
      east = 0.0;
    }
    if ((sia_kernels::MaskOps::icy(mask_c) && sia_kernels::MaskOps::ice_free(mask.value(i - 1, j))) ||
        (sia_kernels::MaskOps::ice_free(mask_c) && sia_kernels::MaskOps::icy(mask.value(i - 1, j)))) {
      west = 0.0;
    }
    if (east + west > 0.0) {
      D_x = 1.0 / (dx * (east + west));
    }

    if (use_upstream_fd) {
      const double vs = 0.5 * (v(i, j - 1, 0) + v(i, j, 0));
      const double vn = 0.5 * (v(i, j, 0) + v(i, j + 1, 0));
      if (vs > 0.0 && vn >= 0.0) {
        south = 1.0;
        north = 0.0;
      } else if (vs <= 0.0 && vn < 0.0) {
        south = 0.0;
        north = 1.0;
      } else {
        south = 1.0;
        north = 1.0;
      }
    }

    if ((sia_kernels::MaskOps::icy(mask_c) && sia_kernels::MaskOps::ice_free(mask.value(i, j + 1))) ||
        (sia_kernels::MaskOps::ice_free(mask_c) && sia_kernels::MaskOps::icy(mask.value(i, j + 1)))) {
      north = 0.0;
    }
    if ((sia_kernels::MaskOps::icy(mask_c) && sia_kernels::MaskOps::ice_free(mask.value(i, j - 1))) ||
        (sia_kernels::MaskOps::ice_free(mask_c) && sia_kernels::MaskOps::icy(mask.value(i, j - 1)))) {
      south = 0.0;
    }
    if (north + south > 0.0) {
      D_y = 1.0 / (dy * (north + south));
    }

    w(i, j, 0) = -basal(i, j);
    double prev = 0.0;
    for (int k = 0; k < Mz; ++k) {
      const double u_x = D_x * (west * (u(i, j, k) - u(i - 1, j, k)) +
                                east * (u(i + 1, j, k) - u(i, j, k)));
      const double v_y = D_y * (south * (v(i, j, k) - v(i, j - 1, k)) +
                                north * (v(i, j + 1, k) - v(i, j, k)));
      const double current = u_x + v_y;
      if (k > 0) {
        const double dz = z[k] - z[k - 1];
        w(i, j, k) = w(i, j, k - 1) - 0.5 * dz * (current + prev);
      }
      prev = current;
    }
  }
};

struct StrainHeatingPB {
  sia_kernels::DeviceMaskView mask;
  sia_kernels::DeviceArray2DConstView u;
  sia_kernels::DeviceArray2DConstView v;
  sia_kernels::DeviceArray2DConstView thk;
  sia_kernels::DeviceArray2DConstView enthalpy;
  sia_kernels::DeviceArray2DView sigma;
  const double *z;
  int Mz;
  double dx;
  double dy;
  double exponent;
  double e_to_a_power;
  int flow_law_mode;
  double A_cold;
  double A_warm;
  double Q_cold;
  double Q_warm;
  double T_crit;
  double gas_const;
  double n;
  double T_melting;
  double beta;
  double c_i;
  double T_0;
  double rho_i;
  double g;
  double p_air;

  __device__ inline void operator()(int i, int j) const {
    double west = 1.0;
    double east = 1.0;
    double south = 1.0;
    double north = 1.0;
    double D_x = 0.0;
    double D_y = 0.0;

    const int mask_c = mask.value(i, j);
    if ((sia_kernels::MaskOps::icy(mask_c) && sia_kernels::MaskOps::ice_free(mask.value(i + 1, j))) ||
        (sia_kernels::MaskOps::ice_free(mask_c) && sia_kernels::MaskOps::icy(mask.value(i + 1, j)))) {
      east = 0.0;
    }
    if ((sia_kernels::MaskOps::icy(mask_c) && sia_kernels::MaskOps::ice_free(mask.value(i - 1, j))) ||
        (sia_kernels::MaskOps::ice_free(mask_c) && sia_kernels::MaskOps::icy(mask.value(i - 1, j)))) {
      west = 0.0;
    }
    if (east + west > 0.0) {
      D_x = 1.0 / (dx * (east + west));
    }

    if ((sia_kernels::MaskOps::icy(mask_c) && sia_kernels::MaskOps::ice_free(mask.value(i, j + 1))) ||
        (sia_kernels::MaskOps::ice_free(mask_c) && sia_kernels::MaskOps::icy(mask.value(i, j + 1)))) {
      north = 0.0;
    }
    if ((sia_kernels::MaskOps::icy(mask_c) && sia_kernels::MaskOps::ice_free(mask.value(i, j - 1))) ||
        (sia_kernels::MaskOps::ice_free(mask_c) && sia_kernels::MaskOps::icy(mask.value(i, j - 1)))) {
      south = 0.0;
    }
    if (north + south > 0.0) {
      D_y = 1.0 / (dy * (north + south));
    }

    const double H = thk(i, j);
    const int ks = k_below_height(z, Mz, H);

    for (int k = 0; k <= ks; ++k) {
      const double depth = H - z[k];
      const double pressure = p_air + rho_i * g * depth;
      const double E = enthalpy(i, j, k);
      const double T = pb_cold_temperature(E, pressure, T_melting, beta, c_i, T_0);
      const double T_pa = T + beta * pressure;
      double A = A_cold;
      double Q = Q_cold;
      if (flow_law_mode == 0) {
        if (T_pa >= T_crit) {
          A = A_warm;
          Q = Q_warm;
        }
      } else if (flow_law_mode == 2) {
        A = A_warm;
        Q = Q_warm;
      }
      const double T_use = T_pa;
      const double softness = pb_softness(T_use, A, Q, gas_const);
      const double hardness = pow(softness, -1.0 / n);

      const double u_x = D_x * (west * (u(i, j, k) - u(i - 1, j, k)) +
                                east * (u(i + 1, j, k) - u(i, j, k)));
      const double u_y = D_y * (south * (u(i, j, k) - u(i, j - 1, k)) +
                                north * (u(i, j + 1, k) - u(i, j, k)));
      const double v_x = D_x * (west * (v(i, j, k) - v(i - 1, j, k)) +
                                east * (v(i + 1, j, k) - v(i, j, k)));
      const double v_y = D_y * (south * (v(i, j, k) - v(i, j - 1, k)) +
                                north * (v(i, j + 1, k) - v(i, j, k)));

      double u_z = 0.0;
      double v_z = 0.0;
      if (k > 0 && k + 1 < Mz) {
        const double dz = z[k + 1] - z[k - 1];
        u_z = (u(i, j, k + 1) - u(i, j, k - 1)) / dz;
        v_z = (v(i, j, k + 1) - v(i, j, k - 1)) / dz;
      } else if (k > 0) {
        const double dz = z[k] - z[k - 1];
        u_z = (u(i, j, k) - u(i, j, k - 1)) / dz;
        v_z = (v(i, j, k) - v(i, j, k - 1)) / dz;
      } else {
        const double dz = z[1] - z[0];
        u_z = (u(i, j, 1) - u(i, j, 0)) / dz;
        v_z = (v(i, j, 1) - v(i, j, 0)) / dz;
      }

      const double D2 = strain_heating_invariant(u_x, u_y, u_z, v_x, v_y, v_z);
      sigma(i, j, k) = 2.0 * e_to_a_power * hardness * pow(D2, exponent);
    }

    for (int k = ks + 1; k < Mz; ++k) {
      sigma(i, j, k) = 0.0;
    }
  }
};

} // namespace

void compute_vertical_velocity(const array::CellType1 &mask,
                               const array::Array3D &u,
                               const array::Array3D &v,
                               const array::Scalar *basal_melt_rate,
                               array::Array3D &result,
                               int xs, int ys, int xm, int ym, int Mz,
                               const double *z,
                               double dx, double dy,
                               int use_upstream_fd) {
  update_vertical_grid(z, Mz);

  CudaMaskView mask_view(mask);
  CudaArrayConstView u_view(u);
  CudaArrayConstView v_view(v);
  CudaArrayWriteView w_view(result);

  if (basal_melt_rate) {
    CudaArrayConstView basal_view(*basal_melt_rate);
    VerticalVelocityBasal functor{mask_view.view, u_view.view, v_view.view, basal_view.view,
                                  w_view.view, g_z_device, Mz, dx, dy, use_upstream_fd};
    launch_2d(functor, xs, ys, xm, ym, 0);
  } else {
    VerticalVelocityNoBasal functor{mask_view.view, u_view.view, v_view.view,
                                    w_view.view, g_z_device, Mz, dx, dy, use_upstream_fd};
    launch_2d(functor, xs, ys, xm, ym, 0);
  }
}

void compute_volumetric_strain_heating(const array::CellType1 &mask,
                                       const array::Array3D &u,
                                       const array::Array3D &v,
                                       const array::Scalar &thickness,
                                       const array::Array3D &enthalpy,
                                       array::Array3D &strain_heating,
                                       int xs, int ys, int xm, int ym, int Mz,
                                       const double *z,
                                       double dx, double dy,
                                       double exponent,
                                       double e_to_a_power,
                                       int flow_law_mode,
                                       double A_cold, double A_warm,
                                       double Q_cold, double Q_warm, double T_crit,
                                       double gas_const, double n,
                                       double T_melting, double beta, double c_i,
                                       double T_0, double rho_i, double g, double p_air) {
  update_vertical_grid(z, Mz);

  CudaMaskView mask_view(mask);
  CudaArrayConstView u_view(u);
  CudaArrayConstView v_view(v);
  CudaArrayConstView thk_view(thickness);
  CudaArrayConstView enth_view(enthalpy);
  CudaArrayWriteView sigma_view(strain_heating);

  StrainHeatingPB functor{mask_view.view, u_view.view, v_view.view, thk_view.view, enth_view.view,
                          sigma_view.view, g_z_device, Mz, dx, dy, exponent, e_to_a_power,
                          flow_law_mode, A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const, n,
                          T_melting, beta, c_i, T_0, rho_i, g, p_air};
  launch_2d(functor, xs, ys, xm, ym, 0);
}

} // namespace cuda
} // namespace stressbalance
} // namespace pism

#else

namespace pism {
namespace stressbalance {
namespace cuda {

void compute_vertical_velocity(const array::CellType1 &, const array::Array3D &,
                               const array::Array3D &, const array::Scalar *,
                               array::Array3D &,
                               int, int, int, int, int,
                               const double *, double, double, int) {}

void compute_volumetric_strain_heating(const array::CellType1 &, const array::Array3D &,
                                       const array::Array3D &, const array::Scalar &,
                                       const array::Array3D &, array::Array3D &,
                                       int, int, int, int, int,
                                       const double *, double, double,
                                       double, double, int,
                                       double, double, double, double, double,
                                       double, double,
                                       double, double, double,
                                       double, double, double, double) {}

} // namespace cuda
} // namespace stressbalance
} // namespace pism

#endif
