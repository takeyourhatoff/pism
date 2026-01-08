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

#include "pism/stressbalance/sia/SIAFD_cuda.hh"

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

constexpr int kBlockX = 16;
constexpr int kBlockY = 8;

double *g_z_device = nullptr;
const double *g_z_host = nullptr;
int g_z_size = 0;
double *g_D_max_device = nullptr;
int *g_high_diffusivity_device = nullptr;

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
  PetscErrorCode ierr = DMDAGetGhostCorners(da, &gxs, &gys, NULL, &gxm, &gym, NULL);
  PISM_CHK(ierr, "DMDAGetGhostCorners");
  int dof = 1;
  ierr = DMDAGetInfo(da, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                     &dof, NULL, NULL, NULL, NULL, NULL);
  PISM_CHK(ierr, "DMDAGetInfo");
  view.gxs = gxs;
  view.gys = gys;
  view.gxm = gxm;
  view.dof = dof;
}

static inline sia_kernels::DeviceArray2DView make_dummy_view(const array::Array &array) {
  sia_kernels::DeviceArray2DView view{};
  fill_layout(view, array);
  view.data = nullptr;
  return view;
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

} // namespace

bool vec_is_cuda(const array::Array &array) {
  PetscBool is_cuda = PETSC_FALSE;
  ::Vec vec = array.vec().get();
  PetscErrorCode ierr = PetscObjectTypeCompare(reinterpret_cast<PetscObject>(vec),
                                               VECSEQCUDA, &is_cuda);
  PISM_CHK(ierr, "PetscObjectTypeCompare");
  if (!is_cuda) {
    ierr = PetscObjectTypeCompare(reinterpret_cast<PetscObject>(vec), VECMPICUDA, &is_cuda);
    PISM_CHK(ierr, "PetscObjectTypeCompare");
  }
  if (!is_cuda) {
    ierr = PetscObjectTypeCompare(reinterpret_cast<PetscObject>(vec), VECCUDA, &is_cuda);
    PISM_CHK(ierr, "PetscObjectTypeCompare");
  }
  return is_cuda == PETSC_TRUE;
}

void update_vertical_grid(const double *z, int Mz) {
  if (Mz <= 0 || z == nullptr) {
    throw RuntimeError::formatted(PISM_ERROR_LOCATION, "invalid vertical grid metadata");
  }
  if (g_z_device == nullptr || g_z_size != Mz) {
    if (g_z_device != nullptr) {
      cudaError_t err = cudaFree(g_z_device);
      check_cuda(err, "cudaFree(g_z_device)");
    }
    cudaError_t err = cudaMalloc(&g_z_device, Mz * sizeof(double));
    check_cuda(err, "cudaMalloc(g_z_device)");
    g_z_size = Mz;
    g_z_host = nullptr;
  }
  if (g_z_host != z) {
    cudaError_t err = cudaMemcpy(g_z_device, z, Mz * sizeof(double), cudaMemcpyHostToDevice);
    check_cuda(err, "cudaMemcpy(g_z_device)");
    g_z_host = z;
  }
}

struct ComputeI {
  sia_kernels::DeviceArray2DConstView thk;
  sia_kernels::DeviceArray2DConstView delta;
  sia_kernels::DeviceArray2DView I;
  const double *z;
  int Mz;
  int o;
  PISM_HOST_DEVICE inline void operator()(int i, int j) const {
    const int oi = 1 - o;
    const int oj = o;
    const double thk_local = 0.5 * (thk(i, j) + thk(i + oi, j + oj));
    const int ks = k_below_height(z, Mz, thk_local);

    I(i, j, 0) = 0.0;
    double I_current = 0.0;
    for (int k = 1; k <= ks; ++k) {
      const double dz = z[k] - z[k - 1];
      I_current += 0.5 * dz * (delta(i, j, k - 1) + delta(i, j, k));
      I(i, j, k) = I_current;
    }
    for (int k = ks + 1; k < Mz; ++k) {
      I(i, j, k) = I_current;
    }
  }
};

struct ComputeHorizontalVelocity3D {
  sia_kernels::DeviceArray2DConstView I0;
  sia_kernels::DeviceArray2DConstView I1;
  sia_kernels::DeviceArray2DConstView h_x;
  sia_kernels::DeviceArray2DConstView h_y;
  sia_kernels::DeviceArray2DConstView sliding;
  sia_kernels::DeviceArray2DView u_out;
  sia_kernels::DeviceArray2DView v_out;
  int Mz;
  PISM_HOST_DEVICE inline void operator()(int i, int j) const {
    const double h_x_w = h_x(i - 1, j, 0);
    const double h_x_e = h_x(i, j, 0);
    const double h_x_n = h_x(i, j, 1);
    const double h_x_s = h_x(i, j - 1, 1);

    const double h_y_w = h_y(i - 1, j, 0);
    const double h_y_e = h_y(i, j, 0);
    const double h_y_n = h_y(i, j, 1);
    const double h_y_s = h_y(i, j - 1, 1);

    const double sliding_u = sliding(i, j, 0);
    const double sliding_v = sliding(i, j, 1);

    for (int k = 0; k < Mz; ++k) {
      const double I_e = I0(i, j, k);
      const double I_w = I0(i - 1, j, k);
      const double I_n = I1(i, j, k);
      const double I_s = I1(i, j - 1, k);
      u_out(i, j, k) = sliding_u - 0.25 * (I_e * h_x_e + I_w * h_x_w +
                                          I_n * h_x_n + I_s * h_x_s);
      v_out(i, j, k) = sliding_v - 0.25 * (I_e * h_y_e + I_w * h_y_w +
                                          I_n * h_y_n + I_s * h_y_s);
    }
  }
};

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

__device__ inline double atomic_max_double(double *address, double val) {
  auto addr_as_ull = reinterpret_cast<unsigned long long int *>(address);
  unsigned long long int old = *addr_as_ull;
  while (true) {
    unsigned long long int assumed = old;
    if (__longlong_as_double(assumed) >= val) {
      break;
    }
    old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(val));
    if (assumed == old) {
      break;
    }
  }
  return __longlong_as_double(old);
}

template <int FlowLawMode, bool FullUpdate, bool ComputeI, bool UseQuadratic>
__device__ inline void compute_diffusivity_orient(
    int i, int j, int o, int oi, int oj,
    double h_x_local, double h_y_local,
    const sia_kernels::DeviceArray2DConstView &thk,
    const sia_kernels::DeviceArray2DConstView &theta,
    const sia_kernels::DeviceArray2DConstView &enthalpy,
    const sia_kernels::DeviceArray2DView &result,
    const sia_kernels::DeviceArray2DView &flux,
    const sia_kernels::DeviceArray2DView &delta,
    const sia_kernels::DeviceArray2DView &I,
    const double *z, int Mz,
    int Mx, int My, int xs, int ys, int xm, int ym,
    int periodic_x, int periodic_y,
    int limit_diffusivity, double D_limit, double e_factor,
    double A_cold, double A_warm,
    double Q_cold, double Q_warm, double T_crit,
    double gas_const, double n_minus_1,
    double T_melting, double beta, double c_i, double T_0,
    double rho_i_g, double p_air,
    double *D_max, int *high_diffusivity_counter) {
  const double thk_local = 0.5 * (thk(i, j) + thk(i + oi, j + oj));

  if (thk_local == 0.0) {
    result(i, j, o) = 0.0;
    if constexpr (FullUpdate) {
      for (int k = 0; k < Mz; ++k) {
        delta(i, j, k) = 0.0;
      }
    }
    if constexpr (ComputeI) {
      for (int k = 0; k < Mz; ++k) {
        I(i, j, k) = 0.0;
      }
    }
    return;
  }

  const int ks = k_below_height(z, Mz, thk_local);
  const double theta_local = 0.5 * (theta(i, j) + theta(i + oi, j + oj));
  const double alpha = sqrt(h_x_local * h_x_local + h_y_local * h_y_local);

  double D = 0.0;
  double delta_prev = 0.0;
  double I_current = 0.0;
  if constexpr (ComputeI) {
    I(i, j, 0) = 0.0;
  }

  for (int k = 0; k <= ks; ++k) {
    const double depth = thk_local - z[k];
    const double pressure = p_air + rho_i_g * depth;
    const double E = 0.5 * (enthalpy(i, j, k) + enthalpy(i + oi, j + oj, k));
    const double stress = alpha * pressure;
    const double T = pb_cold_temperature(E, pressure, T_melting, beta, c_i, T_0);
    double A = A_cold;
    double Q = Q_cold;
    double T_use = T;
    if constexpr (FlowLawMode == 0) {
      const double T_pa = T + beta * pressure;
      T_use = T_pa;
      if (T_pa >= T_crit) {
        A = A_warm;
        Q = Q_warm;
      }
    } else if constexpr (FlowLawMode == 2) {
      A = A_warm;
      Q = Q_warm;
    }
    const double stress_pow = UseQuadratic ? (stress * stress) : pow(stress, n_minus_1);
    const double flow = pb_softness(T_use, A, Q, gas_const) * stress_pow;
    const double delta_k = e_factor * theta_local * 2.0 * pressure * flow;

    if constexpr (FullUpdate) {
      delta(i, j, k) = delta_k;
    }

    if (k >= 1) {
      const double dz = z[k] - z[k - 1];
      D += 0.5 * dz * ((depth + dz) * delta_prev + depth * delta_k);
      if constexpr (ComputeI) {
        I_current += 0.5 * dz * (delta_prev + delta_k);
        I(i, j, k) = I_current;
      }
    }
    delta_prev = delta_k;
  }

  {
    const double dz = thk_local - z[ks];
    D += 0.5 * dz * dz * delta_prev;
  }

  if constexpr (FullUpdate) {
    for (int k = ks + 1; k < Mz; ++k) {
      delta(i, j, k) = 0.0;
    }
  }
  if constexpr (ComputeI) {
    for (int k = ks + 1; k < Mz; ++k) {
      I(i, j, k) = I_current;
    }
  }

  if ((!periodic_x && (i < 0 || i >= Mx - 1)) ||
      (!periodic_y && (j < 0 || j >= My - 1))) {
    D = 0.0;
  }

  if (limit_diffusivity && D >= D_limit) {
    D = D_limit;
    atomicAdd(high_diffusivity_counter, 1);
  }

  result(i, j, o) = D;
  if (i >= xs && i < xs + xm && j >= ys && j < ys + ym) {
    const double slope = (o == 0) ? h_x_local : h_y_local;
    flux(i, j, o) = -D * slope;
  }
  atomic_max_double(D_max, D);
}

template <int FlowLawMode, bool FullUpdate, bool ComputeI, bool UseQuadratic>
struct DiffusivityPBBoth {
  sia_kernels::DeviceArray2DConstView thk;
  sia_kernels::DeviceArray2DConstView theta;
  sia_kernels::DeviceArray2DConstView h_x;
  sia_kernels::DeviceArray2DConstView h_y;
  sia_kernels::DeviceArray2DConstView enthalpy;
  sia_kernels::DeviceArray2DView result;
  sia_kernels::DeviceArray2DView flux;
  sia_kernels::DeviceArray2DView delta0;
  sia_kernels::DeviceArray2DView delta1;
  sia_kernels::DeviceArray2DView I0;
  sia_kernels::DeviceArray2DView I1;
  const double *z;
  int Mz;
  int Mx;
  int My;
  int xs;
  int ys;
  int xm;
  int ym;
  int periodic_x;
  int periodic_y;
  int limit_diffusivity;
  double D_limit;
  double e_factor;
  double A_cold;
  double A_warm;
  double Q_cold;
  double Q_warm;
  double T_crit;
  double gas_const;
  double n_minus_1;
  double T_melting;
  double beta;
  double c_i;
  double T_0;
  double rho_i_g;
  double p_air;
  double *D_max;
  int *high_diffusivity_counter;

  __device__ inline void operator()(int i, int j) const {
    const double h_x_0 = h_x(i, j, 0);
    const double h_y_0 = h_y(i, j, 0);
    compute_diffusivity_orient<FlowLawMode, FullUpdate, ComputeI, UseQuadratic>(
        i, j, 0, 1, 0, h_x_0, h_y_0,
        thk, theta, enthalpy, result, flux, delta0, I0,
        z, Mz, Mx, My, xs, ys, xm, ym,
        periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
        A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const, n_minus_1,
        T_melting, beta, c_i, T_0, rho_i_g, p_air,
        D_max, high_diffusivity_counter);

    const double h_x_1 = h_x(i, j, 1);
    const double h_y_1 = h_y(i, j, 1);
    compute_diffusivity_orient<FlowLawMode, FullUpdate, ComputeI, UseQuadratic>(
        i, j, 1, 0, 1, h_x_1, h_y_1,
        thk, theta, enthalpy, result, flux, delta1, I1,
        z, Mz, Mx, My, xs, ys, xm, ym,
        periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
        A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const, n_minus_1,
        T_melting, beta, c_i, T_0, rho_i_g, p_air,
        D_max, high_diffusivity_counter);
  }
};

template <int FlowLawMode, bool FullUpdate, bool ComputeI, bool UseQuadratic>
struct DiffusivityPBWithGradientBoth {
  sia_kernels::DeviceArray2DConstView thk;
  sia_kernels::DeviceArray2DConstView theta;
  sia_kernels::DeviceArray2DView h_x;
  sia_kernels::DeviceArray2DView h_y;
  sia_kernels::DeviceArray2DConstView surface;
  sia_kernels::DeviceArray2DConstView enthalpy;
  sia_kernels::DeviceArray2DView result;
  sia_kernels::DeviceArray2DView flux;
  sia_kernels::DeviceArray2DView delta0;
  sia_kernels::DeviceArray2DView delta1;
  sia_kernels::DeviceArray2DView I0;
  sia_kernels::DeviceArray2DView I1;
  const double *z;
  int Mz;
  int Mx;
  int My;
  int xs;
  int ys;
  int xm;
  int ym;
  double dx;
  double dy;
  int periodic_x;
  int periodic_y;
  int limit_diffusivity;
  double D_limit;
  double e_factor;
  double A_cold;
  double A_warm;
  double Q_cold;
  double Q_warm;
  double T_crit;
  double gas_const;
  double n_minus_1;
  double T_melting;
  double beta;
  double c_i;
  double T_0;
  double rho_i_g;
  double p_air;
  double *D_max;
  int *high_diffusivity_counter;

  __device__ inline void operator()(int i, int j) const {
    const double h_x_0 = (surface(i + 1, j) - surface(i, j)) / dx;
    const double h_y_0 = (surface(i + 1, j + 1) + surface(i, j + 1) -
                          surface(i + 1, j - 1) - surface(i, j - 1)) / (4.0 * dy);
    h_x(i, j, 0) = h_x_0;
    h_y(i, j, 0) = h_y_0;

    const double h_y_1 = (surface(i, j + 1) - surface(i, j)) / dy;
    const double h_x_1 = (surface(i + 1, j + 1) + surface(i + 1, j) -
                          surface(i - 1, j + 1) - surface(i - 1, j)) / (4.0 * dx);
    h_x(i, j, 1) = h_x_1;
    h_y(i, j, 1) = h_y_1;

    compute_diffusivity_orient<FlowLawMode, FullUpdate, ComputeI, UseQuadratic>(
        i, j, 0, 1, 0, h_x_0, h_y_0,
        thk, theta, enthalpy, result, flux, delta0, I0,
        z, Mz, Mx, My, xs, ys, xm, ym,
        periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
        A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const, n_minus_1,
        T_melting, beta, c_i, T_0, rho_i_g, p_air,
        D_max, high_diffusivity_counter);

    compute_diffusivity_orient<FlowLawMode, FullUpdate, ComputeI, UseQuadratic>(
        i, j, 1, 0, 1, h_x_1, h_y_1,
        thk, theta, enthalpy, result, flux, delta1, I1,
        z, Mz, Mx, My, xs, ys, xm, ym,
        periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
        A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const, n_minus_1,
        T_melting, beta, c_i, T_0, rho_i_g, p_air,
        D_max, high_diffusivity_counter);
  }
};

template <int FlowLawMode, bool FullUpdate, bool ComputeI, bool UseQuadratic>
static inline void launch_diffusivity_pb_no_gradient(
    const sia_kernels::DeviceArray2DConstView &thk,
    const sia_kernels::DeviceArray2DConstView &theta,
    const sia_kernels::DeviceArray2DConstView &h_x,
    const sia_kernels::DeviceArray2DConstView &h_y,
    const sia_kernels::DeviceArray2DConstView &enthalpy,
    const sia_kernels::DeviceArray2DView &result,
    const sia_kernels::DeviceArray2DView &flux,
    const sia_kernels::DeviceArray2DView &delta0,
    const sia_kernels::DeviceArray2DView &delta1,
    const sia_kernels::DeviceArray2DView &I0,
    const sia_kernels::DeviceArray2DView &I1,
    const double *z,
    int Mz, int Mx, int My,
    int xs, int ys, int xm, int ym,
    int periodic_x, int periodic_y,
    int limit_diffusivity,
    double D_limit, double e_factor,
    double A_cold, double A_warm,
    double Q_cold, double Q_warm, double T_crit,
    double gas_const, double n_minus_1,
    double T_melting, double beta, double c_i, double T_0,
    double rho_i_g, double p_air,
    double *D_max, int *high_diffusivity_counter,
    int ghosts) {
  DiffusivityPBBoth<FlowLawMode, FullUpdate, ComputeI, UseQuadratic> functor{
      thk, theta, h_x, h_y, enthalpy, result, flux, delta0, delta1, I0, I1,
      z, Mz, Mx, My, xs, ys, xm, ym,
      periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
      A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
      n_minus_1, T_melting, beta, c_i, T_0,
      rho_i_g, p_air, D_max, high_diffusivity_counter};
  launch_2d(functor, xs, ys, xm, ym, ghosts);
}

template <bool FullUpdate, bool ComputeI>
static inline void dispatch_diffusivity_pb_no_gradient(
    int flow_law_mode,
    const sia_kernels::DeviceArray2DConstView &thk,
    const sia_kernels::DeviceArray2DConstView &theta,
    const sia_kernels::DeviceArray2DConstView &h_x,
    const sia_kernels::DeviceArray2DConstView &h_y,
    const sia_kernels::DeviceArray2DConstView &enthalpy,
    const sia_kernels::DeviceArray2DView &result,
    const sia_kernels::DeviceArray2DView &flux,
    const sia_kernels::DeviceArray2DView &delta0,
    const sia_kernels::DeviceArray2DView &delta1,
    const sia_kernels::DeviceArray2DView &I0,
    const sia_kernels::DeviceArray2DView &I1,
    const double *z,
    int Mz, int Mx, int My,
    int xs, int ys, int xm, int ym,
    int periodic_x, int periodic_y,
    int limit_diffusivity,
    double D_limit, double e_factor,
    double A_cold, double A_warm,
    double Q_cold, double Q_warm, double T_crit,
    double gas_const, double n_minus_1, int use_quadratic,
    double T_melting, double beta, double c_i, double T_0,
    double rho_i_g, double p_air,
    double *D_max, int *high_diffusivity_counter,
    int ghosts) {
  if (use_quadratic) {
    switch (flow_law_mode) {
      case 0:
        launch_diffusivity_pb_no_gradient<0, FullUpdate, ComputeI, true>(
            thk, theta, h_x, h_y, enthalpy, result, flux, delta0, delta1, I0, I1,
            z, Mz, Mx, My, xs, ys, xm, ym,
            periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
            A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
            n_minus_1, T_melting, beta, c_i, T_0,
            rho_i_g, p_air, D_max, high_diffusivity_counter, ghosts);
        break;
      case 1:
        launch_diffusivity_pb_no_gradient<1, FullUpdate, ComputeI, true>(
            thk, theta, h_x, h_y, enthalpy, result, flux, delta0, delta1, I0, I1,
            z, Mz, Mx, My, xs, ys, xm, ym,
            periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
            A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
            n_minus_1, T_melting, beta, c_i, T_0,
            rho_i_g, p_air, D_max, high_diffusivity_counter, ghosts);
        break;
      case 2:
        launch_diffusivity_pb_no_gradient<2, FullUpdate, ComputeI, true>(
            thk, theta, h_x, h_y, enthalpy, result, flux, delta0, delta1, I0, I1,
            z, Mz, Mx, My, xs, ys, xm, ym,
            periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
            A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
            n_minus_1, T_melting, beta, c_i, T_0,
            rho_i_g, p_air, D_max, high_diffusivity_counter, ghosts);
        break;
      default:
        break;
    }
  } else {
    switch (flow_law_mode) {
      case 0:
        launch_diffusivity_pb_no_gradient<0, FullUpdate, ComputeI, false>(
            thk, theta, h_x, h_y, enthalpy, result, flux, delta0, delta1, I0, I1,
            z, Mz, Mx, My, xs, ys, xm, ym,
            periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
            A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
            n_minus_1, T_melting, beta, c_i, T_0,
            rho_i_g, p_air, D_max, high_diffusivity_counter, ghosts);
        break;
      case 1:
        launch_diffusivity_pb_no_gradient<1, FullUpdate, ComputeI, false>(
            thk, theta, h_x, h_y, enthalpy, result, flux, delta0, delta1, I0, I1,
            z, Mz, Mx, My, xs, ys, xm, ym,
            periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
            A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
            n_minus_1, T_melting, beta, c_i, T_0,
            rho_i_g, p_air, D_max, high_diffusivity_counter, ghosts);
        break;
      case 2:
        launch_diffusivity_pb_no_gradient<2, FullUpdate, ComputeI, false>(
            thk, theta, h_x, h_y, enthalpy, result, flux, delta0, delta1, I0, I1,
            z, Mz, Mx, My, xs, ys, xm, ym,
            periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
            A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
            n_minus_1, T_melting, beta, c_i, T_0,
            rho_i_g, p_air, D_max, high_diffusivity_counter, ghosts);
        break;
      default:
        break;
    }
  }
}

template <int FlowLawMode, bool FullUpdate, bool ComputeI, bool UseQuadratic>
static inline void launch_diffusivity_pb_with_gradient(
    const sia_kernels::DeviceArray2DConstView &thk,
    const sia_kernels::DeviceArray2DConstView &theta,
    const sia_kernels::DeviceArray2DView &h_x,
    const sia_kernels::DeviceArray2DView &h_y,
    const sia_kernels::DeviceArray2DConstView &surface,
    const sia_kernels::DeviceArray2DConstView &enthalpy,
    const sia_kernels::DeviceArray2DView &result,
    const sia_kernels::DeviceArray2DView &flux,
    const sia_kernels::DeviceArray2DView &delta0,
    const sia_kernels::DeviceArray2DView &delta1,
    const sia_kernels::DeviceArray2DView &I0,
    const sia_kernels::DeviceArray2DView &I1,
    const double *z,
    int Mz, int Mx, int My,
    int xs, int ys, int xm, int ym,
    double dx, double dy,
    int periodic_x, int periodic_y,
    int limit_diffusivity,
    double D_limit, double e_factor,
    double A_cold, double A_warm,
    double Q_cold, double Q_warm, double T_crit,
    double gas_const, double n_minus_1,
    double T_melting, double beta, double c_i, double T_0,
    double rho_i_g, double p_air,
    double *D_max, int *high_diffusivity_counter,
    int ghosts) {
  DiffusivityPBWithGradientBoth<FlowLawMode, FullUpdate, ComputeI, UseQuadratic> functor{
      thk, theta, h_x, h_y, surface, enthalpy, result, flux, delta0, delta1, I0, I1,
      z, Mz, Mx, My, xs, ys, xm, ym, dx, dy,
      periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
      A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
      n_minus_1, T_melting, beta, c_i, T_0,
      rho_i_g, p_air, D_max, high_diffusivity_counter};
  launch_2d(functor, xs, ys, xm, ym, ghosts);
}

template <bool FullUpdate, bool ComputeI>
static inline void dispatch_diffusivity_pb_with_gradient(
    int flow_law_mode,
    const sia_kernels::DeviceArray2DConstView &thk,
    const sia_kernels::DeviceArray2DConstView &theta,
    const sia_kernels::DeviceArray2DView &h_x,
    const sia_kernels::DeviceArray2DView &h_y,
    const sia_kernels::DeviceArray2DConstView &surface,
    const sia_kernels::DeviceArray2DConstView &enthalpy,
    const sia_kernels::DeviceArray2DView &result,
    const sia_kernels::DeviceArray2DView &flux,
    const sia_kernels::DeviceArray2DView &delta0,
    const sia_kernels::DeviceArray2DView &delta1,
    const sia_kernels::DeviceArray2DView &I0,
    const sia_kernels::DeviceArray2DView &I1,
    const double *z,
    int Mz, int Mx, int My,
    int xs, int ys, int xm, int ym,
    double dx, double dy,
    int periodic_x, int periodic_y,
    int limit_diffusivity,
    double D_limit, double e_factor,
    double A_cold, double A_warm,
    double Q_cold, double Q_warm, double T_crit,
    double gas_const, double n_minus_1, int use_quadratic,
    double T_melting, double beta, double c_i, double T_0,
    double rho_i_g, double p_air,
    double *D_max, int *high_diffusivity_counter,
    int ghosts) {
  if (use_quadratic) {
    switch (flow_law_mode) {
      case 0:
        launch_diffusivity_pb_with_gradient<0, FullUpdate, ComputeI, true>(
            thk, theta, h_x, h_y, surface, enthalpy, result, flux, delta0, delta1, I0, I1,
            z, Mz, Mx, My, xs, ys, xm, ym, dx, dy,
            periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
            A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
            n_minus_1, T_melting, beta, c_i, T_0,
            rho_i_g, p_air, D_max, high_diffusivity_counter, ghosts);
        break;
      case 1:
        launch_diffusivity_pb_with_gradient<1, FullUpdate, ComputeI, true>(
            thk, theta, h_x, h_y, surface, enthalpy, result, flux, delta0, delta1, I0, I1,
            z, Mz, Mx, My, xs, ys, xm, ym, dx, dy,
            periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
            A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
            n_minus_1, T_melting, beta, c_i, T_0,
            rho_i_g, p_air, D_max, high_diffusivity_counter, ghosts);
        break;
      case 2:
        launch_diffusivity_pb_with_gradient<2, FullUpdate, ComputeI, true>(
            thk, theta, h_x, h_y, surface, enthalpy, result, flux, delta0, delta1, I0, I1,
            z, Mz, Mx, My, xs, ys, xm, ym, dx, dy,
            periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
            A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
            n_minus_1, T_melting, beta, c_i, T_0,
            rho_i_g, p_air, D_max, high_diffusivity_counter, ghosts);
        break;
      default:
        break;
    }
  } else {
    switch (flow_law_mode) {
      case 0:
        launch_diffusivity_pb_with_gradient<0, FullUpdate, ComputeI, false>(
            thk, theta, h_x, h_y, surface, enthalpy, result, flux, delta0, delta1, I0, I1,
            z, Mz, Mx, My, xs, ys, xm, ym, dx, dy,
            periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
            A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
            n_minus_1, T_melting, beta, c_i, T_0,
            rho_i_g, p_air, D_max, high_diffusivity_counter, ghosts);
        break;
      case 1:
        launch_diffusivity_pb_with_gradient<1, FullUpdate, ComputeI, false>(
            thk, theta, h_x, h_y, surface, enthalpy, result, flux, delta0, delta1, I0, I1,
            z, Mz, Mx, My, xs, ys, xm, ym, dx, dy,
            periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
            A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
            n_minus_1, T_melting, beta, c_i, T_0,
            rho_i_g, p_air, D_max, high_diffusivity_counter, ghosts);
        break;
      case 2:
        launch_diffusivity_pb_with_gradient<2, FullUpdate, ComputeI, false>(
            thk, theta, h_x, h_y, surface, enthalpy, result, flux, delta0, delta1, I0, I1,
            z, Mz, Mx, My, xs, ys, xm, ym, dx, dy,
            periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
            A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
            n_minus_1, T_melting, beta, c_i, T_0,
            rho_i_g, p_air, D_max, high_diffusivity_counter, ghosts);
        break;
      default:
        break;
    }
  }
}

struct ComputeHorizontalVelocityFromDelta {
  sia_kernels::DeviceArray2DConstView delta0;
  sia_kernels::DeviceArray2DConstView delta1;
  sia_kernels::DeviceArray2DConstView h_x;
  sia_kernels::DeviceArray2DConstView h_y;
  sia_kernels::DeviceArray2DConstView sliding;
  sia_kernels::DeviceArray2DView u_out;
  sia_kernels::DeviceArray2DView v_out;
  const double *z;
  int Mz;
  __device__ inline void operator()(int i, int j) const {
    const double h_x_w = h_x(i - 1, j, 0);
    const double h_x_e = h_x(i, j, 0);
    const double h_x_n = h_x(i, j, 1);
    const double h_x_s = h_x(i, j - 1, 1);

    const double h_y_w = h_y(i - 1, j, 0);
    const double h_y_e = h_y(i, j, 0);
    const double h_y_n = h_y(i, j, 1);
    const double h_y_s = h_y(i, j - 1, 1);

    const double sliding_u = sliding(i, j, 0);
    const double sliding_v = sliding(i, j, 1);

    double I_e = 0.0;
    double I_w = 0.0;
    double I_n = 0.0;
    double I_s = 0.0;

    u_out(i, j, 0) = sliding_u - 0.25 * (I_e * h_x_e + I_w * h_x_w +
                                        I_n * h_x_n + I_s * h_x_s);
    v_out(i, j, 0) = sliding_v - 0.25 * (I_e * h_y_e + I_w * h_y_w +
                                        I_n * h_y_n + I_s * h_y_s);

    for (int k = 1; k < Mz; ++k) {
      const double dz = z[k] - z[k - 1];
      I_e += 0.5 * dz * (delta0(i, j, k - 1) + delta0(i, j, k));
      I_w += 0.5 * dz * (delta0(i - 1, j, k - 1) + delta0(i - 1, j, k));
      I_n += 0.5 * dz * (delta1(i, j, k - 1) + delta1(i, j, k));
      I_s += 0.5 * dz * (delta1(i, j - 1, k - 1) + delta1(i, j - 1, k));

      u_out(i, j, k) = sliding_u - 0.25 * (I_e * h_x_e + I_w * h_x_w +
                                          I_n * h_x_n + I_s * h_x_s);
      v_out(i, j, k) = sliding_v - 0.25 * (I_e * h_y_e + I_w * h_y_w +
                                          I_n * h_y_n + I_s * h_y_s);
    }
  }
};

void surface_gradient_mahaffy(const array::Scalar &ice_surface_elevation,
                              array::Staggered &h_x,
                              array::Staggered &h_y,
                              int xs, int ys, int xm, int ym,
                              double dx, double dy) {
  CudaArrayConstView h(ice_surface_elevation);
  CudaArrayWriteView hx(h_x);
  CudaArrayWriteView hy(h_y);

  sia_kernels::SurfaceGradientMahaffy<sia_kernels::DeviceArray2DConstView,
                                      sia_kernels::DeviceArray2DView>
    functor{h.view, hx.view, hy.view, dx, dy};
  launch_2d(functor, xs, ys, xm, ym, 1);
}

void surface_gradient_eta(const array::Scalar2 &ice_thickness,
                          const array::Scalar2 &bed_elevation,
                          array::Scalar2 &eta,
                          array::Staggered &h_x,
                          array::Staggered &h_y,
                          int xs, int ys, int xm, int ym,
                          int ghosts, double dx, double dy,
                          double invpow, double dinvpow, double etapow) {
  {
    CudaArrayConstView H(ice_thickness);
    CudaArrayWriteView eta_view(eta);
    sia_kernels::EtaFromThickness<sia_kernels::DeviceArray2DConstView,
                                  sia_kernels::DeviceArray2DView>
      functor{H.view, eta_view.view, etapow};
    launch_2d(functor, xs, ys, xm, ym, ghosts);
  }

  {
    CudaArrayConstView bed(bed_elevation);
    CudaArrayConstView eta_in(eta);
    CudaArrayWriteView hx(h_x);
    CudaArrayWriteView hy(h_y);
    sia_kernels::SurfaceGradientEta<sia_kernels::DeviceArray2DConstView,
                                    sia_kernels::DeviceArray2DView>
      functor{bed.view, eta_in.view, hx.view, hy.view, dx, dy, invpow, dinvpow};
    launch_2d(functor, xs, ys, xm, ym, 1);
  }
}

void surface_gradient_haseloff(const array::Scalar2 &ice_surface_elevation,
                               const array::CellType2 &cell_type,
                               array::Scalar1 &w_i,
                               array::Scalar1 &w_j,
                               array::Staggered &h_x,
                               array::Staggered &h_y,
                               int xs, int ys, int xm, int ym,
                               double dx, double dy) {
  {
    CudaArrayConstView h(ice_surface_elevation);
    CudaMaskView mask(cell_type);
    CudaArrayWriteView wi(w_i);
    CudaArrayWriteView wj(w_j);
    CudaArrayWriteView hx(h_x);
    CudaArrayWriteView hy(h_y);

    sia_kernels::SurfaceGradientHaseloffPrimary<sia_kernels::DeviceArray2DConstView,
                                                sia_kernels::DeviceArray2DView,
                                                sia_kernels::DeviceArray2DView,
                                                sia_kernels::DeviceMaskView>
      functor{h.view, hx.view, hy.view, wi.view, wj.view, mask.view, dx, dy};
    launch_2d(functor, xs, ys, xm, ym, 2);
  }

  {
    CudaMaskView mask(cell_type);
    CudaArrayConstView wi(w_i);
    CudaArrayConstView wj(w_j);
    CudaArrayView hx(h_x);
    CudaArrayView hy(h_y);
    sia_kernels::SurfaceGradientHaseloffSecondary<sia_kernels::DeviceArray2DConstView,
                                                  sia_kernels::DeviceArray2DView,
                                                  sia_kernels::DeviceMaskView>
      functor{hx.view, hy.view, wi.view, wj.view, mask.view};
    launch_2d(functor, xs, ys, xm, ym, 1);
  }
}

void diffusive_flux(const array::Staggered &h_x,
                    const array::Staggered &h_y,
                    const array::Staggered &diffusivity,
                    array::Staggered &result,
                    int xs, int ys, int xm, int ym,
                    int o) {
  CudaArrayConstView hx(h_x);
  CudaArrayConstView hy(h_y);
  CudaArrayConstView D(diffusivity);
  CudaArrayWriteView out(result);

  sia_kernels::DiffusiveFlux<sia_kernels::DeviceArray2DConstView,
                             sia_kernels::DeviceArray2DView>
    functor{hx.view, hy.view, D.view, out.view, o};
  launch_2d(functor, xs, ys, xm, ym, 1);
}

void bed_smoother_smoothed_thk(const array::Scalar &usurf,
                               const array::Scalar &thk,
                               const array::Scalar &maxtl,
                               const array::Scalar &topgsmooth,
                               const array::CellType2 &mask,
                               array::Scalar &result,
                               int xs, int ys, int xm, int ym,
                               int ghosts) {
  CudaArrayConstView us(usurf);
  CudaArrayConstView H(thk);
  CudaArrayConstView max_tl(maxtl);
  CudaArrayConstView topg(topgsmooth);
  CudaMaskView m(mask);
  CudaArrayWriteView out(result);

  sia_kernels::BedSmoothThk<sia_kernels::DeviceArray2DConstView,
                            sia_kernels::DeviceArray2DView,
                            sia_kernels::DeviceMaskView>
    functor{us.view, H.view, max_tl.view, topg.view, m.view, out.view};
  launch_2d(functor, xs, ys, xm, ym, ghosts);
}

void bed_smoother_theta(const array::Scalar &usurf,
                        const array::Scalar &maxtl,
                        const array::Scalar &topgsmooth,
                        const array::Scalar &C2,
                        const array::Scalar &C3,
                        const array::Scalar &C4,
                        array::Scalar &result,
                        int xs, int ys, int xm, int ym,
                        int ghosts,
                        double theta_min, double theta_max,
                        double Glen_exponent) {
  CudaArrayConstView us(usurf);
  CudaArrayConstView max_tl(maxtl);
  CudaArrayConstView topg(topgsmooth);
  CudaArrayConstView c2(C2);
  CudaArrayConstView c3(C3);
  CudaArrayConstView c4(C4);
  CudaArrayWriteView out(result);

  sia_kernels::BedSmoothTheta<sia_kernels::DeviceArray2DConstView,
                              sia_kernels::DeviceArray2DView>
    functor{us.view, max_tl.view, topg.view, c2.view, c3.view, c4.view,
            out.view, theta_min, theta_max, Glen_exponent};
  launch_2d(functor, xs, ys, xm, ym, ghosts);
}

void compute_I(const array::Scalar &thk_smooth,
               const array::Array3D &delta,
               array::Array3D &I,
               int xs, int ys, int xm, int ym,
               int ghosts, int o) {
  if (g_z_device == nullptr || g_z_size <= 0) {
    throw RuntimeError::formatted(PISM_ERROR_LOCATION,
                                  "vertical grid metadata is not initialized on device");
  }
  CudaArrayConstView thk(thk_smooth);
  CudaArrayConstView delta_view(delta);
  CudaArrayWriteView I_view(I);
  ComputeI functor{thk.view, delta_view.view, I_view.view, g_z_device, g_z_size, o};
  launch_2d(functor, xs, ys, xm, ym, ghosts);
}

void compute_3d_horizontal_velocity(const array::Array3D &I0,
                                    const array::Array3D &I1,
                                    const array::Staggered &h_x,
                                    const array::Staggered &h_y,
                                    const array::Vector &sliding_velocity,
                                    array::Array3D &u_out,
                                    array::Array3D &v_out,
                                    int xs, int ys, int xm, int ym,
                                    int Mz) {
  CudaArrayConstView I0_view(I0);
  CudaArrayConstView I1_view(I1);
  CudaArrayConstView hx(h_x);
  CudaArrayConstView hy(h_y);
  CudaArrayConstView sliding(sliding_velocity);
  CudaArrayWriteView u(u_out);
  CudaArrayWriteView v(v_out);
  ComputeHorizontalVelocity3D functor{I0_view.view, I1_view.view, hx.view, hy.view,
                                      sliding.view, u.view, v.view, Mz};
  launch_2d(functor, xs, ys, xm, ym, 0);
}

void compute_3d_horizontal_velocity_from_delta(const array::Array3D &delta0,
                                               const array::Array3D &delta1,
                                               const array::Staggered &h_x,
                                               const array::Staggered &h_y,
                                               const array::Vector &sliding_velocity,
                                               array::Array3D &u_out,
                                               array::Array3D &v_out,
                                               int xs, int ys, int xm, int ym,
                                               int Mz) {
  if (g_z_device == nullptr || g_z_size <= 0) {
    throw RuntimeError::formatted(PISM_ERROR_LOCATION,
                                  "vertical grid metadata is not initialized on device");
  }
  CudaArrayConstView d0(delta0);
  CudaArrayConstView d1(delta1);
  CudaArrayConstView hx(h_x);
  CudaArrayConstView hy(h_y);
  CudaArrayConstView sliding(sliding_velocity);
  CudaArrayWriteView u(u_out);
  CudaArrayWriteView v(v_out);
  ComputeHorizontalVelocityFromDelta functor{d0.view, d1.view, hx.view, hy.view,
                                             sliding.view, u.view, v.view,
                                             g_z_device, Mz};
  launch_2d(functor, xs, ys, xm, ym, 0);
}

void compute_diffusivity_pb_both(const array::Scalar &thk_smooth,
                                 const array::Scalar &theta,
                                 const array::Scalar2 &ice_surface_elevation,
                                 array::Staggered &h_x,
                                 array::Staggered &h_y,
                                 const array::Array3D &enthalpy,
                                 array::Staggered &result,
                                 array::Staggered &flux,
                                 array::Array3D &delta0,
                                 array::Array3D &delta1,
                                 array::Array3D &I0,
                                 array::Array3D &I1,
                                 int xs, int ys, int xm, int ym,
                                 int Mx, int My,
                                 int ghosts, int full_update, int compute_I,
                                 int compute_mahaffy, double dx, double dy,
                                 int flow_law_mode,
                                 int periodic_x, int periodic_y,
                                 int limit_diffusivity,
                                 double D_limit, double e_factor,
                                 double A_cold, double A_warm,
                                 double Q_cold, double Q_warm, double T_crit,
                                 double gas_const, double n,
                                 double T_melting, double beta, double c_i,
                                 double T_0, double rho_i, double g, double p_air,
                                 double *D_max, int *high_diffusivity_counter) {
  if (g_z_device == nullptr || g_z_size <= 0) {
    throw RuntimeError::formatted(PISM_ERROR_LOCATION,
                                  "vertical grid metadata is not initialized on device");
  }
  CudaArrayConstView thk_view(thk_smooth);
  CudaArrayConstView theta_view(theta);
  CudaArrayConstView enth(enthalpy);
  CudaArrayWriteView out(result);
  CudaArrayWriteView flux_view(flux);

  if (g_D_max_device == nullptr) {
    cudaError_t err = cudaMalloc(&g_D_max_device, sizeof(double));
    check_cuda(err, "cudaMalloc(g_D_max_device)");
  }
  if (g_high_diffusivity_device == nullptr) {
    cudaError_t err = cudaMalloc(&g_high_diffusivity_device, sizeof(int));
    check_cuda(err, "cudaMalloc(g_high_diffusivity_device)");
  }

  cudaError_t err = cudaMemset(g_D_max_device, 0, sizeof(double));
  check_cuda(err, "cudaMemset(g_D_max_device)");
  err = cudaMemset(g_high_diffusivity_device, 0, sizeof(int));
  check_cuda(err, "cudaMemset(g_high_diffusivity_device)");

  const double n_minus_1 = n - 1.0;
  const int use_quadratic = (n_minus_1 == 2.0) ? 1 : 0;
  const double rho_i_g = rho_i * g;

  auto dispatch = [&](const sia_kernels::DeviceArray2DView &delta0_view,
                      const sia_kernels::DeviceArray2DView &delta1_view,
                      const sia_kernels::DeviceArray2DView &I0_view,
                      const sia_kernels::DeviceArray2DView &I1_view) {
    if (compute_mahaffy) {
      CudaArrayWriteView hx(h_x);
      CudaArrayWriteView hy(h_y);
      CudaArrayConstView surface_view(ice_surface_elevation);
      if (full_update) {
        if (compute_I) {
          dispatch_diffusivity_pb_with_gradient<true, true>(
              flow_law_mode,
              thk_view.view, theta_view.view, hx.view, hy.view, surface_view.view,
              enth.view, out.view, flux_view.view, delta0_view, delta1_view, I0_view, I1_view,
              g_z_device, g_z_size,
              Mx, My, xs, ys, xm, ym, dx, dy,
              periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
              A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
              n_minus_1, use_quadratic, T_melting, beta, c_i, T_0,
              rho_i_g, p_air,
              g_D_max_device, g_high_diffusivity_device, ghosts);
        } else {
          dispatch_diffusivity_pb_with_gradient<true, false>(
              flow_law_mode,
              thk_view.view, theta_view.view, hx.view, hy.view, surface_view.view,
              enth.view, out.view, flux_view.view, delta0_view, delta1_view, I0_view, I1_view,
              g_z_device, g_z_size,
              Mx, My, xs, ys, xm, ym, dx, dy,
              periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
              A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
              n_minus_1, use_quadratic, T_melting, beta, c_i, T_0,
              rho_i_g, p_air,
              g_D_max_device, g_high_diffusivity_device, ghosts);
        }
      } else {
        if (compute_I) {
          dispatch_diffusivity_pb_with_gradient<false, true>(
              flow_law_mode,
              thk_view.view, theta_view.view, hx.view, hy.view, surface_view.view,
              enth.view, out.view, flux_view.view, delta0_view, delta1_view, I0_view, I1_view,
              g_z_device, g_z_size,
              Mx, My, xs, ys, xm, ym, dx, dy,
              periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
              A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
              n_minus_1, use_quadratic, T_melting, beta, c_i, T_0,
              rho_i_g, p_air,
              g_D_max_device, g_high_diffusivity_device, ghosts);
        } else {
          dispatch_diffusivity_pb_with_gradient<false, false>(
              flow_law_mode,
              thk_view.view, theta_view.view, hx.view, hy.view, surface_view.view,
              enth.view, out.view, flux_view.view, delta0_view, delta1_view, I0_view, I1_view,
              g_z_device, g_z_size,
              Mx, My, xs, ys, xm, ym, dx, dy,
              periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
              A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
              n_minus_1, use_quadratic, T_melting, beta, c_i, T_0,
              rho_i_g, p_air,
              g_D_max_device, g_high_diffusivity_device, ghosts);
        }
      }
    } else {
      CudaArrayConstView hx(h_x);
      CudaArrayConstView hy(h_y);
      if (full_update) {
        if (compute_I) {
          dispatch_diffusivity_pb_no_gradient<true, true>(
              flow_law_mode,
              thk_view.view, theta_view.view, hx.view, hy.view, enth.view,
              out.view, flux_view.view, delta0_view, delta1_view, I0_view, I1_view,
              g_z_device, g_z_size,
              Mx, My, xs, ys, xm, ym,
              periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
              A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
              n_minus_1, use_quadratic, T_melting, beta, c_i, T_0,
              rho_i_g, p_air,
              g_D_max_device, g_high_diffusivity_device, ghosts);
        } else {
          dispatch_diffusivity_pb_no_gradient<true, false>(
              flow_law_mode,
              thk_view.view, theta_view.view, hx.view, hy.view, enth.view,
              out.view, flux_view.view, delta0_view, delta1_view, I0_view, I1_view,
              g_z_device, g_z_size,
              Mx, My, xs, ys, xm, ym,
              periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
              A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
              n_minus_1, use_quadratic, T_melting, beta, c_i, T_0,
              rho_i_g, p_air,
              g_D_max_device, g_high_diffusivity_device, ghosts);
        }
      } else {
        if (compute_I) {
          dispatch_diffusivity_pb_no_gradient<false, true>(
              flow_law_mode,
              thk_view.view, theta_view.view, hx.view, hy.view, enth.view,
              out.view, flux_view.view, delta0_view, delta1_view, I0_view, I1_view,
              g_z_device, g_z_size,
              Mx, My, xs, ys, xm, ym,
              periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
              A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
              n_minus_1, use_quadratic, T_melting, beta, c_i, T_0,
              rho_i_g, p_air,
              g_D_max_device, g_high_diffusivity_device, ghosts);
        } else {
          dispatch_diffusivity_pb_no_gradient<false, false>(
              flow_law_mode,
              thk_view.view, theta_view.view, hx.view, hy.view, enth.view,
              out.view, flux_view.view, delta0_view, delta1_view, I0_view, I1_view,
              g_z_device, g_z_size,
              Mx, My, xs, ys, xm, ym,
              periodic_x, periodic_y, limit_diffusivity, D_limit, e_factor,
              A_cold, A_warm, Q_cold, Q_warm, T_crit, gas_const,
              n_minus_1, use_quadratic, T_melting, beta, c_i, T_0,
              rho_i_g, p_air,
              g_D_max_device, g_high_diffusivity_device, ghosts);
        }
      }
    }
  };

  if (full_update || compute_I) {
    CudaArrayWriteView delta0_view(delta0);
    CudaArrayWriteView delta1_view(delta1);
    CudaArrayWriteView I0_view(I0);
    CudaArrayWriteView I1_view(I1);
    dispatch(delta0_view.view, delta1_view.view, I0_view.view, I1_view.view);
  } else {
    const auto delta0_dummy = make_dummy_view(delta0);
    const auto delta1_dummy = make_dummy_view(delta1);
    const auto I0_dummy = make_dummy_view(I0);
    const auto I1_dummy = make_dummy_view(I1);
    dispatch(delta0_dummy, delta1_dummy, I0_dummy, I1_dummy);
  }

  err = cudaMemcpy(D_max, g_D_max_device, sizeof(double), cudaMemcpyDeviceToHost);
  check_cuda(err, "cudaMemcpy(D_max)");
  err = cudaMemcpy(high_diffusivity_counter, g_high_diffusivity_device, sizeof(int), cudaMemcpyDeviceToHost);
  check_cuda(err, "cudaMemcpy(high_diffusivity_counter)");
}

} // namespace cuda
} // namespace stressbalance
} // namespace pism

#else

namespace pism {
namespace stressbalance {
namespace cuda {

bool vec_is_cuda(const array::Array &) {
  return false;
}

void surface_gradient_mahaffy(const array::Scalar &, array::Staggered &, array::Staggered &,
                              int, int, int, int, double, double) {}

void surface_gradient_eta(const array::Scalar2 &, const array::Scalar2 &, array::Scalar2 &,
                          array::Staggered &, array::Staggered &,
                          int, int, int, int, int, double, double,
                          double, double, double) {}

void surface_gradient_haseloff(const array::Scalar2 &, const array::CellType2 &,
                               array::Scalar1 &, array::Scalar1 &,
                               array::Staggered &, array::Staggered &,
                               int, int, int, int, double, double) {}

void diffusive_flux(const array::Staggered &, const array::Staggered &,
                    const array::Staggered &, array::Staggered &,
                    int, int, int, int, int) {}

void bed_smoother_smoothed_thk(const array::Scalar &, const array::Scalar &, const array::Scalar &,
                               const array::Scalar &, const array::CellType2 &, array::Scalar &,
                               int, int, int, int, int) {}

void bed_smoother_theta(const array::Scalar &, const array::Scalar &, const array::Scalar &,
                        const array::Scalar &, const array::Scalar &, const array::Scalar &,
                        array::Scalar &, int, int, int, int, int,
                        double, double, double) {}

void update_vertical_grid(const double *, int) {}

void compute_I(const array::Scalar &, const array::Array3D &, array::Array3D &,
               int, int, int, int, int, int) {}

void compute_3d_horizontal_velocity(const array::Array3D &, const array::Array3D &,
                                    const array::Staggered &, const array::Staggered &,
                                    const array::Vector &, array::Array3D &, array::Array3D &,
                                    int, int, int, int, int) {}

void compute_3d_horizontal_velocity_from_delta(const array::Array3D &, const array::Array3D &,
                                               const array::Staggered &, const array::Staggered &,
                                               const array::Vector &, array::Array3D &,
                                               array::Array3D &,
                                               int, int, int, int, int) {}

void compute_diffusivity_pb_both(const array::Scalar &, const array::Scalar &,
                                 const array::Scalar2 &,
                                 array::Staggered &, array::Staggered &,
                                 const array::Array3D &, array::Staggered &, array::Staggered &,
                                 array::Array3D &, array::Array3D &,
                                 array::Array3D &, array::Array3D &,
                                 int, int, int, int, int, int,
                                 int, int, int,
                                 int, double, double,
                                 int, int, int,
                                 double, double, double, double, double, double, double,
                                 double, double,
                                 double, double, double, double, double,
                                 double, double, double, double, double,
                                 double *, int *) {}

} // namespace cuda
} // namespace stressbalance
} // namespace pism

#endif
