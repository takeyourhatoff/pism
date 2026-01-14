#include "gpism/geometry.h"

#include "gpism/profile.h"

#include <cuda_runtime.h>

namespace gpism {
namespace {

__device__ inline int idx(int i, int j, int gw, int stride) {
  return (j + gw) * stride + (i + gw);
}

__device__ inline bool is_ice_free(int mask) {
  return mask == IceFreeBedrock || mask == IceFreeOcean;
}

__device__ inline bool is_icy(int mask) {
  return mask == GroundedIce || mask == FloatingIce;
}

__device__ inline bool is_grounded(int mask) { return mask == GroundedIce; }

__device__ inline bool is_floating(int mask) { return mask == FloatingIce; }

__device__ inline bool is_ice_free_ocean(int mask) {
  return mask == IceFreeOcean;
}

__device__ inline int weight(bool margin_bc, int M_ij, int M_n, double h_ij,
                             double h_n) {
  if ((is_grounded(M_ij) && is_floating(M_n)) ||
      (is_floating(M_ij) && is_grounded(M_n)) ||
      (is_floating(M_ij) && is_ice_free_ocean(M_n))) {
    return 0;
  }

  if ((is_icy(M_ij) && is_ice_free(M_n) && h_n > h_ij) ||
      (is_ice_free(M_ij) && is_icy(M_n) && h_ij > h_n)) {
    return 0;
  }

  if (margin_bc && ((is_icy(M_ij) && is_ice_free(M_n)) ||
                    (is_ice_free(M_ij) && is_icy(M_n)))) {
    return 0;
  }

  return 1;
}

__device__ inline double diff_uphill(double left, double center, double right) {
  const double d_left = center - left;
  const double d_right = right - center;
  if (d_right * d_left > 0.0) {
    return (d_left < 0.0) ? d_left : d_right;
  }
  return 0.5 * (d_left + d_right);
}

__device__ inline double diff_centered(double left, double /*center*/,
                                       double right) {
  return 0.5 * (right - left);
}

__global__ void cell_type_kernel(int mx, int my, int gw, int stride,
                                 const double* thk, const double* topg,
                                 int* cell_type, double sea_level,
                                 double rho_ice, double rho_water) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int c = idx(i, j, gw, stride);
  const double H = thk[c];
  const double bed = topg[c];
  int mask = IceFreeBedrock;
  if (H > 0.0) {
    const double floatation_thk =
        (sea_level - bed) * (rho_water / rho_ice);
    if (floatation_thk > 0.0 && H <= floatation_thk) {
      mask = FloatingIce;
    } else {
      mask = GroundedIce;
    }
  } else {
    mask = (bed < sea_level) ? IceFreeOcean : IceFreeBedrock;
  }
  cell_type[c] = mask;
}

__global__ void usurf_kernel(int mx, int my, int gw, int stride,
                             const double* thk, const double* topg,
                             const int* cell_type, double sea_level,
                             double rho_ice, double rho_water, double* usurf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int c = idx(i, j, gw, stride);
  const double H = thk[c];
  const double bed = topg[c];
  const int mask = cell_type ? cell_type[c] : GroundedIce;
  if (mask == FloatingIce) {
    usurf[c] = sea_level + (1.0 - (rho_ice / rho_water)) * H;
  } else if (mask == IceFreeOcean) {
    usurf[c] = sea_level;
  } else {
    usurf[c] = bed + H;
  }
}

__global__ void slope_kernel(int mx, int my, int gw, int stride,
                             const double* usurf, double* dhdx, double* dhdy,
                             const int* cell_type, int stride_mask,
                             int surface_gradient_inward, int uphill,
                             int use_cfbc, double inv_dx, double inv_dy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= mx || j >= my) {
    return;
  }
  const int il = (i == 0) ? 0 : (i - 1);
  const int ir = (i == mx - 1) ? (mx - 1) : (i + 1);
  const int jd = (j == 0) ? 0 : (j - 1);
  const int ju = (j == my - 1) ? (my - 1) : (j + 1);
  const int c = idx(i, j, gw, stride);
  const int idx_left = idx(il, j, gw, stride);
  const int idx_right = idx(ir, j, gw, stride);
  const int idx_down = idx(i, jd, gw, stride);
  const int idx_up = idx(i, ju, gw, stride);

  const double h_c = usurf[c];
  const double h_w = usurf[idx_left];
  const double h_e = usurf[idx_right];
  const double h_s = usurf[idx_down];
  const double h_n = usurf[idx_up];

  if (surface_gradient_inward) {
    const double left = (i == 0) ? h_c : h_w;
    const double right = (i == mx - 1) ? h_c : h_e;
    const double down = (j == 0) ? h_c : h_s;
    const double up = (j == my - 1) ? h_c : h_n;
    dhdx[c] = 0.5 * (right - left) * inv_dx;
    dhdy[c] = 0.5 * (up - down) * inv_dy;
    return;
  }

  const int M_c = cell_type ? cell_type[c] : GroundedIce;
  const int M_w = cell_type ? cell_type[idx(il, j, gw, stride_mask)] : M_c;
  const int M_e = cell_type ? cell_type[idx(ir, j, gw, stride_mask)] : M_c;
  const int M_s = cell_type ? cell_type[idx(i, jd, gw, stride_mask)] : M_c;
  const int M_n = cell_type ? cell_type[idx(i, ju, gw, stride_mask)] : M_c;

  const auto diff_grounded = uphill ? diff_uphill : diff_centered;

  double h_x = 0.0;
  {
    const int west = weight(use_cfbc != 0, M_c, M_w, h_c, h_w);
    const int east = weight(use_cfbc != 0, M_c, M_e, h_c, h_e);
    if (east + west == 2 && is_grounded(M_c)) {
      h_x = diff_grounded(h_w, h_c, h_e) * inv_dx;
    } else if (east + west > 0) {
      h_x = (west * (h_c - h_w) + east * (h_e - h_c)) *
            (inv_dx / static_cast<double>(east + west));
      if (is_floating(M_c) &&
          (is_ice_free_ocean(M_e) || is_ice_free_ocean(M_w))) {
        h_x *= 0.5;
      }
    } else {
      h_x = 0.0;
    }
  }

  double h_y = 0.0;
  {
    const int south = weight(use_cfbc != 0, M_c, M_s, h_c, h_s);
    const int north = weight(use_cfbc != 0, M_c, M_n, h_c, h_n);
    if (north + south == 2 && is_grounded(M_c)) {
      h_y = diff_grounded(h_s, h_c, h_n) * inv_dy;
    } else if (north + south > 0) {
      h_y = (south * (h_c - h_s) + north * (h_n - h_c)) *
            (inv_dy / static_cast<double>(north + south));
      if (is_floating(M_c) &&
          (is_ice_free_ocean(M_n) || is_ice_free_ocean(M_s))) {
        h_y *= 0.5;
      }
    } else {
      h_y = 0.0;
    }
  }

  dhdx[c] = h_x;
  dhdy[c] = h_y;
}

}  // namespace

void compute_cell_type_cuda(int mx, int my, int gw, int stride,
                            const double* thk, const double* topg,
                            int* cell_type, double sea_level, double rho_ice,
                            double rho_water) {
  CudaEventTimer timer("geometry_cell_type");
  dim3 block(16, 16);
  dim3 grid_dim((mx + block.x - 1) / block.x,
                (my + block.y - 1) / block.y);
  cell_type_kernel<<<grid_dim, block>>>(mx, my, gw, stride, thk, topg, cell_type,
                                        sea_level, rho_ice, rho_water);
}

void compute_usurf_flotation_cuda(int mx, int my, int gw, int stride,
                                  const double* thk, const double* topg,
                                  const int* cell_type, double sea_level,
                                  double rho_ice, double rho_water,
                                  double* usurf) {
  CudaEventTimer timer("geometry_usurf");
  dim3 block(16, 16);
  dim3 grid_dim((mx + block.x - 1) / block.x,
                (my + block.y - 1) / block.y);
  usurf_kernel<<<grid_dim, block>>>(mx, my, gw, stride, thk, topg, cell_type,
                                    sea_level, rho_ice, rho_water, usurf);
}

void compute_surface_slopes_pism_cuda(
    int mx, int my, int gw, int stride, const double* usurf,
    double* dhdx, double* dhdy, const int* cell_type, int stride_mask,
    int surface_gradient_inward, int uphill, int use_cfbc, double inv_dx,
    double inv_dy) {
  CudaEventTimer timer("geometry_slopes");
  dim3 block(16, 16);
  dim3 grid_dim((mx + block.x - 1) / block.x,
                (my + block.y - 1) / block.y);
  slope_kernel<<<grid_dim, block>>>(
      mx, my, gw, stride, usurf, dhdx, dhdy, cell_type, stride_mask,
      surface_gradient_inward, uphill, use_cfbc, inv_dx, inv_dy);
}

void GeometryDiagnostics::compute_usurf(const Grid2D& grid, const Field2D<double>& thk,
                                        const Field2D<double>& topg,
                                        Field2D<double>& usurf) {
  if (thk.has_device_data() && topg.has_device_data() && usurf.has_device_data()) {
    CudaEventTimer timer("geometry_usurf");
    dim3 block(16, 16);
    dim3 grid_dim((grid.local_mx() + block.x - 1) / block.x,
                  (grid.local_my() + block.y - 1) / block.y);
    usurf_kernel<<<grid_dim, block>>>(grid.local_mx(), grid.local_my(),
                                      thk.ghost_width(), thk.stride(),
                                      thk.device_data(), topg.device_data(),
                                      nullptr, 0.0, 1.0, 1.0,
                                      usurf.device_data());
    return;
  }
  GeometryDiagnostics::compute_usurf_cpu(grid, thk, topg, usurf);
}

void GeometryDiagnostics::compute_surface_slopes(const Grid2D& grid,
                                                 const Field2D<double>& usurf,
                                                 Field2D<double>& dhdx,
                                                 Field2D<double>& dhdy) {
  if (usurf.has_device_data() && dhdx.has_device_data() && dhdy.has_device_data()) {
    CudaEventTimer timer("geometry_slopes");
    dim3 block(16, 16);
    dim3 grid_dim((grid.local_mx() + block.x - 1) / block.x,
                  (grid.local_my() + block.y - 1) / block.y);
    slope_kernel<<<grid_dim, block>>>(grid.local_mx(), grid.local_my(),
                                      usurf.ghost_width(), usurf.stride(),
                                      usurf.device_data(), dhdx.device_data(),
                                      dhdy.device_data(), nullptr, 0, 0, 0, 0,
                                      1.0 / grid.dx(), 1.0 / grid.dy());
    return;
  }
  GeometryDiagnostics::compute_surface_slopes_cpu(grid, usurf, dhdx, dhdy);
}

}  // namespace gpism
