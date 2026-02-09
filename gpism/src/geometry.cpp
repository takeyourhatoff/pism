#include "gpism/geometry.h"

#include "gpism/config.h"
#include <algorithm>
#include <iostream>
#if GPISM_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace gpism {
namespace {

bool is_ice_free(int mask) {
  return mask == IceFreeBedrock || mask == IceFreeOcean;
}

bool is_icy(int mask) {
  return mask == GroundedIce || mask == FloatingIce;
}

bool is_grounded(int mask) { return mask == GroundedIce; }

bool is_floating(int mask) { return mask == FloatingIce; }

bool is_ice_free_ocean(int mask) { return mask == IceFreeOcean; }

int weight(bool margin_bc, int M_ij, int M_n, double h_ij, double h_n) {
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

double diff_uphill(double left, double center, double right) {
  const double d_left = center - left;
  const double d_right = right - center;
  if (d_right * d_left > 0.0) {
    return (d_left < 0.0) ? d_left : d_right;
  }
  return 0.5 * (d_left + d_right);
}

double diff_centered(double left, double /*center*/, double right) {
  return 0.5 * (right - left);
}

}  // namespace

#if GPISM_HAVE_CUDA
void compute_cell_type_cuda(int mx, int my, int gw, int stride,
                            const double* thk, const double* topg,
                            int* cell_type, double sea_level, double rho_ice,
                            double rho_water, double ice_free_thickness_threshold);
void compute_usurf_flotation_cuda(int mx, int my, int gw, int stride,
                                  const double* thk, const double* topg,
                                  const int* cell_type, double sea_level,
                                  double rho_ice, double rho_water,
                                  double* usurf);
void compute_surface_slopes_pism_cuda(
    int mx, int my, int gw, int stride, const double* usurf,
    double* dhdx, double* dhdy, const int* cell_type, int stride_mask,
    int surface_gradient_inward, int uphill, int use_cfbc, int periodic,
    double inv_dx, double inv_dy);
#endif

void compute_cell_type(const Grid2D& grid, const Field2D<double>& thk,
                       const Field2D<double>& topg, double sea_level,
                       double rho_ice, double rho_water,
                       double ice_free_thickness_threshold,
                       Field2D<int>& cell_type) {
#if GPISM_HAVE_CUDA
  if (thk.has_device_data() && topg.has_device_data() &&
      cell_type.has_device_data()) {
    compute_cell_type_cuda(grid.local_mx(), grid.local_my(), thk.ghost_width(),
                           thk.stride(), thk.device_data(),
                           topg.device_data(), cell_type.device_data(),
                           sea_level, rho_ice, rho_water,
                           ice_free_thickness_threshold);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "compute_cell_type_cuda launch failed: "
                << cudaGetErrorString(err) << "\n";
    }
    return;
  }
#endif
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  const double H_thr = std::max(0.0, ice_free_thickness_threshold);
  const double alpha = 1.0 - (rho_ice / rho_water);
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const double H = thk(i, j);
      const double bed = topg(i, j);
      const double hgrounded = bed + H;
      const double hfloating = sea_level + alpha * H;
      const bool is_floating = (hfloating > hgrounded);
      const bool ice_free = (H <= H_thr);

      int mask = IceFreeBedrock;
      if (is_floating) {
        mask = ice_free ? IceFreeOcean : FloatingIce;
      } else {
        mask = ice_free ? IceFreeBedrock : GroundedIce;
      }
      cell_type(i, j) = mask;
    }
  }
}

void compute_usurf_flotation(const Grid2D& grid, const Field2D<double>& thk,
                             const Field2D<double>& topg,
                             const Field2D<int>& cell_type,
                             double sea_level, double rho_ice,
                             double rho_water, Field2D<double>& usurf) {
#if GPISM_HAVE_CUDA
  if (thk.has_device_data() && topg.has_device_data() &&
      usurf.has_device_data() && cell_type.has_device_data()) {
    compute_usurf_flotation_cuda(
        grid.local_mx(), grid.local_my(), thk.ghost_width(), thk.stride(),
        thk.device_data(), topg.device_data(), cell_type.device_data(),
        sea_level, rho_ice, rho_water, usurf.device_data());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "compute_usurf_flotation_cuda launch failed: "
                << cudaGetErrorString(err) << "\n";
    }
    return;
  }
#endif
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  const double flotation_scale = 1.0 - (rho_ice / rho_water);
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const int mask = cell_type(i, j);
      const double H = thk(i, j);
      const double bed = topg(i, j);
      if (mask == FloatingIce || mask == IceFreeOcean) {
        usurf(i, j) = sea_level + flotation_scale * H;
      } else {
        usurf(i, j) = bed + H;
      }
    }
  }
}

void compute_surface_slopes_pism(const Grid2D& grid,
                                 const Field2D<double>& usurf,
                                 const Field2D<int>& cell_type,
                                 Field2D<double>& dhdx,
                                 Field2D<double>& dhdy,
                                 bool surface_gradient_inward,
                                 bool uphill,
                                 bool use_cfbc) {
#if GPISM_HAVE_CUDA
  if (usurf.has_device_data() && dhdx.has_device_data() &&
      dhdy.has_device_data() && cell_type.has_device_data()) {
    const bool periodic = (grid.dims_x() == 1 && grid.dims_y() == 1);
    compute_surface_slopes_pism_cuda(
        grid.local_mx(), grid.local_my(), usurf.ghost_width(), usurf.stride(),
        usurf.device_data(), dhdx.device_data(), dhdy.device_data(),
        cell_type.device_data(), cell_type.stride(),
        surface_gradient_inward ? 1 : 0, uphill ? 1 : 0, use_cfbc ? 1 : 0,
        periodic ? 1 : 0,
        1.0 / grid.dx(), 1.0 / grid.dy());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "compute_surface_slopes_pism_cuda launch failed: "
                << cudaGetErrorString(err) << "\n";
    }
    return;
  }
#endif
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  const double inv_dx = 1.0 / grid.dx();
  const double inv_dy = 1.0 / grid.dy();
  auto diff_grounded = uphill ? diff_uphill : diff_centered;
  const bool periodic =
      (grid.dims_x() == 1 && grid.dims_y() == 1);

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const int il =
          periodic ? ((i == 0) ? (mx - 1) : (i - 1)) : ((i == 0) ? 0 : (i - 1));
      const int ir = periodic ? ((i == mx - 1) ? 0 : (i + 1))
                              : ((i == mx - 1) ? (mx - 1) : (i + 1));
      const int jd =
          periodic ? ((j == 0) ? (my - 1) : (j - 1)) : ((j == 0) ? 0 : (j - 1));
      const int ju = periodic ? ((j == my - 1) ? 0 : (j + 1))
                              : ((j == my - 1) ? (my - 1) : (j + 1));

      const double h_c = usurf(i, j);
      const double h_w = usurf(il, j);
      const double h_e = usurf(ir, j);
      const double h_s = usurf(i, jd);
      const double h_n = usurf(i, ju);

      const int M_c = cell_type(i, j);
      const int M_w = cell_type(il, j);
      const int M_e = cell_type(ir, j);
      const int M_s = cell_type(i, jd);
      const int M_n = cell_type(i, ju);

      if (surface_gradient_inward) {
        dhdx(i, j) = 0.5 * (h_e - h_w) * inv_dx;
        dhdy(i, j) = 0.5 * (h_n - h_s) * inv_dy;
        continue;
      }

      double h_x = 0.0;
      {
        const int west = weight(use_cfbc, M_c, M_w, h_c, h_w);
        const int east = weight(use_cfbc, M_c, M_e, h_c, h_e);
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
        const int south = weight(use_cfbc, M_c, M_s, h_c, h_s);
        const int north = weight(use_cfbc, M_c, M_n, h_c, h_n);
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

      dhdx(i, j) = h_x;
      dhdy(i, j) = h_y;
    }
  }
}

void GeometryDiagnostics::compute_usurf_cpu(const Grid2D& grid,
                                            const Field2D<double>& thk,
                                            const Field2D<double>& topg,
                                            Field2D<double>& usurf) {
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      usurf(i, j) = topg(i, j) + thk(i, j);
    }
  }
}

void GeometryDiagnostics::compute_surface_slopes_cpu(const Grid2D& grid,
                                                     const Field2D<double>& usurf,
                                                     Field2D<double>& dhdx,
                                                     Field2D<double>& dhdy) {
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  const double inv_dx = 1.0 / grid.dx();
  const double inv_dy = 1.0 / grid.dy();

  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      const double left = (i == 0) ? usurf(i, j) : usurf(i - 1, j);
      const double right = (i == mx - 1) ? usurf(i, j) : usurf(i + 1, j);
      const double down = (j == 0) ? usurf(i, j) : usurf(i, j - 1);
      const double up = (j == my - 1) ? usurf(i, j) : usurf(i, j + 1);
      dhdx(i, j) = 0.5 * (right - left) * inv_dx;
      dhdy(i, j) = 0.5 * (up - down) * inv_dy;
    }
  }
}

#if !GPISM_HAVE_CUDA
void GeometryDiagnostics::compute_usurf(const Grid2D& grid, const Field2D<double>& thk,
                                        const Field2D<double>& topg,
                                        Field2D<double>& usurf) {
  compute_usurf_cpu(grid, thk, topg, usurf);
}

void GeometryDiagnostics::compute_surface_slopes(const Grid2D& grid,
                                                 const Field2D<double>& usurf,
                                                 Field2D<double>& dhdx,
                                                 Field2D<double>& dhdy) {
  compute_surface_slopes_cpu(grid, usurf, dhdx, dhdy);
}
#endif

}  // namespace gpism
