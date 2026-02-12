#include "gpism/geometry.h"

#include <algorithm>
#include <stdexcept>

#include <cuda_runtime.h>

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

void compute_cell_type(const Grid2D& grid, const Field2D<double>& thk,
                       const Field2D<double>& topg, double sea_level,
                       double rho_ice, double rho_water,
                       double ice_free_thickness_threshold,
                       Field2D<int>& cell_type) {
  if (!(thk.has_device_data() && topg.has_device_data() &&
        cell_type.has_device_data())) {
    throw std::runtime_error(
        "compute_cell_type requires device-resident thk/topg/cell_type");
  }
  compute_cell_type_cuda(grid.local_mx(), grid.local_my(), thk.ghost_width(),
                         thk.stride(), thk.device_data(), topg.device_data(),
                         cell_type.device_data(), sea_level, rho_ice, rho_water,
                         ice_free_thickness_threshold);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("compute_cell_type_cuda launch failed: ") +
                             cudaGetErrorString(err));
  }
}

void compute_usurf_flotation(const Grid2D& grid, const Field2D<double>& thk,
                             const Field2D<double>& topg,
                             const Field2D<int>& cell_type,
                             double sea_level, double rho_ice,
                             double rho_water, Field2D<double>& usurf) {
  if (!(thk.has_device_data() && topg.has_device_data() &&
        usurf.has_device_data() && cell_type.has_device_data())) {
    throw std::runtime_error(
        "compute_usurf_flotation requires device-resident "
        "thk/topg/cell_type/usurf");
  }
  compute_usurf_flotation_cuda(
      grid.local_mx(), grid.local_my(), thk.ghost_width(), thk.stride(),
      thk.device_data(), topg.device_data(), cell_type.device_data(), sea_level,
      rho_ice, rho_water, usurf.device_data());
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("compute_usurf_flotation_cuda launch failed: ") +
        cudaGetErrorString(err));
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
  if (!(usurf.has_device_data() && dhdx.has_device_data() &&
        dhdy.has_device_data() && cell_type.has_device_data())) {
    throw std::runtime_error(
        "compute_surface_slopes_pism requires device-resident "
        "usurf/cell_type/dhdx/dhdy");
  }
  const bool periodic = (grid.dims_x() == 1 && grid.dims_y() == 1);
  compute_surface_slopes_pism_cuda(
      grid.local_mx(), grid.local_my(), usurf.ghost_width(), usurf.stride(),
      usurf.device_data(), dhdx.device_data(), dhdy.device_data(),
      cell_type.device_data(), cell_type.stride(),
      surface_gradient_inward ? 1 : 0, uphill ? 1 : 0, use_cfbc ? 1 : 0,
      periodic ? 1 : 0, 1.0 / grid.dx(), 1.0 / grid.dy());
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("compute_surface_slopes_pism_cuda launch failed: ") +
        cudaGetErrorString(err));
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

}  // namespace gpism
